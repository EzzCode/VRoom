"""COLMAP point-cloud initialization for per-object scratch training.

Reads the scene's labeled COLMAP sparse reconstruction (points3D_labeled.ply
or similar), selects points belonging to the target object, optionally
upsamples via neighbor interpolation, and returns a PointCloudSample that
seeds the fresh per-object Gaussian model.
"""

import json
import logging
from pathlib import Path
from typing import cast, Any

import cv2
import numpy as np

from gstrain.vroom_core.utilities.utils.utils import PointCloudSample

logger = logging.getLogger(__name__)

_VROOM_ROOT = Path(__file__).resolve().parents[2]



# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _resolve_path(path_value, *, base_dir):
    p = Path(path_value)
    if p.is_absolute():
        return p
    for candidate in (base_dir / p, _VROOM_ROOT / p, Path.cwd() / p):
        if candidate.exists():
            return candidate
    return base_dir / p


def _read_source_path(model_path):
    model_path = Path(model_path)
    config_json_path = model_path / "config.json"

    if not config_json_path.exists():
        return None

    from gstrain.vroom_core.config import load_vroom_config
    try:
        _, model_params, _, _ = load_vroom_config(config_json_path)
        source_path = model_params.get("source_path")
        if source_path:
            return _resolve_path(source_path, base_dir=model_path)
    except Exception:
        pass

    return None


def _read_extraction_manifest(extraction_index_path):
    """Return manifest_dict or None on any failure."""
    if extraction_index_path is None:
        return None
    path = Path(extraction_index_path)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning("Could not read extraction manifest from %s: %s", path, exc)
        return None


# ---------------------------------------------------------------------------
# PLY loading
# ---------------------------------------------------------------------------

def _candidate_labeled_plys(source_path):
    return [
        source_path / "sparse" / "0" / "points3D_corr.ply",
        source_path / "vote_output" / "points3D_labeled.ply",
        source_path / "sparse" / "0" / "points3D_labeled.ply",
        source_path / "sparse" / "0" / "points3D_deva.ply",
        source_path / "sparse" / "points3D_corr.ply",
        source_path / "sparse" / "points3D_labeled.ply",
        source_path / "sparse" / "points3D_deva.ply",
    ]


def _load_labeled_ply(path):
    """Return (xyz float32, rgb float32, labels int64) from a labeled PLY file."""
    try:
        from plyfile import PlyData
    except ImportError as exc:
        raise RuntimeError("plyfile is required to read labeled COLMAP PLY files") from exc

    ply    = PlyData.read(str(path))
    vertex = ply["vertex"].data
    names  = set(vertex.dtype.names or [])
    missing = sorted({"x", "y", "z", "label"} - names)
    if missing:
        raise ValueError(f"{path} is missing required PLY properties: {missing}")

    xyz    = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float32)
    if {"red", "green", "blue"}.issubset(names):
        rgb = np.stack([vertex["red"], vertex["green"], vertex["blue"]], axis=1).astype(np.float32) / 255.0
    else:
        rgb = np.full((xyz.shape[0], 3), 0.8, np.float32)
    labels = np.asarray(vertex["label"]).reshape(-1).astype(np.int64)
    return xyz, rgb, labels


# ---------------------------------------------------------------------------
# Point-cloud manipulation
# ---------------------------------------------------------------------------

def _filter_scope_aabb(xyz, colors, *, scope, min_keep):
    """Crop to object AABB + 20% padding. Returns (xyz, colors, was_filtered)."""
    if scope is None or xyz.size == 0:
        return xyz, colors, False
    aabb_min = np.asarray(scope.aabb_min, np.float32)
    aabb_max = np.asarray(scope.aabb_max, np.float32)
    pad  = 0.20 * np.maximum(aabb_max - aabb_min, 1e-5)
    keep = np.all((xyz >= (aabb_min - pad)) & (xyz <= (aabb_max + pad)), axis=1)
    if int(keep.sum()) >= int(min_keep):
        return xyz[keep], colors[keep], True
    return xyz, colors, False


def _upsample_from_colmap_neighbors(xyz, colors, *, target_points):
    """Interpolate between nearest neighbors until target_points is reached."""
    n = int(xyz.shape[0])
    base_meta = {
        "colmap_upsampled": False,
        "colmap_upsample_target_points": int(target_points),
        "colmap_upsample_source_points": n,
        "colmap_upsample_added_points": 0,
    }
    if n <= 1 or n >= int(target_points):
        return xyz, colors, base_meta

    d2 = ((xyz[:, None, :] - xyz[None, :, :]) ** 2).sum(axis=2)
    np.fill_diagonal(d2, np.inf)
    k         = min(8, max(1, n - 1))
    nbr_idx   = np.argpartition(d2, kth=k - 1, axis=1)[:, :k]
    nn        = np.sqrt(np.min(d2, axis=1))
    finite_nn = nn[np.isfinite(nn)]
    median_nn = float(np.median(finite_nn)) if finite_nn.size else 0.01
    noise_sig = max(median_nn * 0.08, 1e-5)

    n_extra  = int(target_points) - n
    rng      = np.random.default_rng(0)
    base     = rng.integers(0, n, size=n_extra)
    pick     = rng.integers(0, k, size=n_extra)
    nbr      = nbr_idx[base, pick]
    alpha    = rng.random((n_extra, 1), dtype=np.float32)

    extra_xyz    = (1.0 - alpha) * xyz[base] + alpha * xyz[nbr]
    extra_xyz   += rng.normal(0.0, noise_sig, size=extra_xyz.shape)
    extra_colors = (1.0 - alpha) * colors[base] + alpha * colors[nbr]

    xyz_out    = np.concatenate([xyz,    extra_xyz],    axis=0)
    colors_out = np.concatenate([colors, extra_colors], axis=0)
    return xyz_out, colors_out, {
        "colmap_upsampled": True,
        "colmap_upsample_target_points": int(target_points),
        "colmap_upsample_source_points": n,
        "colmap_upsample_added_points": n_extra,
        "colmap_upsample_median_nn": median_nn,
        "colmap_upsample_noise_sigma": noise_sig,
    }


# ---------------------------------------------------------------------------
# Label scoring via extraction mask projection
# ---------------------------------------------------------------------------

def _project_points(points, cam, mask_shape):
    """Project world points through camera; return (u_int, v_int, valid_bool)."""
    eps = 1e-4
    R = np.asarray(cam["R"], np.float32)
    T = np.asarray(cam["T"], np.float32).reshape(1, 3)
    K = np.asarray(cam["K"], np.float32)
    height, width   = int(mask_shape[0]), int(mask_shape[1])
    cam_h = int(cam.get("height", height))
    cam_w = int(cam.get("width",  width))

    pts_c = points @ R.T + T
    z = pts_c[:, 2]
    u = K[0, 0] * pts_c[:, 0] / np.maximum(z, 1e-8) + K[0, 2]
    v = K[1, 1] * pts_c[:, 1] / np.maximum(z, 1e-8) + K[1, 2]

    if cam_w > 0 and cam_h > 0 and (cam_w != width or cam_h != height):
        u = u * (float(width)  / float(cam_w))
        v = v * (float(height) / float(cam_h))

    ui = np.rint(u).astype(np.int64)
    vi = np.rint(v).astype(np.int64)
    valid = (z > eps) & (ui >= 0) & (ui < width) & (vi >= 0) & (vi < height)
    return ui, vi, valid


def _score_labels_against_extraction(xyz_all, labels_all, *, scope,
                                      extraction_index_path, min_points):
    """Return per-label projection score dict by comparing labels to extraction masks."""
    manifest = _read_extraction_manifest(extraction_index_path)
    if manifest is None or scope is None:
        return {}

    manifest_dir = (Path(extraction_index_path).parent
                    if extraction_index_path is not None else _VROOM_ROOT)

    candidates = [
        int(lb) for lb in np.unique(labels_all)
        if int(lb) != 0 and int((labels_all == int(lb)).sum()) >= int(min_points)
    ]
    scores = {
        lb: {
            "label_count": int((labels_all == lb).sum()),
            "projected_votes": 0,
            "inside_votes": 0,
            "mask_frames_seen": 0,
            "inside_fraction": 0.0,
            "projection_score": 0.0,
        }
        for lb in candidates
    }
    if not scores:
        return {}

    for frame in manifest.get("frames", []):
        try:
            cam_index = int(frame["cam_index"])
        except Exception:
            continue
        if cam_index < 0 or cam_index >= len(scope.cameras):
            continue

        rgba_value = frame.get("rgba_path")
        if not rgba_value:
            continue
        mask_path = _resolve_path(rgba_value, base_dir=manifest_dir)
        if not mask_path.exists():
            continue

        img_mat = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if img_mat is None:
            continue
        img = np.asarray(img_mat)
        if img.ndim == 3 and img.shape[2] == 4:
            mask = cast(Any, img)[..., 3] > 127
        elif img.ndim == 3:
            mask = img.mean(axis=2) > 127
        else:
            mask = img > 127

        if int(mask.sum()) < 64:
            continue

        ui, vi, valid = _project_points(xyz_all, scope.cameras[cam_index], mask.shape[:2])
        if not valid.any():
            continue

        inside_all = np.zeros(labels_all.shape[0], bool)
        inside_all[valid] = mask[vi[valid], ui[valid]]

        for lb in candidates:
            lb_mask   = labels_all == lb
            projected = int(np.logical_and(lb_mask, valid).sum())
            if projected <= 0:
                continue
            inside = int(np.logical_and(lb_mask, inside_all).sum())
            scores[lb]["projected_votes"] += projected
            scores[lb]["inside_votes"]    += inside
            scores[lb]["mask_frames_seen"] += 1

    for lb, s in scores.items():
        proj = s["projected_votes"]
        ins  = s["inside_votes"]
        frac = float(ins) / max(float(proj), 1.0)
        s["inside_fraction"]   = frac
        s["projection_score"]  = float(frac * np.log1p(float(ins)))

    return scores


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_colmap_object_point_cloud(*, model_path, object_id, scope,
                                    extraction_index_path=None,
                                    max_points=20000, min_points=16,
                                    target_points=8000):
    """Load object seed points from the scene COLMAP labeled PLY file.

    Parameters
    ----------
    model_path             : path to the trained gstrain model directory
    object_id              : integer label id of the target object
    scope                  : ObjectScope from scene_analysis.py
    extraction_index_path  : path to extraction_index.json (optional; improves
                             label selection when multiple labels are present)
    max_points             : hard cap on returned point count
    min_points             : minimum points for a COLMAP label to be usable
    target_points          : desired point count after neighbour upsampling

    Returns
    -------
    (PointCloudSample, metadata_dict)
    """
    source_path = _read_source_path(model_path)
    if source_path is None:
        raise FileNotFoundError(
            f"Could not determine source_path from config.json in {model_path}"
        )

    ply_path = next(
        (p for p in _candidate_labeled_plys(source_path) if p.exists()), None
    )
    if ply_path is None:
        searched = "\n".join(str(p) for p in _candidate_labeled_plys(source_path))
        raise FileNotFoundError(f"No labeled COLMAP PLY found. Searched:\n{searched}")

    xyz_all, colors_all, labels_all = _load_labeled_ply(ply_path)
    finite   = np.isfinite(xyz_all).all(axis=1)
    xyz_all, colors_all, labels_all = (
        xyz_all[finite], colors_all[finite], labels_all[finite]
    )

    label_priority = [int(object_id)]

    label_counts = {
        int(lb): int((labels_all == lb).sum()) for lb in np.unique(labels_all)
    }
    proj_scores = _score_labels_against_extraction(
        xyz_all, labels_all,
        scope=scope,
        extraction_index_path=extraction_index_path,
        min_points=int(min_points),
    )
    scored = [
        (lb, data) for lb, data in proj_scores.items()
        if int(data.get("inside_votes", 0)) >= max(5, int(min_points) // 2)
    ]

    chosen_label = None
    if scored:
        chosen_label = max(
            scored,
            key=lambda item: (
                float(item[1].get("projection_score", 0.0)),
                float(item[1].get("inside_fraction", 0.0)),
                int(item[1].get("inside_votes", 0)),
            ),
        )[0]
    if chosen_label is None:
        chosen_label = next(
            (lb for lb in label_priority if label_counts.get(lb, 0) >= int(min_points)),
            None,
        )
    if chosen_label is None:
        positive = {
            lb: cnt for lb, cnt in label_counts.items()
            if lb != 0 and cnt >= int(min_points)
        }
        if not positive:
            raise RuntimeError(
                f"Labeled COLMAP PLY {ply_path} has no usable object labels; "
                f"counts={label_counts}"
            )
        chosen_label = max(positive.items(), key=lambda item: item[1])[0]


    lb_mask = labels_all == int(chosen_label)
    xyz     = xyz_all[lb_mask]
    colors  = colors_all[lb_mask]

    if xyz.shape[0] < int(min_points):
        raise RuntimeError(
            f"COLMAP label {chosen_label} has only {xyz.shape[0]} points; "
            f"need at least {min_points}."
        )

    min_keep = max(int(min_points), min(int(xyz.shape[0]), 32))
    n_selected = int(xyz.shape[0])
    xyz, colors, aabb_filtered = _filter_scope_aabb(
        xyz, colors, scope=scope, min_keep=min_keep
    )

    actual_target = min(int(max_points), max(int(target_points), int(xyz.shape[0])))
    xyz, colors, upsample_meta = _upsample_from_colmap_neighbors(
        xyz, colors, target_points=actual_target
    )

    if xyz.shape[0] > int(max_points):
        rng     = np.random.default_rng(0)
        keep    = rng.choice(xyz.shape[0], size=int(max_points), replace=False)
        xyz     = xyz[keep]
        colors  = colors[keep]

    normals   = np.zeros_like(xyz, np.float32)
    label_ids = np.full((xyz.shape[0],), int(object_id), np.uint8)
    pcd = PointCloudSample(
        points=xyz,
        colors=colors,
        normals=normals,
        label_ids=label_ids,
    )
    metadata = {
        "init_source": "colmap_labeled_ply",
        "source_path": str(source_path),
        "colmap_ply_path": str(ply_path),
        "colmap_label_used": int(chosen_label),
        "colmap_label_counts": label_counts,
        "colmap_label_projection_scores": proj_scores,
        "aabb_filtered": aabb_filtered,
        "n_colmap_selected_points": n_selected,
        "n_colmap_seed_points": xyz.shape[0],
        "label_ids_written_as_object_id": int(object_id),
        **upsample_meta,
    }
    logger.info(
        "Scratch init obj %d: %d seed points from %s (label=%d, aabb_filtered=%s).",
        int(object_id), xyz.shape[0], ply_path, int(chosen_label), aabb_filtered,
    )
    return pcd, metadata
