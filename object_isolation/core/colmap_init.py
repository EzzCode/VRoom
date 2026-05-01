"""COLMAP point-cloud initialization for object scratch training."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml

logger = logging.getLogger(__name__)

_VROOM_ROOT = Path(__file__).resolve().parents[2]
_OBJECTGS_DIR = _VROOM_ROOT / "temp_deps" / "ObjectGS"
if str(_OBJECTGS_DIR) not in sys.path:
    sys.path.insert(0, str(_OBJECTGS_DIR))

from utils.graphics_utils import BasicPointCloud  # noqa: E402


def _resolve_path(path_value: str | Path, *, base_dir: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    for candidate in (base_dir / path, _VROOM_ROOT / path, Path.cwd() / path):
        if candidate.exists():
            return candidate
    return base_dir / path


def _read_source_path(model_path: str | Path) -> Optional[Path]:
    model_path = Path(model_path)
    config_path = model_path / "config.yaml"
    if not config_path.exists():
        return None
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader) or {}
    model_params = cfg.get("model_params", {}) or {}
    source_path = model_params.get("source_path")
    if not source_path:
        return None
    return _resolve_path(source_path, base_dir=model_path)


def _read_phase3_module_label(extraction_index_path: str | Path | None) -> Optional[int]:
    if extraction_index_path is None:
        return None
    path = Path(extraction_index_path)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        value = manifest.get("module1_obj_id")
        return int(value) if value is not None else None
    except Exception as exc:
        logger.warning("Could not read Phase-3 module label from %s: %s", path, exc)
        return None


def _read_phase3_manifest(extraction_index_path: str | Path | None) -> tuple[Optional[dict], Optional[int]]:
    if extraction_index_path is None:
        return None, None
    path = Path(extraction_index_path)
    if not path.exists():
        return None, None
    try:
        with open(path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        value = manifest.get("module1_obj_id")
        module_label = int(value) if value is not None else None
        return manifest, module_label
    except Exception as exc:
        logger.warning("Could not read Phase-3 manifest from %s: %s", path, exc)
        return None, None


def _candidate_labeled_plys(source_path: Path) -> list[Path]:
    return [
        source_path / "sparse" / "0" / "points3D_corr.ply",
        source_path / "vote_output" / "points3D_labeled.ply",
        source_path / "sparse" / "0" / "points3D_labeled.ply",
        source_path / "sparse" / "0" / "points3D_deva.ply",
        source_path / "sparse" / "points3D_corr.ply",
        source_path / "sparse" / "points3D_labeled.ply",
        source_path / "sparse" / "points3D_deva.ply",
    ]


def _load_labeled_ply(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        from plyfile import PlyData
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("plyfile is required to read labeled COLMAP PLY files") from exc

    ply = PlyData.read(str(path))
    vertex = ply["vertex"].data
    names = set(vertex.dtype.names or [])
    required = {"x", "y", "z", "label"}
    missing = sorted(required - names)
    if missing:
        raise ValueError(f"{path} is missing required PLY properties: {missing}")

    xyz = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float32)
    if {"red", "green", "blue"}.issubset(names):
        rgb = np.stack([vertex["red"], vertex["green"], vertex["blue"]], axis=1).astype(np.float32) / 255.0
    else:
        rgb = np.full((xyz.shape[0], 3), 0.8, dtype=np.float32)
    labels = np.asarray(vertex["label"]).reshape(-1).astype(np.int64)
    return xyz, rgb, labels


def _filter_scope_aabb(
    xyz: np.ndarray,
    colors: np.ndarray,
    *,
    scope,
    min_keep: int,
) -> tuple[np.ndarray, np.ndarray, bool]:
    if scope is None or xyz.size == 0:
        return xyz, colors, False
    aabb_min = np.asarray(scope.aabb_min_W, dtype=np.float32)
    aabb_max = np.asarray(scope.aabb_max_W, dtype=np.float32)
    extent = np.maximum(aabb_max - aabb_min, 1e-5)
    pad = 0.20 * extent
    keep = np.all((xyz >= (aabb_min - pad)) & (xyz <= (aabb_max + pad)), axis=1)
    if int(keep.sum()) >= int(min_keep):
        return xyz[keep], colors[keep], True
    return xyz, colors, False


def _upsample_from_colmap_neighbors(
    xyz: np.ndarray,
    colors: np.ndarray,
    *,
    target_points: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    n_points = int(xyz.shape[0])
    if n_points <= 1 or n_points >= int(target_points):
        return xyz, colors, {
            "colmap_upsampled": False,
            "colmap_upsample_target_points": int(target_points),
            "colmap_upsample_source_points": int(n_points),
            "colmap_upsample_added_points": 0,
        }

    d2 = ((xyz[:, None, :] - xyz[None, :, :]) ** 2).sum(axis=2)
    np.fill_diagonal(d2, np.inf)
    k = min(8, max(1, n_points - 1))
    neighbor_idx = np.argpartition(d2, kth=k - 1, axis=1)[:, :k]
    nn = np.sqrt(np.min(d2, axis=1))
    finite_nn = nn[np.isfinite(nn)]
    median_nn = float(np.median(finite_nn)) if finite_nn.size else 0.01
    noise_sigma = max(median_nn * 0.08, 1e-5)

    n_extra = int(target_points) - n_points
    rng = np.random.default_rng(0)
    base_idx = rng.integers(0, n_points, size=n_extra)
    pick_idx = rng.integers(0, k, size=n_extra)
    nbr_idx = neighbor_idx[base_idx, pick_idx]
    alpha = rng.random((n_extra, 1), dtype=np.float32)
    extra_xyz = (1.0 - alpha) * xyz[base_idx] + alpha * xyz[nbr_idx]
    extra_xyz = extra_xyz + rng.normal(0.0, noise_sigma, size=extra_xyz.shape).astype(np.float32)
    extra_colors = (1.0 - alpha) * colors[base_idx] + alpha * colors[nbr_idx]

    xyz_out = np.concatenate([xyz, extra_xyz.astype(np.float32)], axis=0)
    colors_out = np.concatenate([colors, extra_colors.astype(np.float32)], axis=0)
    return xyz_out, colors_out, {
        "colmap_upsampled": True,
        "colmap_upsample_target_points": int(target_points),
        "colmap_upsample_source_points": int(n_points),
        "colmap_upsample_added_points": int(n_extra),
        "colmap_upsample_median_nn": float(median_nn),
        "colmap_upsample_noise_sigma": float(noise_sigma),
    }


def _project_points(points: np.ndarray, cam: dict, mask_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    R = np.asarray(cam["R"], dtype=np.float32)
    T = np.asarray(cam["T"], dtype=np.float32).reshape(1, 3)
    K = np.asarray(cam["K"], dtype=np.float32)
    height, width = int(mask_shape[0]), int(mask_shape[1])
    cam_height = int(cam.get("height", height))
    cam_width = int(cam.get("width", width))

    cam_pts = points @ R.T + T
    z = cam_pts[:, 2]
    u = K[0, 0] * cam_pts[:, 0] / np.maximum(z, 1e-8) + K[0, 2]
    v = K[1, 1] * cam_pts[:, 1] / np.maximum(z, 1e-8) + K[1, 2]
    if cam_width > 0 and cam_height > 0 and (cam_width != width or cam_height != height):
        u = u * (float(width) / float(cam_width))
        v = v * (float(height) / float(cam_height))
    ui = np.rint(u).astype(np.int64)
    vi = np.rint(v).astype(np.int64)
    valid = (z > 1e-4) & (ui >= 0) & (ui < width) & (vi >= 0) & (vi < height)
    return ui, vi, valid


def _score_colmap_labels_against_phase3(
    xyz_all: np.ndarray,
    labels_all: np.ndarray,
    *,
    scope,
    extraction_index_path: str | Path | None,
    min_points: int,
) -> dict[int, dict]:
    manifest, _module_label = _read_phase3_manifest(extraction_index_path)
    if manifest is None or scope is None:
        return {}

    manifest_dir = Path(extraction_index_path).parent if extraction_index_path is not None else _VROOM_ROOT
    candidate_labels = [
        int(label) for label in np.unique(labels_all)
        if int(label) != 0 and int((labels_all == int(label)).sum()) >= int(min_points)
    ]
    scores = {
        int(label): {
            "label_count": int((labels_all == int(label)).sum()),
            "projected_votes": 0,
            "inside_votes": 0,
            "mask_frames_seen": 0,
            "inside_fraction": 0.0,
            "projection_score": 0.0,
        }
        for label in candidate_labels
    }
    if not scores:
        return {}

    frames = list(manifest.get("frames", []))
    for frame in frames:
        try:
            cam_index = int(frame["cam_index"])
        except Exception:
            continue
        if cam_index < 0 or cam_index >= len(scope.cameras):
            continue
        mask_value = frame.get("out_mask_path") or frame.get("out_rgba_path")
        if not mask_value:
            continue
        mask_path = _resolve_path(mask_value, base_dir=manifest_dir)
        mask_img = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED) if mask_path.exists() else None
        if mask_img is None:
            continue
        if mask_img.ndim == 3 and mask_img.shape[2] == 4:
            mask = mask_img[..., 3] > 127
        elif mask_img.ndim == 3:
            mask = mask_img.mean(axis=2) > 127
        else:
            mask = mask_img > 127
        if int(mask.sum()) < 64:
            continue

        ui, vi, valid = _project_points(xyz_all, scope.cameras[cam_index], mask.shape[:2])
        if not valid.any():
            continue
        inside_all = np.zeros(labels_all.shape[0], dtype=bool)
        inside_all[valid] = mask[vi[valid], ui[valid]]
        for label in candidate_labels:
            label_mask = labels_all == int(label)
            projected = int(np.logical_and(label_mask, valid).sum())
            if projected <= 0:
                continue
            inside = int(np.logical_and(label_mask, inside_all).sum())
            scores[label]["projected_votes"] += projected
            scores[label]["inside_votes"] += inside
            scores[label]["mask_frames_seen"] += 1

    for label, score in scores.items():
        projected = int(score["projected_votes"])
        inside = int(score["inside_votes"])
        inside_fraction = float(inside) / max(float(projected), 1.0)
        score["inside_fraction"] = inside_fraction
        score["projection_score"] = float(inside_fraction * np.log1p(float(inside)))
    return scores


def load_colmap_object_point_cloud(
    *,
    model_path: str | Path,
    object_id: int,
    scope,
    extraction_index_path: str | Path | None = None,
    max_points: int = 20000,
    min_points: int = 16,
    target_points: int = 8000,
) -> tuple[BasicPointCloud, dict]:
    """Load object seed points from the scene COLMAP point cloud, not ObjectGS anchors."""
    source_path = _read_source_path(model_path)
    if source_path is None:
        raise FileNotFoundError(f"Could not determine source_path from {Path(model_path) / 'config.yaml'}")

    ply_path = next((path for path in _candidate_labeled_plys(source_path) if path.exists()), None)
    if ply_path is None:
        searched = "\n".join(str(p) for p in _candidate_labeled_plys(source_path))
        raise FileNotFoundError(f"No labeled COLMAP PLY found. Searched:\n{searched}")

    xyz_all, colors_all, labels_all = _load_labeled_ply(ply_path)
    finite = np.isfinite(xyz_all).all(axis=1)
    xyz_all = xyz_all[finite]
    colors_all = colors_all[finite]
    labels_all = labels_all[finite]

    _manifest, module_label = _read_phase3_manifest(extraction_index_path)
    label_priority: list[int] = []
    for value in (module_label, int(object_id)):
        if value is not None and int(value) not in label_priority:
            label_priority.append(int(value))

    label_counts = {int(label): int((labels_all == label).sum()) for label in np.unique(labels_all)}
    label_projection_scores = _score_colmap_labels_against_phase3(
        xyz_all,
        labels_all,
        scope=scope,
        extraction_index_path=extraction_index_path,
        min_points=int(min_points),
    )
    scored_candidates = [
        (label, data) for label, data in label_projection_scores.items()
        if int(data.get("inside_votes", 0)) >= max(5, int(min_points) // 2)
    ]
    chosen_label = None
    if scored_candidates:
        chosen_label = max(
            scored_candidates,
            key=lambda item: (
                float(item[1].get("projection_score", 0.0)),
                float(item[1].get("inside_fraction", 0.0)),
                int(item[1].get("inside_votes", 0)),
            ),
        )[0]
    if chosen_label is None:
        chosen_label = next((label for label in label_priority if label_counts.get(label, 0) >= int(min_points)), None)
    if chosen_label is None:
        positive = {label: count for label, count in label_counts.items() if label != 0 and count >= int(min_points)}
        if not positive:
            raise RuntimeError(
                f"Labeled COLMAP PLY {ply_path} has no usable object labels; counts={label_counts}"
            )
        chosen_label = max(positive.items(), key=lambda item: item[1])[0]
        logger.warning(
            "No preferred COLMAP label for object %d (phase3=%s). Falling back to largest label %d.",
            int(object_id), module_label, int(chosen_label),
        )

    label_mask = labels_all == int(chosen_label)
    xyz = xyz_all[label_mask]
    colors = colors_all[label_mask]
    if xyz.shape[0] < int(min_points):
        raise RuntimeError(
            f"COLMAP label {chosen_label} has only {xyz.shape[0]} points; need at least {min_points}."
        )

    min_keep = max(int(min_points), min(int(xyz.shape[0]), 32))
    n_colmap_selected_points = int(xyz.shape[0])
    xyz, colors, aabb_filtered = _filter_scope_aabb(xyz, colors, scope=scope, min_keep=min_keep)

    target_points = min(int(max_points), max(int(target_points), int(xyz.shape[0])))
    xyz, colors, upsample_meta = _upsample_from_colmap_neighbors(
        xyz,
        colors,
        target_points=target_points,
    )

    if xyz.shape[0] > int(max_points):
        rng = np.random.default_rng(0)
        keep_idx = rng.choice(xyz.shape[0], size=int(max_points), replace=False)
        xyz = xyz[keep_idx]
        colors = colors[keep_idx]

    normals = np.zeros_like(xyz, dtype=np.float32)
    label_ids = np.full((xyz.shape[0],), int(object_id), dtype=np.uint8)
    pcd = BasicPointCloud(points=xyz.astype(np.float32), colors=colors.astype(np.float32), normals=normals, label_ids=label_ids)
    metadata = {
        "init_source": "colmap_labeled_ply",
        "source_path": str(source_path),
        "colmap_ply_path": str(ply_path),
        "phase3_module_label": int(module_label) if module_label is not None else None,
        "colmap_label_used": int(chosen_label),
        "colmap_label_counts": label_counts,
        "colmap_label_projection_scores": label_projection_scores,
        "aabb_filtered": bool(aabb_filtered),
        "n_colmap_selected_points": int(n_colmap_selected_points),
        "n_colmap_seed_points": int(xyz.shape[0]),
        "label_ids_written_as_object_id": int(object_id),
        **upsample_meta,
    }
    logger.info(
        "Scratch init obj %d: loaded %d COLMAP seed points from %s (label=%d, aabb_filtered=%s).",
        int(object_id), int(xyz.shape[0]), ply_path, int(chosen_label), bool(aabb_filtered),
    )
    return pcd, metadata