"""Phase 1 (real-mask variant) — extract per-object views from real
photos + DEVA semantic masks instead of ObjectGS renders.

Why:
    ObjectGS renders contain "smoke" floaters from mislabeled anchors.
    The actual training masks at ``data/replica/<scene>/object_mask/``
    are clean DEVA segmentations of the original photos.

Two complications:
    1. The DEVA mask label IDs are NOT the same as the ObjectGS
       per-anchor ``object_id``. We auto-discover the mapping by
       projecting object anchors into a sample of training cameras
       and voting on which mask label they hit.
    2. The mask resolution (480x640) differs slightly from the camera
       intrinsics image size (479x638). We always work at the camera
       image size and resize masks with nearest-neighbour.

Outputs match :mod:`object_isolation.extraction` so the rest of the
pipeline (reference picker, Z123 prep, pose alignment) is unchanged.
"""
from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from target_replenishment.core import objectgs_bridge as bridge

from object_isolation.extraction import (
    ObjectFrame, ViewMeta,
    compute_object_frame, crop_and_letterbox,
    _write_simple_ply,
)

logger = logging.getLogger(__name__)


def _project_points(P_world: np.ndarray, R_w2c: np.ndarray,
                    T_w2c: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project (N,3) world points to (N,2) pixels using OpenCV/COLMAP convention.

    Returns ``(pix, in_front)`` where ``pix`` is (N,2) float and
    ``in_front`` is the boolean mask of points with ``z>0`` in cam frame.
    """
    Pc = (R_w2c @ P_world.T).T + T_w2c[None, :]
    z = Pc[:, 2]
    in_front = z > 1e-6
    uv = (K @ Pc.T).T
    pix = uv[:, :2] / np.maximum(np.abs(uv[:, 2:3]), 1e-9)
    return pix, in_front


def _read_labeled_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load vote.py's ``points3D_labeled.ply`` -> ``(xyz (N,3), labels (N,))``.

    The PLY layout is the one produced by ``Module-1/vote.py::save_labeled_ply``
    (vertex props x,y,z, nx,ny,nz, red,green,blue, label).
    """
    from plyfile import PlyData  # local import; heavy dependency

    ply = PlyData.read(str(path))
    v = ply["vertex"].data
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
    labels = np.asarray(v["label"], dtype=np.int32)
    return xyz, labels


def discover_label_via_vote_ply(
    object_id: int,
    object_anchors_world: np.ndarray,
    vote_ply: Path,
    nn_radius: float = 0.05,
    min_neighbors: int = 5,
) -> tuple[int, dict]:
    """Use vote.py's labeled COLMAP cloud as ground truth for DEVA labels.

    For each ObjectGS anchor, find the nearest labeled COLMAP point within
    ``nn_radius`` metres and inherit its DEVA label. Majority over all
    anchors selects the winning label.

    This is far more robust than per-camera mode voting because vote.py
    has already aggregated mask observations across ~all training views
    (with optional alias merging).
    """
    from scipy.spatial import cKDTree  # local import

    xyz, labels = _read_labeled_ply(vote_ply)
    if xyz.shape[0] == 0:
        raise RuntimeError(f"empty labeled PLY: {vote_ply}")

    tree = cKDTree(xyz)
    dists, idxs = tree.query(object_anchors_world, k=1)
    inside = dists < nn_radius
    n_match = int(inside.sum())
    if n_match < min_neighbors:
        # Loosen the radius automatically to whatever the median distance is.
        med = float(np.median(dists))
        nn_radius = max(nn_radius, med * 1.5)
        inside = dists < nn_radius
        n_match = int(inside.sum())
    if n_match < min_neighbors:
        raise RuntimeError(
            f"too few labeled COLMAP neighbours for object_id={object_id}: "
            f"{n_match} within {nn_radius:.3f}m"
        )

    nbr_labels = labels[idxs[inside]]
    # Drop background (0) — never a valid object id.
    nbr_labels = nbr_labels[nbr_labels != 0]
    if nbr_labels.size == 0:
        raise RuntimeError(
            f"all matched COLMAP points are background for object_id={object_id}"
        )
    vals, counts = np.unique(nbr_labels, return_counts=True)
    order = np.argsort(-counts)
    deva_label = int(vals[order[0]])
    n = int(counts[order[0]])
    runners = [(int(vals[i]), int(counts[i])) for i in order[1:5]]
    stats = {
        "source": "vote_ply",
        "vote_ply": str(vote_ply),
        "deva_label": deva_label,
        "objectgs_object_id": int(object_id),
        "n_anchors": int(object_anchors_world.shape[0]),
        "n_matched": int(n_match),
        "winner_votes": n,
        "winner_share": float(n) / float(max(1, nbr_labels.size)),
        "runners_up": runners,
        "nn_radius": float(nn_radius),
    }
    logger.info(
        "Label discovery via vote.ply: object_id=%d → DEVA label %d "
        "(%d/%d anchors; runners-up: %s)",
        object_id, deva_label, n, int(nbr_labels.size), runners[:3],
    )
    return deva_label, stats


def discover_label_mapping(
    object_id: int,
    object_anchors_world: np.ndarray,
    cam_data: list[dict],
    mask_dir: Path,
    sample_cams: int = 30,
    min_anchors_per_cam: int = 50,
) -> tuple[int, dict]:
    """Return ``(deva_label, stats)``: the DEVA mask label that the
    ObjectGS object's anchors most consistently project onto.
    """
    n_total = len(cam_data)
    if n_total == 0:
        raise ValueError("cam_data is empty")
    step = max(1, n_total // sample_cams)
    label_votes: Counter = Counter()
    cam_votes = 0
    for i in range(0, n_total, step):
        cam = cam_data[i]
        img_name = str(cam["img_name"])
        mask_path = mask_dir / f"{img_name}.png"
        if not mask_path.exists():
            continue
        m = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if m is None:
            continue
        H, W = int(cam["height"]), int(cam["width"])
        if m.shape != (H, W):
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        fx, fy = float(cam["fx"]), float(cam["fy"])
        K = np.array([[fx, 0.0, W * 0.5],
                      [0.0, fy, H * 0.5],
                      [0.0, 0.0, 1.0]], dtype=np.float64)
        R = np.asarray(cam["rotation"], dtype=np.float64)
        T = np.asarray(cam["position"], dtype=np.float64)
        pix, in_front = _project_points(object_anchors_world, R, T, K)
        u = np.round(pix[:, 0]).astype(int)
        v = np.round(pix[:, 1]).astype(int)
        good = in_front & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        if int(good.sum()) < min_anchors_per_cam:
            continue
        labels_at_anchors = m[v[good], u[good]]
        # Per-cam mode → 1 vote for that label
        vals, counts = np.unique(labels_at_anchors, return_counts=True)
        # ignore label 0 (background)
        valid_idx = vals != 0
        if not valid_idx.any():
            continue
        vals, counts = vals[valid_idx], counts[valid_idx]
        winner = int(vals[np.argmax(counts)])
        label_votes[winner] += 1
        cam_votes += 1

    if not label_votes:
        raise RuntimeError(
            f"Could not discover DEVA label for object_id={object_id}: "
            f"no successful projections in {n_total} cameras"
        )
    deva_label, n = label_votes.most_common(1)[0]
    stats = {
        "deva_label": deva_label,
        "objectgs_object_id": int(object_id),
        "n_cams_voted": cam_votes,
        "winner_votes": int(n),
        "all_votes": dict(label_votes),
    }
    logger.info(
        "Label discovery: object_id=%d → DEVA label %d "
        "(%d/%d cam votes; runners-up: %s)",
        object_id, deva_label, n, cam_votes,
        sorted(label_votes.items(), key=lambda kv: -kv[1])[1:4],
    )
    return deva_label, stats


def _load_real_view(
    cam_entry: dict,
    images_dir: Path,
    mask_dir: Path,
    deva_label: int,
    soften_px: int = 1,
) -> Optional[dict]:
    """Load the real photo + mask, apply mask as alpha. Returns dict with
    ``rgb (H,W,3) float32``, ``alpha (H,W) float32``, ``K_full``,
    ``R_w2c``, ``T_w2c``, ``width``, ``height``.

    ``None`` if either file is missing or the mask is empty for the target.
    """
    img_name = str(cam_entry["img_name"])
    img_path = images_dir / f"{img_name}.png"
    mask_path = mask_dir / f"{img_name}.png"
    if not img_path.exists() or not mask_path.exists():
        return None

    bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        return None

    H, W = int(cam_entry["height"]), int(cam_entry["width"])
    if rgb.shape[:2] != (H, W):
        rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA)
    if mask.shape != (H, W):
        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)

    binary = (mask == int(deva_label)).astype(np.float32)
    if binary.sum() < 1:
        return None
    if soften_px > 0:
        # Slight gaussian feather for clean edges in Z123 input
        k = 2 * soften_px + 1
        binary = cv2.GaussianBlur(binary, (k, k), soften_px * 0.5)
    alpha = np.clip(binary, 0.0, 1.0)

    fx, fy = float(cam_entry["fx"]), float(cam_entry["fy"])
    K = np.array([[fx, 0.0, W * 0.5],
                  [0.0, fy, H * 0.5],
                  [0.0, 0.0, 1.0]], dtype=np.float32)
    R_w2c = np.asarray(cam_entry["rotation"], dtype=np.float32)
    T_w2c = np.asarray(cam_entry["position"], dtype=np.float32)
    return {
        "rgb": rgb,
        "alpha": alpha,
        "K_full": K,
        "R_w2c": R_w2c,
        "T_w2c": T_w2c,
        "width": W,
        "height": H,
    }


# ── Phase 1 driver (real mask variant) ─────────────────────────────────────


def extract_from_real_masks(
    model_path: str,
    object_id: int,
    output_dir: str,
    scene_dir: str,
    *,
    images_subdir: str = "images_all",
    mask_subdir: str = "object_mask",
    deva_label: Optional[int] = None,
    vote_ply: Optional[str] = None,
    render_size: int = 512,
    crop_pad_frac: float = 0.15,
    min_visible_pixels: int = 256,
    alpha_thr: float = 0.05,
    iteration: int = -1,
    mask_soften_px: int = 1,
) -> dict:
    """Like :func:`object_isolation.extraction.extract` but uses real
    photos + DEVA masks for the per-view RGBA tiles.

    The ObjectGS model is still loaded (just for anchors → object_frame
    + label discovery + saving ``object_anchors.ply``); we never call
    its renderer.
    """
    out_root = Path(output_dir) / f"obj_{int(object_id)}"
    real_views_dir = out_root / "real_views"
    real_views_dir.mkdir(parents=True, exist_ok=True)

    scene = Path(scene_dir)
    images_dir = scene / images_subdir
    mask_dir = scene / mask_subdir
    if not images_dir.exists():
        raise FileNotFoundError(f"images dir not found: {images_dir}")
    if not mask_dir.exists():
        raise FileNotFoundError(f"mask dir not found: {mask_dir}")

    # Load model (anchors only)
    gaussians, _pp = bridge.load_gaussians(model_path, iteration=iteration)
    anchor_xyz = bridge.get_anchor_positions(gaussians)
    label_ids = bridge.get_label_ids(gaussians)
    obj_anchors = anchor_xyz[label_ids == int(object_id)]
    if obj_anchors.shape[0] == 0:
        raise RuntimeError(f"No anchors for object_id={object_id}")
    _write_simple_ply(out_root / "object_anchors.ply", obj_anchors)

    cam_path = Path(model_path) / "cameras.json"
    with open(cam_path, "r", encoding="utf-8") as f:
        cam_data = json.load(f)

    obj_frame = compute_object_frame(object_id, anchor_xyz, label_ids, cam_data)

    # Discover (or accept user-provided) DEVA label
    if deva_label is None:
        # Prefer vote.py's labeled cloud if available — far more robust.
        candidate_paths: list[Path] = []
        if vote_ply is not None:
            candidate_paths.append(Path(vote_ply))
        candidate_paths.extend([
            scene / "vote_output" / "points3D_labeled.ply",
            scene / "output" / "points3D_labeled.ply",
        ])
        chosen_vote_ply: Optional[Path] = None
        for p in candidate_paths:
            if p.exists():
                chosen_vote_ply = p
                break

        if chosen_vote_ply is not None:
            try:
                deva_label, label_stats = discover_label_via_vote_ply(
                    object_id=object_id,
                    object_anchors_world=obj_anchors,
                    vote_ply=chosen_vote_ply,
                )
            except Exception as exc:
                logger.warning(
                    "vote-ply label discovery failed (%s); falling back to "
                    "per-camera projection voting.", exc,
                )
                deva_label, label_stats = discover_label_mapping(
                    object_id=object_id,
                    object_anchors_world=obj_anchors,
                    cam_data=cam_data,
                    mask_dir=mask_dir,
                )
        else:
            deva_label, label_stats = discover_label_mapping(
                object_id=object_id,
                object_anchors_world=obj_anchors,
                cam_data=cam_data,
                mask_dir=mask_dir,
            )
    else:
        label_stats = {"deva_label": int(deva_label),
                       "objectgs_object_id": int(object_id),
                       "user_provided": True}
    with open(out_root / "label_mapping.json", "w", encoding="utf-8") as f:
        json.dump(label_stats, f, indent=2)

    # Per-view real RGBA tiles
    views: list[ViewMeta] = []
    drop_reasons: dict[str, int] = {}
    for cam_entry in cam_data:
        idx = int(cam_entry.get("id", -1))
        img_name = str(cam_entry.get("img_name", f"frame_{idx:05d}"))
        loaded = _load_real_view(cam_entry, images_dir, mask_dir,
                                 deva_label=int(deva_label),
                                 soften_px=mask_soften_px)
        if loaded is None:
            drop_reasons["no_image_or_mask_or_empty"] = (
                drop_reasons.get("no_image_or_mask_or_empty", 0) + 1)
            continue

        cropped = crop_and_letterbox(
            loaded["rgb"], loaded["alpha"], loaded["K_full"],
            pad_frac=crop_pad_frac, out_size=render_size, alpha_thr=alpha_thr,
        )
        if cropped is None:
            drop_reasons["empty_alpha"] = drop_reasons.get("empty_alpha", 0) + 1
            continue
        if cropped["visible_pixel_count"] < min_visible_pixels:
            drop_reasons["too_few_pixels"] = drop_reasons.get("too_few_pixels", 0) + 1
            continue

        view_path = real_views_dir / f"{idx:05d}.png"
        bgra = cv2.cvtColor(cropped["rgba"], cv2.COLOR_RGBA2BGRA)
        cv2.imwrite(str(view_path), bgra)

        views.append(ViewMeta(
            frame_index=idx,
            img_name=img_name,
            image_path=f"real_views/{view_path.name}",
            width=int(render_size),
            height=int(render_size),
            R_w2c=loaded["R_w2c"].tolist(),
            T_w2c=loaded["T_w2c"].tolist(),
            K=cropped["K"].tolist(),
            crop_bbox_in_full=cropped["crop_bbox_in_full"],
            visible_pixel_count=cropped["visible_pixel_count"],
            mean_alpha=cropped["mean_alpha"],
        ))

    with open(out_root / "meta.json", "w", encoding="utf-8") as f:
        json.dump([asdict(v) for v in views], f, indent=2)

    summary = {
        "object_frame": asdict(obj_frame),
        "n_real_views": len(views),
        "n_total_cameras": len(cam_data),
        "drop_reasons": drop_reasons,
        "render_size": int(render_size),
        "crop_pad_frac": float(crop_pad_frac),
        "full_render_scale": 1.0,
        "min_visible_pixels": int(min_visible_pixels),
        "alpha_threshold": float(alpha_thr),
        "model_path": str(model_path),
        "object_id": int(object_id),
        "extraction_mode": "real_masks",
        "scene_dir": str(scene),
        "deva_label": int(deva_label),
        "label_mapping": label_stats,
    }
    with open(out_root / "extraction_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(
        "Real-mask extract for object %d: kept %d/%d cams (drops: %s)",
        object_id, len(views), len(cam_data), drop_reasons,
    )
    return summary
