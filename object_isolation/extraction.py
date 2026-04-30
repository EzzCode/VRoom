"""Phase 1 — Extraction (per-object dataset assembly).

For a (room ObjectGS, object_id) pair, write a per-object dataset folder:

    obj_<id>/
        real_views/
            <frame_id>.png         # RGBA, tightly cropped, letterboxed to render_size
        meta.json                  # one entry per real view (R_w2c, T_w2c, K, bbox, ...)
        object_anchors.ply         # label-filtered anchor cloud (debug / 2DGS init seed)
        extraction_summary.json    # counts, per-frame keep/drop reasons, object frame stats
"""
from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

# Reuse the existing ObjectGS bridge — same Python path setup as
# target_replenishment.
_VROOM_ROOT = Path(__file__).resolve().parent.parent
if str(_VROOM_ROOT) not in sys.path:
    sys.path.insert(0, str(_VROOM_ROOT))

from target_replenishment.core import objectgs_bridge as bridge  # noqa: E402
from target_replenishment.core import diagnostics as diag        # noqa: E402

logger = logging.getLogger(__name__)


# ── Object-frame summary statistics (Phase 1.4) ─────────────────────────────


@dataclass
class ObjectFrame:
    """Per-object world-frame stats persisted in extraction_summary.json."""

    object_id: int
    object_center: list  # length-3 list[float]
    object_radius: float
    object_extent_aabb: list  # [[xmin,ymin,zmin],[xmax,ymax,zmax]]
    object_up_world: list  # length-3 list[float]
    n_object_anchors: int


def compute_object_frame(
    object_id: int,
    anchor_xyz: np.ndarray,
    label_ids: np.ndarray,
    cam_data: list,
) -> ObjectFrame:
    """Compute the canonical object frame in world coordinates.

    ``object_center`` uses the median of label-masked anchor positions to
    resist outliers; ``object_radius`` = max anchor-to-center distance.
    ``object_up_world`` reuses the camera-local-up consensus from
    ``diagnostics.estimate_scene_up_from_cameras``. The W -> O basis itself
    is constructed in ``pose_alignment.compute_W_to_O`` once a reference
    camera is chosen (Phase 2).
    """
    mask = (label_ids == object_id)
    pts = anchor_xyz[mask].astype(np.float64)
    if pts.shape[0] == 0:
        raise ValueError(f"No anchors with label_id={object_id} in this model")

    center = np.median(pts, axis=0)
    radii = np.linalg.norm(pts - center, axis=1)
    radius = float(np.percentile(radii, 99.0))  # robust max
    aabb = np.stack([pts.min(axis=0), pts.max(axis=0)], axis=0)
    up_world = diag.estimate_scene_up_from_cameras(cam_data).astype(np.float64)

    return ObjectFrame(
        object_id=int(object_id),
        object_center=center.tolist(),
        object_radius=radius,
        object_extent_aabb=aabb.tolist(),
        object_up_world=up_world.tolist(),
        n_object_anchors=int(pts.shape[0]),
    )


# ── Per-view object-only RGBA render + tight crop (Phase 1.3) ───────────────


@dataclass
class ViewMeta:
    """One entry of meta.json (one real view of the object)."""

    frame_index: int
    img_name: str
    image_path: str  # relative to obj_<id>/
    width: int
    height: int
    R_w2c: list      # 3x3
    T_w2c: list      # length-3
    K: list          # 3x3 (already rescaled for crop+letterbox)
    crop_bbox_in_full: list  # [x0, y0, x1, y1] in original full-image px
    visible_pixel_count: int
    mean_alpha: float


def render_object_view(
    gaussians,
    pp,
    cam_entry: dict,
    object_id: int,
    full_render_scale: float = 1.0,
):
    """Render the room ObjectGS at the given training camera, with all
    non-target objects masked out (``object_label_id`` argument). Returns a
    dict with ``rgb (H,W,3) float32`` and ``alpha (H,W) float32`` in NumPy.
    """
    # Optionally render at lower resolution to save VRAM
    w = max(1, int(round(cam_entry["width"] * full_render_scale)))
    h = max(1, int(round(cam_entry["height"] * full_render_scale)))
    fx = float(cam_entry["fx"]) * full_render_scale
    fy = float(cam_entry["fy"]) * full_render_scale
    cx = w * 0.5
    cy = h * 0.5

    R_w2c = np.asarray(cam_entry["rotation"], dtype=np.float32)
    T_w2c = np.asarray(cam_entry["position"], dtype=np.float32)
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
    cam = bridge.create_virtual_camera(R_w2c, T_w2c, K, w, h)

    out = bridge.render_view(gaussians, cam, pp, object_label_id=int(object_id))
    rgb = out["rgb"].detach().cpu().numpy()  # (3,H,W)
    rgb = np.clip(np.transpose(rgb, (1, 2, 0)), 0.0, 1.0).astype(np.float32)
    alpha = out["alpha"].detach().cpu().numpy()
    if alpha.ndim == 3:
        alpha = alpha[0]
    alpha = np.clip(alpha, 0.0, 1.0).astype(np.float32)
    return {
        "rgb": rgb,
        "alpha": alpha,
        "K_full": K,
        "width": w,
        "height": h,
        "R_w2c": R_w2c,
        "T_w2c": T_w2c,
    }


def _bbox_from_alpha(alpha: np.ndarray, alpha_thr: float = 0.05) -> Optional[tuple[int, int, int, int]]:
    """Tight bbox of pixels with ``alpha > alpha_thr``. Returns None if empty."""
    mask = alpha > alpha_thr
    if not mask.any():
        return None
    ys, xs = np.where(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _expand_bbox_square(
    bbox: tuple[int, int, int, int],
    width: int,
    height: int,
    pad_frac: float,
) -> tuple[int, int, int, int]:
    """Pad bbox by ``pad_frac`` of its current size on each side, then make
    it square by extending the shorter axis. Result is clamped to the image.
    """
    x0, y0, x1, y1 = bbox
    bw = x1 - x0
    bh = y1 - y0
    side = max(bw, bh) * (1.0 + 2.0 * pad_frac)
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    half = 0.5 * side
    nx0 = int(np.floor(cx - half))
    ny0 = int(np.floor(cy - half))
    nx1 = int(np.ceil(cx + half))
    ny1 = int(np.ceil(cy + half))
    nx0 = max(0, nx0)
    ny0 = max(0, ny0)
    nx1 = min(width, nx1)
    ny1 = min(height, ny1)
    return nx0, ny0, nx1, ny1


def _resize_K_for_crop(
    K_full: np.ndarray,
    crop_bbox: tuple[int, int, int, int],
    out_size: int,
) -> np.ndarray:
    """Adjust intrinsics for (1) cropping to ``crop_bbox`` then (2) resizing
    the (square) crop to ``out_size x out_size``.

    Cropping shifts the principal point: ``cx -> cx - x0``.
    Resizing scales focal and principal point: ``f -> f * out/(x1-x0)``.
    """
    x0, y0, x1, y1 = crop_bbox
    bw = float(x1 - x0)
    bh = float(y1 - y0)
    if bw <= 0 or bh <= 0:
        raise ValueError(f"Bad crop_bbox: {crop_bbox}")

    sx = out_size / bw
    sy = out_size / bh
    K = K_full.copy().astype(np.float64)
    K[0, 0] *= sx
    K[1, 1] *= sy
    K[0, 2] = (K[0, 2] - x0) * sx
    K[1, 2] = (K[1, 2] - y0) * sy
    return K


def _composite_rgba(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Stack RGB + alpha into a (H,W,4) uint8 image."""
    rgba = np.concatenate([rgb, alpha[..., None]], axis=2)
    rgba = np.clip(rgba * 255.0, 0, 255).astype(np.uint8)
    return rgba


def crop_and_letterbox(
    rgb: np.ndarray,
    alpha: np.ndarray,
    K_full: np.ndarray,
    pad_frac: float,
    out_size: int,
    alpha_thr: float = 0.05,
) -> Optional[dict]:
    """Tight-crop the RGBA render around the object, square-pad, resize to
    ``out_size``. Returns ``None`` if the alpha mask is empty.
    """
    h, w = alpha.shape
    bbox = _bbox_from_alpha(alpha, alpha_thr=alpha_thr)
    if bbox is None:
        return None

    crop = _expand_bbox_square(bbox, width=w, height=h, pad_frac=pad_frac)
    x0, y0, x1, y1 = crop

    rgb_crop = rgb[y0:y1, x0:x1]
    alpha_crop = alpha[y0:y1, x0:x1]

    rgb_resized = cv2.resize(rgb_crop, (out_size, out_size), interpolation=cv2.INTER_AREA)
    alpha_resized = cv2.resize(alpha_crop, (out_size, out_size), interpolation=cv2.INTER_AREA)

    K_out = _resize_K_for_crop(K_full, crop, out_size)
    rgba = _composite_rgba(rgb_resized, alpha_resized)
    return {
        "rgba": rgba,
        "K": K_out,
        "crop_bbox_in_full": list(crop),
        "visible_pixel_count": int((alpha_resized > alpha_thr).sum()),
        "mean_alpha": float(alpha_resized.mean()),
    }


# ── Phase 1 driver ──────────────────────────────────────────────────────────


def extract(
    model_path: str,
    object_id: int,
    output_dir: str,
    render_size: int = 512,
    crop_pad_frac: float = 0.15,
    full_render_scale: float = 1.0,
    min_visible_pixels: int = 256,
    alpha_thr: float = 0.05,
    iteration: int = -1,
) -> dict:
    """Run Phase 1. Returns the loaded summary dict.

    The output folder ``<output_dir>/obj_<object_id>/`` is created and
    populated with ``real_views/``, ``meta.json``, ``object_anchors.ply``,
    and ``extraction_summary.json``.
    """
    out_root = Path(output_dir) / f"obj_{int(object_id)}"
    real_views_dir = out_root / "real_views"
    real_views_dir.mkdir(parents=True, exist_ok=True)

    # ── Load room model ────────────────────────────────────────────────────
    gaussians, pp = bridge.load_gaussians(model_path, iteration=iteration)
    anchor_xyz = bridge.get_anchor_positions(gaussians)
    label_ids = bridge.get_label_ids(gaussians)

    # ── Object frame stats (Phase 1.4) ─────────────────────────────────────
    cam_path = Path(model_path) / "cameras.json"
    if not cam_path.exists():
        raise FileNotFoundError(f"cameras.json not found at {cam_path}")
    with open(cam_path, "r", encoding="utf-8") as f:
        cam_data = json.load(f)

    obj_frame = compute_object_frame(object_id, anchor_xyz, label_ids, cam_data)
    logger.info(
        "Object %d: %d anchors, center=%s, radius=%.3f",
        object_id, obj_frame.n_object_anchors, obj_frame.object_center, obj_frame.object_radius,
    )

    # ── Save label-filtered anchor cloud (Phase 1.5) ───────────────────────
    object_pts = anchor_xyz[label_ids == object_id]
    _write_simple_ply(out_root / "object_anchors.ply", object_pts)

    # ── Per-view object-only RGBA + tight crop (Phase 1.3) ─────────────────
    views: list[ViewMeta] = []
    drop_reasons: dict[str, int] = {}

    for cam_entry in cam_data:
        idx = int(cam_entry.get("id", -1))
        img_name = str(cam_entry.get("img_name", f"frame_{idx:05d}"))

        try:
            r = render_object_view(
                gaussians, pp, cam_entry, object_id, full_render_scale=full_render_scale
            )
        except Exception as exc:
            logger.warning("Render failed for cam %s: %s", img_name, exc)
            drop_reasons["render_failed"] = drop_reasons.get("render_failed", 0) + 1
            continue

        cropped = crop_and_letterbox(
            r["rgb"], r["alpha"], r["K_full"],
            pad_frac=crop_pad_frac, out_size=render_size, alpha_thr=alpha_thr,
        )
        if cropped is None:
            drop_reasons["empty_alpha"] = drop_reasons.get("empty_alpha", 0) + 1
            continue
        if cropped["visible_pixel_count"] < min_visible_pixels:
            drop_reasons["too_few_pixels"] = drop_reasons.get("too_few_pixels", 0) + 1
            continue

        # Write the RGBA tile
        view_path = real_views_dir / f"{idx:05d}.png"
        # cv2.imwrite expects BGRA
        bgra = cv2.cvtColor(cropped["rgba"], cv2.COLOR_RGBA2BGRA)
        cv2.imwrite(str(view_path), bgra)

        views.append(ViewMeta(
            frame_index=idx,
            img_name=img_name,
            image_path=f"real_views/{view_path.name}",
            width=int(render_size),
            height=int(render_size),
            R_w2c=r["R_w2c"].tolist(),
            T_w2c=r["T_w2c"].tolist(),
            K=cropped["K"].tolist(),
            crop_bbox_in_full=cropped["crop_bbox_in_full"],
            visible_pixel_count=cropped["visible_pixel_count"],
            mean_alpha=cropped["mean_alpha"],
        ))

    # ── Persist meta.json + extraction_summary.json ────────────────────────
    meta_path = out_root / "meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump([asdict(v) for v in views], f, indent=2)

    summary = {
        "object_frame": asdict(obj_frame),
        "n_real_views": len(views),
        "n_total_cameras": len(cam_data),
        "drop_reasons": drop_reasons,
        "render_size": int(render_size),
        "crop_pad_frac": float(crop_pad_frac),
        "full_render_scale": float(full_render_scale),
        "min_visible_pixels": int(min_visible_pixels),
        "alpha_threshold": float(alpha_thr),
        "model_path": str(model_path),
        "object_id": int(object_id),
    }
    with open(out_root / "extraction_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(
        "Phase 1 done: %d views written, drops=%s", len(views), drop_reasons,
    )

    # Free GPU memory before subsequent phases
    del gaussians
    torch.cuda.empty_cache()
    return summary


# ── Tiny PLY writer (no open3d dep) ─────────────────────────────────────────


def _write_simple_ply(path: Path, pts: np.ndarray) -> None:
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 3)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {pts.shape[0]}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "end_header\n"
        )
        f.write(header.encode("ascii"))
        f.write(pts.tobytes())
