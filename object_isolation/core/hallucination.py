"""
Diffusion-prior orchestration for novel-view hallucination.

Workflow:
    1. Load the frame-scoring top-1 frame (best SV3D conditioning view).
    2. Crop tight + pad onto neutral background → square conditioning image.
    3. Instantiate the chosen DiffusionPriorBackend (SV3D-u by default).
    4. Run backend → list of HallucinatedView at (azimuth_V, elevation_V).
    5. For each output: map V→W camera, render ObjectGS at that pose to get
       a reference silhouette M_objgs.
    6. Extract M_sv3d from each output (white-bg subtraction).
    7. Apply silhouette filter:
           — If |M_objgs| < min_objgs_pixels  → KEEP (back-side hallucination,
             nothing to compare against).
           — Else require IoU(M_sv3d, M_objgs) ≥ iou_threshold.
    8. Save accepted frames as RGBA + manifest `hallucination_index.json`.

Outputs at <out_root>/obj_<id>/03_novel_views/::

    conditioning.png                        SV3D input (square, padded)
    hallucinated/<seq>__az<DEG>.png         RGBA hallucinated view
    objgs_refs/<seq>__az<DEG>.png           RGB ObjectGS reference render
    sv3d_raw/<seq>__az<DEG>.png             raw SV3D RGB output
    hallucination_index.json                manifest

Run via pipeline orchestrator (recommended)::

    python -m object_isolation.run_pipeline \\
        --model_path temp_deps/ObjectGS/outputs/3dovs/.../2026-03-19_04-01-38 \\
        --object_id 8 \\
        --output_root object_isolation/outputs \\
        [--reuse_sv3d]
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from .coordinate_frames import LocalSV3D
from .diffusion_priors.base import DiffusionPriorBackend, HallucinatedView
from .object_scope import ObjectScope

logger = logging.getLogger(__name__)

# ── Shared constants — must match dataset_builder.py exactly ──────────────────
# Fraction of the SV3D native resolution that the object fills in conditioning
# and reference renders.  Must match _SV3D_FILL_FRAC in dataset_builder.py.
_SV3D_FILL_FRAC: float = 0.85

# Alpha threshold used to binarise ObjectGS reference renders into a mask.
_ALPHA_THRESHOLD: float = 0.4

# Minimum number of foreground pixels in the SV3D output mask; frames below
# this are treated as empty (no object visible) and rejected.
_MIN_SV3D_MASK_PIXELS: int = 200


# ─────────────────────────────────────────────────────────────────────────────
# Conditioning image prep
# ─────────────────────────────────────────────────────────────────────────────

def _prepare_conditioning(rgba_path: Path, target_size: int = 576,
                          fill_frac: float = _SV3D_FILL_FRAC,
                          bg_value: int = 255) -> np.ndarray:
    """Crop tight on alpha, pad to square, composite on neutral bg."""
    rgba = cv2.imread(str(rgba_path), cv2.IMREAD_UNCHANGED)
    if rgba is None or rgba.shape[2] != 4:
        raise ValueError(f"Bad RGBA: {rgba_path}")
    bgr = rgba[..., :3]
    a = rgba[..., 3]
    ys, xs = np.where(a > 127)
    if len(xs) == 0:
        raise ValueError(f"Empty alpha in {rgba_path}")
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    obj_w, obj_h = x1 - x0, y1 - y0
    side = max(obj_w, obj_h)
    pad = int(round(side * (1.0 - fill_frac) / (2.0 * fill_frac)))
    # square crop centered on object
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    half = side // 2 + pad
    sx0 = max(0, cx - half); sy0 = max(0, cy - half)
    sx1 = min(rgba.shape[1], cx + half); sy1 = min(rgba.shape[0], cy + half)
    crop_bgr = bgr[sy0:sy1, sx0:sx1]
    crop_a = a[sy0:sy1, sx0:sx1].astype(np.float32) / 255.0

    # square pad
    h, w = crop_bgr.shape[:2]
    side2 = max(h, w)
    pad_top = (side2 - h) // 2; pad_bot = side2 - h - pad_top
    pad_left = (side2 - w) // 2; pad_right = side2 - w - pad_left
    crop_bgr = cv2.copyMakeBorder(crop_bgr, pad_top, pad_bot, pad_left, pad_right,
                                  cv2.BORDER_CONSTANT, value=(bg_value,)*3)
    crop_a = cv2.copyMakeBorder(crop_a, pad_top, pad_bot, pad_left, pad_right,
                                cv2.BORDER_CONSTANT, value=0.0)

    # composite on bg
    bg = np.full_like(crop_bgr, bg_value)
    a3 = crop_a[..., None]
    comp = (a3 * crop_bgr + (1 - a3) * bg).astype(np.uint8)

    # resize to target
    comp = cv2.resize(comp, (target_size, target_size), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(comp, cv2.COLOR_BGR2RGB)


# ─────────────────────────────────────────────────────────────────────────────
# White-background mask extraction
# ─────────────────────────────────────────────────────────────────────────────

def _alpha_from_white_bg(rgb: np.ndarray, sat_thresh: int = 12,
                         val_thresh: int = 245) -> np.ndarray:
    """Estimate fg mask from a render on (near-)white background."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[..., 1]
    val = hsv[..., 2]
    fg = (sat > sat_thresh) | (val < val_thresh)
    fg = fg.astype(np.uint8)
    # cleanup
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    return fg.astype(bool)


def _normalize_framing(rgb: np.ndarray, alpha: np.ndarray,
                       target_size: int, fill_frac: float = _SV3D_FILL_FRAC,
                       bg_value: int = 255) -> tuple[np.ndarray, np.ndarray]:
    """Tight-crop on alpha, square-pad to give ``fill_frac`` coverage, resize.

    Mirrors `_prepare_conditioning` so ObjectGS reference renders use the same
    framing convention as the SV3D conditioning input. Returns
    (rgb_uint8 HxWx3 RGB, alpha_float HxW in [0,1]).
    """
    a01 = alpha.astype(np.float32)
    if a01.max() > 1.5:
        a01 = a01 / 255.0
    a01 = np.clip(a01, 0.0, 1.0)
    mask = a01 > _ALPHA_THRESHOLD
    # Drop floaters: keep only the largest connected component for the bbox.
    # (Stray Gaussians outside the object inflate the bbox and squash the
    # actual silhouette into a corner — visible as apparent mirror/flip on
    # frames where there are sparse floaters off to one side.)
    mask_u8 = mask.astype(np.uint8)
    n_lbl, lbls, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if n_lbl > 1:
        # stats[0] is background; pick the largest non-background area.
        areas = stats[1:, cv2.CC_STAT_AREA]
        biggest = 1 + int(np.argmax(areas))
        mask = (lbls == biggest)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        # Fall back: blank frame at target_size with white bg.
        empty_rgb = np.full((target_size, target_size, 3), bg_value, np.uint8)
        empty_a = np.zeros((target_size, target_size), np.float32)
        return empty_rgb, empty_a

    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    obj_w, obj_h = x1 - x0, y1 - y0
    side = max(obj_w, obj_h)
    pad = int(round(side * (1.0 - fill_frac) / (2.0 * fill_frac)))
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    half = side // 2 + pad
    H, W = rgb.shape[:2]
    sx0 = max(0, cx - half); sy0 = max(0, cy - half)
    sx1 = min(W, cx + half); sy1 = min(H, cy + half)
    crop_rgb = rgb[sy0:sy1, sx0:sx1]
    crop_a = a01[sy0:sy1, sx0:sx1]

    h, w = crop_rgb.shape[:2]
    side2 = max(h, w)
    pad_top = (side2 - h) // 2; pad_bot = side2 - h - pad_top
    pad_left = (side2 - w) // 2; pad_right = side2 - w - pad_left
    crop_rgb = cv2.copyMakeBorder(crop_rgb, pad_top, pad_bot, pad_left, pad_right,
                                  cv2.BORDER_CONSTANT, value=(bg_value,) * 3)
    crop_a = cv2.copyMakeBorder(crop_a, pad_top, pad_bot, pad_left, pad_right,
                                cv2.BORDER_CONSTANT, value=0.0)

    # Composite on white bg so RGB visually matches conditioning.
    a3 = crop_a[..., None]
    bg = np.full_like(crop_rgb, bg_value)
    comp = (a3 * crop_rgb.astype(np.float32) + (1 - a3) * bg).astype(np.uint8)

    comp = cv2.resize(comp, (target_size, target_size), interpolation=cv2.INTER_AREA)
    crop_a = cv2.resize(crop_a, (target_size, target_size), interpolation=cv2.INTER_AREA)
    return comp, crop_a


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool); b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / max(union, 1))


def _load_sv3d_cache(out_raw: Path, cond_az: float, cond_el: float,
                     expected_n: int) -> List[HallucinatedView]:
    """Reload previously saved SV3D outputs as HallucinatedView list."""
    files = sorted(Path(out_raw).glob("*.png"))
    if not files:
        raise FileNotFoundError(f"No cached SV3D outputs in {out_raw}")
    if len(files) != expected_n:
        logger.warning("Cached SV3D count=%d, expected %d", len(files), expected_n)
    n = len(files)
    views: List[HallucinatedView] = []
    for i, p in enumerate(files):
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        # Reproduce sv3d_p azimuth schedule: az_off = (i+1)*360/n.
        az_off = (((i + 1) * 360.0 / n) % 360.0)
        az_abs = ((cond_az + az_off + 180.0) % 360.0) - 180.0
        views.append(HallucinatedView(
            rgb=rgb, azimuth_V_deg=float(az_abs),
            elevation_V_deg=float(cond_el), is_conditioning=False,
        ))
    return views


def _signed_angle_delta_deg(a: float, b: float) -> float:
    """Shortest signed angular difference a-b in degrees."""
    return float(((float(a) - float(b) + 180.0) % 360.0) - 180.0)


# ─────────────────────────────────────────────────────────────────────────────
# Render ObjectGS at a hallucinated V-pose
# ─────────────────────────────────────────────────────────────────────────────

def _render_reference(scope: ObjectScope, local_sv3d, gaussians, pipe_config,
                      object_label_id: int,
                      az_V_deg: float, el_V_deg: float,
                      resolution: int = 576,
                      fov_y_deg: float = 50.0,
                      up_W_override: Optional[np.ndarray] = None):
    """Render ObjectGS at a V-frame pose; returns (rgb uint8 HxWx3, alpha float HxW).

    Distance from object centroid is local_sv3d.world_local.radius
    (baked into the SV3D virtual orbit at construction time)."""
    from .gs_renderer import create_camera, render_rgba

    R_w2c, T_w2c, _C_W = local_sv3d.sv3d_view_to_world_camera(
        az_V_deg, el_V_deg,
    )
    if up_W_override is not None:
        # Re-roll the look-at using the conditioning camera's own world-up so
        # that ObjectGS reference renders match SV3D's inherited roll.
        from .coordinate_frames import look_at_w2c
        R_w2c, T_w2c = look_at_w2c(
            np.asarray(_C_W, dtype=np.float64),
            np.asarray(local_sv3d.world_local.centroid_W, dtype=np.float64),
            np.asarray(up_W_override, dtype=np.float64),
        )

    # Build K from desired FOV.
    fov_y = math.radians(fov_y_deg)
    fy = 0.5 * resolution / math.tan(0.5 * fov_y)
    fx = fy
    cx = cy = resolution / 2.0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    cam = create_camera(R_w2c, T_w2c, K, resolution, resolution)
    out = render_rgba(gaussians, cam, pipe_config, bg_white=True, object_label_id=object_label_id)
    rgb_t = out["rgb"].detach().clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
    rgb_u8 = (rgb_t * 255.0 + 0.5).astype(np.uint8)
    alpha = out["alpha"].detach().cpu().numpy()
    if alpha.ndim == 3:
        alpha = alpha[0]
    alpha = np.asarray(alpha)

    # Normalize framing to match SV3D conditioning convention (≈85% fill, square).
    rgb_u8, alpha = _normalize_framing(rgb_u8, alpha, target_size=resolution,
                                       fill_frac=_SV3D_FILL_FRAC, bg_value=255)
    return rgb_u8, alpha


# ─────────────────────────────────────────────────────────────────────────────
# Manifest dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class HallucinatedFrame:
    index: int
    azimuth_V_deg: float
    elevation_V_deg: float
    is_conditioning: bool
    iou_with_objgs: float
    n_pixels_sv3d: int
    n_pixels_objgs: int
    accepted: bool
    reject_reason: str
    out_rgba_path: str
    sv3d_raw_path: str
    objgs_ref_path: str


# ─────────────────────────────────────────────────────────────────────────────
# Main entry
# ─────────────────────────────────────────────────────────────────────────────

def run_hallucination(
    scope: ObjectScope,
    local_sv3d: LocalSV3D,
    gaussians,
    pipe_config,
    *,
    scores_json_path: Path,
    output_dir: Path,
    object_label_id: int,
    backend: Optional[DiffusionPriorBackend] = None,
    iou_threshold: float = 0.20,
    min_objgs_pixels: int = 600,
    fov_y_deg: float = 50.0,
    seed: int = 0,
    save_dropped: bool = True,
    reuse_sv3d: bool = False,
) -> dict:
    """Run novel-view hallucination on the top-ranked conditioning view."""
    output_dir = Path(output_dir)
    out_halluc = output_dir / "hallucinated"
    out_raw = output_dir / "sv3d_raw"
    out_ref = output_dir / "objgs_refs"
    for d in (out_halluc, out_raw, out_ref):
        d.mkdir(parents=True, exist_ok=True)

    # Load frame-scoring results → pick the top conditioning frame.
    with open(scores_json_path) as f:
        scores = json.load(f)
    if not scores["top_k"]:
        raise RuntimeError("No frames in scores.json; rerun frame scoring.")
    top1 = scores["top_k"][0]
    # Find full record for paths.
    full = next((fr for fr in scores["frames"] if fr["cam_index"] == top1["cam_index"]), None)
    if full is None:
        raise RuntimeError("top1 cam not in scores.frames")

    rgba_path = Path(full["out_rgba_path"])
    cond_az = float(full.get("azimuth_V_deg", top1["azimuth_V_deg"]))
    cond_el = float(full.get("elevation_V_deg", 0.0))
    if not math.isfinite(cond_el):
        cond_el = 0.0
    if not math.isfinite(cond_az):
        cond_az = 0.0

    cond_cam_dict = scope.cameras[int(top1["cam_index"])]
    current_az, current_el = local_sv3d.world_camera_to_sv3d_view(cond_cam_dict["position"])
    current_az = ((float(current_az) + 180.0) % 360.0) - 180.0
    current_el = float(current_el)
    stale_az = abs(_signed_angle_delta_deg(cond_az, current_az))
    stale_el = abs(float(cond_el) - current_el)
    if stale_az > 0.5 or stale_el > 0.5:
        logger.warning(
            "Frame-scoring pose for conditioning cam %d is stale under current coordinate frame: "
            "scores az/el=(%.2f, %.2f), current az/el=(%.2f, %.2f). "
            "Using current pose for SV3D.",
            int(top1["cam_index"]), cond_az, cond_el, current_az, current_el,
        )
        cond_az = current_az
        cond_el = current_el

    logger.info("Conditioning: cam=%d az_V=%.1f el_V=%.1f score=%.3f from %s",
                top1["cam_index"], cond_az, cond_el, top1["score"], rgba_path.name)

    # Extract the conditioning camera's own world-up (= -R_w2c[1] in OpenCV).
    # SV3D outputs inherit the cond image's roll; reference renders must use
    # the same up so the silhouettes align (otherwise scene-averaged up_W can
    # be tens of degrees off-axis from any individual camera).
    R_cond = np.asarray(cond_cam_dict["R"], dtype=np.float64)  # already R_w2c
    cond_cam_up_W = (-R_cond[1]).astype(np.float64)
    cond_cam_up_W = cond_cam_up_W / max(np.linalg.norm(cond_cam_up_W), 1e-9)
    ang_deg = float(np.degrees(np.arccos(np.clip(
        cond_cam_up_W @ np.asarray(scope.up_W, dtype=np.float64), -1.0, 1.0))))
    logger.info("Cond cam up_W=%s (\u2220 scope.up_W = %.1f\u00b0)",
                np.round(cond_cam_up_W, 3).tolist(), ang_deg)

    # Backend.
    if backend is None:
        from .diffusion_priors.sv3d import SV3DBackend
        backend = SV3DBackend()

    cond_rgb = _prepare_conditioning(rgba_path, target_size=backend.native_resolution)
    cv2.imwrite(str(output_dir / "conditioning.png"),
                cv2.cvtColor(cond_rgb, cv2.COLOR_RGB2BGR))

    # Run prior (or reuse cached outputs).
    if reuse_sv3d:
        views = _load_sv3d_cache(out_raw, cond_az, cond_el, backend.output_count)
        logger.info("Reusing %d cached SV3D views from %s.", len(views), out_raw)
    else:
        views = backend.hallucinate(
            cond_rgb, cond_elevation_deg=cond_el, cond_azimuth_deg=cond_az, seed=seed,
        )
        logger.info("Backend produced %d views.", len(views))
        # Free SV3D before heavy renders.
        backend.unload()

    # Render references + filter.
    frames: List[HallucinatedFrame] = []
    res = views[0].rgb.shape[0]
    n_kept = 0
    for i, v in enumerate(views):
        # Reference render at the same V-pose (white bg).
        ref_rgb, ref_alpha = _render_reference(
            scope, local_sv3d, gaussians, pipe_config,
            object_label_id=object_label_id,
            az_V_deg=v.azimuth_V_deg, el_V_deg=v.elevation_V_deg,
            resolution=res, fov_y_deg=fov_y_deg,
            up_W_override=cond_cam_up_W,
        )
        m_objgs = ref_alpha > _ALPHA_THRESHOLD
        m_sv3d = _alpha_from_white_bg(v.rgb)

        # Resize masks to common shape (should match res anyway).
        if m_objgs.shape != m_sv3d.shape:
            m_objgs = cv2.resize(m_objgs.astype(np.uint8),
                                 (m_sv3d.shape[1], m_sv3d.shape[0]),
                                 interpolation=cv2.INTER_NEAREST).astype(bool)
        iou = _iou(m_sv3d, m_objgs)
        n_objgs = int(m_objgs.sum())
        n_sv3d = int(m_sv3d.sum())

        # Accept logic.
        accepted = True
        reason = ""
        if n_sv3d < _MIN_SV3D_MASK_PIXELS:
            accepted = False
            reason = "sv3d_empty"
        elif n_objgs < min_objgs_pixels:
            accepted = True
            reason = "back_side_no_ref"
        elif iou < iou_threshold:
            accepted = False
            reason = f"iou_low_{iou:.2f}"

        if not accepted and not save_dropped:
            continue
        if accepted:
            n_kept += 1

        # Filenames.
        az_tag = int(round(v.azimuth_V_deg))
        stem = f"{i:02d}__az{az_tag:+04d}"
        sv3d_path = out_raw / f"{stem}.png"
        ref_path = out_ref / f"{stem}.png"
        rgba_path_out = out_halluc / f"{stem}.png"

        cv2.imwrite(str(sv3d_path), cv2.cvtColor(v.rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(ref_path), cv2.cvtColor(ref_rgb, cv2.COLOR_RGB2BGR))
        # RGBA: SV3D RGB + computed alpha
        rgba_out = np.dstack([cv2.cvtColor(v.rgb, cv2.COLOR_RGB2BGR),
                              (m_sv3d * 255).astype(np.uint8)])
        cv2.imwrite(str(rgba_path_out), rgba_out)

        frames.append(HallucinatedFrame(
            index=i,
            azimuth_V_deg=float(v.azimuth_V_deg),
            elevation_V_deg=float(v.elevation_V_deg),
            is_conditioning=bool(v.is_conditioning),
            iou_with_objgs=float(iou),
            n_pixels_sv3d=n_sv3d,
            n_pixels_objgs=n_objgs,
            accepted=accepted,
            reject_reason=reason,
            out_rgba_path=str(rgba_path_out),
            sv3d_raw_path=str(sv3d_path),
            objgs_ref_path=str(ref_path),
        ))

    # Manifest.
    manifest = {
        "backend": backend.name,
        "object_label_id": object_label_id,
        "conditioning": {
            "cam_index": top1["cam_index"],
            "img_name": top1["img_name"],
            "azimuth_V_deg": cond_az,
            "elevation_V_deg": cond_el,
            "score": top1["score"],
            "image_path": str(output_dir / "conditioning.png"),
        },
        "params": {
            "iou_threshold": iou_threshold,
            "min_objgs_pixels": min_objgs_pixels,
            "fov_y_deg": fov_y_deg,
            "resolution": res,
            "seed": seed,
        },
        "n_views": len(views),
        "n_kept": n_kept,
        "frames": [asdict(fr) for fr in frames],
    }
    with open(output_dir / "hallucination_index.json", "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Novel-view synthesis kept %d / %d views (manifest: %s)",
                n_kept, len(views), output_dir / "hallucination_index.json")
    return manifest
