import hashlib
import json
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from .utils.gstrain_wrapper import make_camera, render_rgba
from .utils.sv3d_prior import HallucinatedView, SV3DBackend
from .utils.transforms import look_at

logger = logging.getLogger(__name__)

_SV3D_FILL_FRAC   = 0.85
_ALPHA_THRESHOLD  = 0.4
_MIN_SV3D_MASK_PX = 200


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------

def _prepare_conditioning(rgba_path, target_size=576, fill_frac=_SV3D_FILL_FRAC, bg_value=255):
    """Load RGBA, tight-crop on alpha, pad to square, composite on white, resize.

    Returns an (H, W, 3) uint8 RGB array ready to pass to SV3DBackend.
    """
    img = cv2.imread(str(rgba_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot load conditioning frame: {rgba_path}")

    if img.ndim == 2:
        # Greyscale — treat as opaque
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        alpha = np.ones(rgb.shape[:2], np.uint8) * 255
    elif img.shape[2] == 3:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        alpha = np.ones(rgb.shape[:2], np.uint8) * 255
    else:
        alpha = img[:, :, 3]
        rgb = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2RGB)

    mask = alpha > 0
    ys, xs = np.where(mask)
    if len(ys) == 0:
        # No foreground — return a blank white square
        return np.full((target_size, target_size, 3), bg_value, np.uint8)

    y0, y1, x0, x1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
    crop_rgb = rgb[y0:y1 + 1, x0:x1 + 1]
    crop_alpha = alpha[y0:y1 + 1, x0:x1 + 1]

    h, w = crop_rgb.shape[:2]
    side = max(h, w)
    pad_top  = (side - h) // 2
    pad_bot  = side - h - pad_top
    pad_left = (side - w) // 2
    pad_rig  = side - w - pad_left

    def _pad_channel(ch, bg):
        return cv2.copyMakeBorder(ch, pad_top, pad_bot, pad_left, pad_rig,
                                  cv2.BORDER_CONSTANT, value=bg)

    sq_alpha = _pad_channel(crop_alpha, 0).astype(np.float32) / 255.0
    sq_rgb   = np.stack([_pad_channel(crop_rgb[:, :, c], bg_value) for c in range(3)], axis=-1)

    # Composite on white background
    bg  = np.full((side, side, 3), bg_value, np.uint8)
    a3  = sq_alpha[:, :, None]
    out = (sq_rgb.astype(np.float32) * a3 + bg.astype(np.float32) * (1.0 - a3)).clip(0, 255).astype(np.uint8)

    # Shrink so that object fills fill_frac of the square
    obj_side = int(target_size * fill_frac)
    if obj_side < side:
        scale = obj_side / side
        obj_side_px = int(round(side * scale))
        resized = cv2.resize(out, (obj_side_px, obj_side_px), interpolation=cv2.INTER_AREA)
        pad = (target_size - obj_side_px) // 2
        pad_r = target_size - obj_side_px - pad
        out = cv2.copyMakeBorder(resized, pad, pad_r, pad, pad_r,
                                  cv2.BORDER_CONSTANT, value=bg_value)
    else:
        out = cv2.resize(out, (target_size, target_size), interpolation=cv2.INTER_AREA)

    return out


def _alpha_from_white_bg(rgb, sat_thresh=12, val_thresh=245):
    """Estimate a foreground mask from a white-background RGB image via HSV.

    Returns a bool array (H, W).
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    s = hsv[:, :, 1].astype(np.float32)
    v = hsv[:, :, 2].astype(np.float32)
    # Foreground: high saturation OR low brightness
    mask = (s > sat_thresh) | (v < val_thresh)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_u8 = mask.astype(np.uint8) * 255
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN,  kernel)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
    return mask_u8 > 0


def _normalize_framing(rgb, alpha, target_size, fill_frac=_SV3D_FILL_FRAC, bg_value=255):
    """Tight-crop on largest CC of alpha, pad to square, composite on white, resize.

    Drops floaters by keeping only the largest connected component before computing bbox.
    Returns (rgb_uint8 HxWx3, alpha_float HxW).
    """
    bin_mask = (alpha > _ALPHA_THRESHOLD).astype(np.uint8) * 255
    n, labels = cv2.connectedComponents(bin_mask, connectivity=8)
    if n < 2:
        # No foreground — return blank
        blank = np.full((target_size, target_size, 3), bg_value, np.uint8)
        return blank, np.zeros((target_size, target_size), np.float32)

    # Largest CC by pixel count (label 0 = background)
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    best = int(np.argmax(counts))
    fg_mask = (labels == best)

    ys, xs = np.where(fg_mask)
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())

    crop_rgb   = rgb[y0:y1 + 1, x0:x1 + 1]
    crop_alpha = alpha[y0:y1 + 1, x0:x1 + 1]
    h, w = crop_rgb.shape[:2]
    side = max(h, w)

    pad_top  = (side - h) // 2
    pad_bot  = side - h - pad_top
    pad_left = (side - w) // 2
    pad_rig  = side - w - pad_left

    def _pad_f(arr, bg):
        return cv2.copyMakeBorder(arr, pad_top, pad_bot, pad_left, pad_rig,
                                  cv2.BORDER_CONSTANT, value=bg)

    sq_alpha = _pad_f(crop_alpha.astype(np.float32), 0.0)
    sq_rgb   = np.stack([_pad_f(crop_rgb[:, :, c].astype(np.float32), float(bg_value))
                         for c in range(3)], axis=-1)

    bg  = np.full((side, side, 3), float(bg_value), np.float32)
    a3  = sq_alpha[:, :, None]
    out = (sq_rgb * a3 + bg * (1.0 - a3)).clip(0, 255).astype(np.uint8)

    # Scale so object fills fill_frac of target
    obj_side = int(target_size * fill_frac)
    if obj_side < side:
        obj_side_px = int(round(side * obj_side / side))
        obj_side_px = max(obj_side_px, 2)
        out_r     = cv2.resize(out,         (obj_side_px, obj_side_px), interpolation=cv2.INTER_AREA)
        alpha_r   = cv2.resize(sq_alpha,    (obj_side_px, obj_side_px), interpolation=cv2.INTER_AREA)
        pad       = (target_size - obj_side_px) // 2
        pad_r     = target_size - obj_side_px - pad
        out_pad   = cv2.copyMakeBorder(out_r,   pad, pad_r, pad, pad_r,
                                        cv2.BORDER_CONSTANT, value=bg_value)
        alpha_pad = cv2.copyMakeBorder(alpha_r, pad, pad_r, pad, pad_r,
                                        cv2.BORDER_CONSTANT, value=0.0)
    else:
        out_pad   = cv2.resize(out,      (target_size, target_size), interpolation=cv2.INTER_AREA)
        alpha_pad = cv2.resize(sq_alpha, (target_size, target_size), interpolation=cv2.INTER_AREA)

    return out_pad, alpha_pad.astype(np.float32)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _iou(a, b):
    """Intersection over union of two boolean masks."""
    a = np.asarray(a, bool)
    b = np.asarray(b, bool)
    inter = int((a & b).sum())
    union = int((a | b).sum())
    return float(inter / max(union, 1))


def _md5_file(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# SV3D cache management
# ---------------------------------------------------------------------------

def _load_cache(out_dir, cond_az, cond_el, expected_n):
    """Reload a previous SV3D run from disk.

    Prefers the manifest-driven path; falls back to a sorted glob for legacy
    outputs without a manifest.  Raises RuntimeError / FileNotFoundError with
    actionable messages when the cache is stale or incomplete.
    """
    out_dir = Path(out_dir)
    manifest_path = out_dir / "hallucination_index.json"

    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)

        m_n = int(manifest.get("n_views", -1))
        if m_n != expected_n:
            raise RuntimeError(
                f"Cached manifest has n_views={m_n} but backend expects {expected_n}. "
                "Delete the cache or re-run without reuse_sv3d=True."
            )

        cond = manifest.get("conditioning", {})
        m_az = float(cond.get("azimuth_deg", float("nan")))
        m_el = float(cond.get("elevation_deg", float("nan")))
        daz  = abs(((m_az - float(cond_az) + 180.0) % 360.0) - 180.0)
        if daz > 0.5 or abs(m_el - float(cond_el)) > 0.5:
            raise RuntimeError(
                f"Cached conditioning az/el=({m_az:.2f}, {m_el:.2f}) differs from "
                f"current ({float(cond_az):.2f}, {float(cond_el):.2f}). "
                "Re-run without reuse_sv3d=True."
            )

        entries = sorted(manifest.get("frames", []), key=lambda e: int(e["index"]))
        if not entries:
            raise RuntimeError(f"Manifest at {manifest_path} has no frames.")

        views = []
        for entry in entries:
            p = Path(entry["sv3d_raw_path"])
            if not p.exists():
                raise FileNotFoundError(
                    f"Missing cached frame: {p}. Re-run without reuse_sv3d=True."
                )
            bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if bgr is None:
                raise RuntimeError(f"Cannot read cached frame: {p}")
            views.append(HallucinatedView(
                rgb=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
                azimuth_deg=float(entry["azimuth_deg"]),
                elevation_deg=float(entry["elevation_deg"]),
                is_conditioning=bool(entry.get("is_conditioning", False)),
            ))
        logger.info("Cache reuse: loaded %d frames from %s.", len(views), manifest_path)
        return views

    # ── Legacy fallback: sorted glob ──────────────────────────────────────
    out_raw = out_dir / "sv3d_raw"
    files   = sorted(out_raw.glob("*.png"))
    if not files:
        raise FileNotFoundError(
            f"No cached frames in {out_raw} and no manifest. "
            "Re-run without reuse_sv3d=True."
        )
    if len(files) != expected_n:
        raise RuntimeError(
            f"Legacy cache has {len(files)} PNGs but backend expects {expected_n}. "
            "Delete the cache and re-run without reuse_sv3d=True."
        )

    n = len(files)
    views = []
    for i, p in enumerate(files):
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Cannot read legacy cached frame: {p}")
        az_off = ((i + 1) * 360.0 / n) % 360.0
        az_abs = ((float(cond_az) + az_off + 180.0) % 360.0) - 180.0
        views.append(HallucinatedView(
            rgb=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
            azimuth_deg=float(az_abs),
            elevation_deg=float(cond_el),
            is_conditioning=False,
        ))
    logger.warning("Legacy cache reuse (no manifest) from %s: %d frames.", out_raw, n)
    return views


# ---------------------------------------------------------------------------
# Reference renderer
# ---------------------------------------------------------------------------

def _render_reference(scope, frame, gaussians, pipe_config, object_label_id,
                       az_deg, el_deg, resolution=576, fov_y_deg=50.0,
                       up_override=None):
    """Render ObjectGS at a V-frame pose, normalize framing, return (rgb, alpha).

    rgb  — (H, W, 3) uint8 white-background composite
    alpha — (H, W) float32 in [0, 1]
    """
    fov_y = math.radians(float(fov_y_deg))
    fy    = 0.5 * resolution / math.tan(0.5 * fov_y)
    K = np.array([[fy, 0.0, resolution / 2.0],
                  [0.0, fy, resolution / 2.0],
                  [0.0, 0.0, 1.0]], dtype=np.float32)

    R_w2c, T_w2c, C_W = frame.virtual_to_world_camera(float(az_deg), float(el_deg))

    if up_override is not None:
        R_w2c, T_w2c = look_at(
            np.asarray(C_W, np.float32),
            np.asarray(scope.centroid, np.float32),
            np.asarray(up_override, np.float32),
        )

    cam = make_camera(R_w2c, T_w2c, K, resolution, resolution)
    out = render_rgba(gaussians, cam, pipe_config, bg_white=True,
                      object_label_id=object_label_id)

    rgb = (out["rgb"].detach().clamp(0.0, 1.0)
                     .permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    alpha = out["alpha"].detach().cpu().numpy()
    if alpha.ndim == 3:
        alpha = alpha[0]

    rgb, alpha = _normalize_framing(rgb, alpha, resolution)
    return rgb, alpha.astype(np.float32)


# ---------------------------------------------------------------------------
# Hallucinated-frame dataclass
# ---------------------------------------------------------------------------

@dataclass
class HallucinatedFrame:
    index:           int
    azimuth_deg:     float
    elevation_deg:   float
    is_conditioning: bool
    iou_with_objgs:  float
    accepted:        bool
    reject_reason:   str
    out_rgba_path:   str
    sv3d_raw_path:   str
    objgs_ref_path:  str


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_hallucination(scope, frame, gaussians, pipe_config, *,
                      scores_json_path, output_dir, object_label_id,
                      backend=None, iou_threshold=0.20, min_objgs_pixels=600,
                      fov_y_deg=50.0, seed=0, save_dropped=True,
                      reuse_sv3d=False):
    """Run SV3D novel-view synthesis and filter outputs against the ObjectGS model.

    Parameters
    ----------
    scope            : ObjectScope (from scene_analysis.py)
    frame            : ObjectFrame (from transforms.py)
    gaussians        : GaussianModel loaded by load_gaussians
    pipe_config      : PipelineConfig from load_gaussians
    scores_json_path : path to scores.json written by view_selection.run_scoring
    output_dir       : directory where outputs are written
    object_label_id  : integer label id of the target object
    backend          : SV3DBackend instance; created on-demand if None
    iou_threshold    : minimum IoU between SV3D mask and ObjectGS mask for acceptance
    min_objgs_pixels : ObjectGS pixel count below which a frame is accepted as
                       "back side with no reference" rather than rejected
    fov_y_deg        : vertical field-of-view used for reference renders
    seed             : RNG seed passed to SV3D
    save_dropped     : whether to write rejected frames to disk (default True)
    reuse_sv3d       : if True, reload cached SV3D outputs instead of re-generating

    Returns
    -------
    manifest dict (same content as hallucination_index.json)
    """
    output_dir = Path(output_dir)
    out_halluc = output_dir / "hallucinated"
    out_raw    = output_dir / "sv3d_raw"
    out_ref    = output_dir / "objgs_refs"
    for d in (out_halluc, out_raw, out_ref):
        d.mkdir(parents=True, exist_ok=True)

    if not reuse_sv3d:
        # Clear stale outputs
        stale = (output_dir / "hallucination_index.json",)
        for f in stale:
            if f.exists():
                f.unlink()
        for d in (out_halluc, out_raw, out_ref):
            for p in d.glob("*.png"):
                p.unlink()

    # ── Load scores and pick conditioning frame ───────────────────────────
    with open(scores_json_path) as f:
        scores = json.load(f)

    if not scores.get("top_k"):
        raise RuntimeError(f"scores file has no top_k entries: {scores_json_path}")

    top1     = scores["top_k"][0]
    rgba_path = Path(top1["out_rgba_path"])

    # Conditioning pose in V-frame
    cond_az = float(top1.get("azimuth_deg", 0.0))
    cond_el = 0.0   # view_selection always evaluates at elevation 0

    # Verify the RGBA file still exists (guard against stale scores)
    if not rgba_path.exists():
        raise FileNotFoundError(
            f"Conditioning RGBA not found at {rgba_path}. "
            "Re-run view_selection before hallucination."
        )

    # Up vector from the conditioning camera row
    cond_cam_idx = int(top1["cam_index"])
    cond_cam_up  = None
    if scope.cameras and len(scope.cameras) > cond_cam_idx:
        cam_dict = scope.cameras[cond_cam_idx]
        R_cam = np.asarray(cam_dict.get("R"), np.float32)
        if R_cam is not None and R_cam.shape == (3, 3):
            # Camera -Y axis in world coords is the camera up direction
            cond_cam_up = -R_cam[1]

    # ── Conditioning image ───────────────────────────────────────────────
    own_backend = backend is None
    if own_backend:
        backend = SV3DBackend()

    cond_rgb = _prepare_conditioning(rgba_path, target_size=backend.native_resolution)
    cond_png = output_dir / "conditioning.png"
    cv2.imwrite(str(cond_png), cv2.cvtColor(cond_rgb, cv2.COLOR_RGB2BGR))
    cond_png_md5 = _md5_file(cond_png)
    logger.info("Conditioning image: %s  (md5=%s)", cond_png, cond_png_md5)

    # ── Run or reload SV3D ───────────────────────────────────────────────
    if reuse_sv3d:
        views = _load_cache(output_dir, cond_az, cond_el, backend._num_frames)
    else:
        views = backend.hallucinate(cond_rgb, cond_el, cond_az, seed=int(seed))
        if own_backend:
            backend.unload()

    if not views:
        raise RuntimeError("SV3D returned no views.")

    res = views[0].rgb.shape[0]

    # ── Filter views against ObjectGS reference renders ──────────────────
    frames  = []
    n_kept  = 0

    for i, v in enumerate(views):
        ref_rgb, ref_alpha = _render_reference(
            scope, frame, gaussians, pipe_config,
            object_label_id=object_label_id,
            az_deg=v.azimuth_deg, el_deg=v.elevation_deg,
            resolution=res, fov_y_deg=fov_y_deg,
            up_override=cond_cam_up,
        )

        m_objgs = ref_alpha > _ALPHA_THRESHOLD
        m_sv3d  = _alpha_from_white_bg(v.rgb)

        if m_objgs.shape != m_sv3d.shape:
            m_objgs = cv2.resize(m_objgs.astype(np.uint8),
                                 (m_sv3d.shape[1], m_sv3d.shape[0]),
                                 interpolation=cv2.INTER_NEAREST).astype(bool)

        iou     = _iou(m_sv3d, m_objgs)
        n_objgs = int(m_objgs.sum())
        n_sv3d  = int(m_sv3d.sum())

        accepted = True
        reason   = ""
        if n_sv3d < _MIN_SV3D_MASK_PX:
            accepted = False
            reason   = "sv3d_empty"
        elif n_objgs < min_objgs_pixels:
            accepted = True
            reason   = "back_side_no_ref"
        elif iou < iou_threshold:
            accepted = False
            reason   = f"iou_low_{iou:.2f}"

        if not accepted and not save_dropped:
            continue
        if accepted:
            n_kept += 1

        az_tag  = int(round(v.azimuth_deg))
        stem    = f"{i:02d}__az{az_tag:+04d}"
        sv3d_p  = out_raw    / f"{stem}.png"
        ref_p   = out_ref    / f"{stem}.png"
        rgba_p  = out_halluc / f"{stem}.png"

        cv2.imwrite(str(sv3d_p), cv2.cvtColor(v.rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(ref_p),  cv2.cvtColor(ref_rgb, cv2.COLOR_RGB2BGR))

        # RGBA out: SV3D RGB + computed alpha
        rgba_out = np.dstack([
            cv2.cvtColor(v.rgb, cv2.COLOR_RGB2BGR),
            (m_sv3d.astype(np.uint8) * 255),
        ])
        cv2.imwrite(str(rgba_p), rgba_out)

        frames.append(HallucinatedFrame(
            index=i,
            azimuth_deg=float(v.azimuth_deg),
            elevation_deg=float(v.elevation_deg),
            is_conditioning=bool(v.is_conditioning),
            iou_with_objgs=float(iou),
            accepted=accepted,
            reject_reason=reason,
            out_rgba_path=str(rgba_p),
            sv3d_raw_path=str(sv3d_p),
            objgs_ref_path=str(ref_p),
        ))

    logger.info("Novel-view synthesis: kept %d / %d views (threshold IoU=%.2f).",
                n_kept, len(views), iou_threshold)

    # ── Write manifest ────────────────────────────────────────────────────
    manifest = {
        "backend": "sv3d",
        "n_views": len(views),
        "n_kept": n_kept,
        "object_label_id": object_label_id,
        "conditioning": {
            "cam_index": int(top1["cam_index"]),
            "image_name": top1.get("image_name", ""),
            "azimuth_deg": float(cond_az),
            "elevation_deg": float(cond_el),
            "score": float(top1.get("score", 0.0)),
            "image_path": str(cond_png),
            "md5": cond_png_md5,
        },
        "params": {
            "iou_threshold": float(iou_threshold),
            "min_objgs_pixels": int(min_objgs_pixels),
            "fov_y_deg": float(fov_y_deg),
            "resolution": int(res),
            "seed": int(seed),
        },
        "frames": [asdict(fr) for fr in frames],
    }

    index_path = output_dir / "hallucination_index.json"
    with open(index_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Hallucination manifest: %s", index_path)
    return manifest
