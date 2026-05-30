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

_ALPHA_THRESHOLD = 0.4
_FOV_Y_DEG       = 50.0


def _tight_crop_pad_resize(rgb, alpha, target_size):
    fill_frac = 0.85
    bg_value  = 255
    if alpha.max() > 1.5:
        alpha = alpha.astype(np.float32) / 255.0
 
    alpha = np.clip(alpha, 0.0, 1.0)
    
    
    mask = alpha > _ALPHA_THRESHOLD

    labels_n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if labels_n > 1:
        mask = labels == (1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])))

    ys, xs = np.where(mask)
    if len(xs) == 0:
        return np.full((target_size, target_size, 3), bg_value, np.uint8), np.zeros((target_size, target_size), np.float32)

    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    side = max(x1 - x0, y1 - y0)
    pad  = int(round(side * (1.0 - fill_frac) / (2.0 * fill_frac)))
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    half   = side // 2 + pad
    h, w   = rgb.shape[:2]
    sx0, sy0 = max(0, cx - half), max(0, cy - half)
    sx1, sy1 = min(w, cx + half), min(h, cy + half)
    crop_rgb   = rgb[sy0:sy1, sx0:sx1]
    crop_alpha = alpha[sy0:sy1, sx0:sx1]

    ch, cw = crop_rgb.shape[:2]
    s  = max(ch, cw)
    pt = (s - ch) // 2
    pl = (s - cw) // 2
    crop_rgb   = cv2.copyMakeBorder(crop_rgb,   pt, s - ch - pt, pl, s - cw - pl, cv2.BORDER_CONSTANT, value=(bg_value,) * 3)
    crop_alpha = cv2.copyMakeBorder(crop_alpha, pt, s - ch - pt, pl, s - cw - pl, cv2.BORDER_CONSTANT, value=0.0)

    bg   = np.full_like(crop_rgb, bg_value)
    comp = (crop_alpha[..., None] * crop_rgb.astype(np.float32) + (1.0 - crop_alpha[..., None]) * bg).astype(np.uint8)
    comp       = cv2.resize(comp,       (target_size, target_size), interpolation=cv2.INTER_AREA)
    crop_alpha = cv2.resize(crop_alpha, (target_size, target_size), interpolation=cv2.INTER_AREA)
    return comp, crop_alpha



def _alpha_from_white_bg(rgb):
    sat_thresh, val_thresh = 12, 245
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    s   = hsv[:, :, 1]
    v   = hsv[:, :, 2]
    mask = ((s > sat_thresh) | (v < val_thresh)).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask > 0


def _iou(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    return float((a & b).sum() / max((a | b).sum(), 1))


def _load_cache(out_dir, cond_az, cond_el):
    expected_n = 21
    out_dir       = Path(out_dir)
    manifest_path = out_dir / "hallucination_index.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No cache manifest at {manifest_path}. Re-run without reuse_sv3d=True.")

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
    daz  = abs(((m_az - cond_az + 180.0) % 360.0) - 180.0)
    if daz > 0.5 or abs(m_el - cond_el) > 0.5:
        raise RuntimeError(
            f"Cached conditioning az/el=({m_az:.2f}, {m_el:.2f}) differs from "
            f"current ({cond_az:.2f}, {cond_el:.2f}). Re-run without reuse_sv3d=True."
        )

    views = []
    for entry in sorted(manifest.get("frames", []), key=lambda e: int(e["index"])):
        p = Path(entry["sv3d_raw_path"])
        if not p.exists():
            raise FileNotFoundError(f"Missing cached frame: {p}. Re-run without reuse_sv3d=True.")
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Cannot read cached frame: {p}")
        views.append(HallucinatedView(
            rgb=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
            azimuth_deg=float(entry["azimuth_deg"]),
            elevation_deg=float(entry["elevation_deg"]),
            is_conditioning=bool(entry.get("is_conditioning", False)),
        ))
    if not views:
        raise RuntimeError(f"Manifest at {manifest_path} has no frames.")
    logger.info("Cache reuse: loaded %d frames from %s.", len(views), manifest_path)
    return views


def _render_reference(scope, frame, gaussians, pipe_config,
                       az_deg, el_deg, resolution, up_override=None):
    fov_y = math.radians(_FOV_Y_DEG)
    fy    = 0.5 * resolution / math.tan(0.5 * fov_y)
    K = np.array([[fy, 0.0, resolution / 2.0],
                  [0.0, fy, resolution / 2.0],
                  [0.0, 0.0, 1.0]], dtype=np.float32)

    R_w2c, T_w2c, C_W = frame.virtual_to_world_camera(az_deg, el_deg)
    if up_override is not None:
        R_w2c, T_w2c = look_at(
            np.asarray(C_W, np.float32),
            np.asarray(scope.centroid, np.float32),
            np.asarray(up_override, np.float32),
        )

    cam = make_camera(R_w2c, T_w2c, K, resolution, resolution)
    out = render_rgba(gaussians, cam, pipe_config, bg_white=True, object_label_id=scope.object_label_id)

    rgb   = (out["rgb"].detach().clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    alpha = out["alpha"].detach().cpu().numpy()
    if alpha.ndim == 3:
        alpha = alpha[0]

    rgb, alpha = _tight_crop_pad_resize(rgb, alpha, resolution)
    return rgb, alpha


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


def run_hallucination(scope, frame, gaussians, pipe_config, *, scores, output_dir, reuse_sv3d=False):
    min_sv3d_mask_px = 200
    min_objgs_pixels = 600
    iou_threshold    = 0.20
    output_dir       = Path(output_dir)
    output_generated = output_dir / "generated"
    out_raw          = output_dir / "sv3d_raw"
    out_ref          = output_dir / "objgs_refs"
    for d in (output_generated, out_raw, out_ref):
        d.mkdir(parents=True, exist_ok=True)

    if not reuse_sv3d:
        index_file = output_dir / "hallucination_index.json"
        if index_file.exists():
            index_file.unlink()
        for d in (output_generated, out_raw, out_ref):
            for p in d.glob("*.png"):
                p.unlink()

    if not scores or not scores.get("top_k"):
        logger.warning("scores object has no top_k")
        raise RuntimeError("scores object has no top_k")

    top = scores["top_k"][0]
    rgba_path = Path(top["out_rgba_path"])
    if not rgba_path.exists():
        raise FileNotFoundError(
            f"input RGBA not found at {rgba_path}.")

    top_cam_idx = int(top["cam_index"])
    top_azimuth = float(top["azimuth_deg"])
    top_elevation = float(top["elevation_deg"])
    if not math.isfinite(top_azimuth):
        logger.warning("Conditioning azimuth is not finite: %s.", top_azimuth)
        top_azimuth = 0.0
    if not math.isfinite(top_elevation):
        logger.warning("Conditioning elevation is not finite: %s.", top_elevation)
        top_elevation = 0.0

    top_cam_up = None
    R = np.asarray(scope.cameras[top_cam_idx]["R"], np.float32)
    if R is not None:
        top_cam_up = -R[1]


    size = 512
    rgba = cv2.imread(str(rgba_path), cv2.IMREAD_UNCHANGED)
    rgb   = cv2.cvtColor(rgba[..., :3], cv2.COLOR_BGR2RGB)
    alpha = rgba[..., 3].astype(np.float32) / 255.0
    input_rgb, _ = _tight_crop_pad_resize(rgb, alpha, size)


    input_png = output_dir / "input.png"
    cv2.imwrite(str(input_png), cv2.cvtColor(input_rgb, cv2.COLOR_RGB2BGR))
    logger.info("input image: %s", input_png)

    if reuse_sv3d:
        views = _load_cache(output_dir, top_azimuth, top_elevation)
    else:
        backend = SV3DBackend()
        views = backend.hallucinate(input_rgb, top_elevation, top_azimuth, seed=0)
        backend.unload()

    if not views:
        raise RuntimeError("SV3D returned no views.")

    res    = views[0].rgb.shape[0]
    frames = []
    n_kept = 0

    for i, view in enumerate(views):
        ref_rgb, ref_alpha = _render_reference(
            scope, frame, gaussians, pipe_config,
            az_deg=view.azimuth_deg, el_deg=view.elevation_deg,
            resolution=res,
            up_override=top_cam_up,
        )

        m_objgs = ref_alpha > _ALPHA_THRESHOLD
        m_sv3d  = _alpha_from_white_bg(view.rgb)

        if m_objgs.shape != m_sv3d.shape:
            m_objgs = cv2.resize(m_objgs.astype(np.uint8),
                                 (m_sv3d.shape[1], m_sv3d.shape[0]),
                                 interpolation=cv2.INTER_NEAREST).astype(bool)

        iou     = _iou(m_sv3d, m_objgs)
        n_objgs = int(m_objgs.sum())
        n_sv3d  = int(m_sv3d.sum())

        accepted = True
        reason = "accepted"
        if n_sv3d < min_sv3d_mask_px:
            accepted = False
            reason = "sv3d_empty"
        elif n_objgs < min_objgs_pixels:
            accepted = False
            reason = "back_side_no_ref"
        elif iou < iou_threshold:
            accepted = False
            reason = f"iou_low_{iou:.2f}"

        if accepted:
            n_kept += 1

        stem   = f"{i:02d}__az{int(round(view.azimuth_deg)):+04d}"
        sv3d_p = out_raw          / f"{stem}.png"
        ref_p  = out_ref          / f"{stem}.png"
        rgba_p = output_generated / f"{stem}.png"

        cv2.imwrite(str(sv3d_p), cv2.cvtColor(view.rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(ref_p),  cv2.cvtColor(ref_rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(rgba_p), np.dstack([
            cv2.cvtColor(view.rgb, cv2.COLOR_RGB2BGR),
            (m_sv3d.astype(np.uint8) * 255),
        ]))

        frames.append(HallucinatedFrame(
            index=i,
            azimuth_deg=view.azimuth_deg,
            elevation_deg=view.elevation_deg,
            is_conditioning=view.is_conditioning,
            iou_with_objgs=iou,
            accepted=accepted,
            reject_reason=reason,
            out_rgba_path=str(rgba_p),
            sv3d_raw_path=str(sv3d_p),
            objgs_ref_path=str(ref_p),
        ))

    logger.info("Novel-view synthesis: kept %d / %d views (threshold IoU=%.2f).",
                n_kept, len(views), iou_threshold)

    result = {
        "backend": "sv3d",
        "n_views": len(views),
        "n_kept": n_kept,
        "object_label_id": scope.object_label_id,
        "conditioning": {
            "cam_index": top["cam_index"],
            "image_name": top.get("image_name", ""),
            "azimuth_deg": top_azimuth,
            "elevation_deg": top_elevation,
            "score": top.get("score", 0.0),
            "image_path": str(input_png),
            "rgba_path": str(rgba_path),
        },
        "params": {
            "iou_threshold": iou_threshold,
            "min_objgs_pixels": min_objgs_pixels,
            "fov_y_deg": _FOV_Y_DEG,
            "resolution": int(res),
            "seed": 0,
        },
        "frames": [asdict(fr) for fr in frames],
    }

    index_path = output_dir / "hallucination_index.json"
    with open(index_path, "w") as f:
        json.dump(result, f, indent=2)

    logger.info("Hallucination manifest: %s", index_path)
    return result
