from ModuleTBD.utils.scene_analysis import ObjectScope
from ModuleTBD.utils.transforms import ObjectFrame
import json
import logging
import math
from pathlib import Path
from typing import cast, Any

import cv2
import numpy as np

from ModuleTBD.utils.helpers import load_cache

from .constants import ALPHA_THRESH, FOV_Y_DEG, SV3D_FILL_FRAC
from .utils.gstrain_wrapper import make_camera, render_rgba
from .utils.sv3d_wrapper import SV3DBackend
from .utils.transforms import look_at

logger = logging.getLogger(__name__)


SV3D_INPUT_SIZE      = 576
MIN_SV3D_MASK_PIXELS = 200
MIN_OBJGS_PIXELS     = 600
IOU_THRESHOLD        = 0.20


def _resize(rgb, alpha, target_size):
    #resize and pad and center object and fill a fraction of the image
    fill_frac = SV3D_FILL_FRAC
    if alpha.max() > 1.5:
        alpha = alpha.astype(np.float32) / 255.0

    alpha = np.clip(alpha, 0.0, 1.0)
    mask = alpha > ALPHA_THRESH

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n_labels > 1:
        largest_component_idx = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
        mask = labels == largest_component_idx

    
    #if no pixels above threshold, make blank image
    y, x = np.where(mask)
    bg_value  = 255
    if len(x) == 0:
        return np.full((target_size, target_size, 3), bg_value, np.uint8), np.zeros((target_size, target_size), np.float32)

    x0, y0, x1, y1 = x.min(), y.min(), x.max() + 1, y.max() + 1
    
    longest = max(x1 - x0, y1 - y0)
    pad_amount  = int(round(longest * (1.0 - fill_frac) / (2.0 * fill_frac)))

    center_x, center_y = (x0 + x1) // 2, (y0 + y1) // 2
    half   = longest // 2 + pad_amount
    height, width   = rgb.shape[:2]
    src_x0, src_y0 = max(0, center_x - half), max(0, center_y - half)
    src_x1, src_y1 = min(width, center_x + half), min(height, center_y + half)
    crop_rgb   = rgb[src_y0:src_y1, src_x0:src_x1]
    crop_alpha = alpha[src_y0:src_y1, src_x0:src_x1]

    current_h, current_w = crop_rgb.shape[:2]
    size  = max(current_h, current_w)
    top_pad = (size - current_h) // 2
    left_pad = (size - current_w) // 2
    crop_rgb   = cv2.copyMakeBorder(crop_rgb,   top_pad, size - current_h - top_pad, left_pad, size - current_w - left_pad, cv2.BORDER_CONSTANT, value=(bg_value,) * 3)
    crop_alpha = cv2.copyMakeBorder(crop_alpha, top_pad, size - current_h - top_pad, left_pad, size - current_w - left_pad, cv2.BORDER_CONSTANT, value=0.0)

    bg = np.full_like(crop_rgb, bg_value)
    alpha_blending = (crop_alpha[..., None] * crop_rgb + (1.0 - crop_alpha[..., None]) * bg).astype(np.uint8)
    img = cv2.resize(alpha_blending,       (target_size, target_size), interpolation=cv2.INTER_AREA)
    crop_alpha = cv2.resize(crop_alpha, (target_size, target_size), interpolation=cv2.INTER_AREA)
    
    return img, crop_alpha


def _alpha_mask(rgb):
    saturation_thresh = 12
    value_thresh = 245
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    mask = ((s > saturation_thresh) | (v < value_thresh)).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask > 0



def _reference_alpha(scope, frame: ObjectFrame, gaussians, pipe_config, az_deg, el_deg, resolution, up_override=None):
    
    fov_y = math.radians(FOV_Y_DEG)
    fy = 0.5 * resolution / math.tan(0.5 * fov_y)
    K = np.array([[fy, 0.0, resolution / 2.0],
                  [0.0, fy, resolution / 2.0],
                  [0.0, 0.0, 1.0]])

    # R and T is for world to camera and C is camera position in world 
    R, T, C = frame.virtual_to_world_camera(az_deg, el_deg)
    if up_override is not None:
        R, T = look_at(C, scope.centroid, up_override)

    cam = make_camera(R, T, K, resolution, resolution)
    render = render_rgba(gaussians, cam, pipe_config, bg_white=True, object_label_id=scope.object_label_id)

    rgb   = (render["rgb"].detach().clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    alpha = render["alpha"].detach().cpu().numpy()
    if alpha.ndim == 3:
        alpha = alpha[0]

    _, alpha = _resize(rgb, alpha, resolution)
    return alpha


def run_hallucination(scope: ObjectScope, frame: ObjectFrame, gaussians, pipe_config, *, scores, output_dir, reuse_sv3d=False):
    output_dir    = Path(output_dir)
    generated_dir = output_dir / "generated"
    sv3d_raw_dir  = output_dir / "sv3d_raw"
    manifest_path = output_dir / "hallucination_index.json"
    for dir in (generated_dir, sv3d_raw_dir):
        dir.mkdir(parents=True, exist_ok=True)

    if not reuse_sv3d:
        if manifest_path.exists():
            manifest_path.unlink()
        for dir in (generated_dir, sv3d_raw_dir):
            for img in dir.glob("*.png"):
                img.unlink()

    if not scores or not scores.get("top_k"):
        logger.warning("scores object has no top_k")
        raise RuntimeError("scores object has no top_k")

    top = scores["top_k"][0]
    rgba_path = Path(top["rgba_path"])
    if not rgba_path.exists():
        raise FileNotFoundError(
            f"input RGBA not found at {rgba_path}.")

    top_cam_index = int(top["cam_index"])
    top_azimuth_deg = float(top["azimuth_deg"])
    top_elevation_deg = float(top["elevation_deg"])

    if not math.isfinite(top_azimuth_deg):
        raise ValueError(f"Conditioning azimuth is not finite: {top_azimuth_deg}")
    if not math.isfinite(top_elevation_deg):
        raise ValueError(f"Conditioning elevation is not finite: {top_elevation_deg}")


    top_cam_up = None
    R = np.asarray(scope.cameras[top_cam_index]["R"])
    if R is not None:
        top_cam_up = -R[1]

    rgba_matrix = cv2.imread(str(rgba_path), cv2.IMREAD_UNCHANGED)
    if rgba_matrix is None:
        raise ValueError(f"Failed to load image from {rgba_path}")
    rgba = np.asarray(rgba_matrix)
    rgb = cv2.cvtColor(cast(Any, rgba)[..., :3], cv2.COLOR_BGR2RGB)
    alpha = cast(Any, rgba)[..., 3].astype(np.float32) / 255.0
    input_rgb, _ = _resize(rgb, alpha, SV3D_INPUT_SIZE)

    if reuse_sv3d:
        views = load_cache(output_dir, top_azimuth_deg, top_elevation_deg)
    else:
        backend = SV3DBackend()
        views = backend.hallucinate(input_rgb, top_elevation_deg, top_azimuth_deg, seed=0)
        backend.unload()

    if not views:
        raise RuntimeError("SV3D returned no views.")

    resolution = views[0].rgb.shape[0]
    frames = []
    n_kept = 0

    for i, view in enumerate(views):
        ref_alpha = _reference_alpha(
            scope, frame, gaussians, pipe_config,
            az_deg=view.azimuth_deg,
            el_deg=view.elevation_deg,
            resolution=resolution,
            up_override=top_cam_up,
        )

        mask_ref = ref_alpha > ALPHA_THRESH
        mask_sv3d = _alpha_mask(view.rgb)

        if mask_ref.shape != mask_sv3d.shape:
            mask_ref = cv2.resize(mask_ref.astype(np.uint8), (mask_sv3d.shape[1], mask_sv3d.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)

        iou = float((mask_sv3d & mask_ref).sum() / max((mask_sv3d | mask_ref).sum(), 1))
        n_ref = int(mask_ref.sum())
        n_sv3d = int(mask_sv3d.sum())

        accepted = n_sv3d >= MIN_SV3D_MASK_PIXELS and n_ref >= MIN_OBJGS_PIXELS and iou >= IOU_THRESHOLD

        if accepted:
            n_kept += 1

        stem = f"{i:02d}__az{round(view.azimuth_deg):+04d}"
        sv3d_raw_path = sv3d_raw_dir / f"{stem}.png"
        generated_path = generated_dir / f"{stem}.png"

        cv2.imwrite(str(sv3d_raw_path), cv2.cvtColor(view.rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(generated_path), np.dstack([
            cv2.cvtColor(view.rgb, cv2.COLOR_RGB2BGR),
            (mask_sv3d.astype(np.uint8) * 255),
        ]))

        frames.append({
            "index": i,
            "azimuth_deg": view.azimuth_deg,
            "elevation_deg": view.elevation_deg,
            "is_conditioning": view.is_conditioning,
            "accepted": accepted,
            "rgba_path": str(generated_path),
        })

    logger.info("Novel-view synthesis: kept %d / %d views (threshold IoU=%.2f).",
                n_kept, len(views), IOU_THRESHOLD)

    result = {
        "n_views": len(views),
        "n_kept": n_kept,
        "conditioning": {
            "cam_index": top_cam_index,
            "azimuth_deg": top_azimuth_deg,
            "elevation_deg": top_elevation_deg,
        },
        "frames": frames,
    }

    with open(manifest_path, "w") as f:
        json.dump(result, f, indent=2)

    logger.info("Hallucination manifest: %s", manifest_path)
    return result
