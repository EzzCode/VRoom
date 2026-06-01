import json
import logging
import math
from pathlib import Path

import cv2
import numpy as np

from .constants import (
    SEED_DEPTH_MIN,
    SEED_MIN_IN_FRONT,
    SEED_PERCENTILE_HI,
    SEED_PERCENTILE_LO,
    SV3D_FILL_FRAC,
    WS_CLIP_MAX,
    WS_CLIP_MIN,
    FOV_Y_DEG,
)
RESOLUTION = 576

from .utils.helpers import resolve_path
from .utils.transforms import look_at

logger = logging.getLogger(__name__)


def _rgba_to_rgb_mask(rgba):
    """convert rgba to rgb and mask (bool)"""
    if rgba.ndim == 2:
        rgba = cv2.cvtColor(rgba, cv2.COLOR_GRAY2BGR)
        
    if rgba.shape[2] == 3:
        # convert bgr to RGB and white pixels as background
        rgb = cv2.cvtColor(rgba, cv2.COLOR_BGR2RGB)
        mask = np.mean(rgb, axis=2) < 250.0
        return rgb, mask
        
    # rgba color channels over white background
    colors = rgba[..., :3].astype(np.float32)
    alpha = rgba[..., 3:4].astype(np.float32) / 255.0
    rgb = cv2.cvtColor((alpha * colors + (1.0 - alpha) * 255.0).astype(np.uint8), cv2.COLOR_BGR2RGB)
    mask = alpha[..., 0] > 0.5
    return rgb, mask


def build_supervision_views(generation_log_path, extraction_path,
                             scope, frame, cloud_points,
                             real_weight=1.0, generated_weight=1.0,
                             up_override=None):


    generation_log_path = Path(generation_log_path)
    if not generation_log_path.exists():
        raise FileNotFoundError(f"Generation log not found: {generation_log_path}")
        
    with open(generation_log_path) as f:
        generation_manifest = json.load(f)

    generation_frames = generation_manifest.get("frames", [])
    accepted_frames = [f for f in generation_frames if f.get("accepted", False)]

    if not accepted_frames:
        raise RuntimeError(f"No accepted frames in {generation_log_path}.")

    if cloud_points is None or len(cloud_points) == 0:
        raise ValueError("cloud_points cant be empty")

    real_views = []

    extraction_path = Path(extraction_path)
    
    if not extraction_path.exists():
        logger.warning("Extraction json not found: %s", extraction_path)
    else:
        with open(extraction_path) as f:
            extraction_data = json.load(f)

        # process real views
        for extracted_frame in extraction_data.get("frames", []):
            cam_index = int(extracted_frame["cam_index"])
            camera = scope.cameras[cam_index]
            rgba_path = resolve_path(extracted_frame["rgba_path"], manifest_dir=extraction_path)
            
            if not rgba_path.exists():
                logger.warning("Missing real RGBA %s.", rgba_path)
                continue

            rgba = cv2.imread(str(rgba_path), cv2.IMREAD_UNCHANGED)
            if rgba is None:
                logger.warning("Cannot read %s.", rgba_path)
                continue

            rgb, mask = _rgba_to_rgb_mask(rgba)
            K = np.asarray(camera["K"], np.float32)
            
            orig_height, orig_width = rgb.shape[:2]
            
            # downscale image
            scale = min(1.0, float(RESOLUTION) / float(max(orig_width, orig_height)))
            if scale < 1.0:
                width = max(1, int(round(orig_width * scale)))
                height = max(1, int(round(orig_height * scale)))
                scale_x = float(width) / float(orig_width)
                scale_y = float(height) / float(orig_height)
                
                rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
                mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST) > 0
                
                K_scaled = K.copy()
                K_scaled[0, :] *= scale_x
                K_scaled[1, :] *= scale_y
                K = K_scaled
            else:
                width, height = orig_width, orig_height

            # pad to square
            if height != width:
                max_dim = max(height, width)
                pad_top = (max_dim - height) // 2
                pad_bottom = max_dim - height - pad_top
                pad_left = (max_dim - width) // 2
                pad_right = max_dim - width - pad_left
                
                rgb = cv2.copyMakeBorder(
                    rgb, pad_top, pad_bottom, pad_left, pad_right,
                    cv2.BORDER_CONSTANT, value=(255, 255, 255))
                mask = cv2.copyMakeBorder(
                    mask.astype(np.uint8), pad_top, pad_bottom, pad_left, pad_right,
                    cv2.BORDER_CONSTANT, value=0).astype(bool)
                
                K_padded = K.copy()
                K_padded[0, 2] += float(pad_left)
                K_padded[1, 2] += float(pad_top)
                K = K_padded
                width = max_dim
                height = max_dim

            real_views.append({
                "source": "real",
                "rgb": rgb,
                "mask": mask,
                "camera": {
                    "R": np.asarray(camera["R"], np.float32),
                    "T": np.asarray(camera["T"], np.float32),
                    "K": K,
                    "width": int(width),
                    "height": int(height),
                    "position": np.asarray(camera["position"], np.float32),
                    "azimuth_offset_deg": float(camera.get("azimuth_deg", extracted_frame["azimuth_deg"])),
                    "elevation_offset_deg": float(camera.get("elevation_deg", extracted_frame["elevation_deg"])),
                    "frame_index": cam_index,
                },
                "weight": float(real_weight),
            })

    # Setup SV3D intrinsic matrix using vertical FOV
    target_res = int(RESOLUTION)
    focal_y = 0.5 * target_res / math.tan(0.5 * math.radians(FOV_Y_DEG))
    K_sv3d = np.array([[focal_y, 0.0, target_res / 2.0],
                       [0.0, focal_y, target_res / 2.0],
                       [0.0, 0.0, 1.0]], dtype=np.float32)

    generated_views = []
    
    # Process generated camera views
    for gen_frame in accepted_frames:
        rgba_path = resolve_path(gen_frame["rgba_path"], manifest_dir=generation_log_path.parent)
        if not rgba_path.exists():
            raise FileNotFoundError(f"Accepted generated image missing: {rgba_path}.")

        rgba = cv2.imread(str(rgba_path), cv2.IMREAD_UNCHANGED)
        if rgba is None:
            raise RuntimeError(f"cv2.imread returned None for: {rgba_path}")

        rgb, mask = _rgba_to_rgb_mask(rgba)
        if rgb.shape[0] != target_res or rgb.shape[1] != target_res:
            rgb = cv2.resize(rgb, (target_res, target_res), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask.astype(np.uint8), (target_res, target_res), interpolation=cv2.INTER_NEAREST) > 0

        azimuth_deg = float(gen_frame["azimuth_deg"])
        elevation_deg = float(gen_frame["elevation_deg"])

        R_world_to_cam, T_world_to_cam, cam_pos_world = frame.virtual_to_world_camera(azimuth_deg, elevation_deg)

        if up_override is not None:
            up_vector = np.asarray(up_override, np.float32)
            up_vector = up_vector / max(float(np.linalg.norm(up_vector)), 1e-9)
            R_world_to_cam, T_world_to_cam = look_at(cam_pos_world, frame.centroid, up_vector)

        # Compute projection scale to match the target filling fraction
        R = np.asarray(R_world_to_cam, np.float64)
        T = np.asarray(T_world_to_cam, np.float64).reshape(3)
        K = np.asarray(K_sv3d, np.float64)
        fx = float(K[0, 0])
        fy = float(K[1, 1])
        center_x = float(K[0, 2])
        center_y = float(K[1, 2])
        sv3d_px = SV3D_FILL_FRAC * float(target_res)

        points_world = np.asarray(cloud_points, np.float64)
        points_cam = (R @ points_world.T).T + T
        is_in_front = points_cam[:, 2] > SEED_DEPTH_MIN
        num_front_points = int(is_in_front.sum())

        if num_front_points < SEED_MIN_IN_FRONT:
            raise RuntimeError(
                f"Only {num_front_points} / {len(points_world)} seed points are in front of this "
                f"camera (depth > {SEED_DEPTH_MIN}). Expected >= {SEED_MIN_IN_FRONT}."
            )

        front_points_cam = points_cam[is_in_front]
        projected_u = front_points_cam[:, 0] / front_points_cam[:, 2] * fx + center_x
        projected_v = front_points_cam[:, 1] / front_points_cam[:, 2] * fy + center_y
        
        u_lo = float(np.percentile(projected_u, SEED_PERCENTILE_LO))
        u_hi = float(np.percentile(projected_u, SEED_PERCENTILE_HI))
        v_lo = float(np.percentile(projected_v, SEED_PERCENTILE_LO))
        v_hi = float(np.percentile(projected_v, SEED_PERCENTILE_HI))
        
        projected_span = max(u_hi - u_lo, v_hi - v_lo)
        world_scale = float(np.clip(projected_span / max(sv3d_px, 1.0), WS_CLIP_MIN, WS_CLIP_MAX))

        K_view = K_sv3d.copy()
        K_view[0, 0] = float(K_sv3d[0, 0] / world_scale)
        K_view[1, 1] = float(K_sv3d[1, 1] / world_scale)

        # Center-align the projected point cloud bounding box with the actual mask image bounding box
        ys, xs = np.where(mask)
        if len(xs) > 0:
            # Re-project using the newly adjusted focal lengths
            projected_u = front_points_cam[:, 0] / front_points_cam[:, 2] * float(K_view[0, 0]) + float(K_view[0, 2])
            projected_v = front_points_cam[:, 1] / front_points_cam[:, 2] * float(K_view[1, 1]) + float(K_view[1, 2])
            in_bounds = (projected_u >= 0) & (projected_u < target_res) & (projected_v >= 0) & (projected_v < target_res)
            
            if int(in_bounds.sum()) >= SEED_MIN_IN_FRONT:
                valid_u = projected_u[in_bounds]
                valid_v = projected_v[in_bounds]
                proj_bbox = (
                    float(np.percentile(valid_u, SEED_PERCENTILE_LO)),
                    float(np.percentile(valid_v, SEED_PERCENTILE_LO)),
                    float(np.percentile(valid_u, SEED_PERCENTILE_HI)),
                    float(np.percentile(valid_v, SEED_PERCENTILE_HI)),
                )
                img_bbox = (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))
                proj_cx = 0.5 * (proj_bbox[0] + proj_bbox[2])
                proj_cy = 0.5 * (proj_bbox[1] + proj_bbox[3])
                img_cx = 0.5 * (img_bbox[0] + img_bbox[2])
                img_cy = 0.5 * (img_bbox[1] + img_bbox[3])
                
                # Limit shift to 25% of the frame resolution
                shift_x = float(np.clip(img_cx - proj_cx, -0.25 * target_res, 0.25 * target_res))
                shift_y = float(np.clip(img_cy - proj_cy, -0.25 * target_res, 0.25 * target_res))
                K_view[0, 2] += shift_x
                K_view[1, 2] += shift_y

        generated_views.append({
            "source": "SV3D",
            "rgb": rgb,
            "mask": mask,
            "camera": {
                "R": np.asarray(R_world_to_cam, np.float32),
                "T": np.asarray(T_world_to_cam, np.float32),
                "K": K_view,
                "width": target_res,
                "height": target_res,
                "position": np.asarray(cam_pos_world, np.float32),
                "azimuth_offset_deg": azimuth_deg,
                "elevation_offset_deg": elevation_deg,
                "frame_index": int(gen_frame.get("index", 0)),
            },
            "weight": float(generated_weight),
        })

    views = real_views + generated_views
    logger.info("Supervision views ready: total=%d  real=%d  generated=%d.",
                len(views), len(real_views), len(generated_views))
    return views
