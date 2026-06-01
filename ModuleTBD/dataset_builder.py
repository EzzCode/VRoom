import json
import logging
import math
from pathlib import Path

import cv2
import numpy as np

from .constants import (
    SV3D_FILL_FRAC,
    SEED_DEPTH_MIN,
    SEED_MIN_IN_FRONT,
    SEED_PERCENTILE_LO,
    SEED_PERCENTILE_HI,
    WS_CLIP_MIN,
    WS_CLIP_MAX,
)
from .utils.transforms import look_at

logger = logging.getLogger(__name__)

_VROOM_ROOT = Path(__file__).resolve().parents[1]


def _resolve_path(path_value, *, manifest_dir):
    p = Path(path_value)
    if p.is_absolute():
        return p
    for candidate in (manifest_dir / p, Path.cwd() / p, _VROOM_ROOT / p):
        if candidate.exists():
            return candidate
    return _VROOM_ROOT / p


def _rgba_to_rgb_mask(rgba):
    if rgba.ndim == 2:
        rgba = cv2.cvtColor(rgba, cv2.COLOR_GRAY2BGR)
    if rgba.shape[2] == 3:
        rgb  = cv2.cvtColor(rgba, cv2.COLOR_BGR2RGB)
        mask = rgb.mean(axis=2) < 250
        return rgb, mask
    bgr = rgba[..., :3].astype(np.float32)
    a   = rgba[..., 3:4].astype(np.float32) / 255.0
    out = (a * bgr + (1.0 - a) * 255.0).astype(np.uint8)
    rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    return rgb, (a[..., 0] > 0.5)


def _resize_rgb_mask_camera(rgb, mask, K, *, target_long_edge):
    h, w = rgb.shape[:2]
    if target_long_edge is None or int(target_long_edge) <= 0:
        return rgb, mask, K.astype(np.float32), w, h

    scale = min(1.0, float(target_long_edge) / float(max(w, h)))
    if scale >= 0.999:
        return rgb, mask, K.astype(np.float32), w, h

    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    sx = float(nw) / float(w)
    sy = float(nh) / float(h)

    rgb  = cv2.resize(rgb,  (nw, nh), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask.astype(np.uint8), (nw, nh),
                      interpolation=cv2.INTER_NEAREST) > 0
    K2 = K.astype(np.float32).copy()
    K2[0, :] *= sx
    K2[1, :] *= sy
    return rgb, mask, K2, nw, nh


def _compute_world_scale_px(seed_points_W, R_w2c, T_w2c, K, target_size):
    R   = np.asarray(R_w2c, np.float64)
    T   = np.asarray(T_w2c, np.float64).reshape(3)
    K64 = np.asarray(K, np.float64)
    fx  = float(K64[0, 0])
    fy  = float(K64[1, 1])
    cx  = float(K64[0, 2])
    cy  = float(K64[1, 2])
    sv3d_px = SV3D_FILL_FRAC * float(target_size)

    pts     = np.asarray(seed_points_W, np.float64)
    pts_c   = (R @ pts.T).T + T
    in_front = pts_c[:, 2] > SEED_DEPTH_MIN
    n_in_front = int(in_front.sum())

    if n_in_front < SEED_MIN_IN_FRONT:
        raise RuntimeError(
            f"Only {n_in_front} / {len(pts)} COLMAP seed points are in front of this "
            f"camera (depth > {SEED_DEPTH_MIN}). Expected >= {SEED_MIN_IN_FRONT}. "
            "Check that seed_points_W and the camera share the same world frame."
        )

    pf  = pts_c[in_front]
    u   = pf[:, 0] / pf[:, 2] * fx + cx
    v   = pf[:, 1] / pf[:, 2] * fy + cy
    u_lo = float(np.percentile(u, SEED_PERCENTILE_LO))
    u_hi = float(np.percentile(u, SEED_PERCENTILE_HI))
    v_lo = float(np.percentile(v, SEED_PERCENTILE_LO))
    v_hi = float(np.percentile(v, SEED_PERCENTILE_HI))
    world_px = max(u_hi - u_lo, v_hi - v_lo)
    return float(np.clip(world_px / max(sv3d_px, 1.0), WS_CLIP_MIN, WS_CLIP_MAX))


def build_supervision_views(halluc_index_path, extraction_index_path,
                             scope, frame, seed_points_W,
                             real_weight=1.0, hallucination_weight=1.0,
                             fov_y_deg=50.0, resolution=576,
                             real_target_long_edge=576,
                             up_override=None):
    if seed_points_W is None or len(seed_points_W) == 0:
        raise ValueError("seed_points_W must be a non-empty array.")

    real_views = []
    extraction_index_path = Path(extraction_index_path)
    if not extraction_index_path.exists():
        logger.warning("Extraction manifest not found: %s", extraction_index_path)
    else:
        with open(extraction_index_path) as f:
            manifest = json.load(f)

        for fr in manifest.get("frames", []):
            cam_index = int(fr["cam_index"])
            if cam_index < 0 or cam_index >= len(scope.cameras):
                logger.warning("Skipping real frame with invalid cam_index=%d.", cam_index)
                continue

            cam_p     = scope.cameras[cam_index]
            rgba_path = _resolve_path(fr["rgba_path"], manifest_dir=extraction_index_path.parent)
            if not rgba_path.exists():
                logger.warning("Missing real RGBA %s; skipping.", rgba_path)
                continue

            rgba = cv2.imread(str(rgba_path), cv2.IMREAD_UNCHANGED)
            if rgba is None:
                logger.warning("Cannot read %s; skipping.", rgba_path)
                continue

            rgb, mask = _rgba_to_rgb_mask(rgba)
            K = np.asarray(cam_p["K"], np.float32)
            rgb, mask, K, width, height = _resize_rgb_mask_camera(
                rgb, mask, K, target_long_edge=real_target_long_edge
            )

            if int(height) != int(width):
                side      = max(int(height), int(width))
                pad_top   = (side - int(height)) // 2
                pad_bot   = side - int(height) - pad_top
                pad_left  = (side - int(width))  // 2
                pad_right = side - int(width)  - pad_left
                rgb = cv2.copyMakeBorder(rgb, pad_top, pad_bot, pad_left, pad_right,
                                         cv2.BORDER_CONSTANT, value=(255, 255, 255))
                mask = cv2.copyMakeBorder(mask.astype(np.uint8),
                                          pad_top, pad_bot, pad_left, pad_right,
                                          cv2.BORDER_CONSTANT, value=0).astype(bool)
                K = K.copy()
                K[0, 2] += float(pad_left)
                K[1, 2] += float(pad_top)
                width  = side
                height = side

            real_views.append({
                "source": "real",
                "rgb": rgb,
                "mask": mask,
                "camera": {
                    "R": np.asarray(cam_p["R"], np.float32),
                    "T": np.asarray(cam_p["T"], np.float32),
                    "K": K,
                    "width": int(width),
                    "height": int(height),
                    "position": np.asarray(cam_p["position"], np.float32),
                    "azimuth_offset_deg": float(cam_p.get("azimuth_deg", fr.get("azimuth_deg", 0.0))),
                    "elevation_offset_deg": float(cam_p.get("elevation_deg", 0.0)),
                    "is_conditioning": False,
                    "frame_index": cam_index,
                },
                "weight": float(real_weight),
            })

    halluc_index_path = Path(halluc_index_path)
    if not halluc_index_path.exists():
        raise FileNotFoundError(f"Hallucination manifest not found: {halluc_index_path}")

    with open(halluc_index_path) as f:
        manifest = json.load(f)

    frames = manifest.get("frames", [])
    accepted = [fr for fr in frames if fr.get("accepted", False)]

    if not accepted:
        raise RuntimeError(f"No accepted hallucinated frames in {halluc_index_path}.")

    res  = int(resolution)
    fy_  = 0.5 * res / math.tan(0.5 * math.radians(float(fov_y_deg)))
    K_sv3d = np.array([[fy_, 0.0, res / 2.0],
                       [0.0, fy_, res / 2.0],
                       [0.0, 0.0, 1.0]], dtype=np.float32)

    halluc_views = []
    for fr in accepted:
        rgba_path = _resolve_path(fr["rgba_path"], manifest_dir=halluc_index_path.parent)
        if not rgba_path.exists():
            raise FileNotFoundError(
                f"Accepted hallucination RGBA missing: {rgba_path}. "
                "Re-run novel-view synthesis or check the output directory."
            )

        rgba = cv2.imread(str(rgba_path), cv2.IMREAD_UNCHANGED)
        if rgba is None:
            raise RuntimeError(f"cv2.imread returned None for: {rgba_path}")

        rgb, mask = _rgba_to_rgb_mask(rgba)
        if rgb.shape[0] != res or rgb.shape[1] != res:
            rgb  = cv2.resize(rgb,  (res, res), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask.astype(np.uint8), (res, res),
                              interpolation=cv2.INTER_NEAREST) > 0

        az_V = float(fr["azimuth_deg"])
        el_V = float(fr["elevation_deg"])

        R_w2c, T_w2c, C_W = frame.virtual_to_world_camera(az_V, el_V)

        if up_override is not None:
            up = np.asarray(up_override, np.float32)
            up = up / max(float(np.linalg.norm(up)), 1e-9)
            R_w2c, T_w2c = look_at(C_W, frame.centroid, up)

        ws = _compute_world_scale_px(
            seed_points_W,
            np.asarray(R_w2c, np.float64),
            np.asarray(T_w2c, np.float64),
            K_sv3d,
            res,
        )
        K_view = K_sv3d.copy()
        K_view[0, 0] = float(K_sv3d[0, 0] / ws)
        K_view[1, 1] = float(K_sv3d[1, 1] / ws)

        pts = np.asarray(seed_points_W, np.float64)
        pts_c = (np.asarray(R_w2c, np.float64) @ pts.T).T + np.asarray(T_w2c, np.float64).reshape(3)
        in_front = pts_c[:, 2] > SEED_DEPTH_MIN
        ys, xs = np.where(mask)
        valid = np.asarray([], bool)
        u = np.asarray([], np.float64)
        v = np.asarray([], np.float64)
        if int(in_front.sum()) >= SEED_MIN_IN_FRONT and len(xs) > 0:
            pts_f = pts_c[in_front]
            u = pts_f[:, 0] / pts_f[:, 2] * float(K_view[0, 0]) + float(K_view[0, 2])
            v = pts_f[:, 1] / pts_f[:, 2] * float(K_view[1, 1]) + float(K_view[1, 2])
            valid = (u >= 0) & (u < res) & (v >= 0) & (v < res)

        if int(valid.sum()) >= SEED_MIN_IN_FRONT:
            u = u[valid]
            v = v[valid]
            proj_bbox = (
                float(np.percentile(u, SEED_PERCENTILE_LO)),
                float(np.percentile(v, SEED_PERCENTILE_LO)),
                float(np.percentile(u, SEED_PERCENTILE_HI)),
                float(np.percentile(v, SEED_PERCENTILE_HI)),
            )
            img_bbox = (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))
            proj_cx = 0.5 * (proj_bbox[0] + proj_bbox[2])
            proj_cy = 0.5 * (proj_bbox[1] + proj_bbox[3])
            img_cx = 0.5 * (img_bbox[0] + img_bbox[2])
            img_cy = 0.5 * (img_bbox[1] + img_bbox[3])
            shift_x = float(np.clip(img_cx - proj_cx, -0.25 * res, 0.25 * res))
            shift_y = float(np.clip(img_cy - proj_cy, -0.25 * res, 0.25 * res))
            K_view[0, 2] += shift_x
            K_view[1, 2] += shift_y

        halluc_views.append({
            "source": "hallucinated",
            "rgb": rgb,
            "mask": mask,
            "camera": {
                "R": np.asarray(R_w2c, np.float32),
                "T": np.asarray(T_w2c, np.float32),
                "K": K_view,
                "width": res,
                "height": res,
                "position": np.asarray(C_W, np.float32),
                "azimuth_offset_deg": az_V,
                "elevation_offset_deg": el_V,
                "is_conditioning": bool(fr.get("is_conditioning", False)),
                "frame_index": int(fr.get("index", 0)),
            },
            "weight": float(hallucination_weight),
        })

    views = real_views + halluc_views
    logger.info("Supervision views ready: total=%d  real=%d  hallucinated=%d.",
                len(views), len(real_views), len(halluc_views))
    return views
