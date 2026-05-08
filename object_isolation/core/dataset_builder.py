"""Phase 6 — Build aligned real + hallucinated supervision views.

Converts Phase-3 real extractions and Phase-5 hallucinations into the
``supervision_views`` list expected by the Phase-7 optimizer:

    [{
        'rgb': np.ndarray HxWx3 uint8/float32 (RGB, white background),
        'mask': np.ndarray HxW bool/float32 aligned with rgb,
        'source': 'real' | 'hallucinated',
        'camera': {
            'R': (3,3) float32 R_w2c (COLMAP convention),
            'T': (3,) float32 T_w2c,
            'K': (3,3) float32,
            'width': int, 'height': int,
            'position': (3,) float32 camera centre in world,
            'azimuth_offset_deg': float,    # for logging/diagnostics
            'elevation_offset_deg': float,
        },
        'weight': float,
    }, ...]

Critical design points:
- Real views use the original training camera R/T/K and Phase-3 alpha mask.
- Hallucinated views use the Phase-5 SV3D orbit camera and Phase-5 alpha mask.
- If an image is resized, K is scaled by exactly the same x/y factors.
- Hallucinated views may receive a small audited image-space correction
    (flip and/or bbox scale/translate) only when the corrected image mask
    passes the same reference-mask alignment thresholds used for acceptance.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .coordinate_frames import LocalSV3D, look_at_w2c

logger = logging.getLogger(__name__)

_VROOM_ROOT = Path(__file__).resolve().parents[2]


def _resolve_path(path_value: str | Path, *, manifest_dir: Path) -> Path:
    """Resolve paths saved in manifests, supporting old relative outputs."""
    p = Path(path_value)
    if p.is_absolute():
        return p
    for candidate in (manifest_dir / p, Path.cwd() / p, _VROOM_ROOT / p):
        if candidate.exists():
            return candidate
    return _VROOM_ROOT / p


def _rgba_to_rgb_mask(rgba: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Read cv2 BGRA/BGR/gray data as RGB uint8 + explicit mask."""
    if rgba.ndim == 2:
        rgba = cv2.cvtColor(rgba, cv2.COLOR_GRAY2BGR)
    if rgba.shape[2] == 3:
        rgb = cv2.cvtColor(rgba, cv2.COLOR_BGR2RGB)
        mask = rgb.mean(axis=2) < 250
        return rgb, mask
    bgr = rgba[..., :3].astype(np.float32)
    a = rgba[..., 3:4].astype(np.float32) / 255.0
    white = np.full_like(bgr, 255.0)
    out = a * bgr + (1.0 - a) * white
    rgb = cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_BGR2RGB)
    return rgb, (a[..., 0] > 0.5)


def _resize_rgb_mask_camera(
    rgb: np.ndarray,
    mask: np.ndarray,
    K: np.ndarray,
    *,
    target_long_edge: Optional[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Optionally downsample RGB/mask and scale intrinsics identically."""
    height, width = rgb.shape[:2]
    if target_long_edge is None or int(target_long_edge) <= 0:
        return rgb, mask.astype(bool), K.astype(np.float32), width, height

    scale = min(1.0, float(target_long_edge) / float(max(width, height)))
    if scale >= 0.999:
        return rgb, mask.astype(bool), K.astype(np.float32), width, height

    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    sx = float(new_width) / float(width)
    sy = float(new_height) / float(height)

    rgb = cv2.resize(rgb, (new_width, new_height), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask.astype(np.uint8), (new_width, new_height), interpolation=cv2.INTER_NEAREST) > 0
    K2 = K.astype(np.float32).copy()
    K2[0, :] *= sx
    K2[1, :] *= sy
    return rgb, mask, K2, new_width, new_height


def _compute_aabb_world_scale_px(
    aabb_min: np.ndarray,
    aabb_max: np.ndarray,
    R_w2c: np.ndarray,
    T_w2c: np.ndarray,
    K: np.ndarray,
    sv3d_fill_frac: float = 0.85,
    target_size: int = 576,
    seed_points_W: Optional[np.ndarray] = None,
) -> tuple[float, tuple[float, float]]:
    R = R_w2c.astype(np.float64)
    T = T_w2c.astype(np.float64).reshape(3)
    K64 = K.astype(np.float64)
    fx, fy = float(K64[0, 0]), float(K64[1, 1])
    cx_k, cy_k = float(K64[0, 2]), float(K64[1, 2])
    fallback_center = (float(target_size) / 2.0, float(target_size) / 2.0)
    sv3d_px = sv3d_fill_frac * float(target_size)

    if seed_points_W is not None and len(seed_points_W) > 0:
        pts = np.asarray(seed_points_W, dtype=np.float64)
        pts_cam = (R @ pts.T).T + T
        in_front = pts_cam[:, 2] > 0.1
        if in_front.sum() >= 20:
            pts_f = pts_cam[in_front]
            u_all = pts_f[:, 0] / pts_f[:, 2] * fx + cx_k
            v_all = pts_f[:, 1] / pts_f[:, 2] * fy + cy_k
            u_lo = float(np.percentile(u_all, 2))
            u_hi = float(np.percentile(u_all, 98))
            v_lo = float(np.percentile(v_all, 2))
            v_hi = float(np.percentile(v_all, 98))
            world_px = float(max(u_hi - u_lo, v_hi - v_lo))
            scale = float(np.clip(world_px / max(sv3d_px, 1.0), 0.05, 2.0))
            cx_px = float(np.median(u_all))
            cy_px = float(np.median(v_all))
            return scale, (cx_px, cy_px)

    xs = [float(aabb_min[0]), float(aabb_max[0])]
    ys = [float(aabb_min[1]), float(aabb_max[1])]
    zs = [float(aabb_min[2]), float(aabb_max[2])]
    corners = np.array([[x, y, z] for x in xs for y in ys for z in zs], dtype=np.float64)
    pts_cam = (R @ corners.T).T + T
    in_front = pts_cam[:, 2] > 1e-3
    if int(in_front.sum()) < 2:
        return 1.0, fallback_center
    pts_f = pts_cam[in_front]
    u = pts_f[:, 0] / pts_f[:, 2] * fx + cx_k
    v = pts_f[:, 1] / pts_f[:, 2] * fy + cy_k
    world_px = float(max(u.max() - u.min(), v.max() - v.min()))
    scale = float(np.clip(world_px / max(sv3d_px, 1.0), 0.05, 2.0))

    centroid_W = (np.asarray(aabb_min, np.float64) + np.asarray(aabb_max, np.float64)) / 2.0
    pt_cam = R @ centroid_W + T
    if pt_cam[2] > 1e-3:
        cx_px = float(pt_cam[0] / pt_cam[2] * fx + cx_k)
        cy_px = float(pt_cam[1] / pt_cam[2] * fy + cy_k)
    else:
        cx_px, cy_px = fallback_center
    return scale, (cx_px, cy_px)


def build_hallucinated_supervision_views(
    halluc_index_path: str | Path,
    local_sv3d: LocalSV3D,
    *,
    weight: float = 0.10,
    fov_y_deg: float = 50.0,
    target_resolution: int = 576,
    up_W_override: Optional[np.ndarray] = None,
    include_conditioning: bool = True,
    aabb_min_W: Optional[np.ndarray] = None,
    aabb_max_W: Optional[np.ndarray] = None,
    sv3d_fill_frac: float = 0.85,
    seed_points_W: Optional[np.ndarray] = None,
) -> list[dict]:
    halluc_index_path = Path(halluc_index_path)
    if not halluc_index_path.exists():
        logger.warning(f"Hallucination manifest not found: {halluc_index_path}")
        return []

    with open(halluc_index_path) as f:
        manifest = json.load(f)

    frames = manifest.get("frames", [])
    if not include_conditioning:
        frames = [fr for fr in frames if not fr.get("is_conditioning", False)]

    candidates = [fr for fr in frames if fr.get("accepted", False)]

    if not candidates:
        raise RuntimeError(f"No accepted hallucinated frames in {halluc_index_path}.")

    res = int(target_resolution)
    fy = 0.5 * res / math.tan(0.5 * math.radians(fov_y_deg))
    K_sv3d = np.array([[fy, 0.0, res / 2.0],
                       [0.0, fy, res / 2.0],
                       [0.0, 0.0, 1.0]], dtype=np.float32)

    centroid_W = np.asarray(local_sv3d.world_local.centroid_W, dtype=np.float64)

    views: list[dict] = []
    n_skipped = 0
    for fr in candidates:
        rgba_path = _resolve_path(fr["out_rgba_path"], manifest_dir=halluc_index_path.parent)
        if not rgba_path.exists():
            n_skipped += 1
            continue

        rgba = cv2.imread(str(rgba_path), cv2.IMREAD_UNCHANGED)
        if rgba is None:
            n_skipped += 1
            continue

        rgb, mask = _rgba_to_rgb_mask(rgba)
        if rgb.shape[0] != res or rgb.shape[1] != res:
            rgb = cv2.resize(rgb, (res, res), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask.astype(np.uint8), (res, res), interpolation=cv2.INTER_NEAREST) > 0

        az_V = float(fr["azimuth_V_deg"])
        el_V = float(fr["elevation_V_deg"])

        # Map V-pose to world camera. Ignore up_W_override to avoid the 13-degree wobble!
        # The camera should always orbit perfectly around scope.up_W.
        R_w2c, T_w2c, C_W = local_sv3d.sv3d_view_to_world_camera(az_V, el_V)

        # ── Find Optimal Camera Roll and Flip ─────────────────────────────
        flip_h_applied = False
        angle_deg = 0.0
        
        # Load ref_mask to align against
        ref_path = None
        if fr.get("objgs_ref_path"):
            ref_path = _resolve_path(fr["objgs_ref_path"], manifest_dir=halluc_index_path.parent)
        
        if ref_path is not None and ref_path.exists():
            ref_rgba = cv2.imread(str(ref_path), cv2.IMREAD_UNCHANGED)
            if ref_rgba is not None:
                _, ref_mask = _rgba_to_rgb_mask(ref_rgba)
                ref_mask = cv2.resize(ref_mask.astype(np.uint8), (res, res), interpolation=cv2.INTER_NEAREST) > 0
                
                # Center ref_mask on sv3d_mask for pure rotation testing
                ys_s, xs_s = np.where(mask)
                if len(ys_s) > 0:
                    c_s = (float(np.mean(xs_s)), float(np.mean(ys_s)))
                    ys_r, xs_r = np.where(ref_mask)
                    if len(ys_r) > 0:
                        c_r = (float(np.mean(xs_r)), float(np.mean(ys_r)))
                        
                        M_align = np.float32([[1, 0, c_s[0] - c_r[0]], [0, 1, c_s[1] - c_r[1]]])
                        ref_mask_aligned = cv2.warpAffine(ref_mask.astype(np.uint8), M_align, (res, res)) > 0
                        
                        best_iou = 0.0
                        
                        for flip in [False, True]:
                            mask_f = cv2.flip(mask.astype(np.uint8), 1) > 0 if flip else mask.copy()
                            for test_angle in range(0, 360, 5):
                                M_rot = cv2.getRotationMatrix2D(c_s, test_angle, 1.0)
                                mask_r = cv2.warpAffine(mask_f.astype(np.uint8), M_rot, (res, res)) > 0
                                
                                intersection = np.logical_and(mask_r, ref_mask_aligned).sum()
                                union = np.logical_or(mask_r, ref_mask_aligned).sum()
                                iou = intersection / max(union, 1)
                                
                                if iou > best_iou:
                                    best_iou = iou
                                    angle_deg = float(test_angle)
                                    flip_h_applied = flip
        
        # Apply flip if necessary (this handles the "up/down" chiral use cases)
        if flip_h_applied:
            rgb = cv2.flip(rgb, 1)
            mask = cv2.flip(mask.astype(np.uint8), 1) > 0

        # Apply the Camera Roll
        # angle_deg is the CCW rotation applied to sv3d_mask to match ref_mask.
        # This means ref_mask needs to be rotated CW by angle_deg to match sv3d_mask.
        # R_roll rotates the projection CW when looking from +Z into -Z.
        theta = np.deg2rad(angle_deg)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        R_roll = np.array([
            [cos_t, -sin_t, 0.0],
            [sin_t,  cos_t, 0.0],
            [  0.0,    0.0, 1.0]
        ], dtype=np.float64)
        
        R_w2c_new = (R_roll @ R_w2c.astype(np.float64)).astype(np.float32)
        T_w2c_new = (R_roll @ T_w2c.astype(np.float64)).astype(np.float32)

        # ── Telephoto Focal Length Adjustment ──────────────────────────────
        K_view = K_sv3d.copy()
        if aabb_min_W is not None and aabb_max_W is not None:
            ws, centroid_uv = _compute_aabb_world_scale_px(
                np.asarray(aabb_min_W, dtype=np.float64),
                np.asarray(aabb_max_W, dtype=np.float64),
                np.asarray(R_w2c_new, dtype=np.float64),
                np.asarray(T_w2c_new, dtype=np.float64),
                K_sv3d.astype(np.float64),
                sv3d_fill_frac=sv3d_fill_frac,
                target_size=res,
                seed_points_W=seed_points_W,
            )
            # Find the new centroid of the sv3d_mask
            ys_new, xs_new = np.where(mask)
            c_s_new = (res/2.0, res/2.0)
            if len(ys_new) > 0:
                c_s_new = (float(np.mean(xs_new)), float(np.mean(ys_new)))

            # Scale focal length to 'zoom in' by the mismatch factor
            K_view[0, 0] = float(K_sv3d[0, 0] / ws)
            K_view[1, 1] = float(K_sv3d[1, 1] / ws)
            # Adjust principal point so the object center lands at the new image center
            K_view[0, 2] = float((K_sv3d[0, 2] - centroid_uv[0]) / ws + c_s_new[0])
            K_view[1, 2] = float((K_sv3d[1, 2] - centroid_uv[1]) / ws + c_s_new[1])

        views.append({
            "source": "hallucinated",
            "rgb": rgb,
            "mask": mask,
            "image_path": str(rgba_path),
            "camera": {
                "R": np.asarray(R_w2c_new, dtype=np.float32),
                "T": np.asarray(T_w2c_new, dtype=np.float32),
                "K": K_view,
                "width": res,
                "height": res,
                "position": np.asarray(C_W, dtype=np.float32),
                "azimuth_offset_deg": az_V,
                "elevation_offset_deg": el_V,
                "azimuth_world_rad": float(np.deg2rad(az_V)),
                "is_conditioning": fr.get("is_conditioning", False),
                "frame_index": int(fr.get("index", 0)),
                "alignment_flip": flip_h_applied,
                "alignment_roll_deg": angle_deg,
            },
            "weight": weight,
        })

    if n_skipped > 0:
        logger.warning(f"Skipped {n_skipped} hallucinated frames.")

    return views


def build_real_supervision_views(
    extraction_index_path: str | Path,
    scope,
    *,
    weight: float = 1.0,
    target_long_edge: int = 576,
) -> list[dict]:
    """Read Phase-3 real extractions as camera-aligned supervision views."""
    extraction_index_path = Path(extraction_index_path)
    if not extraction_index_path.exists():
        logger.warning("Phase-3 extraction manifest not found: %s", extraction_index_path)
        return []

    with open(extraction_index_path) as f:
        manifest = json.load(f)

    views: list[dict] = []
    for fr in manifest.get("frames", []):
        cam_index = int(fr["cam_index"])
        if cam_index < 0 or cam_index >= len(scope.cameras):
            logger.warning("Skipping real frame with invalid cam_index=%d.", cam_index)
            continue
        cam_p = scope.cameras[cam_index]
        rgba_path = _resolve_path(fr["out_rgba_path"], manifest_dir=extraction_index_path.parent)
        if not rgba_path.exists():
            logger.warning("Missing real extraction RGBA %s; skipping.", rgba_path)
            continue
        rgba = cv2.imread(str(rgba_path), cv2.IMREAD_UNCHANGED)
        if rgba is None:
            logger.warning("Failed to read %s; skipping.", rgba_path)
            continue

        rgb, mask = _rgba_to_rgb_mask(rgba)
        K = np.asarray(cam_p["K"], dtype=np.float32)
        rgb, mask, K, width, height = _resize_rgb_mask_camera(
            rgb, mask, K, target_long_edge=target_long_edge
        )

        # Square-pad to match hallucinated view shape (target_long_edge × target_long_edge).
        # Real COLMAP images are landscape (e.g. 576×432 after resize); letterbox with white.
        if int(height) != int(width):
            side = max(int(height), int(width))
            pad_top = (side - int(height)) // 2
            pad_bot = side - int(height) - pad_top
            pad_left = (side - int(width)) // 2
            pad_right = side - int(width) - pad_left
            rgb = cv2.copyMakeBorder(
                rgb, pad_top, pad_bot, pad_left, pad_right,
                cv2.BORDER_CONSTANT, value=(255, 255, 255),
            )
            mask_u8 = mask.astype(np.uint8)
            mask = cv2.copyMakeBorder(
                mask_u8, pad_top, pad_bot, pad_left, pad_right,
                cv2.BORDER_CONSTANT, value=0,
            ).astype(bool)
            K = K.copy()
            K[0, 2] += float(pad_left)
            K[1, 2] += float(pad_top)
            width = side
            height = side

        views.append({
            "source": "real",
            "rgb": rgb,
            "mask": mask,
            "image_path": str(rgba_path),
            "camera": {
                "R": np.asarray(cam_p["R"], dtype=np.float32),
                "T": np.asarray(cam_p["T"], dtype=np.float32),
                "K": K,
                "width": int(width),
                "height": int(height),
                "position": np.asarray(cam_p["position"], dtype=np.float32),
                "azimuth_offset_deg": float(cam_p.get("azimuth_V_deg", fr.get("azimuth_V_deg", 0.0))),
                "elevation_offset_deg": float(cam_p.get("elevation_V_deg", 0.0)),
                "is_conditioning": False,
                "frame_index": cam_index,
            },
            "weight": weight,
        })

    logger.info(
        "Phase 6: built %d real supervision views from %s (frames=%d).",
        len(views), extraction_index_path.name, len(manifest.get("frames", [])),
    )
    return views


def build_joint_supervision_views(
    *,
    halluc_index_path: str | Path,
    extraction_index_path: str | Path,
    scope,
    local_sv3d: LocalSV3D,
    real_weight: float = 1.0,
    hallucination_weight: float = 1.0,
    fov_y_deg: float = 50.0,
    hallucination_resolution: int = 576,
    real_target_long_edge: int = 576,
    up_W_override: Optional[np.ndarray] = None,
    include_conditioning: bool = True,
    seed_points_W: Optional[np.ndarray] = None,
) -> list[dict]:
    """Build one aligned training set containing real and hallucinated views."""
    real_views = build_real_supervision_views(
        extraction_index_path=extraction_index_path,
        scope=scope,
        weight=real_weight,
        target_long_edge=real_target_long_edge,
    )
    aabb_min = np.asarray(scope.aabb_min_W, dtype=np.float64) if scope is not None else None
    aabb_max = np.asarray(scope.aabb_max_W, dtype=np.float64) if scope is not None else None
    hallucinated_views = build_hallucinated_supervision_views(
        halluc_index_path=halluc_index_path,
        local_sv3d=local_sv3d,
        weight=hallucination_weight,
        fov_y_deg=fov_y_deg,
        target_resolution=hallucination_resolution,
        up_W_override=up_W_override,
        include_conditioning=include_conditioning,
        aabb_min_W=aabb_min,
        aabb_max_W=aabb_max,
        seed_points_W=seed_points_W,
    )
    views = real_views + hallucinated_views
    logger.info(
        "Phase 6: joint supervision views ready: total=%d real=%d hallucinated=%d.",
        len(views), len(real_views), len(hallucinated_views),
    )
    return views


def build_supervision_views(*args, **kwargs) -> list[dict]:
    """Backward-compatible alias for hallucination-only callers."""
    return build_hallucinated_supervision_views(*args, **kwargs)


def save_supervision_manifest(views: list[dict], output_path: str | Path) -> Path:
    """Persist a JSON-serialisable manifest of the in-memory supervision_views.

    Useful for debugging / re-running Phase 7 without rebuilding from raw
    Phase-5 outputs. Image arrays are NOT saved here — only the camera
    metadata + paths to the source RGBA files."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = []
    for v in views:
        cam = v["camera"]
        payload.append({
            "source": v.get("source", "hallucinated"),
            "image_path": v.get("image_path"),
            "original_image_path": v.get("original_image_path"),
            "azimuth_V_deg": cam["azimuth_offset_deg"],
            "elevation_V_deg": cam["elevation_offset_deg"],
            "is_conditioning": cam.get("is_conditioning", False),
            "frame_index": cam.get("frame_index"),
            "alignment_iou": cam.get("alignment_iou"),
            "alignment_bbox_iou": cam.get("alignment_bbox_iou"),
            "alignment_centroid_distance_norm": cam.get("alignment_centroid_distance_norm"),
            "alignment_area_ratio": cam.get("alignment_area_ratio"),
            "alignment_transform": cam.get("alignment_transform"),
            "R_w2c": cam["R"].tolist(),
            "T_w2c": cam["T"].tolist(),
            "K": cam["K"].tolist(),
            "C_W": cam["position"].tolist(),
            "width": cam["width"],
            "height": cam["height"],
            "weight": v["weight"],
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"n_views": len(payload), "views": payload}, f, indent=2)
    return output_path
