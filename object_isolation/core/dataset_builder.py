"""Build aligned real + hallucinated supervision views.

Converts extraction outputs and novel-view hallucinations into the
``supervision_views`` list expected by the object-training optimizer:

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
- Real views use the original training camera R/T/K and extraction alpha mask.
- Hallucinated views use the novel-view SV3D orbit camera and novel-view alpha mask.
- If an image is resized, K is scaled by exactly the same x/y factors.
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
from .object_scope import ObjectScope

logger = logging.getLogger(__name__)

_VROOM_ROOT = Path(__file__).resolve().parents[2]

# ── Shared constants — must match hallucination.py exactly ────────────────────
# SV3D conditioning and output properties
_SV3D_FILL_FRAC: float = 0.85    # fraction of output resolution the object fills

# Seed-point projection thresholds
_SEED_DEPTH_MIN: float = 0.1     # minimum camera-space depth to treat a point as "in front"
_SEED_MIN_IN_FRONT: int = 20     # minimum in-front points required to trust p2–p98 estimate
_SEED_PERCENTILE_LO: float = 2   # lower percentile for outlier-robust extent
_SEED_PERCENTILE_HI: float = 98  # upper percentile for outlier-robust extent

# World-scale clamp: prevents absurd telephoto/wide-angle K adjustments
_WS_CLIP_MIN: float = 0.05
_WS_CLIP_MAX: float = 2.0

# Camera-pose optimisation (CPO): maximum elevation correction allowed per
# hallucinated view.  SV3D elevation drift is typically < 5°; 12° is a
# conservative safety bound that prevents the optimiser from over-fitting.
_CPO_MAX_DELTA_EL_DEG: float = 12.0


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


def _mask_fill_stats(mask: np.ndarray, min_pixels: int = 64) -> tuple[float, float, float]:
    """Return (cx_px, cy_px, fill_px) measured from an SV3D boolean mask.

    cx_px, cy_px — 2D centroid of the foreground in pixel coordinates.
    fill_px      — max(bbox_width, bbox_height) of the foreground region.

    Falls back to image-centre cx/cy and _SV3D_FILL_FRAC-based fill when
    fewer than ``min_pixels`` foreground pixels are present.
    """
    res_h, res_w = mask.shape[:2]
    fallback_cx = float(res_w) / 2.0
    fallback_cy = float(res_h) / 2.0
    fallback_fill = _SV3D_FILL_FRAC * float(max(res_h, res_w))

    ys, xs = np.where(mask)
    if len(xs) < min_pixels:
        return fallback_cx, fallback_cy, fallback_fill

    cx = float(xs.mean())
    cy = float(ys.mean())
    fill = float(max(int(xs.max()) - int(xs.min()) + 1,
                     int(ys.max()) - int(ys.min()) + 1))
    if fill < 1.0:
        return fallback_cx, fallback_cy, fallback_fill

    return cx, cy, fill


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


def _compute_world_scale_px(
    seed_points_W: np.ndarray,
    R_w2c: np.ndarray,
    T_w2c: np.ndarray,
    K: np.ndarray,
    target_size: int,
    *,
    sv3d_px_override: float | None = None,
) -> float:
    """Return ws from COLMAP seed point projection.

    ws  — ratio of world object extent in K_sv3d pixels to sv3d_px.
          K_view = K_sv3d / ws gives telephoto intrinsics that make the
          projected seed-point bounding box span exactly sv3d_px pixels.

    sv3d_px_override — if given, use this as the target pixel extent instead
          of the fixed _SV3D_FILL_FRAC * target_size assumption.  Pass the
          mask bounding-box extent from _mask_fill_stats for a per-view,
          data-driven estimate.

    Raises RuntimeError if fewer than _SEED_MIN_IN_FRONT points are in
    front of this camera (pathological case — signals a data problem).
    """
    R = R_w2c.astype(np.float64)
    T = T_w2c.astype(np.float64).reshape(3)
    K64 = K.astype(np.float64)
    fx, fy = float(K64[0, 0]), float(K64[1, 1])
    cx_k, cy_k = float(K64[0, 2]), float(K64[1, 2])
    sv3d_px = float(sv3d_px_override) if sv3d_px_override is not None else _SV3D_FILL_FRAC * float(target_size)

    pts = np.asarray(seed_points_W, dtype=np.float64)
    pts_cam = (R @ pts.T).T + T
    in_front = pts_cam[:, 2] > _SEED_DEPTH_MIN
    n_in_front = int(in_front.sum())
    if n_in_front < _SEED_MIN_IN_FRONT:
        raise RuntimeError(
            f"Only {n_in_front} of {len(pts)} COLMAP seed points are in front of this camera "
            f"(depth > {_SEED_DEPTH_MIN}).  Expected >= {_SEED_MIN_IN_FRONT}.  "
            "Check that seed_points_W and the camera are in the same world frame."
        )

    pts_f = pts_cam[in_front]
    u_all = pts_f[:, 0] / pts_f[:, 2] * fx + cx_k
    v_all = pts_f[:, 1] / pts_f[:, 2] * fy + cy_k
    u_lo = float(np.percentile(u_all, _SEED_PERCENTILE_LO))
    u_hi = float(np.percentile(u_all, _SEED_PERCENTILE_HI))
    v_lo = float(np.percentile(v_all, _SEED_PERCENTILE_LO))
    v_hi = float(np.percentile(v_all, _SEED_PERCENTILE_HI))
    world_px = float(max(u_hi - u_lo, v_hi - v_lo))
    ws = float(np.clip(world_px / max(sv3d_px, 1.0), _WS_CLIP_MIN, _WS_CLIP_MAX))
    return ws


def _cpo_refine_elevation(
    seed_points_W: np.ndarray,
    K_view: np.ndarray,
    centroid_W: np.ndarray,
    up_W: np.ndarray,
    local_sv3d: "LocalSV3D",
    az_V: float,
    el_V: float,
    mask_cx: float,
    mask_cy: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Optimise the SV3D elevation offset (cause #3 correction).

    SV3D is a diffusion model — its rendered elevation is not guaranteed to
    match the geometrically-computed angle.  This finds the small δel that
    minimises the distance between the median projected-seed-point position
    and the mask centroid (mask_cx, mask_cy).

    Returns (R_w2c, T_w2c, C_W, delta_el_deg) all as float64 arrays.
    On any failure the nominal pose is returned with delta_el_deg = 0.0.
    """
    from scipy.optimize import minimize_scalar

    fx = float(K_view[0, 0])
    fy = float(K_view[1, 1])
    cx = float(K_view[0, 2])
    cy = float(K_view[1, 2])
    pts = np.asarray(seed_points_W, dtype=np.float64)

    def _residual(delta_el: float) -> float:
        R_c, T_c, C_c = local_sv3d.sv3d_view_to_world_camera(az_V, el_V + delta_el)
        R_c, T_c = look_at_w2c(np.asarray(C_c, dtype=np.float64), centroid_W, up_W)
        R_c = np.asarray(R_c, dtype=np.float64)
        T_c = np.asarray(T_c, dtype=np.float64).reshape(3)
        pts_cam = (R_c @ pts.T).T + T_c
        in_front = pts_cam[:, 2] > _SEED_DEPTH_MIN
        if in_front.sum() < _SEED_MIN_IN_FRONT:
            return 1e9
        pf = pts_cam[in_front]
        u = pf[:, 0] / pf[:, 2] * fx + cx
        v = pf[:, 1] / pf[:, 2] * fy + cy
        # Use percentile-based centroid to match _compute_world_scale_px and be
        # robust to outlier COLMAP points behind thin surfaces.
        mu_u = float(np.percentile(u, 50))
        mu_v = float(np.percentile(v, 50))
        return float((mu_u - mask_cx) ** 2 + (mu_v - mask_cy) ** 2)

    def _make_pose(delta_el: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        R_c, T_c, C_c = local_sv3d.sv3d_view_to_world_camera(az_V, el_V + delta_el)
        R_ref, T_ref = look_at_w2c(np.asarray(C_c, dtype=np.float64), centroid_W, up_W)
        return (
            np.asarray(R_ref, dtype=np.float64),
            np.asarray(T_ref, dtype=np.float64).reshape(3),
            np.asarray(C_c, dtype=np.float64).reshape(3),
        )

    try:
        result = minimize_scalar(
            _residual,
            bounds=(-_CPO_MAX_DELTA_EL_DEG, _CPO_MAX_DELTA_EL_DEG),
            method="bounded",
            options={"xatol": 0.05},   # 0.05° precision is more than enough
        )
        delta_el = float(result.x)
    except Exception as exc:
        logger.debug("CPO elevation search failed (%s); using nominal pose.", exc)
        R_n, T_n, C_n = _make_pose(0.0)
        return R_n, T_n, C_n, 0.0

    if abs(delta_el) >= _CPO_MAX_DELTA_EL_DEG * 0.99:
        logger.warning(
            "CPO reached elevation bound (δel=%.2f°) for az=%.1f el=%.1f — "
            "check seed point coverage for this view.",
            delta_el, az_V, el_V,
        )
    R_ref, T_ref, C_ref = _make_pose(delta_el)
    return R_ref, T_ref, C_ref, delta_el


def build_hallucinated_supervision_views(
    halluc_index_path: str | Path,
    local_sv3d: LocalSV3D,
    *,
    seed_points_W: np.ndarray,
    weight: float = 0.10,
    fov_y_deg: float = 50.0,
    target_resolution: int = 576,
    up_W_override: Optional[np.ndarray] = None,
    include_conditioning: bool = True,
    enable_cpo: bool = True,
) -> list[dict]:
    """Build hallucinated supervision views from novel-view outputs.

    seed_points_W is required — it is used to compute per-view telephoto
    intrinsics (K_view) that make the object span _SV3D_FILL_FRAC of the
    target image exactly as SV3D rendered it.

    enable_cpo — when True (default) a 1-D elevation search is run per frame
    (cause #3 correction): the elevation angle is perturbed by up to
    ±_CPO_MAX_DELTA_EL_DEG to minimise the distance between the median
    projected seed-point position and the mask centroid.
    """
    if seed_points_W is None or len(seed_points_W) == 0:
        raise ValueError(
            "seed_points_W must be a non-empty array of COLMAP seed points.  "
            "They are required to compute per-view telephoto K_view."
        )

    halluc_index_path = Path(halluc_index_path)
    if not halluc_index_path.exists():
        raise FileNotFoundError(f"Hallucination manifest not found: {halluc_index_path}")

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
    for fr in candidates:
        rgba_path = _resolve_path(fr["out_rgba_path"], manifest_dir=halluc_index_path.parent)
        if not rgba_path.exists():
            raise FileNotFoundError(
                f"Accepted hallucination RGBA missing: {rgba_path}\n"
                f"  Frame: az={fr.get('azimuth_V_deg')} el={fr.get('elevation_V_deg')}\n"
                "  Re-run novel-view synthesis or check the output directory."
            )

        rgba = cv2.imread(str(rgba_path), cv2.IMREAD_UNCHANGED)
        if rgba is None:
            raise RuntimeError(f"cv2.imread failed (returned None) for accepted frame: {rgba_path}")

        rgb, mask = _rgba_to_rgb_mask(rgba)
        if rgb.shape[0] != res or rgb.shape[1] != res:
            rgb = cv2.resize(rgb, (res, res), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask.astype(np.uint8), (res, res), interpolation=cv2.INTER_NEAREST) > 0

        az_V = float(fr["azimuth_V_deg"])
        el_V = float(fr["elevation_V_deg"])

        R_w2c, T_w2c, C_W = local_sv3d.sv3d_view_to_world_camera(az_V, el_V)
        # Resolve the up vector used for look_at; needed by CPO to stay consistent.
        if up_W_override is not None:
            up_W = np.asarray(up_W_override, dtype=np.float64).reshape(3)
            up_W = up_W / max(float(np.linalg.norm(up_W)), 1e-9)
            R_w2c, T_w2c = look_at_w2c(np.asarray(C_W, dtype=np.float64), centroid_W, up_W)
        else:
            up_W = np.asarray(local_sv3d.world_local.up_W, dtype=np.float64)

        # ── Compute telephoto intrinsics K_view ───────────────────────────
        # Focal length is scaled so the projected COLMAP seed-point bounding
        # box spans _SV3D_FILL_FRAC of the target resolution.
        # The mask centroid (mask_cx, mask_cy) is read for CPO targeting only —
        # the principal point stays at image centre (K_sv3d default).
        mask_cx, mask_cy, _ = _mask_fill_stats(mask)
        ws = _compute_world_scale_px(
            seed_points_W=seed_points_W,
            R_w2c=np.asarray(R_w2c, dtype=np.float64),
            T_w2c=np.asarray(T_w2c, dtype=np.float64),
            K=K_sv3d.astype(np.float64),
            target_size=res,
        )
        K_view = K_sv3d.copy()
        K_view[0, 0] = float(K_sv3d[0, 0] / ws)
        K_view[1, 1] = float(K_sv3d[1, 1] / ws)

        # ── Cause #3: CPO elevation refinement ────────────────────────────
        if enable_cpo:
            R_w2c, T_w2c, C_W, cpo_delta_el = _cpo_refine_elevation(
                seed_points_W=seed_points_W,
                K_view=K_view,
                centroid_W=centroid_W,
                up_W=up_W,
                local_sv3d=local_sv3d,
                az_V=az_V,
                el_V=el_V,
                mask_cx=mask_cx,
                mask_cy=mask_cy,
            )
            logger.debug(
                "CPO  az=%+6.1f  el=%+6.1f  →  δel=%+.2f°",
                az_V, el_V, cpo_delta_el,
            )
        else:
            cpo_delta_el = 0.0

        views.append({
            "source": "hallucinated",
            "rgb": rgb,
            "mask": mask,
            "image_path": str(rgba_path),
            "camera": {
                "R": np.asarray(R_w2c, dtype=np.float32),
                "T": np.asarray(T_w2c, dtype=np.float32),
                "K": K_view,
                "width": res,
                "height": res,
                "position": np.asarray(C_W, dtype=np.float32),
                "azimuth_offset_deg": az_V,
                "elevation_offset_deg": el_V,
                "elevation_cpo_delta_deg": cpo_delta_el,
                "azimuth_world_rad": float(np.deg2rad(az_V)),
                "is_conditioning": fr.get("is_conditioning", False),
                "frame_index": int(fr.get("index", 0)),
            },
            "weight": weight,
        })

    return views


def build_real_supervision_views(
    extraction_index_path: str | Path,
    scope: ObjectScope,
    *,
    weight: float = 1.0,
    target_long_edge: int = 576,
) -> list[dict]:
    """Read extraction outputs as camera-aligned supervision views."""
    extraction_index_path = Path(extraction_index_path)
    if not extraction_index_path.exists():
        logger.warning("Extraction manifest not found: %s", extraction_index_path)
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
        "Built %d real supervision views from %s (frames=%d).",
        len(views), extraction_index_path.name, len(manifest.get("frames", [])),
    )
    return views


def build_joint_supervision_views(
    *,
    halluc_index_path: str | Path,
    extraction_index_path: str | Path,
    scope: ObjectScope,
    local_sv3d: LocalSV3D,
    seed_points_W: np.ndarray,
    real_weight: float = 1.0,
    hallucination_weight: float = 1.0,
    fov_y_deg: float = 50.0,
    hallucination_resolution: int = 576,
    real_target_long_edge: int = 576,
    up_W_override: Optional[np.ndarray] = None,
    include_conditioning: bool = True,
) -> list[dict]:
    """Build one aligned training set containing real and hallucinated views."""
    real_views = build_real_supervision_views(
        extraction_index_path=extraction_index_path,
        scope=scope,
        weight=real_weight,
        target_long_edge=real_target_long_edge,
    )
    hallucinated_views = build_hallucinated_supervision_views(
        halluc_index_path=halluc_index_path,
        local_sv3d=local_sv3d,
        seed_points_W=seed_points_W,
        weight=hallucination_weight,
        fov_y_deg=fov_y_deg,
        target_resolution=hallucination_resolution,
        up_W_override=up_W_override,
        include_conditioning=include_conditioning,
    )
    views = real_views + hallucinated_views
    logger.info(
        "Joint supervision views ready: total=%d real=%d hallucinated=%d.",
        len(views), len(real_views), len(hallucinated_views),
    )
    return views


def write_projection_overlays(
    xyz_W: np.ndarray,
    supervision_views: list[dict],
    output_dir: str | Path,
) -> None:
    """Project COLMAP seed points onto every supervision view and save overlay images.

    Called automatically after supervision building to verify coordinate-frame alignment.
    Colour-codes each projected point by depth (JET colormap: blue=near, red=far).
    Green contour = supervision mask.  Yellow text = source / az / el / stats.

    If the dots DON'T trace the object silhouette the camera coordinate frame
    is broken and training data is incorrect.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    xyz = np.asarray(xyz_W, dtype=np.float64)
    for i, view in enumerate(supervision_views):
        cam = view["camera"]
        R = np.asarray(cam["R"], dtype=np.float64)
        T = np.asarray(cam["T"], dtype=np.float64).flatten()
        K = np.asarray(cam["K"], dtype=np.float64)
        W, H = int(cam["width"]), int(cam["height"])
        source = view.get("source", "?")
        az = float(cam.get("azimuth_offset_deg", 0.0))
        el = float(cam.get("elevation_offset_deg", 0.0))

        # Project
        pts_c = (R @ xyz.T).T + T.reshape(1, 3)
        in_front = pts_c[:, 2] > _SEED_DEPTH_MIN
        pts_f = pts_c[in_front]

        # Build BGR overlay
        rgb = np.asarray(view["rgb"], dtype=np.uint8)
        if rgb.shape[0] != H or rgb.shape[1] != W:
            rgb = cv2.resize(rgb, (W, H))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        # Green mask contour
        mask_u8 = np.asarray(view["mask"], dtype=np.uint8) * 255
        if mask_u8.shape[0] != H or mask_u8.shape[1] != W:
            mask_u8 = cv2.resize(mask_u8, (W, H), interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(bgr, contours, -1, (0, 255, 0), 2)

        n_in_frame = 0
        if pts_f.shape[0] > 0:
            x = pts_f[:, 0] / pts_f[:, 2]
            y = pts_f[:, 1] / pts_f[:, 2]
            u = (K[0, 0] * x + K[0, 2]).astype(np.float32)
            v = (K[1, 1] * y + K[1, 2]).astype(np.float32)
            valid = (u >= 0) & (u < W) & (v >= 0) & (v < H)
            depths = pts_f[valid, 2]
            u_v = u[valid].astype(np.int32)
            v_v = v[valid].astype(np.int32)
            n_in_frame = int(valid.sum())
            if depths.size > 0:
                d_lo, d_hi = float(depths.min()), float(depths.max())
                d_norm = ((depths - d_lo) / max(d_hi - d_lo, 1e-6)).clip(0, 1)
                cmap = cv2.applyColorMap(
                    (d_norm * 255).astype(np.uint8).reshape(-1, 1), cv2.COLORMAP_JET
                )
                for j, (pu, pv) in enumerate(zip(u_v, v_v)):
                    cv2.circle(bgr, (int(pu), int(pv)), 3, tuple(int(c) for c in cmap[j, 0]), -1)
                cv2.putText(
                    bgr,
                    f"pts={n_in_frame}  d=[{d_lo:.2f},{d_hi:.2f}]",
                    (6, H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 0), 1,
                )

        cv2.putText(
            bgr,
            f"{source}  az={az:.0f}  el={el:.0f}",
            (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1,
        )

        flag = "" if n_in_frame > 10 else "  *** FEW POINTS ***"
        cv2.putText(
            bgr,
            f"in-frame={n_in_frame}  behind={int((~in_front).sum())}{flag}",
            (6, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1,
        )

        fname = f"{i:03d}_{source}_az{az:.0f}_el{el:.0f}.jpg"
        cv2.imwrite(str(output_dir / fname), bgr)

    logger.info(
        "Projection overlays: %d views saved to %s",
        len(supervision_views), output_dir,
    )


def save_supervision_manifest(views: list[dict], output_path: str | Path) -> Path:
    """Persist a JSON-serialisable manifest of the in-memory supervision_views.

    Useful for debugging / re-running training without rebuilding from raw
    novel-view outputs. Image arrays are NOT saved here — only the camera
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
