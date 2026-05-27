"""Build aligned real + hallucinated supervision views for object training.

Each supervision view is a dict::

    {
        'rgb':    np.ndarray (H, W, 3) uint8, white-background RGB,
        'mask':   np.ndarray (H, W) bool,
        'source': 'real' | 'hallucinated',
        'camera': {
            'R': (3,3) float32   R_w2c,
            'T': (3,)  float32   T_w2c,
            'K': (3,3) float32,
            'width': int, 'height': int,
            'position': (3,) float32  camera centre in world coords,
            'azimuth_offset_deg': float,
            'elevation_offset_deg': float,
        },
        'weight': float,
        'image_path': str,
    }
"""

import json
import logging
import math
from pathlib import Path

import cv2
import numpy as np

from .utils.transforms import look_at

logger = logging.getLogger(__name__)

_VROOM_ROOT = Path(__file__).resolve().parents[1]

# ── Seed-point projection constants ──────────────────────────────────────────
_SV3D_FILL_FRAC      = 0.85   # must match hallucination.py
_SEED_DEPTH_MIN      = 0.1
_SEED_MIN_IN_FRONT   = 20
_SEED_PERCENTILE_LO  = 2
_SEED_PERCENTILE_HI  = 98
_WS_CLIP_MIN         = 0.05
_WS_CLIP_MAX         = 2.0


# ---------------------------------------------------------------------------
# Path and image utilities
# ---------------------------------------------------------------------------

def _resolve_path(path_value, *, manifest_dir):
    """Resolve paths stored in manifests, supporting old relative paths."""
    p = Path(path_value)
    if p.is_absolute():
        return p
    for candidate in (manifest_dir / p, Path.cwd() / p, _VROOM_ROOT / p):
        if candidate.exists():
            return candidate
    return _VROOM_ROOT / p


def _rgba_to_rgb_mask(rgba):
    """Convert cv2 BGRA/BGR/gray data to (rgb uint8, mask bool)."""
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
    """Optionally downsample and scale intrinsics identically.

    Returns (rgb, mask, K, width, height).
    """
    h, w = rgb.shape[:2]
    if target_long_edge is None or int(target_long_edge) <= 0:
        return rgb, mask.astype(bool), K.astype(np.float32), w, h

    scale = min(1.0, float(target_long_edge) / float(max(w, h)))
    if scale >= 0.999:
        return rgb, mask.astype(bool), K.astype(np.float32), w, h

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


# ---------------------------------------------------------------------------
# Telephoto intrinsic scaling
# ---------------------------------------------------------------------------

def _compute_world_scale_px(seed_points_W, R_w2c, T_w2c, K, target_size):
    """Compute the world-to-pixel scale ratio for telephoto K adjustment.

    Returns ws — ratio of world object extent (in K_sv3d pixels) to the
    expected object extent in the SV3D image (fill_frac * target_size).
    K_view = K_sv3d / ws gives correct telephoto intrinsics so the object
    spans exactly _SV3D_FILL_FRAC of the image.

    Only focal lengths are scaled; cx/cy are unchanged because look_at()
    places scope.centroid on the optical axis, so centroid always projects
    to (res/2, res/2) regardless of seed-point distribution.
    """
    R   = np.asarray(R_w2c, np.float64)
    T   = np.asarray(T_w2c, np.float64).reshape(3)
    K64 = np.asarray(K, np.float64)
    fx  = float(K64[0, 0])
    fy  = float(K64[1, 1])
    cx  = float(K64[0, 2])
    cy  = float(K64[1, 2])
    sv3d_px = _SV3D_FILL_FRAC * float(target_size)

    pts     = np.asarray(seed_points_W, np.float64)
    pts_c   = (R @ pts.T).T + T
    in_front = pts_c[:, 2] > _SEED_DEPTH_MIN
    n_in_front = int(in_front.sum())

    if n_in_front < _SEED_MIN_IN_FRONT:
        raise RuntimeError(
            f"Only {n_in_front} / {len(pts)} COLMAP seed points are in front of this "
            f"camera (depth > {_SEED_DEPTH_MIN}). Expected >= {_SEED_MIN_IN_FRONT}. "
            "Check that seed_points_W and the camera share the same world frame."
        )

    pf  = pts_c[in_front]
    u   = pf[:, 0] / pf[:, 2] * fx + cx
    v   = pf[:, 1] / pf[:, 2] * fy + cy
    u_lo = float(np.percentile(u, _SEED_PERCENTILE_LO))
    u_hi = float(np.percentile(u, _SEED_PERCENTILE_HI))
    v_lo = float(np.percentile(v, _SEED_PERCENTILE_LO))
    v_hi = float(np.percentile(v, _SEED_PERCENTILE_HI))
    world_px = float(max(u_hi - u_lo, v_hi - v_lo))
    return float(np.clip(world_px / max(sv3d_px, 1.0), _WS_CLIP_MIN, _WS_CLIP_MAX))


def _project_seed_bbox(seed_points_W, R_w2c, T_w2c, K, width, height):
    R = np.asarray(R_w2c, np.float64)
    T = np.asarray(T_w2c, np.float64).reshape(3)
    K64 = np.asarray(K, np.float64)
    pts = np.asarray(seed_points_W, np.float64)
    pts_c = (R @ pts.T).T + T
    in_front = pts_c[:, 2] > _SEED_DEPTH_MIN
    if int(in_front.sum()) < _SEED_MIN_IN_FRONT:
        return None
    pts_f = pts_c[in_front]
    u = pts_f[:, 0] / pts_f[:, 2] * float(K64[0, 0]) + float(K64[0, 2])
    v = pts_f[:, 1] / pts_f[:, 2] * float(K64[1, 1]) + float(K64[1, 2])
    valid = (u >= 0) & (u < int(width)) & (v >= 0) & (v < int(height))
    if int(valid.sum()) < _SEED_MIN_IN_FRONT:
        return None
    u = u[valid]
    v = v[valid]
    return (
        float(np.percentile(u, _SEED_PERCENTILE_LO)),
        float(np.percentile(v, _SEED_PERCENTILE_LO)),
        float(np.percentile(u, _SEED_PERCENTILE_HI)),
        float(np.percentile(v, _SEED_PERCENTILE_HI)),
    )


def _mask_bbox(mask):
    ys, xs = np.where(np.asarray(mask).astype(bool))
    if len(xs) == 0:
        return None
    return float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)


# ---------------------------------------------------------------------------
# Hallucinated views
# ---------------------------------------------------------------------------

def build_hallucinated_views(halluc_index_path, frame, *,
                              seed_points_W, weight=0.10, fov_y_deg=50.0,
                              resolution=576, up_override=None,
                              include_conditioning=True):
    """Build supervision view dicts from SV3D hallucination outputs.

    Parameters
    ----------
    halluc_index_path   : path to hallucination_index.json
    frame               : ObjectFrame (from transforms.py)
    seed_points_W       : (N, 3) COLMAP seed points in world coords; required
                          to compute per-view telephoto intrinsics
    weight              : loss weight assigned to each hallucinated view
    fov_y_deg           : vertical FOV used during hallucination rendering
    resolution          : SV3D output resolution (default 576)
    up_override         : if given, override the look_at up vector
    include_conditioning: whether to include the conditioning view
    """
    if seed_points_W is None or len(seed_points_W) == 0:
        raise ValueError("seed_points_W must be a non-empty array.")

    halluc_index_path = Path(halluc_index_path)
    if not halluc_index_path.exists():
        raise FileNotFoundError(f"Hallucination manifest not found: {halluc_index_path}")

    with open(halluc_index_path) as f:
        manifest = json.load(f)

    frames = manifest.get("frames", [])
    if not include_conditioning:
        frames = [fr for fr in frames if not fr.get("is_conditioning", False)]
    accepted = [fr for fr in frames if fr.get("accepted", False)]

    if not accepted:
        raise RuntimeError(f"No accepted hallucinated frames in {halluc_index_path}.")

    res  = int(resolution)
    fy_  = 0.5 * res / math.tan(0.5 * math.radians(float(fov_y_deg)))
    K_sv3d = np.array([[fy_, 0.0, res / 2.0],
                       [0.0, fy_, res / 2.0],
                       [0.0, 0.0, 1.0]], dtype=np.float32)

    views = []
    for fr in accepted:
        rgba_path = _resolve_path(fr["out_rgba_path"],
                                  manifest_dir=halluc_index_path.parent)
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
            R_w2c, T_w2c = look_at(
                np.asarray(C_W, np.float32),
                np.asarray(frame.centroid, np.float32),
                up,
            )

        ws = _compute_world_scale_px(
            seed_points_W,
            np.asarray(R_w2c, np.float64),
            np.asarray(T_w2c, np.float64),
            K_sv3d.astype(np.float64),
            res,
        )
        K_view = K_sv3d.copy()
        K_view[0, 0] = float(K_sv3d[0, 0] / ws)
        K_view[1, 1] = float(K_sv3d[1, 1] / ws)
        proj_bbox = _project_seed_bbox(seed_points_W, R_w2c, T_w2c, K_view, res, res)
        img_bbox = _mask_bbox(mask)
        alignment_shift = [0.0, 0.0]
        if proj_bbox is not None and img_bbox is not None:
            proj_cx = 0.5 * (proj_bbox[0] + proj_bbox[2])
            proj_cy = 0.5 * (proj_bbox[1] + proj_bbox[3])
            img_cx = 0.5 * (img_bbox[0] + img_bbox[2])
            img_cy = 0.5 * (img_bbox[1] + img_bbox[3])
            shift_x = float(np.clip(img_cx - proj_cx, -0.25 * res, 0.25 * res))
            shift_y = float(np.clip(img_cy - proj_cy, -0.25 * res, 0.25 * res))
            K_view[0, 2] += shift_x
            K_view[1, 2] += shift_y
            alignment_shift = [shift_x, shift_y]

        views.append({
            "source": "hallucinated",
            "rgb": rgb,
            "mask": mask,
            "image_path": str(rgba_path),
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
                "alignment_transform": "principal_point_shift",
                "alignment_shift_px": alignment_shift,
                "seed_projection_bbox_before_shift": proj_bbox,
                "image_mask_bbox": img_bbox,
            },
            "weight": float(weight),
        })

    logger.info("Built %d hallucinated supervision views.", len(views))
    return views


# ---------------------------------------------------------------------------
# Real views
# ---------------------------------------------------------------------------

def build_real_views(extraction_index_path, scope, *,
                     weight=1.0, target_long_edge=576):
    """Build supervision view dicts from real COLMAP-camera extraction outputs."""
    extraction_index_path = Path(extraction_index_path)
    if not extraction_index_path.exists():
        logger.warning("Extraction manifest not found: %s", extraction_index_path)
        return []

    with open(extraction_index_path) as f:
        manifest = json.load(f)

    views = []
    for fr in manifest.get("frames", []):
        cam_index = int(fr["cam_index"])
        if cam_index < 0 or cam_index >= len(scope.cameras):
            logger.warning("Skipping real frame with invalid cam_index=%d.", cam_index)
            continue

        cam_p     = scope.cameras[cam_index]
        rgba_path = _resolve_path(fr["out_rgba_path"],
                                  manifest_dir=extraction_index_path.parent)
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
            rgb, mask, K, target_long_edge=target_long_edge
        )

        # Square-pad landscape images so real views match hallucinated resolution
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

        az = float(cam_p.get("azimuth_deg", fr.get("azimuth_deg", 0.0)))
        el = float(cam_p.get("elevation_deg", 0.0))

        views.append({
            "source": "real",
            "rgb": rgb,
            "mask": mask,
            "image_path": str(rgba_path),
            "camera": {
                "R": np.asarray(cam_p["R"], np.float32),
                "T": np.asarray(cam_p["T"], np.float32),
                "K": K,
                "width": int(width),
                "height": int(height),
                "position": np.asarray(cam_p["position"], np.float32),
                "azimuth_offset_deg": az,
                "elevation_offset_deg": el,
                "is_conditioning": False,
                "frame_index": cam_index,
            },
            "weight": float(weight),
        })

    logger.info("Built %d real supervision views from %s.",
                len(views), extraction_index_path.name)
    return views


# ---------------------------------------------------------------------------
# Combined entry point (used by pipeline.py)
# ---------------------------------------------------------------------------

def build_supervision_views(halluc_index_path, extraction_index_path,
                             scope, frame, seed_points_W,
                             real_weight=1.0, hallucination_weight=1.0,
                             fov_y_deg=50.0, resolution=576,
                             real_target_long_edge=576,
                             up_override=None, include_conditioning=True):
    """Return one aligned list containing real and hallucinated supervision views."""
    real_views = build_real_views(
        extraction_index_path,
        scope,
        weight=real_weight,
        target_long_edge=real_target_long_edge,
    )
    halluc_views = build_hallucinated_views(
        halluc_index_path,
        frame,
        seed_points_W=seed_points_W,
        weight=hallucination_weight,
        fov_y_deg=fov_y_deg,
        resolution=resolution,
        up_override=up_override,
        include_conditioning=include_conditioning,
    )
    views = real_views + halluc_views
    logger.info("Supervision views ready: total=%d  real=%d  hallucinated=%d.",
                len(views), len(real_views), len(halluc_views))
    return views


# ---------------------------------------------------------------------------
# Debug overlay
# ---------------------------------------------------------------------------

def write_projection_overlays(xyz_W, supervision_views, output_dir):
    """Project COLMAP seed points onto each supervision view and save JPGs.

    Colour-codes points by depth (COLORMAP_JET: blue=near, red=far).
    Green contour = supervision mask.  If dots don't trace the silhouette
    the coordinate frame is misaligned and training data is incorrect.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for p in output_dir.glob("*.jpg"):
        p.unlink()

    xyz = np.asarray(xyz_W, np.float64)

    for i, view in enumerate(supervision_views):
        cam    = view["camera"]
        R      = np.asarray(cam["R"], np.float64)
        T      = np.asarray(cam["T"], np.float64).flatten()
        K      = np.asarray(cam["K"], np.float64)
        W, H   = int(cam["width"]), int(cam["height"])
        source = view.get("source", "?")
        az     = float(cam.get("azimuth_offset_deg",   0.0))
        el     = float(cam.get("elevation_offset_deg", 0.0))

        pts_c   = (R @ xyz.T).T + T
        in_front = pts_c[:, 2] > _SEED_DEPTH_MIN
        pts_f   = pts_c[in_front]

        rgb = np.asarray(view["rgb"], np.uint8)
        if rgb.shape[0] != H or rgb.shape[1] != W:
            rgb = cv2.resize(rgb, (W, H))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        mask_u8 = np.asarray(view["mask"], np.uint8) * 255
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
                d_lo  = float(depths.min())
                d_hi  = float(depths.max())
                d_norm = ((depths - d_lo) / max(d_hi - d_lo, 1e-6)).clip(0, 1)
                cmap = cv2.applyColorMap(
                    (d_norm * 255).astype(np.uint8).reshape(-1, 1), cv2.COLORMAP_JET
                )
                for j, (pu, pv) in enumerate(zip(u_v, v_v)):
                    cv2.circle(bgr, (int(pu), int(pv)), 3,
                               tuple(int(c) for c in cmap[j, 0]), -1)
                cv2.putText(bgr,
                            f"pts={n_in_frame}  d=[{d_lo:.2f},{d_hi:.2f}]",
                            (6, H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 0), 1)

        cv2.putText(bgr, f"{source}  az={az:.0f}  el={el:.0f}",
                    (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        flag = "" if n_in_frame > 10 else "  *** FEW POINTS ***"
        cv2.putText(bgr,
                    f"in-frame={n_in_frame}  behind={int((~in_front).sum())}{flag}",
                    (6, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        fname = f"{i:03d}_{source}_az{az:.0f}_el{el:.0f}.jpg"
        cv2.imwrite(str(output_dir / fname), bgr)

    logger.info("Projection overlays: %d views → %s", len(supervision_views), output_dir)


# ---------------------------------------------------------------------------
# Manifest persistence
# ---------------------------------------------------------------------------

def save_supervision_manifest(views, output_path):
    """Save a JSON manifest of supervision views (no image arrays, only metadata)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = []
    for v in views:
        cam = v["camera"]
        payload.append({
            "source":          v.get("source", "hallucinated"),
            "image_path":      v.get("image_path"),
            "azimuth_deg":     float(cam.get("azimuth_offset_deg",   0.0)),
            "elevation_deg":   float(cam.get("elevation_offset_deg", 0.0)),
            "is_conditioning": bool(cam.get("is_conditioning", False)),
            "frame_index":     cam.get("frame_index"),
            "alignment_transform": cam.get("alignment_transform"),
            "alignment_shift_px": cam.get("alignment_shift_px"),
            "seed_projection_bbox_before_shift": cam.get("seed_projection_bbox_before_shift"),
            "image_mask_bbox": cam.get("image_mask_bbox"),
            "R_w2c":           np.asarray(cam["R"]).tolist(),
            "T_w2c":           np.asarray(cam["T"]).tolist(),
            "K":               np.asarray(cam["K"]).tolist(),
            "C_W":             np.asarray(cam["position"]).tolist(),
            "width":           int(cam["width"]),
            "height":          int(cam["height"]),
            "weight":          float(v["weight"]),
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"n_views": len(payload), "views": payload}, f, indent=2)

    return output_path
