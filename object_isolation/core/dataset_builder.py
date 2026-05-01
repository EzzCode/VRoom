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
- No bbox alignment is applied anywhere; image, alpha and camera always move
    together.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import List, Optional

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


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a).astype(bool)
    b = np.asarray(b).astype(bool)
    if a.shape != b.shape:
        b = cv2.resize(b.astype(np.uint8), (a.shape[1], a.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    inter = float(np.logical_and(a, b).sum())
    union = float(np.logical_or(a, b).sum())
    return inter / max(union, 1.0)


def build_hallucinated_supervision_views(
    halluc_index_path: str | Path,
    local_sv3d: LocalSV3D,
    *,
    weight: float = 0.10,
    fov_y_deg: float = 50.0,
    target_resolution: int = 576,
    up_W_override: Optional[np.ndarray] = None,
    include_conditioning: bool = True,
    min_alignment_iou: float = 0.55,
) -> List[dict]:
    """Read a Phase-5 hallucination manifest and return hallucinated views.

    Args:
        halluc_index_path: path to ``hallucination_index.json``.
        local_sv3d: same coordinate-frame helper used during Phase 5.
        weight: per-view loss weight (matches ``hallucination_weight``).
        fov_y_deg: vertical FOV used for SV3D output (matches Phase 5 default).
        target_resolution: square output resolution.
        up_W_override: if given, recompute (R, T) via look-at with this up
            vector instead of the scope's averaged up. Must match the up
            used during Phase 5 reference rendering.
        include_conditioning: if False, skip the cond frame (frame index n-1
            in SV3D's orbit). Default True since the cond frame is the
            highest-confidence supervision signal.
    """
    halluc_index_path = Path(halluc_index_path)
    if not halluc_index_path.exists():
        raise FileNotFoundError(f"hallucination_index.json not found: {halluc_index_path}")

    with open(halluc_index_path) as f:
        manifest = json.load(f)

    frames = manifest.get("frames", [])
    accepted = [fr for fr in frames if bool(fr.get("accepted"))]
    if not include_conditioning:
        accepted = [fr for fr in accepted if not bool(fr.get("is_conditioning"))]

    if not accepted:
        raise RuntimeError(
            f"No accepted frames in {halluc_index_path}. "
            "Re-run Phase 5 with looser IoU threshold or more cond views."
        )

    # Build K from FOV.
    res = int(target_resolution)
    fy = 0.5 * res / math.tan(0.5 * math.radians(fov_y_deg))
    K = np.array([[fy, 0.0, res / 2.0],
                  [0.0, fy, res / 2.0],
                  [0.0, 0.0, 1.0]], dtype=np.float32)

    centroid_W = np.asarray(local_sv3d.world_local.centroid_W, dtype=np.float64)

    views: List[dict] = []
    for fr in accepted:
        rgba_path = _resolve_path(fr["out_rgba_path"], manifest_dir=halluc_index_path.parent)
        if not rgba_path.exists():
            logger.warning("Missing supervision RGBA %s; skipping.", rgba_path)
            continue

        rgba = cv2.imread(str(rgba_path), cv2.IMREAD_UNCHANGED)
        if rgba is None:
            logger.warning("Failed to read %s; skipping.", rgba_path)
            continue

        rgb, mask = _rgba_to_rgb_mask(rgba)
        if rgb.shape[0] != res or rgb.shape[1] != res:
            rgb = cv2.resize(rgb, (res, res), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask.astype(np.uint8), (res, res), interpolation=cv2.INTER_NEAREST) > 0

        ref_iou = None
        if fr.get("objgs_ref_path"):
            ref_path = _resolve_path(fr["objgs_ref_path"], manifest_dir=halluc_index_path.parent)
            ref_rgba = cv2.imread(str(ref_path), cv2.IMREAD_UNCHANGED) if ref_path.exists() else None
            if ref_rgba is not None:
                _ref_rgb, ref_mask = _rgba_to_rgb_mask(ref_rgba)
                if ref_mask.shape != mask.shape:
                    ref_mask = cv2.resize(ref_mask.astype(np.uint8), (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
                ref_iou = _mask_iou(mask, ref_mask)
                if ref_iou < float(min_alignment_iou):
                    logger.warning(
                        "Skipping hallucinated frame %s: mask/ref IoU %.3f < %.3f.",
                        fr.get("index"), ref_iou, float(min_alignment_iou),
                    )
                    continue

        az_V = float(fr["azimuth_V_deg"])
        el_V = float(fr["elevation_V_deg"])

        # Map V-pose to world camera, optionally overriding the up axis.
        R_w2c, T_w2c, C_W = local_sv3d.sv3d_view_to_world_camera(az_V, el_V)
        if up_W_override is not None:
            up = np.asarray(up_W_override, dtype=np.float64).reshape(3)
            up = up / max(np.linalg.norm(up), 1e-9)
            R_w2c, T_w2c = look_at_w2c(np.asarray(C_W, dtype=np.float64), centroid_W, up)

        views.append({
            "source": "hallucinated",
            "rgb": rgb,
            "mask": mask,
            "image_path": str(rgba_path),
            "camera": {
                "R": np.asarray(R_w2c, dtype=np.float32),
                "T": np.asarray(T_w2c, dtype=np.float32),
                "K": K.copy(),
                "width": res,
                "height": res,
                "position": np.asarray(C_W, dtype=np.float32),
                "azimuth_offset_deg": az_V,
                "elevation_offset_deg": el_V,
                "azimuth_world_rad": float(np.deg2rad(az_V)),
                "is_conditioning": bool(fr.get("is_conditioning", False)),
                "frame_index": int(fr.get("index", 0)),
                "alignment_iou": ref_iou,
            },
            "weight": float(weight),
        })

    logger.info(
        "Phase 6: built %d supervision views from %s (manifest_accepted=%d/%d, alignment_retained=%d).",
        len(views), halluc_index_path.name, len(accepted), len(frames), len(views),
    )
    return views


def build_real_supervision_views(
    extraction_index_path: str | Path,
    scope,
    *,
    weight: float = 1.0,
    target_long_edge: int = 576,
) -> List[dict]:
    """Read Phase-3 real extractions as camera-aligned supervision views."""
    extraction_index_path = Path(extraction_index_path)
    if not extraction_index_path.exists():
        logger.warning("Phase-3 extraction manifest not found: %s", extraction_index_path)
        return []

    with open(extraction_index_path) as f:
        manifest = json.load(f)

    views: List[dict] = []
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
            rgb, mask, K, target_long_edge=int(target_long_edge)
        )

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
            "weight": float(weight),
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
    min_hallucination_alignment_iou: float = 0.55,
) -> List[dict]:
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
        weight=hallucination_weight,
        fov_y_deg=fov_y_deg,
        target_resolution=hallucination_resolution,
        up_W_override=up_W_override,
        include_conditioning=include_conditioning,
        min_alignment_iou=float(min_hallucination_alignment_iou),
    )
    views = real_views + hallucinated_views
    logger.info(
        "Phase 6: joint supervision views ready: total=%d real=%d hallucinated=%d.",
        len(views), len(real_views), len(hallucinated_views),
    )
    return views


def build_supervision_views(*args, **kwargs) -> List[dict]:
    """Backward-compatible alias for hallucination-only callers."""
    return build_hallucinated_supervision_views(*args, **kwargs)


def save_supervision_manifest(views: List[dict], output_path: str | Path) -> Path:
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
            "azimuth_V_deg": cam["azimuth_offset_deg"],
            "elevation_V_deg": cam["elevation_offset_deg"],
            "is_conditioning": cam.get("is_conditioning", False),
            "frame_index": cam.get("frame_index"),
            "alignment_iou": cam.get("alignment_iou"),
            "R_w2c": cam["R"].tolist(),
            "T_w2c": cam["T"].tolist(),
            "K": cam["K"].tolist(),
            "C_W": cam["position"].tolist(),
            "width": cam["width"],
            "height": cam["height"],
            "weight": float(v["weight"]),
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"n_views": len(payload), "views": payload}, f, indent=2)
    return output_path
