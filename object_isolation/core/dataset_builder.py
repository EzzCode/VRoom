"""Phase 6 — Build supervision views from Phase-5 hallucinations.

Converts ``hallucination_index.json`` (produced by Phase 5) into the
``supervision_views`` list expected by
``target_replenishment.core.optimizer.optimize_with_novel_views``:

    [{
        'rgb': np.ndarray HxWx3 uint8 (RGB, white background),
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
- The look-at convention used here MUST match Phase-5 reference renders
  (otherwise the supervision RGB is rendered from a different roll than
  the camera the optimizer renders from). We therefore optionally accept a
  ``up_W_override`` matching the conditioning camera's own world-up axis.
- Only frames with ``accepted=True`` are emitted.
- Output RGB is composited onto a white background (matches what the
  optimizer's ``gt_object_mask = (rgb < 0.98).any()`` heuristic expects).
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


def _composite_rgba_on_white(rgba: np.ndarray) -> np.ndarray:
    """Composite an RGBA (uint8 BGRA from cv2.imread) image on a white bg.

    Returns RGB uint8 (HxWx3, RGB order)."""
    if rgba.ndim == 2:
        rgba = cv2.cvtColor(rgba, cv2.COLOR_GRAY2BGR)
    if rgba.shape[2] == 3:
        return cv2.cvtColor(rgba, cv2.COLOR_BGR2RGB)
    bgr = rgba[..., :3].astype(np.float32)
    a = rgba[..., 3:4].astype(np.float32) / 255.0
    white = np.full_like(bgr, 255.0)
    out = a * bgr + (1.0 - a) * white
    return cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_BGR2RGB)


def build_supervision_views(
    halluc_index_path: str | Path,
    local_sv3d: LocalSV3D,
    *,
    weight: float = 0.10,
    fov_y_deg: float = 50.0,
    target_resolution: int = 576,
    up_W_override: Optional[np.ndarray] = None,
    include_conditioning: bool = True,
) -> List[dict]:
    """Read a Phase-5 hallucination manifest and return supervision_views.

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
        rgba_path = Path(fr["out_rgba_path"])
        if not rgba_path.exists():
            logger.warning("Missing supervision RGBA %s; skipping.", rgba_path)
            continue

        rgba = cv2.imread(str(rgba_path), cv2.IMREAD_UNCHANGED)
        if rgba is None:
            logger.warning("Failed to read %s; skipping.", rgba_path)
            continue

        rgb = _composite_rgba_on_white(rgba)
        if rgb.shape[0] != res or rgb.shape[1] != res:
            rgb = cv2.resize(rgb, (res, res), interpolation=cv2.INTER_AREA)

        az_V = float(fr["azimuth_V_deg"])
        el_V = float(fr["elevation_V_deg"])

        # Map V-pose to world camera, optionally overriding the up axis.
        R_w2c, T_w2c, C_W = local_sv3d.sv3d_view_to_world_camera(az_V, el_V)
        if up_W_override is not None:
            up = np.asarray(up_W_override, dtype=np.float64).reshape(3)
            up = up / max(np.linalg.norm(up), 1e-9)
            R_w2c, T_w2c = look_at_w2c(np.asarray(C_W, dtype=np.float64), centroid_W, up)

        views.append({
            "rgb": rgb,
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
            },
            "weight": float(weight),
        })

    logger.info(
        "Phase 6: built %d supervision views from %s (accepted=%d/%d).",
        len(views), halluc_index_path.name, len(accepted), len(frames),
    )
    return views


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
            "azimuth_V_deg": cam["azimuth_offset_deg"],
            "elevation_V_deg": cam["elevation_offset_deg"],
            "is_conditioning": cam.get("is_conditioning", False),
            "frame_index": cam.get("frame_index"),
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
