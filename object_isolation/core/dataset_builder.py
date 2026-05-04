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


def _mask_bbox(mask: np.ndarray) -> Optional[tuple[int, int, int, int]]:
    ys, xs = np.where(np.asarray(mask).astype(bool))
    if xs.size == 0 or ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _bbox_iou(a: Optional[tuple[int, int, int, int]], b: Optional[tuple[int, int, int, int]]) -> float:
    if a is None or b is None:
        return 0.0
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    area_a = max(0, ax1 - ax0) * max(0, ay1 - ay0)
    area_b = max(0, bx1 - bx0) * max(0, by1 - by0)
    return float(inter) / max(float(area_a + area_b - inter), 1.0)


def _mask_centroid(mask: np.ndarray) -> Optional[np.ndarray]:
    ys, xs = np.where(np.asarray(mask).astype(bool))
    if xs.size == 0 or ys.size == 0:
        return None
    return np.array([float(xs.mean()), float(ys.mean())], dtype=np.float32)


def _audit_mask_alignment(
    mask: np.ndarray,
    ref_mask: np.ndarray,
    *,
    min_iou: float,
    min_bbox_iou: float,
    max_centroid_distance: float,
    min_area_ratio: float,
    max_area_ratio: float,
) -> dict:
    mask = np.asarray(mask).astype(bool)
    ref_mask = np.asarray(ref_mask).astype(bool)
    if ref_mask.shape != mask.shape:
        ref_mask = cv2.resize(
            ref_mask.astype(np.uint8),
            (mask.shape[1], mask.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ) > 0

    mask_area = int(mask.sum())
    ref_area = int(ref_mask.sum())
    iou = _mask_iou(mask, ref_mask)
    bbox_iou = _bbox_iou(_mask_bbox(mask), _mask_bbox(ref_mask))
    centroid = _mask_centroid(mask)
    ref_centroid = _mask_centroid(ref_mask)
    diag = float(np.hypot(mask.shape[1], mask.shape[0]))
    if centroid is None or ref_centroid is None:
        centroid_distance = 1.0
    else:
        centroid_distance = float(np.linalg.norm(centroid - ref_centroid) / max(diag, 1.0))
    area_ratio = float(mask_area) / max(float(ref_area), 1.0)

    reasons: list[str] = []
    if mask_area < 200:
        reasons.append("hallucination_mask_empty")
    if ref_area < 200:
        reasons.append("reference_mask_empty")
    if iou < float(min_iou):
        reasons.append(f"mask_iou_{iou:.3f}_lt_{float(min_iou):.3f}")
    if bbox_iou < float(min_bbox_iou):
        reasons.append(f"bbox_iou_{bbox_iou:.3f}_lt_{float(min_bbox_iou):.3f}")
    if centroid_distance > float(max_centroid_distance):
        reasons.append(f"centroid_dist_{centroid_distance:.3f}_gt_{float(max_centroid_distance):.3f}")
    if area_ratio < float(min_area_ratio) or area_ratio > float(max_area_ratio):
        reasons.append(f"area_ratio_{area_ratio:.3f}_outside_{float(min_area_ratio):.3f}_{float(max_area_ratio):.3f}")

    return {
        "accepted": not reasons,
        "reject_reasons": reasons,
        "mask_iou": float(iou),
        "bbox_iou": float(bbox_iou),
        "centroid_distance_norm": float(centroid_distance),
        "area_ratio": float(area_ratio),
        "mask_pixels": int(mask_area),
        "reference_pixels": int(ref_area),
    }


def _apply_flip(rgb: np.ndarray, mask: np.ndarray, flip_code: int | None) -> tuple[np.ndarray, np.ndarray]:
    if flip_code is None:
        return rgb.copy(), mask.astype(bool).copy()
    return cv2.flip(rgb, flip_code), cv2.flip(mask.astype(np.uint8), flip_code) > 0


def _fit_bbox_warp(
    mask: np.ndarray,
    ref_mask: np.ndarray,
) -> tuple[np.ndarray | None, dict]:
    src_bbox = _mask_bbox(mask)
    ref_bbox = _mask_bbox(ref_mask)
    if src_bbox is None or ref_bbox is None:
        return None, {"bbox_warp_reasonable": False}

    sx0, sy0, sx1, sy1 = src_bbox
    rx0, ry0, rx1, ry1 = ref_bbox
    src_w = max(float(sx1 - sx0), 1.0)
    src_h = max(float(sy1 - sy0), 1.0)
    ref_w = max(float(rx1 - rx0), 1.0)
    ref_h = max(float(ry1 - ry0), 1.0)

    scale_x = ref_w / src_w
    scale_y = ref_h / src_h
    src_cx = float(sx0) + 0.5 * src_w
    src_cy = float(sy0) + 0.5 * src_h
    ref_cx = float(rx0) + 0.5 * ref_w
    ref_cy = float(ry0) + 0.5 * ref_h
    translate_x = ref_cx - scale_x * src_cx
    translate_y = ref_cy - scale_y * src_cy

    diag = float(np.hypot(mask.shape[1], mask.shape[0]))
    translation_norm = float(np.hypot(translate_x, translate_y) / max(diag, 1.0))
    reasonable = (
        0.55 <= float(scale_x) <= 1.80
        and 0.55 <= float(scale_y) <= 1.80
        and translation_norm <= 0.20
    )
    matrix = np.array([[scale_x, 0.0, translate_x], [0.0, scale_y, translate_y]], dtype=np.float32)
    return matrix, {
        "bbox_warp_reasonable": bool(reasonable),
        "bbox_warp_scale_x": float(scale_x),
        "bbox_warp_scale_y": float(scale_y),
        "bbox_warp_translate_x": float(translate_x),
        "bbox_warp_translate_y": float(translate_y),
        "bbox_warp_translation_norm": float(translation_norm),
    }


def _warp_rgb_mask(rgb: np.ndarray, mask: np.ndarray, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height, width = mask.shape[:2]
    warped_rgb = cv2.warpAffine(
        rgb,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    warped_mask = cv2.warpAffine(
        mask.astype(np.uint8),
        matrix,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ) > 0
    return warped_rgb, warped_mask


def _centroid_refine_warp(mask: np.ndarray, ref_mask: np.ndarray) -> tuple[np.ndarray | None, dict]:
    centroid = _mask_centroid(mask)
    ref_centroid = _mask_centroid(ref_mask)
    if centroid is None or ref_centroid is None:
        return None, {"centroid_refine_applied": False}
    delta = ref_centroid - centroid
    diag = float(np.hypot(mask.shape[1], mask.shape[0]))
    shift_norm = float(np.linalg.norm(delta) / max(diag, 1.0))
    if shift_norm > 0.08:
        return None, {
            "centroid_refine_applied": False,
            "centroid_refine_shift_norm": float(shift_norm),
        }
    matrix = np.array([[1.0, 0.0, float(delta[0])], [0.0, 1.0, float(delta[1])]], dtype=np.float32)
    return matrix, {
        "centroid_refine_applied": True,
        "centroid_refine_dx": float(delta[0]),
        "centroid_refine_dy": float(delta[1]),
        "centroid_refine_shift_norm": float(shift_norm),
    }


def _similarity_centroid_warp(
    mask: np.ndarray,
    ref_mask: np.ndarray,
    *,
    scale: float,
) -> tuple[np.ndarray | None, dict]:
    centroid = _mask_centroid(mask)
    ref_centroid = _mask_centroid(ref_mask)
    if centroid is None or ref_centroid is None:
        return None, {"similarity_refine_applied": False}
    delta = ref_centroid - centroid
    diag = float(np.hypot(mask.shape[1], mask.shape[0]))
    shift_norm = float(np.linalg.norm(delta) / max(diag, 1.0))
    if shift_norm > 0.08 or not (0.84 <= float(scale) <= 1.16):
        return None, {
            "similarity_refine_applied": False,
            "similarity_scale": float(scale),
            "centroid_refine_shift_norm": float(shift_norm),
        }
    matrix = np.array(
        [
            [float(scale), 0.0, float(ref_centroid[0] - float(scale) * centroid[0])],
            [0.0, float(scale), float(ref_centroid[1] - float(scale) * centroid[1])],
        ],
        dtype=np.float32,
    )
    return matrix, {
        "similarity_refine_applied": True,
        "similarity_scale": float(scale),
        "centroid_refine_dx": float(delta[0]),
        "centroid_refine_dy": float(delta[1]),
        "centroid_refine_shift_norm": float(shift_norm),
    }


def _save_aligned_rgba(rgb: np.ndarray, mask: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    alpha = (mask.astype(np.uint8) * 255)[..., None]
    rgba = np.concatenate([rgb.astype(np.uint8), alpha], axis=2)
    cv2.imwrite(str(out_path), cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))


def _compute_aabb_world_scale_px(
    aabb_min: np.ndarray,
    aabb_max: np.ndarray,
    R_w2c: np.ndarray,
    T_w2c: np.ndarray,
    K: np.ndarray,
    sv3d_fill_frac: float = 0.85,
    target_size: int = 576,
) -> float:
    """Return world_bbox_px / sv3d_normalized_px for a given camera.

    Phase 5 normalises both the SV3D output and the ObjectGS reference renders
    so the object fills ``sv3d_fill_frac`` of the target resolution.  Phase 6
    training cameras use the real focal length, so the same object appears
    ~2-3× smaller in the ObjectGS render than in the SV3D supervision image.
    This function computes the scale factor needed to shrink the SV3D image
    back to world-camera scale before training.
    """
    xs = [float(aabb_min[0]), float(aabb_max[0])]
    ys = [float(aabb_min[1]), float(aabb_max[1])]
    zs = [float(aabb_min[2]), float(aabb_max[2])]
    corners = np.array([[x, y, z] for x in xs for y in ys for z in zs], dtype=np.float64)
    pts_cam = (R_w2c.astype(np.float64) @ corners.T).T + T_w2c.astype(np.float64)
    in_front = pts_cam[:, 2] > 1e-3
    if int(in_front.sum()) < 2:
        return 1.0
    pts_f = pts_cam[in_front]
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    u = pts_f[:, 0] / pts_f[:, 2] * fx + cx
    v = pts_f[:, 1] / pts_f[:, 2] * fy + cy
    world_px = float(max(u.max() - u.min(), v.max() - v.min()))
    sv3d_px = sv3d_fill_frac * float(target_size)
    return float(np.clip(world_px / max(sv3d_px, 1.0), 0.05, 2.0))


def _denormalize_to_world_scale(
    rgb: np.ndarray,
    mask: np.ndarray,
    world_scale: float,
    target_size: int = 576,
) -> tuple[np.ndarray, np.ndarray]:
    """Shrink an SV3D-normalised supervision image to world-camera scale.

    Resizes the image by ``world_scale`` and embeds it centred in a black
    ``target_size × target_size`` background (alpha = 0 outside the object).
    """
    if abs(world_scale - 1.0) < 0.02:
        return rgb, mask
    H, W = rgb.shape[:2]
    new_H = max(1, int(round(H * world_scale)))
    new_W = max(1, int(round(W * world_scale)))
    interp = cv2.INTER_AREA if world_scale < 1.0 else cv2.INTER_LINEAR
    rgb_s = cv2.resize(rgb, (new_W, new_H), interpolation=interp)
    mask_s = (cv2.resize(mask.astype(np.uint8), (new_W, new_H),
                         interpolation=cv2.INTER_NEAREST) > 0)
    out_rgb = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    out_mask = np.zeros((target_size, target_size), dtype=bool)
    y0 = max(0, (target_size - new_H) // 2)
    x0 = max(0, (target_size - new_W) // 2)
    y1 = min(target_size, y0 + new_H)
    x1 = min(target_size, x0 + new_W)
    out_rgb[y0:y1, x0:x1] = rgb_s[:y1 - y0, :x1 - x0]
    out_mask[y0:y1, x0:x1] = mask_s[:y1 - y0, :x1 - x0]
    return out_rgb, out_mask


def _recover_hallucination_alignment(
    rgb: np.ndarray,
    mask: np.ndarray,
    ref_mask: np.ndarray,
    *,
    min_iou: float,
    min_bbox_iou: float,
    max_centroid_distance: float,
    min_area_ratio: float,
    max_area_ratio: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Try conservative image-space recovery for SV3D orientation/scale drift."""
    if ref_mask.shape != mask.shape:
        ref_mask = cv2.resize(
            ref_mask.astype(np.uint8),
            (mask.shape[1], mask.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ) > 0

    def audit(candidate_mask: np.ndarray) -> dict:
        return _audit_mask_alignment(
            candidate_mask,
            ref_mask,
            min_iou=min_iou,
            min_bbox_iou=min_bbox_iou,
            max_centroid_distance=max_centroid_distance,
            min_area_ratio=min_area_ratio,
            max_area_ratio=max_area_ratio,
        )

    candidates: list[tuple[np.ndarray, np.ndarray, dict]] = []
    flips = [
        ("identity", None),
        ("flip_h", 1),
        ("flip_v", 0),
        ("flip_hv", -1),
    ]
    for transform_name, flip_code in flips:
        flip_rgb, flip_mask = _apply_flip(rgb, mask, flip_code)
        direct_audit = audit(flip_mask)
        direct_audit.update({
            "alignment_transform": transform_name,
            "bbox_warp_applied": False,
            "centroid_refine_applied": False,
        })
        candidates.append((flip_rgb, flip_mask, direct_audit))

        centroid_matrix, centroid_meta = _centroid_refine_warp(flip_mask, ref_mask)
        if centroid_matrix is not None:
            centroid_rgb, centroid_mask = _warp_rgb_mask(flip_rgb, flip_mask, centroid_matrix)
            centroid_audit = audit(centroid_mask)
            centroid_audit.update({
                "alignment_transform": f"{transform_name}+centroid",
                "bbox_warp_applied": False,
                **centroid_meta,
            })
            candidates.append((centroid_rgb, centroid_mask, centroid_audit))

        for scale in (0.86, 0.90, 0.94, 0.96, 0.98, 1.02, 1.04, 1.06, 1.10, 1.14):
            sim_matrix, sim_meta = _similarity_centroid_warp(flip_mask, ref_mask, scale=float(scale))
            if sim_matrix is None:
                continue
            sim_rgb, sim_mask = _warp_rgb_mask(flip_rgb, flip_mask, sim_matrix)
            sim_audit = audit(sim_mask)
            sim_audit.update({
                "alignment_transform": f"{transform_name}+scale{float(scale):.2f}+centroid",
                "bbox_warp_applied": False,
                "centroid_refine_applied": True,
                **sim_meta,
            })
            candidates.append((sim_rgb, sim_mask, sim_audit))

        matrix, warp_meta = _fit_bbox_warp(flip_mask, ref_mask)
        if matrix is None or not bool(warp_meta.get("bbox_warp_reasonable", False)):
            continue
        warp_rgb, warp_mask = _warp_rgb_mask(flip_rgb, flip_mask, matrix)
        warp_audit = audit(warp_mask)
        warp_audit.update({
            "alignment_transform": f"{transform_name}+bbox",
            "bbox_warp_applied": True,
            "centroid_refine_applied": False,
            **warp_meta,
        })
        candidates.append((warp_rgb, warp_mask, warp_audit))

        centroid_matrix, centroid_meta = _centroid_refine_warp(warp_mask, ref_mask)
        if centroid_matrix is not None:
            refined_rgb, refined_mask = _warp_rgb_mask(warp_rgb, warp_mask, centroid_matrix)
            refined_audit = audit(refined_mask)
            refined_audit.update({
                "alignment_transform": f"{transform_name}+bbox+centroid",
                "bbox_warp_applied": True,
                **warp_meta,
                **centroid_meta,
            })
            candidates.append((refined_rgb, refined_mask, refined_audit))

    original_audit = candidates[0][2]
    accepted_candidates = [c for c in candidates if bool(c[2].get("accepted", False))]
    if bool(original_audit.get("accepted", False)):
        best_rgb, best_mask, best_audit = candidates[0]
    elif accepted_candidates:
        best_rgb, best_mask, best_audit = max(
            accepted_candidates,
            key=lambda item: (
                float(item[2].get("mask_iou", 0.0)),
                float(item[2].get("bbox_iou", 0.0)),
                -float(item[2].get("centroid_distance_norm", 1.0)),
            ),
        )
    else:
        best_rgb, best_mask, best_audit = rgb, mask, original_audit

    best_seen = max(
        candidates,
        key=lambda item: (
            float(item[2].get("mask_iou", 0.0)),
            float(item[2].get("bbox_iou", 0.0)),
            -float(item[2].get("centroid_distance_norm", 1.0)),
        ),
    )[2]
    best_audit["original_alignment"] = {
        "mask_iou": float(original_audit.get("mask_iou", 0.0)),
        "bbox_iou": float(original_audit.get("bbox_iou", 0.0)),
        "centroid_distance_norm": float(original_audit.get("centroid_distance_norm", 1.0)),
        "area_ratio": float(original_audit.get("area_ratio", 0.0)),
    }
    best_audit["best_rejected_candidate"] = {
        "alignment_transform": best_seen.get("alignment_transform", "identity"),
        "mask_iou": float(best_seen.get("mask_iou", 0.0)),
        "bbox_iou": float(best_seen.get("bbox_iou", 0.0)),
        "centroid_distance_norm": float(best_seen.get("centroid_distance_norm", 1.0)),
        "area_ratio": float(best_seen.get("area_ratio", 0.0)),
    }
    return best_rgb, best_mask.astype(bool), best_audit


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
    min_alignment_bbox_iou: float = 0.55,
    max_alignment_centroid_distance: float = 0.045,
    min_alignment_area_ratio: float = 0.65,
    max_alignment_area_ratio: float = 1.45,
    alignment_audit_path: str | Path | None = None,
    aabb_min_W: Optional[np.ndarray] = None,
    aabb_max_W: Optional[np.ndarray] = None,
    sv3d_fill_frac: float = 0.85,
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
    candidates = list(frames)
    if not include_conditioning:
        candidates = [fr for fr in candidates if not bool(fr.get("is_conditioning"))]

    if not candidates:
        raise RuntimeError(
            f"No hallucinated frames in {halluc_index_path}. Run Phase 5 first."
        )

    # Build K from FOV.
    res = int(target_resolution)
    fy = 0.5 * res / math.tan(0.5 * math.radians(fov_y_deg))
    K = np.array([[fy, 0.0, res / 2.0],
                  [0.0, fy, res / 2.0],
                  [0.0, 0.0, 1.0]], dtype=np.float32)

    centroid_W = np.asarray(local_sv3d.world_local.centroid_W, dtype=np.float64)

    views: List[dict] = []
    audits: list[dict] = []
    for fr in candidates:
        rgba_path = _resolve_path(fr["out_rgba_path"], manifest_dir=halluc_index_path.parent)
        if not rgba_path.exists():
            logger.warning("Missing supervision RGBA %s; skipping.", rgba_path)
            audits.append({
                "frame_index": int(fr.get("index", -1)),
                "accepted": False,
                "reject_reasons": ["missing_hallucinated_rgba"],
                "image_path": str(rgba_path),
            })
            continue

        rgba = cv2.imread(str(rgba_path), cv2.IMREAD_UNCHANGED)
        if rgba is None:
            logger.warning("Failed to read %s; skipping.", rgba_path)
            audits.append({
                "frame_index": int(fr.get("index", -1)),
                "accepted": False,
                "reject_reasons": ["unreadable_hallucinated_rgba"],
                "image_path": str(rgba_path),
            })
            continue

        rgb, mask = _rgba_to_rgb_mask(rgba)
        if rgb.shape[0] != res or rgb.shape[1] != res:
            rgb = cv2.resize(rgb, (res, res), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask.astype(np.uint8), (res, res), interpolation=cv2.INTER_NEAREST) > 0

        if not fr.get("objgs_ref_path"):
            audits.append({
                "frame_index": int(fr.get("index", -1)),
                "accepted": False,
                "reject_reasons": ["missing_reference_path"],
                "image_path": str(rgba_path),
            })
            continue
        ref_path = _resolve_path(fr["objgs_ref_path"], manifest_dir=halluc_index_path.parent)
        ref_rgba = cv2.imread(str(ref_path), cv2.IMREAD_UNCHANGED) if ref_path.exists() else None
        if ref_rgba is None:
            audits.append({
                "frame_index": int(fr.get("index", -1)),
                "accepted": False,
                "reject_reasons": ["unreadable_reference_rgba"],
                "image_path": str(rgba_path),
                "reference_path": str(ref_path),
            })
            continue
        _ref_rgb, ref_mask = _rgba_to_rgb_mask(ref_rgba)
        rgb, mask, audit = _recover_hallucination_alignment(
            rgb,
            mask,
            ref_mask,
            min_iou=float(min_alignment_iou),
            min_bbox_iou=float(min_alignment_bbox_iou),
            max_centroid_distance=float(max_alignment_centroid_distance),
            min_area_ratio=float(min_alignment_area_ratio),
            max_area_ratio=float(max_alignment_area_ratio),
        )
        original_rgba_path = rgba_path
        if audit["accepted"] and str(audit.get("alignment_transform", "identity")) != "identity":
            aligned_dir = (
                Path(alignment_audit_path).parent / "aligned_hallucinated"
                if alignment_audit_path is not None
                else halluc_index_path.parent / "aligned_hallucinated"
            )
            rgba_path = aligned_dir / rgba_path.name
            _save_aligned_rgba(rgb, mask, rgba_path)
        audit.update({
            "frame_index": int(fr.get("index", -1)),
            "manifest_accepted": bool(fr.get("accepted", False)),
            "manifest_iou": fr.get("iou_with_objgs"),
            "azimuth_V_deg": float(fr["azimuth_V_deg"]),
            "elevation_V_deg": float(fr["elevation_V_deg"]),
            "image_path": str(rgba_path),
            "original_image_path": str(original_rgba_path),
            "reference_path": str(ref_path),
        })
        audits.append(audit)
        if not audit["accepted"]:
            logger.warning(
                "Skipping hallucinated frame %s after image audit: %s (mask_iou=%.3f bbox_iou=%.3f centroid=%.3f area=%.3f).",
                fr.get("index"), ",".join(audit["reject_reasons"]),
                audit["mask_iou"], audit["bbox_iou"], audit["centroid_distance_norm"], audit["area_ratio"],
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

        # Undo Phase-5 normalisation: both the SV3D output and the ObjectGS
        # reference renders are cropped so the object fills sv3d_fill_frac of
        # the frame.  Training cameras use the real focal length where the
        # object is much smaller — without this correction the supervision
        # image shows a ~2-3× oversized banana, driving splats to grow and
        # producing the blue-disk artifact.
        if aabb_min_W is not None and aabb_max_W is not None:
            ws = _compute_aabb_world_scale_px(
                np.asarray(aabb_min_W, dtype=np.float64),
                np.asarray(aabb_max_W, dtype=np.float64),
                np.asarray(R_w2c, dtype=np.float64),
                np.asarray(T_w2c, dtype=np.float64),
                K.astype(np.float64),
                sv3d_fill_frac=float(sv3d_fill_frac),
                target_size=int(res),
            )
            rgb, mask = _denormalize_to_world_scale(rgb, mask, ws, target_size=int(res))

        views.append({
            "source": "hallucinated",
            "rgb": rgb,
            "mask": mask,
            "image_path": str(rgba_path),
            "original_image_path": str(original_rgba_path),
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
                "alignment_iou": audit["mask_iou"],
                "alignment_bbox_iou": audit["bbox_iou"],
                "alignment_centroid_distance_norm": audit["centroid_distance_norm"],
                "alignment_area_ratio": audit["area_ratio"],
                "alignment_transform": audit.get("alignment_transform", "identity"),
            },
            "weight": float(weight),
        })

    if alignment_audit_path is not None:
        audit_path = Path(alignment_audit_path)
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        with open(audit_path, "w", encoding="utf-8") as f:
            json.dump({
                "hallucination_index_path": str(halluc_index_path),
                "n_frames_audited": len(audits),
                "n_retained": len(views),
                "thresholds": {
                    "min_mask_iou": float(min_alignment_iou),
                    "min_bbox_iou": float(min_alignment_bbox_iou),
                    "max_centroid_distance_norm": float(max_alignment_centroid_distance),
                    "min_area_ratio": float(min_alignment_area_ratio),
                    "max_area_ratio": float(max_alignment_area_ratio),
                },
                "frames": audits,
            }, f, indent=2)

    logger.info(
        "Phase 6: built %d hallucinated supervision views from %s after auditing %d frame files.",
        len(views), halluc_index_path.name, len(audits),
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
    min_halluc_area_ratio: float = 0.65,
    max_halluc_area_ratio: float = 1.45,
    hallucination_alignment_audit_path: str | Path | None = None,
) -> List[dict]:
    """Build one aligned training set containing real and hallucinated views."""
    real_views = build_real_supervision_views(
        extraction_index_path=extraction_index_path,
        scope=scope,
        weight=real_weight,
        target_long_edge=real_target_long_edge,
    )
    _aabb_min = np.asarray(scope.aabb_min_W, dtype=np.float64) if scope is not None else None
    _aabb_max = np.asarray(scope.aabb_max_W, dtype=np.float64) if scope is not None else None
    hallucinated_views = build_hallucinated_supervision_views(
        halluc_index_path=halluc_index_path,
        local_sv3d=local_sv3d,
        weight=hallucination_weight,
        fov_y_deg=fov_y_deg,
        target_resolution=hallucination_resolution,
        up_W_override=up_W_override,
        include_conditioning=include_conditioning,
        min_alignment_iou=float(min_hallucination_alignment_iou),
        min_alignment_area_ratio=float(min_halluc_area_ratio),
        max_alignment_area_ratio=float(max_halluc_area_ratio),
        aabb_min_W=_aabb_min,
        aabb_max_W=_aabb_max,
        alignment_audit_path=hallucination_alignment_audit_path,
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
            "weight": float(v["weight"]),
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"n_views": len(payload), "views": payload}, f, indent=2)
    return output_path
