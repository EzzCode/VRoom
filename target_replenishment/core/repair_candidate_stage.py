"""
Aligned repair-candidate stage.

Read-only module that, for each Zero123++ candidate view:
  1. Re-renders the current ObjectGS object from the candidate camera.
  2. Extracts a target foreground mask from the generated RGB.
  3. Builds a *conservative* repair mask (missing regions only where they
     touch / closely neighbor the current ObjectGS render support).
  4. Builds a floater-candidate mask (outside-of-target render alpha).
  5. Scores each view and emits an accept/inspect/reject recommendation.

The output is intended to drive verified seeding and floater-suppression
diagnostics without mutating the model. No tensors are updated; no anchors,
offsets, scales, gates, MLPs, or optimizer state are touched.

This module deliberately does NOT trust raw Zero123++ foreground masks as
ground truth. It only keeps mask components that are anchored to current
object support, preventing crescent-scale or far-away mask islands from
becoming supervision targets.
"""

from __future__ import annotations

__all__ = ["analyze_aligned_repair_candidates"]

import logging
from pathlib import Path

import cv2
import numpy as np

from target_replenishment.core.repair_diagnostics import (
    _binary_iou,
    _largest_component_mask,
    _target_mask_from_rgb,
    _tensor_to_hw_np,
    _tensor_to_hwc_uint8,
    _json_float,
)
from target_replenishment.core.image_alignment import align_image_to_render_bbox

logger = logging.getLogger(__name__)


def analyze_aligned_repair_candidates(
    gaussians,
    pipe_config,
    supervision_views: list,
    object_id: int,
    object_anchors,
    object_radius: float,
    output_dir=None,
    target_mask_erode_px: int = 0,
    alpha_threshold: float = 0.03,
    support_dilate_px: int = 12,
    min_repair_component_px: int = 32,
    max_repair_components: int = 6,
    max_repair_area_ratio: float = 0.40,
    min_target_render_iou: float = 0.20,
    max_target_render_area_ratio: float = 2.25,
    max_outside_alpha_ratio: float = 0.20,
    min_target_area_px: int = 64,
    floater_min_component_px: int = 32,
    save_debug_images: bool = True,
) -> dict:
    """Build conservative aligned repair masks per candidate view.

    Args:
        gaussians: Loaded ObjectGS GaussianModel (read-only).
        pipe_config: ObjectGS render pipeline config.
        supervision_views: List with ``rgb`` and ``camera`` entries.
        object_id: Semantic object label to render in isolation.
        object_anchors: Anchor positions for the target object (read-only).
        object_radius: Coverage-estimated object radius (read-only).
        output_dir: Optional directory for debug PNGs and aux JSON.
        target_mask_erode_px: Erode generated target masks at the boundary.
        alpha_threshold: Render alpha threshold for the binary render mask.
        support_dilate_px: Pixels of dilation applied to the current render
            mask to define the "support zone". Repair components are only
            accepted when they fall inside this zone (i.e. they touch or
            closely neighbor existing object support).
        min_repair_component_px: Minimum connected-component size for a
            missing region to be considered a repair candidate.
        max_repair_components: Cap on how many repair components are kept
            per view (largest by area).
        max_repair_area_ratio: Reject views whose total repair-mask area is
            larger than this fraction of the target-mask area.
        min_target_render_iou: Minimum target/render IoU required to even
            consider a view for repair (sanity gate).
        max_target_render_area_ratio: Reject/inspect views where the generated
            target mask is much larger than the cleaned render support.
        max_outside_alpha_ratio: Reject/inspect views with too much render
            alpha in floater components outside the generated target.
        min_target_area_px: Minimum target-mask area in pixels.
        floater_min_component_px: Minimum component size for floater
            candidates reported in the outside-alpha region.
        save_debug_images: Save per-view masks/overlays when ``output_dir``
            is provided.

    Returns:
        JSON-serializable dict with per-view scores and a summary.
    """
    import torch
    from target_replenishment.core.objectgs_bridge import (
        create_virtual_camera,
        render_view,
    )

    out_dir = Path(output_dir) if output_dir is not None else None
    if out_dir is not None and save_debug_images:
        out_dir.mkdir(parents=True, exist_ok=True)

    params = {
        "alpha_threshold": float(alpha_threshold),
        "target_mask_erode_px": int(max(0, target_mask_erode_px)),
        "support_dilate_px": int(max(0, support_dilate_px)),
        "min_repair_component_px": int(max(1, min_repair_component_px)),
        "max_repair_components": int(max(1, max_repair_components)),
        "max_repair_area_ratio": float(max_repair_area_ratio),
        "min_target_render_iou": float(min_target_render_iou),
        "max_target_render_area_ratio": float(max_target_render_area_ratio),
        "max_outside_alpha_ratio": float(max_outside_alpha_ratio),
        "min_target_area_px": int(max(0, min_target_area_px)),
        "floater_min_component_px": int(max(1, floater_min_component_px)),
    }

    if not supervision_views:
        return {
            "n_views": 0,
            "params": params,
            "view_scores": [],
            "summary": {
                "n_accept_repair_views": 0,
                "n_inspect_repair_views": 0,
                "n_reject_repair_views": 0,
                "mean_repair_area_ratio": 0.0,
                "mean_target_render_iou": 0.0,
                "total_repair_area_px": 0,
            },
        }

    bg_color = torch.ones(3, dtype=torch.float32, device="cuda")

    view_scores: list = []
    repair_area_ratios: list = []
    target_render_ious: list = []
    total_repair_area_px = 0

    for idx, view in enumerate(supervision_views):
        cam_data = view["camera"]
        cam = create_virtual_camera(
            cam_data["R"],
            cam_data["T"],
            cam_data["K"],
            cam_data["width"],
            cam_data["height"],
        )

        render_result = render_view(
            gaussians,
            cam,
            pipe_config,
            bg_color,
            object_label_id=int(object_id),
        )
        alpha = _tensor_to_hw_np(render_result.get("alpha"))
        rgb = _tensor_to_hwc_uint8(render_result.get("rgb"))
        height, width = alpha.shape

        # Align the raw Zero123++ tile to the current-render bbox so the target
        # mask lives in the same image-plane frame as the render. Without this,
        # the Zero123 tile is in its own canonical pose/scale/resolution and any
        # mask diff against the render mixes content from different frames.
        target_rgb_raw = view.get("rgb")
        target_rgb_aligned, align_dx, align_dy, align_scale = _prepare_aligned_target_rgb(
            target_rgb_raw, rgb,
        )
        target_mask = _target_mask_from_rgb(
            target_rgb_aligned, erode_px=target_mask_erode_px,
        )
        if target_mask.shape != (height, width):
            target_mask = cv2.resize(
                target_mask.astype(np.uint8),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            ) > 0

        render_mask = alpha > float(alpha_threshold)
        raw_outside_mask = render_mask & (~target_mask)
        floater_mask = _large_component_mask(
            raw_outside_mask,
            min_pixels=int(params["floater_min_component_px"]),
        )
        render_support_mask = render_mask & (~_dilate_mask(floater_mask, 2))
        missing_mask_raw = target_mask & (~render_support_mask)

        target_area = float(target_mask.sum())
        render_area = float(render_mask.sum())
        render_support_area = float(render_support_mask.sum())
        target_render_area_ratio = float(target_area / max(render_support_area, 1.0))

        raw_target_render_iou = _binary_iou(target_mask, render_mask)
        target_render_iou = _binary_iou(target_mask, render_support_mask)
        target_render_ious.append(target_render_iou)

        support_zone = _dilate_mask(render_support_mask, support_dilate_px)
        missing_in_support = missing_mask_raw & support_zone

        repair_mask, repair_components = _filter_repair_components(
            missing_in_support,
            render_support_mask,
            support_dilate_px=support_dilate_px,
            min_component_px=int(params["min_repair_component_px"]),
            max_components=int(params["max_repair_components"]),
        )
        repair_area = float(repair_mask.sum())
        repair_area_ratio = float(repair_area / max(target_area, 1.0))
        repair_area_ratios.append(repair_area_ratio)

        floater_components = _component_stats(
            floater_mask,
            alpha,
            min_pixels=int(params["floater_min_component_px"]),
            limit=int(params["max_repair_components"]),
        )

        outside_alpha_mass = float(alpha[floater_mask].sum()) if floater_mask.any() else 0.0
        total_alpha_mass = float(alpha[render_mask].sum()) if render_mask.any() else 0.0
        outside_alpha_ratio = float(outside_alpha_mass / max(total_alpha_mass, 1e-6))

        recommendation, reason = _recommend(
            target_area=target_area,
            target_render_iou=target_render_iou,
            target_render_area_ratio=target_render_area_ratio,
            repair_mask_area=repair_area,
            repair_area_ratio=repair_area_ratio,
            outside_alpha_ratio=outside_alpha_ratio,
            min_target_area_px=int(params["min_target_area_px"]),
            min_target_render_iou=float(params["min_target_render_iou"]),
            max_target_render_area_ratio=float(params["max_target_render_area_ratio"]),
            max_repair_area_ratio=float(params["max_repair_area_ratio"]),
            max_outside_alpha_ratio=float(params["max_outside_alpha_ratio"]),
        )

        view_score = {
            "view_index": int(idx),
            "azimuth_offset_deg": _json_float(cam_data.get("azimuth_offset_deg")),
            "elevation_offset_deg": _json_float(cam_data.get("elevation_offset_deg")),
            "image_size_hw": [int(height), int(width)],
            "alignment": {
                "dx_px": float(align_dx),
                "dy_px": float(align_dy),
                "scale": float(align_scale),
                "raw_size_hw": (
                    [int(target_rgb_raw.shape[0]), int(target_rgb_raw.shape[1])]
                    if target_rgb_raw is not None and hasattr(target_rgb_raw, "shape")
                    else None
                ),
            },
            "target_area_px": int(target_area),
            "render_area_px": int(render_area),
            "render_support_area_px": int(render_support_area),
            "raw_target_render_iou": float(raw_target_render_iou),
            "target_render_iou": float(target_render_iou),
            "target_render_area_ratio": float(target_render_area_ratio),
            "outside_alpha_ratio": float(outside_alpha_ratio),
            "floater_area_px": int(floater_mask.sum()),
            "repair_mask_area_px": int(repair_area),
            "repair_area_ratio": float(repair_area_ratio),
            "n_repair_components": int(len(repair_components)),
            "repair_components": repair_components,
            "n_floater_components": int(len(floater_components)),
            "floater_components": floater_components,
            "recommendation": recommendation,
            "reason": reason,
        }
        view_scores.append(view_score)
        if recommendation == "accept_repair":
            total_repair_area_px += int(repair_area)

        if out_dir is not None and save_debug_images:
            prefix = out_dir / f"view_{idx:02d}"
            _save_debug_images(
                prefix,
                rgb=rgb,
                alpha=alpha,
                target_rgb=target_rgb_aligned,
                target_mask=target_mask,
                render_mask=render_support_mask,
                outside_mask=floater_mask,
                missing_mask_raw=missing_mask_raw,
                repair_mask=repair_mask,
                support_zone=support_zone,
                recommendation=recommendation,
            )

    n_accept = sum(1 for v in view_scores if v["recommendation"] == "accept_repair")
    n_inspect = sum(1 for v in view_scores if v["recommendation"] == "inspect_repair")
    n_reject = sum(1 for v in view_scores if v["recommendation"] == "reject_repair")

    summary = {
        "n_accept_repair_views": int(n_accept),
        "n_inspect_repair_views": int(n_inspect),
        "n_reject_repair_views": int(n_reject),
        "mean_repair_area_ratio": float(np.mean(repair_area_ratios)) if repair_area_ratios else 0.0,
        "max_repair_area_ratio": float(np.max(repair_area_ratios)) if repair_area_ratios else 0.0,
        "mean_target_render_iou": float(np.mean(target_render_ious)) if target_render_ious else 0.0,
        "total_repair_area_px": int(total_repair_area_px),
    }

    logger.info(
        "Aligned repair candidates: %d views, accept=%d, inspect=%d, reject=%d, "
        "mean_repair_area_ratio=%.3f, mean_target_render_iou=%.3f",
        len(view_scores), n_accept, n_inspect, n_reject,
        summary["mean_repair_area_ratio"], summary["mean_target_render_iou"],
    )

    return {
        "n_views": int(len(view_scores)),
        "params": params,
        "summary": summary,
        "view_scores": view_scores,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Helpers (private)
# ──────────────────────────────────────────────────────────────────────────────


def _prepare_aligned_target_rgb(target_rgb, render_rgb_uint8: np.ndarray):
    """Align Zero123++ tile to the current-render bbox.

    Zero123 tiles arrive in their own canonical frame (and often a different
    resolution, e.g. 320x320), so a raw mask diff against a 512x512 render is
    incoherent. We resize the tile to the render resolution and bbox-align it,
    rejecting wildly out-of-range scale corrections (e.g. when the render is
    empty pre-seeding).

    Returns ``(aligned_rgb_uint8, dx, dy, scale)`` with the original (resized)
    image returned unchanged when alignment fails or is implausible.
    """
    if target_rgb is None:
        return render_rgb_uint8.copy(), 0.0, 0.0, 1.0
    arr = np.asarray(target_rgb)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        return render_rgb_uint8.copy(), 0.0, 0.0, 1.0
    target_u8 = arr if arr.dtype == np.uint8 else (
        np.clip(arr.astype(np.float32), 0.0, 1.0) * 255.0
    ).astype(np.uint8)
    rh, rw = render_rgb_uint8.shape[:2]
    if target_u8.shape[:2] != (rh, rw):
        target_u8 = cv2.resize(target_u8, (rw, rh), interpolation=cv2.INTER_LINEAR)
    try:
        aligned, dx, dy, scale = align_image_to_render_bbox(
            target_u8, render_rgb_uint8,
            bg_color=(255, 255, 255), return_diag=True,
        )
    except Exception:
        return target_u8, 0.0, 0.0, 1.0
    if not (0.25 <= float(scale) <= 4.0):
        return target_u8, 0.0, 0.0, 1.0
    return aligned, float(dx), float(dy), float(scale)


def _dilate_mask(mask: np.ndarray, radius_px: int) -> np.ndarray:
    radius_px = int(max(0, radius_px))
    mask_u8 = (np.asarray(mask).astype(np.uint8) > 0).astype(np.uint8)
    if radius_px <= 0 or mask_u8.sum() == 0:
        return mask_u8.astype(bool)
    kernel_size = 2 * radius_px + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    return cv2.dilate(mask_u8, kernel, iterations=1).astype(bool)


def _large_component_mask(mask: np.ndarray, min_pixels: int) -> np.ndarray:
    mask_u8 = (np.asarray(mask).astype(np.uint8) > 0).astype(np.uint8)
    if int(mask_u8.sum()) < int(min_pixels):
        return np.zeros_like(mask_u8, dtype=bool)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if n_labels <= 1:
        return np.zeros_like(mask_u8, dtype=bool)
    out = np.zeros_like(mask_u8, dtype=bool)
    for label in range(1, n_labels):
        if int(stats[label, cv2.CC_STAT_AREA]) >= int(min_pixels):
            out |= labels == label
    return out


def _filter_repair_components(
    missing_in_support: np.ndarray,
    render_mask: np.ndarray,
    support_dilate_px: int,
    min_component_px: int,
    max_components: int,
) -> tuple:
    """Return (repair_mask, components) keeping only components that touch the
    dilated render-mask boundary and exceed ``min_component_px`` pixels."""
    mask_u8 = (np.asarray(missing_in_support).astype(np.uint8) > 0).astype(np.uint8)
    if int(mask_u8.sum()) < int(min_component_px):
        return np.zeros_like(mask_u8, dtype=bool), []

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if n_labels <= 1:
        return np.zeros_like(mask_u8, dtype=bool), []

    # Slightly tighter "must-touch-support" check: a component must intersect
    # the render mask dilated by ``support_dilate_px`` pixels (which guarantees
    # adjacency to current ObjectGS evidence, not a far-floating island).
    touch_zone = _dilate_mask(render_mask, max(1, int(support_dilate_px)))

    candidates = []
    for label in range(1, n_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < int(min_component_px):
            continue
        comp_mask = labels == label
        if not bool((comp_mask & touch_zone).any()):
            # Disconnected island -> reject.
            continue
        candidates.append({
            "label": int(label),
            "area_px": int(area),
            "bbox_xywh": [
                int(stats[label, cv2.CC_STAT_LEFT]),
                int(stats[label, cv2.CC_STAT_TOP]),
                int(stats[label, cv2.CC_STAT_WIDTH]),
                int(stats[label, cv2.CC_STAT_HEIGHT]),
            ],
            "centroid_xy": [float(centroids[label][0]), float(centroids[label][1])],
        })

    candidates.sort(key=lambda item: item["area_px"], reverse=True)
    candidates = candidates[: int(max(1, max_components))]
    if not candidates:
        return np.zeros_like(mask_u8, dtype=bool), []

    keep_labels = {c["label"] for c in candidates}
    repair_mask = np.zeros_like(mask_u8, dtype=bool)
    for label in keep_labels:
        repair_mask |= labels == label

    out_components = [
        {k: v for k, v in c.items() if k != "label"}
        for c in candidates
    ]
    return repair_mask, out_components


def _component_stats(
    mask: np.ndarray,
    alpha: np.ndarray,
    min_pixels: int = 16,
    limit: int = 5,
) -> list:
    mask_u8 = (np.asarray(mask).astype(np.uint8) > 0).astype(np.uint8)
    if int(mask_u8.sum()) < int(min_pixels):
        return []
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    comps = []
    for label in range(1, n_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < int(min_pixels):
            continue
        comp_mask = labels == label
        comps.append({
            "area_px": int(area),
            "bbox_xywh": [
                int(stats[label, cv2.CC_STAT_LEFT]),
                int(stats[label, cv2.CC_STAT_TOP]),
                int(stats[label, cv2.CC_STAT_WIDTH]),
                int(stats[label, cv2.CC_STAT_HEIGHT]),
            ],
            "centroid_xy": [float(centroids[label][0]), float(centroids[label][1])],
            "alpha_mass": float(alpha[comp_mask].sum()),
        })
    comps.sort(key=lambda item: item["alpha_mass"], reverse=True)
    return comps[: int(limit)]


def _recommend(
    target_area: float,
    target_render_iou: float,
    target_render_area_ratio: float,
    repair_mask_area: float,
    repair_area_ratio: float,
    outside_alpha_ratio: float,
    min_target_area_px: int,
    min_target_render_iou: float,
    max_target_render_area_ratio: float,
    max_repair_area_ratio: float,
    max_outside_alpha_ratio: float,
) -> tuple:
    """Return (recommendation, reason)."""
    if float(target_area) < float(min_target_area_px):
        return "reject_repair", "target_mask_too_small"
    if float(target_render_iou) < float(min_target_render_iou):
        return "reject_repair", "low_target_render_iou"
    if float(target_render_area_ratio) > float(max_target_render_area_ratio):
        return "inspect_repair", "target_mask_too_large_for_render"
    if float(repair_mask_area) <= 0.0:
        return "inspect_repair", "no_repair_region"
    if float(repair_area_ratio) > float(max_repair_area_ratio):
        return "inspect_repair", "repair_region_too_large"
    if float(outside_alpha_ratio) > float(max_outside_alpha_ratio):
        return "inspect_repair", "high_outside_alpha"
    return "accept_repair", "ok"


def _save_debug_images(
    prefix: Path,
    rgb: np.ndarray,
    alpha: np.ndarray,
    target_rgb: np.ndarray,
    target_mask: np.ndarray,
    render_mask: np.ndarray,
    outside_mask: np.ndarray,
    missing_mask_raw: np.ndarray,
    repair_mask: np.ndarray,
    support_zone: np.ndarray,
    recommendation: str,
) -> None:
    cv2.imwrite(
        str(prefix.with_name(prefix.name + "_render.png")),
        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
    )
    if target_rgb is not None and getattr(target_rgb, "ndim", 0) == 3:
        target_rgb_u8 = target_rgb if target_rgb.dtype == np.uint8 else (
            np.clip(target_rgb.astype(np.float32), 0.0, 1.0) * 255.0
        ).astype(np.uint8)
        if target_rgb_u8.shape[:2] != rgb.shape[:2]:
            target_rgb_u8 = cv2.resize(target_rgb_u8, (rgb.shape[1], rgb.shape[0]))
        cv2.imwrite(
            str(prefix.with_name(prefix.name + "_target_rgb_aligned.png")),
            cv2.cvtColor(target_rgb_u8, cv2.COLOR_RGB2BGR),
        )
    cv2.imwrite(
        str(prefix.with_name(prefix.name + "_alpha.png")),
        (np.clip(alpha, 0.0, 1.0) * 255.0).astype(np.uint8),
    )
    cv2.imwrite(
        str(prefix.with_name(prefix.name + "_target_mask.png")),
        target_mask.astype(np.uint8) * 255,
    )
    cv2.imwrite(
        str(prefix.with_name(prefix.name + "_render_mask.png")),
        render_mask.astype(np.uint8) * 255,
    )
    cv2.imwrite(
        str(prefix.with_name(prefix.name + "_missing_raw.png")),
        missing_mask_raw.astype(np.uint8) * 255,
    )
    cv2.imwrite(
        str(prefix.with_name(prefix.name + "_repair_mask.png")),
        repair_mask.astype(np.uint8) * 255,
    )
    cv2.imwrite(
        str(prefix.with_name(prefix.name + "_outside_mask.png")),
        outside_mask.astype(np.uint8) * 255,
    )

    cleaned = rgb.copy()
    if outside_mask.any():
        cleaned[outside_mask] = 255
    cv2.imwrite(
        str(prefix.with_name(prefix.name + "_cleaned_render.png")),
        cv2.cvtColor(cleaned, cv2.COLOR_RGB2BGR),
    )

    overlay = cleaned.copy()
    if str(recommendation) == "accept_repair" and repair_mask.any():
        overlay[repair_mask] = (
            0.45 * overlay[repair_mask] + 0.55 * np.array([64, 128, 255])
        ).astype(np.uint8)
    boundary = support_zone & (~render_mask) if str(recommendation) == "accept_repair" else None
    if boundary is not None and boundary.any():
        overlay[boundary] = (
            0.80 * overlay[boundary] + 0.20 * np.array([64, 200, 64])
        ).astype(np.uint8)
    cv2.imwrite(
        str(prefix.with_name(prefix.name + "_repair_overlay.png")),
        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
    )


# Re-export private helpers used in tests / wiring.
__all__.extend([
    "_dilate_mask",
    "_large_component_mask",
    "_filter_repair_components",
    "_component_stats",
    "_recommend",
])
