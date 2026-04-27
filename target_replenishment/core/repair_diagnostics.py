"""
Read-only verification diagnostics for candidate target replenishment views.

This module does not edit anchors, offsets, scales, gates, or MLPs. It renders
the current Scaffold-GS/ObjectGS object from each candidate camera and compares
that render against the generated candidate foreground. The output is intended
to decide which views are trustworthy enough for later repair and where the
current model is producing likely floaters.
"""

__all__ = ["analyze_repair_candidates"]

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def analyze_repair_candidates(
    gaussians,
    pipe_config,
    supervision_views: list,
    object_id: int,
    object_anchors,
    object_radius: float,
    output_dir=None,
    target_mask_erode_px: int = 0,
    alpha_threshold: float = 0.03,
    save_debug_images: bool = True,
) -> dict:
    """Score candidate repair views against the current 2DGS render.

    Args:
        gaussians: Loaded ObjectGS GaussianModel. Read-only in this function.
        pipe_config: ObjectGS render pipeline config.
        supervision_views: List with ``rgb`` and ``camera`` entries.
        object_id: Semantic object label to render in isolation.
        object_anchors: Anchor positions for the target object before repair.
        object_radius: Coverage-estimated object radius.
        output_dir: Optional directory for JSON-adjacent debug PNGs.
        target_mask_erode_px: Erode generated masks to reduce edge uncertainty.
        alpha_threshold: Render alpha threshold for binary diagnostics.
        save_debug_images: Save per-view masks/overlays when ``output_dir`` is set.

    Returns:
        JSON-serializable summary with per-view scores and likely floater anchors.
    """
    import torch
    from target_replenishment.core.objectgs_bridge import (
        build_anchor_id_map,
        create_virtual_camera,
        project_anchor_silhouette,
        render_view,
    )

    out_dir = Path(output_dir) if output_dir is not None else None
    if out_dir is not None and save_debug_images:
        out_dir.mkdir(parents=True, exist_ok=True)

    if not supervision_views:
        return {
            "n_views": 0,
            "view_scores": [],
            "summary": {},
            "top_suspicious_anchors": [],
        }

    target_masks = []
    target_areas = []
    for view in supervision_views:
        mask = _target_mask_from_rgb(view.get("rgb"), erode_px=target_mask_erode_px)
        target_masks.append(mask)
        target_areas.append(float(mask.sum()))

    positive_areas = [area for area in target_areas if area > 0.0]
    median_target_area = float(np.median(positive_areas)) if positive_areas else 0.0

    bg_color = torch.ones(3, dtype=torch.float32, device="cuda")
    object_anchors_np = np.asarray(object_anchors, dtype=np.float32)
    n_anchors = int(getattr(gaussians, "_anchor").shape[0]) if hasattr(gaussians, "_anchor") else 0
    n_original = getattr(gaussians, "n_original_anchors", None)
    if n_original is not None:
        n_original = int(n_original)

    view_scores = []
    aggregate_anchor_votes = {}

    for idx, (view, target_mask) in enumerate(zip(supervision_views, target_masks)):
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
        if target_mask.shape != (height, width):
            target_mask = cv2.resize(
                target_mask.astype(np.uint8),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            ) > 0

        render_mask = alpha > float(alpha_threshold)
        anchor_silhouette = project_anchor_silhouette(
            cam,
            object_anchors_np,
            object_radius=float(object_radius),
            height=height,
            width=width,
        ) > 0.10

        outside_mask = render_mask & (~target_mask)
        missing_mask = target_mask & (~render_mask)
        target_area = float(target_mask.sum())
        render_area = float(render_mask.sum())
        outside_area = float(outside_mask.sum())
        missing_area = float(missing_mask.sum())

        target_render_iou = _binary_iou(target_mask, render_mask)
        target_silhouette_iou = _binary_iou(target_mask, anchor_silhouette)
        render_silhouette_iou = _binary_iou(render_mask, anchor_silhouette)
        outside_alpha_mass = float(alpha[outside_mask].sum()) if outside_mask.any() else 0.0
        total_alpha_mass = float(alpha[render_mask].sum()) if render_mask.any() else 0.0
        outside_alpha_ratio = outside_alpha_mass / max(total_alpha_mass, 1e-6)
        missing_ratio = missing_area / max(target_area, 1.0)
        area_consistency = _area_consistency(target_area, median_target_area)
        neighbor_area_consistency = _neighbor_area_consistency(target_areas, idx)

        components = _component_stats(outside_mask, alpha, min_pixels=16, limit=5)
        max_component_area = float(components[0]["area_px"]) if components else 0.0

        anchor_map = build_anchor_id_map(render_result, height, width, n_anchors)
        suspicious_anchors = _top_anchor_ids(
            anchor_map,
            outside_mask,
            alpha,
            n_original_anchors=n_original,
            limit=8,
        )
        for item in suspicious_anchors:
            anchor_id = str(item["anchor_id"])
            aggregate_anchor_votes[anchor_id] = aggregate_anchor_votes.get(anchor_id, 0.0) + float(item["alpha_mass"])

        trust_score = _trust_score(
            target_render_iou=target_render_iou,
            target_silhouette_iou=target_silhouette_iou,
            outside_alpha_ratio=outside_alpha_ratio,
            missing_ratio=missing_ratio,
            area_consistency=area_consistency,
            neighbor_area_consistency=neighbor_area_consistency,
        )
        recommendation = _recommendation(
            target_area=target_area,
            target_silhouette_iou=target_silhouette_iou,
            outside_alpha_ratio=outside_alpha_ratio,
            max_component_area=max_component_area,
            target_render_iou=target_render_iou,
            trust_score=trust_score,
        )

        view_score = {
            "view_index": int(idx),
            "azimuth_offset_deg": _json_float(cam_data.get("azimuth_offset_deg")),
            "elevation_offset_deg": _json_float(cam_data.get("elevation_offset_deg")),
            "target_area_px": int(target_area),
            "render_area_px": int(render_area),
            "outside_area_px": int(outside_area),
            "missing_area_px": int(missing_area),
            "target_render_iou": float(target_render_iou),
            "target_silhouette_iou": float(target_silhouette_iou),
            "render_silhouette_iou": float(render_silhouette_iou),
            "outside_alpha_ratio": float(outside_alpha_ratio),
            "missing_ratio": float(missing_ratio),
            "area_consistency": float(area_consistency),
            "neighbor_area_consistency": float(neighbor_area_consistency),
            "trust_score": float(trust_score),
            "recommendation": recommendation,
            "outside_components": components,
            "suspicious_anchors": suspicious_anchors,
        }
        view_scores.append(view_score)

        if out_dir is not None and save_debug_images:
            prefix = out_dir / f"view_{idx:02d}"
            _save_debug_images(prefix, rgb, alpha, target_mask, render_mask, outside_mask, missing_mask)

    usable = [v for v in view_scores if v["recommendation"] == "usable_prior"]
    floater = [v for v in view_scores if v["recommendation"] == "inspect_floaters"]
    rejected = [v for v in view_scores if v["recommendation"] == "reject_prior"]
    trust_values = [float(v["trust_score"]) for v in view_scores]
    outside_values = [float(v["outside_alpha_ratio"]) for v in view_scores]
    missing_values = [float(v["missing_ratio"]) for v in view_scores]

    top_suspicious_anchors = [
        {"anchor_id": int(anchor_id), "alpha_mass": float(mass)}
        for anchor_id, mass in sorted(
            aggregate_anchor_votes.items(), key=lambda item: item[1], reverse=True
        )[:12]
    ]

    summary = {
        "mean_trust_score": float(np.mean(trust_values)) if trust_values else 0.0,
        "min_trust_score": float(np.min(trust_values)) if trust_values else 0.0,
        "mean_outside_alpha_ratio": float(np.mean(outside_values)) if outside_values else 0.0,
        "mean_missing_ratio": float(np.mean(missing_values)) if missing_values else 0.0,
        "n_usable_prior_views": int(len(usable)),
        "n_floater_inspection_views": int(len(floater)),
        "n_rejected_prior_views": int(len(rejected)),
        "median_target_area_px": float(median_target_area),
    }

    logger.info(
        "Repair diagnostics: %d views, usable=%d, floater_inspect=%d, rejected=%d, mean_trust=%.3f",
        len(view_scores), len(usable), len(floater), len(rejected), summary["mean_trust_score"],
    )

    return {
        "n_views": int(len(view_scores)),
        "alpha_threshold": float(alpha_threshold),
        "target_mask_erode_px": int(max(0, target_mask_erode_px)),
        "summary": summary,
        "top_suspicious_anchors": top_suspicious_anchors,
        "view_scores": view_scores,
    }


def _target_mask_from_rgb(rgb, erode_px: int = 0) -> np.ndarray:
    if rgb is None:
        return np.zeros((0, 0), dtype=bool)
    arr = np.asarray(rgb)
    if arr.ndim != 3:
        return np.zeros(arr.shape[:2], dtype=bool)
    if arr.dtype == np.uint8:
        arr_f = arr.astype(np.float32) / 255.0
    else:
        arr_f = np.clip(arr.astype(np.float32), 0.0, 1.0)
    mask = arr_f.mean(axis=2) < 0.98
    mask = _largest_component_mask(mask, min_pixels=64)
    if int(erode_px) > 0 and mask.any():
        kernel_size = 2 * int(erode_px) + 1
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1) > 0
        if int(eroded.sum()) >= 64:
            mask = eroded
    return mask.astype(bool)


def _tensor_to_hw_np(tensor) -> np.ndarray:
    if tensor is None:
        return np.zeros((1, 1), dtype=np.float32)
    arr = tensor.detach().float().cpu().numpy()
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected alpha tensor convertible to HxW, got {arr.shape}")
    return np.clip(arr.astype(np.float32), 0.0, 1.0)


def _tensor_to_hwc_uint8(tensor) -> np.ndarray:
    if tensor is None:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    arr = tensor.detach().float().cpu().numpy()
    arr = np.squeeze(arr)
    if arr.ndim == 3 and arr.shape[0] == 3:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"Expected RGB tensor convertible to HxWx3, got {arr.shape}")
    return (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)


def _binary_iou(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a).astype(bool)
    b = np.asarray(b).astype(bool)
    union = float((a | b).sum())
    if union <= 0.0:
        return 1.0
    return float((a & b).sum()) / union


def _area_consistency(area: float, median_area: float) -> float:
    area = float(area)
    median_area = float(median_area)
    if area <= 0.0 or median_area <= 0.0:
        return 0.0
    return float(np.clip(min(area, median_area) / max(area, median_area), 0.0, 1.0))


def _neighbor_area_consistency(areas: list, idx: int) -> float:
    if len(areas) <= 1:
        return 1.0
    cur = float(areas[idx])
    if cur <= 0.0:
        return 0.0
    prev_area = float(areas[(idx - 1) % len(areas)])
    next_area = float(areas[(idx + 1) % len(areas)])
    neighbor = float(np.mean([a for a in (prev_area, next_area) if a > 0.0])) if (prev_area > 0.0 or next_area > 0.0) else 0.0
    if neighbor <= 0.0:
        return 0.0
    return float(np.clip(min(cur, neighbor) / max(cur, neighbor), 0.0, 1.0))


def _trust_score(
    target_render_iou: float,
    target_silhouette_iou: float,
    outside_alpha_ratio: float,
    missing_ratio: float,
    area_consistency: float,
    neighbor_area_consistency: float,
) -> float:
    score = (
        0.30 * float(target_render_iou)
        + 0.25 * float(target_silhouette_iou)
        + 0.20 * (1.0 - float(outside_alpha_ratio))
        + 0.10 * (1.0 - float(missing_ratio))
        + 0.10 * float(area_consistency)
        + 0.05 * float(neighbor_area_consistency)
    )
    return float(np.clip(score, 0.0, 1.0))


def _recommendation(
    target_area: float,
    target_silhouette_iou: float,
    outside_alpha_ratio: float,
    max_component_area: float,
    target_render_iou: float,
    trust_score: float,
) -> str:
    if target_area < 64.0 or target_silhouette_iou < 0.12:
        return "reject_prior"
    large_outside_component = max_component_area > max(128.0, 0.06 * target_area)
    if outside_alpha_ratio > 0.25 or large_outside_component:
        return "inspect_floaters"
    if trust_score >= 0.55 and target_render_iou >= 0.15:
        return "usable_prior"
    return "inspect_prior"


def _component_stats(mask: np.ndarray, alpha: np.ndarray, min_pixels: int = 16, limit: int = 5) -> list:
    mask_u8 = np.asarray(mask).astype(np.uint8)
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


def _top_anchor_ids(anchor_map: np.ndarray, mask: np.ndarray, alpha: np.ndarray, n_original_anchors=None, limit: int = 8) -> list:
    if anchor_map is None or anchor_map.shape != mask.shape:
        return []
    valid = mask & (anchor_map >= 0)
    if not valid.any():
        return []
    ids = anchor_map[valid].astype(np.int64)
    weights = alpha[valid].astype(np.float64)
    totals = {}
    pixels = {}
    for anchor_id, weight in zip(ids, weights):
        key = int(anchor_id)
        totals[key] = totals.get(key, 0.0) + float(weight)
        pixels[key] = pixels.get(key, 0) + 1
    out = []
    for anchor_id, mass in sorted(totals.items(), key=lambda item: item[1], reverse=True)[: int(limit)]:
        item = {
            "anchor_id": int(anchor_id),
            "alpha_mass": float(mass),
            "pixel_count": int(pixels.get(anchor_id, 0)),
        }
        if n_original_anchors is not None:
            item["is_seeded_anchor"] = bool(int(anchor_id) >= int(n_original_anchors))
        out.append(item)
    return out


def _largest_component_mask(mask: np.ndarray, min_pixels: int = 16) -> np.ndarray:
    mask_u8 = (np.asarray(mask).astype(np.uint8) > 0).astype(np.uint8)
    if int(mask_u8.sum()) < int(min_pixels):
        return mask_u8.astype(bool)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if n_labels <= 1:
        return mask_u8.astype(bool)
    areas = stats[1:, cv2.CC_STAT_AREA]
    if areas.size == 0:
        return mask_u8.astype(bool)
    keep_label = 1 + int(np.argmax(areas))
    keep = labels == keep_label
    if int(keep.sum()) < int(min_pixels):
        return mask_u8.astype(bool)
    return keep


def _save_debug_images(prefix: Path, rgb: np.ndarray, alpha: np.ndarray, target: np.ndarray, render_mask: np.ndarray, outside: np.ndarray, missing: np.ndarray) -> None:
    cv2.imwrite(str(prefix.with_name(prefix.name + "_render.png")), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(prefix.with_name(prefix.name + "_alpha.png")), (np.clip(alpha, 0.0, 1.0) * 255.0).astype(np.uint8))
    cv2.imwrite(str(prefix.with_name(prefix.name + "_target_mask.png")), target.astype(np.uint8) * 255)
    cv2.imwrite(str(prefix.with_name(prefix.name + "_render_mask.png")), render_mask.astype(np.uint8) * 255)
    overlay = rgb.copy()
    overlay[target] = (0.65 * overlay[target] + 0.35 * np.array([0, 180, 0])).astype(np.uint8)
    overlay[outside] = (255, 64, 64)
    overlay[missing] = (64, 128, 255)
    cv2.imwrite(str(prefix.with_name(prefix.name + "_verify_overlay.png")), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


def _json_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

# ──────────────────────────────────────────────────────────────────────────────
# Supervision-view filter (extracted from run_replenishment.py)
# ──────────────────────────────────────────────────────────────────────────────


def filter_supervision_views(
    supervision_views,
    repair_diag_result,
    min_trust: float = 0.45,
    max_outside_alpha_ratio: float = 0.20,
    max_missing_ratio: float = 0.55,
    min_target_render_iou: float = 0.30,
    min_kept_views: int = 2,
    allow_inspect_prior: bool = True,
):
    """Keep only candidate views that passed read-only repair diagnostics.

    The filter is intentionally conservative: views flagged as floaters or
    rejected priors are not allowed to drive seeding/optimization. Kept views
    are reweighted by their trust score so marginal priors contribute less.
    """
    view_scores = repair_diag_result.get("view_scores", []) if repair_diag_result else []
    scores_by_index = {
        int(score.get("view_index")): score
        for score in view_scores
        if score.get("view_index") is not None
    }

    kept_views = []
    kept_entries = []
    rejected_entries = []

    for idx, view in enumerate(supervision_views):
        score = scores_by_index.get(idx)
        if score is None:
            rejected_entries.append({
                "view_index": int(idx),
                "reason": "missing_repair_diagnostic_score",
            })
            continue

        recommendation = str(score.get("recommendation", ""))
        trust_score = float(score.get("trust_score", 0.0))
        outside_alpha_ratio = float(score.get("outside_alpha_ratio", 1.0))
        missing_ratio = float(score.get("missing_ratio", 1.0))
        target_render_iou = float(score.get("target_render_iou", 0.0))

        metric_ok = (
            trust_score >= float(min_trust)
            and outside_alpha_ratio <= float(max_outside_alpha_ratio)
            and missing_ratio <= float(max_missing_ratio)
            and target_render_iou >= float(min_target_render_iou)
        )
        keep = recommendation == "usable_prior" and metric_ok
        if allow_inspect_prior and recommendation == "inspect_prior":
            keep = metric_ok

        entry = {
            "view_index": int(idx),
            "azimuth_offset_deg": score.get("azimuth_offset_deg"),
            "elevation_offset_deg": score.get("elevation_offset_deg"),
            "recommendation": recommendation,
            "trust_score": trust_score,
            "outside_alpha_ratio": outside_alpha_ratio,
            "missing_ratio": missing_ratio,
            "target_render_iou": target_render_iou,
        }

        if keep:
            filtered_view = dict(view)
            original_weight = float(filtered_view.get("weight", 1.0))
            filtered_view["weight"] = original_weight * max(trust_score, 1e-3)
            filtered_view["repair_diagnostic"] = entry
            entry["original_weight"] = original_weight
            entry["filtered_weight"] = float(filtered_view["weight"])
            kept_views.append(filtered_view)
            kept_entries.append(entry)
        else:
            if recommendation in {"inspect_floaters", "reject_prior"}:
                reason = recommendation
            elif trust_score < float(min_trust):
                reason = "below_min_trust"
            elif outside_alpha_ratio > float(max_outside_alpha_ratio):
                reason = "above_max_outside_alpha"
            elif missing_ratio > float(max_missing_ratio):
                reason = "above_max_missing_ratio"
            elif target_render_iou < float(min_target_render_iou):
                reason = "below_min_target_render_iou"
            else:
                reason = "not_allowed_by_filter"
            entry["reason"] = reason
            rejected_entries.append(entry)

    min_kept_views = max(0, int(min_kept_views))
    if len(kept_views) < min_kept_views:
        for entry in kept_entries:
            rejected = dict(entry)
            rejected["reason"] = "below_min_filtered_view_count"
            rejected_entries.append(rejected)
        kept_views = []
        kept_entries = []

    filter_result = {
        "enabled": True,
        "raw_view_count": int(len(supervision_views)),
        "kept_view_count": int(len(kept_views)),
        "rejected_view_count": int(len(rejected_entries)),
        "min_trust": float(min_trust),
        "max_outside_alpha_ratio": float(max_outside_alpha_ratio),
        "max_missing_ratio": float(max_missing_ratio),
        "min_target_render_iou": float(min_target_render_iou),
        "min_kept_views": int(min_kept_views),
        "allow_inspect_prior": bool(allow_inspect_prior),
        "kept_view_indices": [int(entry["view_index"]) for entry in kept_entries],
        "kept_views": kept_entries,
        "rejected_views": rejected_entries,
    }
    return kept_views, filter_result
