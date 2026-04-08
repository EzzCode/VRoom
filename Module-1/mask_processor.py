"""
VRoom Mask Post-Processing Pipeline

Consolidates all mask post-processing: background filtering, containment-based
merging, disconnected-island splitting, and debug visualization. Reads raw SAM
output from sam_inference.py and produces clean per-frame masks ready for tracking.

Usage:
    python mask_processor.py --input_dir data/images --output_dir data/sam_output

Outputs:
    <output_dir>/masks/        — compressed .npz files (stacked bool masks per frame)
    <output_dir>/debug_vis/    — overlay PNGs showing each mask in a unique color
"""

import sys
import argparse
import logging
from pathlib import Path
import numpy as np
import cv2

from sam_inference import load_sam, generate_masks

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)


# ── Background Filtering ─────────────────────────────────────────────────────

def filter_background_masks(sam_masks, img_h, img_w, max_area_ratio=0.40, border_touch_threshold=0.25):
    """Filter SAM mask dicts, returning only foreground binary masks based on area and border touch."""
    foreground_masks = []
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    for mask_data in sam_masks:
        mask = mask_data['segmentation']
        
        # 1. Area check
        mask_area = np.count_nonzero(mask)
        if (mask_area / (img_h * img_w)) > max_area_ratio:
            continue

        # 2. Border touch check
        mask_uint8 = mask.astype(np.uint8)
        # outline of the object
        perimeter_mask = cv2.morphologyEx(mask_uint8, cv2.MORPH_GRADIENT, kernel)
        total_perimeter = np.count_nonzero(perimeter_mask)
        
        if total_perimeter == 0:
            continue

        border_pixels = (
            np.count_nonzero(perimeter_mask[0, :]) +
            np.count_nonzero(perimeter_mask[img_h - 1, :]) +
            np.count_nonzero(perimeter_mask[:, 0]) +
            np.count_nonzero(perimeter_mask[:, img_w - 1])
        )
        
        if (border_pixels / total_perimeter) > border_touch_threshold:
            continue

        foreground_masks.append(mask)

    return foreground_masks


# ── Mask Merging ──────────────────────────────────────────────────────────────

def merge_overlapping_masks(masks, containment_thresh=0.7):
    """Merge masks where a smaller mask is mostly contained within a larger one.

    If mask A's overlap with mask B is > containment_thresh of A's area,
    A is absorbed into B (union). This merges sub-parts of objects into
    whole-object masks.
    """
    if len(masks) <= 1:
        return masks

    # Sort largest first so small parts get absorbed into bigger masks
    areas = [m.sum() for m in masks]
    order = np.argsort(areas)[::-1]
    masks = [masks[i] for i in order]
    areas = [areas[i] for i in order]

    merged = [True] * len(masks)

    for i in range(len(masks)):
        if not merged[i]:
            continue
        for j in range(i + 1, len(masks)):
            if not merged[j]:
                continue
            overlap = np.logical_and(masks[i], masks[j]).sum()
            if areas[j] > 0 and (overlap / areas[j]) > containment_thresh:
                masks[i] = np.logical_or(masks[i], masks[j])
                areas[i] = masks[i].sum()
                merged[j] = False

    return [m for m, alive in zip(masks, merged) if alive]


def merge_by_proximity(masks, frame_hsv, max_gap_px=15, color_thresh=0.4):
    """Merge masks that are spatially close AND visually similar.

    Catches adjacent-but-non-overlapping parts of the same object that
    containment merge misses (e.g. left/right halves of a shoe).
    """
    if len(masks) <= 1:
        return masks

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max_gap_px, max_gap_px))
    alive = [True] * len(masks)

    # Pre-compute dilated masks and histograms
    dilated = [cv2.dilate(m.astype(np.uint8), kernel) > 0 for m in masks]
    hists = []
    for m in masks:
        h = cv2.calcHist([frame_hsv], [0, 1], m.astype(np.uint8), [32, 32], [0, 180, 0, 256])
        cv2.normalize(h, h, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        hists.append(h)

    for i in range(len(masks)):
        if not alive[i]:
            continue
        for j in range(i + 1, len(masks)):
            if not alive[j]:
                continue
            # Check if dilated versions overlap (masks are nearby)
            if not np.logical_and(dilated[i], dilated[j]).any():
                continue
            # Check color similarity
            dist = cv2.compareHist(hists[i], hists[j], cv2.HISTCMP_BHATTACHARYYA)
            if dist < color_thresh:
                masks[i] = np.logical_or(masks[i], masks[j])
                dilated[i] = cv2.dilate(masks[i].astype(np.uint8), kernel) > 0
                # Recompute histogram for merged mask
                h = cv2.calcHist([frame_hsv], [0, 1], masks[i].astype(np.uint8), [32, 32], [0, 180, 0, 256])
                cv2.normalize(h, h, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                hists[i] = h
                alive[j] = False

    return [m for m, a in zip(masks, alive) if a]


# ── Disconnected Island Splitting ─────────────────────────────────────────────

def split_disconnected_masks(masks, min_area=100):
    """Split masks containing multiple disconnected islands into separate masks."""
    refined = []
    for mask in masks:
        mask_uint8 = (mask * 255).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask_uint8, connectivity=8
        )
        if num_labels == 2:
            refined.append(mask)
        elif num_labels > 2:
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] >= min_area:
                    refined.append(labels == i)
    return refined


# ── Visualization ─────────────────────────────────────────────────────────────

def generate_colors(n):
    """Generate N visually distinct colors using HSV spacing."""
    colors = []
    for i in range(n):
        hue = int(180 * i / max(n, 1))
        color = cv2.cvtColor(
            np.array([[[hue, 200, 255]]], dtype=np.uint8), cv2.COLOR_HSV2BGR
        )[0][0]
        colors.append(tuple(int(c) for c in color))
    return colors


def draw_mask_overlay(image, masks, alpha=0.5):
    """Draw colored semi-transparent masks on top of the original image."""
    overlay = image.copy()
    colors = generate_colors(len(masks))
    for mask, color in zip(masks, colors):
        overlay[mask] = color
    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)


def save_frame_masks(path, masks):
    """Save masks in a compressed dense format to avoid object-array bloat."""
    if masks:
        stacked = np.stack([m.astype(np.bool_) for m in masks], axis=0)
    else:
        stacked = np.zeros((0, 0, 0), dtype=np.bool_)
    np.savez_compressed(
        str(path),
        masks=stacked,
    )


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_pipeline(args):
    """Full pipeline: SAM → filter background → merge → split → save masks + debug visualization."""
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    mask_dir = output_dir / "masks"
    vis_dir = output_dir / "visible_masks"

    if not input_dir.exists():
        sys.exit(logger.error(f"Input directory missing: {input_dir}"))

    mask_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    # SAM2
    mask_generator = load_sam(
        args.model_cfg, args.sam_ckpt, args.device,
        points_per_side=args.points_per_side,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
        min_mask_region_area=args.min_mask_area,
    )


    image_paths = sorted([
        p for p in input_dir.iterdir()
        if p.suffix.lower() in ['.png', '.jpg', '.jpeg']
    ])
    if not image_paths:
        sys.exit(logger.error(f"No images found in {input_dir}"))

    logger.info(f"Processing {len(image_paths)} frames...")

    for frame_idx, img_path in enumerate(image_paths):
        frame = cv2.imread(str(img_path))
        if frame is None:
            logger.warning(f"Could not read image: {img_path}")
            continue

        frame_h, frame_w = frame.shape[:2]
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        min_area = max(80, int(args.min_area_ratio * frame_h * frame_w))

        # SAM inference
        raw_masks = generate_masks(mask_generator, frame)

        # Post-processing chain: filter → containment merge → proximity merge → split
        fg_masks = filter_background_masks(
            raw_masks, frame_h, frame_w, 
            max_area_ratio=args.max_area, 
            border_touch_threshold=args.border_touch
        )
        merged_masks = merge_overlapping_masks(fg_masks, containment_thresh=args.merge_thresh)
        merged_masks = merge_by_proximity(merged_masks, frame_hsv,
                                          max_gap_px=args.proximity_gap,
                                          color_thresh=args.proximity_color_thresh)
        final_masks = split_disconnected_masks(merged_masks, min_area=min_area)

        logger.info(
            f"Frame {frame_idx:05d} ({img_path.name}) | "
            f"Raw: {len(raw_masks)}, FG: {len(fg_masks)}, "
            f"Merged: {len(merged_masks)}, Final: {len(final_masks)}"
        )

        # Save masks using compressed dense storage to avoid multi-GB pickle payloads.
        save_frame_masks(mask_dir / f"masks_{frame_idx:05d}.npz", final_masks)

        # Save debug overlay
        vis = draw_mask_overlay(frame, final_masks)
        cv2.putText(
            vis, f"Frame {frame_idx:05d} | {len(final_masks)} objects",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
        )
        cv2.imwrite(str(vis_dir / f"vis_{frame_idx:05d}.png"), vis)

    logger.info(f"Done. Masks: {mask_dir}, Visualizations: {vis_dir}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VRoom Mask Post-Processing Pipeline")
    parser.add_argument("--input_dir", required=True, help="Path to input images directory")
    parser.add_argument("--output_dir", required=True, help="Path to save masks and visualizations")

    # Background filter
    parser.add_argument("--max_area", type=float, default=0.50, help="Max image area %% for foreground")
    parser.add_argument("--border_touch", type=float, default=0.35, help="Max perimeter %% touching edges")

    # SAM 2 model
    parser.add_argument("--model_cfg", default="sam2.1_hiera_l", help="SAM 2 config key or YAML path (sam2.1_hiera_t/s/b+/l)")
    parser.add_argument("--sam_ckpt", default=r"Module-1\models\sam2.1_hiera_large.pt", help="SAM 2 checkpoint path (.pt)")
    parser.add_argument("--device", default="cuda", help="Device ('cuda' or 'cpu')")

    # SAM tuning
    parser.add_argument("--points_per_side", type=int, default=32, help="Grid points per side (default SAM: 32)")
    parser.add_argument("--pred_iou_thresh", type=float, default=0.88, help="Min predicted IoU (default SAM: 0.88)")
    parser.add_argument("--stability_score_thresh", type=float, default=0.95, help="Min stability score (default SAM: 0.95)")
    parser.add_argument("--min_mask_area", type=int, default=300, help="Min mask area in pixels (default SAM: 0)")

    # Merge
    parser.add_argument("--merge_thresh", type=float, default=0.78, help="Containment ratio to merge masks (0-1)")
    parser.add_argument("--proximity_gap", type=int, default=20, help="Max pixel gap for proximity merge")
    parser.add_argument("--proximity_color_thresh", type=float, default=0.32, help="Max color distance for proximity merge (0-1)")
    parser.add_argument("--min_area_ratio", type=float, default=0.0035, help="Min mask area as fraction of image (replaces absolute min_area)")

    run_pipeline(parser.parse_args())
