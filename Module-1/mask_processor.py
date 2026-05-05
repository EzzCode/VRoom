"""Module-1 mask generation with SAM3 text prompting and deterministic cleanup.

This module is responsible for producing per-frame binary instance masks that
are consumed by the vanilla tracker. It intentionally keeps a stable output
contract and uses rule-based postprocessing to improve temporal robustness.

Output contract:
- `masks/masks_%05d.npz`: compressed bool tensor with shape `(N, H, W)`.
- `visible_masks/vis_%05d.png`: visualization overlay for inspection.

Processing pipeline:
1. SAM3 semantic segmentation from text prompts.
2. Background and border-touch filtering.
3. Overlap-based merge for containment cases.
4. Proximity and color-consistency merge.
5. Optional connected-component split.
"""

import sys
import os
import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from sam_inference import SAM3TextSegmenter

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

def _mask_to_box(mask: np.ndarray) -> Optional[List[int]]:
	"""Return [x1, y1, x2, y2] for a boolean mask, or None if empty."""
	ys, xs = np.where(mask)
	if xs.size == 0:
		return None
	return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def _bbox_gap_px(a: Sequence[int], b: Sequence[int]) -> float:
	"""Compute edge-to-edge Euclidean gap between two XYXY boxes."""
	ax1, ay1, ax2, ay2 = a
	bx1, by1, bx2, by2 = b
	dx = max(0, max(ax1 - bx2, bx1 - ax2))
	dy = max(0, max(ay1 - by2, by1 - ay2))
	return float(np.hypot(dx, dy))


def _mean_hsv(frame_hsv: np.ndarray, mask: np.ndarray) -> np.ndarray:
	"""Compute masked mean HSV color; returns zeros for empty masks."""
	px = frame_hsv[mask]
	if px.size == 0:
		return np.array([0.0, 0.0, 0.0], dtype=np.float32)
	return px.mean(axis=0).astype(np.float32)


def filter_background_masks(
	masks: List[np.ndarray],
	max_area_ratio: float = 0.50,
	border_touch_threshold: float = 0.35,
) -> List[np.ndarray]:
	"""Drop likely background masks using area and border-touch heuristics."""
	if not masks:
		return []

	stack = np.stack(masks, axis=0).astype(bool, copy=False)
	n, h, w = stack.shape

	img_area = float(h * w)
	areas = stack.reshape(n, -1).sum(axis=1).astype(np.float32)
	area_ok = (areas / max(1.0, img_area)) <= float(max_area_ratio)

	border_len = float((w * 2) + (h * 2) - 4)
	top_hits = stack[:, 0, :].sum(axis=1)
	bottom_hits = stack[:, h - 1, :].sum(axis=1)
	if h > 2:
		left_hits = stack[:, 1:h - 1, 0].sum(axis=1)
		right_hits = stack[:, 1:h - 1, w - 1].sum(axis=1)
	else:
		left_hits = np.zeros(n, dtype=np.int64)
		right_hits = np.zeros(n, dtype=np.int64)
	border_ratio = (top_hits + bottom_hits + left_hits + right_hits) / max(1.0, border_len)
	border_ok = border_ratio <= float(border_touch_threshold)

	keep = area_ok & border_ok
	idx = np.flatnonzero(keep)
	return [stack[i] for i in idx]


def merge_overlapping_masks(masks: List[np.ndarray], thresh: float = 0.78) -> List[np.ndarray]:
	"""Merge masks when one is strongly contained in another."""
	if not masks:
		return []

	merged = [m.copy() for m in masks]
	changed = True
	while changed:
		changed = False
		out: List[np.ndarray] = []
		used = [False] * len(merged)

		for i in range(len(merged)):
			if used[i]:
				continue
			base = merged[i].copy()
			used[i] = True
			area_base = float(base.sum())

			for j in range(i + 1, len(merged)):
				if used[j]:
					continue
				other = merged[j]
				inter = np.logical_and(base, other).sum()
				min_area = max(1.0, min(area_base, float(other.sum())))
				containment = float(inter) / min_area
				if containment >= thresh:
					base = np.logical_or(base, other)
					used[j] = True
					changed = True
					area_base = float(base.sum())

			out.append(base)

		merged = out

	return merged


def merge_by_proximity(
	masks: List[np.ndarray],
	frame_bgr: np.ndarray,
	gap_px: int = 20,
	color_thresh: float = 0.32,
) -> List[np.ndarray]:
	"""Merge nearby masks if mean HSV distance is below threshold."""
	if len(masks) <= 1:
		return masks

	frame_hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
	boxes = [_mask_to_box(m) for m in masks]
	means = [_mean_hsv(frame_hsv, m) for m in masks]

	hsv_norm = float(np.linalg.norm(np.array([180.0, 255.0, 255.0], dtype=np.float32)))
	merged = [m.copy() for m in masks]
	changed = True
	while changed:
		changed = False
		out: List[np.ndarray] = []
		used = [False] * len(merged)

		for i in range(len(merged)):
			if used[i]:
				continue
			base = merged[i].copy()
			used[i] = True

			for j in range(i + 1, len(merged)):
				if used[j]:
					continue
				if boxes[i] is None or boxes[j] is None:
					continue
				gap = _bbox_gap_px(boxes[i], boxes[j])
				if gap > float(gap_px):
					continue
				color_dist = np.linalg.norm(means[i] - means[j]) / hsv_norm
				if color_dist > float(color_thresh):
					continue

				base = np.logical_or(base, merged[j])
				used[j] = True
				changed = True

			out.append(base)

		merged = out
		boxes = [_mask_to_box(m) for m in merged]
		means = [_mean_hsv(frame_hsv, m) for m in merged]

	return merged


def split_disconnected(masks: List[np.ndarray], min_component_area: int = 1) -> List[np.ndarray]:
	"""Split each mask into connected components and keep sufficiently large ones."""
	if not masks:
		return []

	out: List[np.ndarray] = []
	for mask in masks:
		num_labels, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
		if num_labels <= 2:
			out.append(mask)
			continue
		for label in range(1, num_labels):
			comp = labels == label
			if int(comp.sum()) >= int(min_component_area):
				out.append(comp)

	return out


def postprocess_masks(
	raw_masks: List[np.ndarray],
	frame_bgr: np.ndarray,
	min_mask_area: int = 120,
	max_area_ratio: float = 0.50,
	border_touch_threshold: float = 0.35,
	merge_thresh: float = 0.78,
	proximity_gap: int = 20,
	proximity_color_thresh: float = 0.32,
	split_components: bool = True,
) -> Tuple[List[np.ndarray], Dict[str, int]]:
	"""Clean up raw SAM3 masks using rule-based spatial processing.

	Args:
		raw_masks: Unprocessed boolean masks from SAM3.
		frame_bgr: Input frame in BGR format.
		min_mask_area: Minimum kept component area in pixels.
		max_area_ratio: Maximum allowed mask area ratio.
		border_touch_threshold: Maximum allowed border-touch ratio.
		merge_thresh: Containment threshold for overlap merging.
		proximity_gap: Max pixel gap for proximity merge.
		proximity_color_thresh: Max normalized HSV distance for merge.
		split_components: Whether to split disconnected components.

	Returns:
		Tuple of `(final_masks, debug_stats)` where masks are boolean arrays.
	"""
	masks = filter_background_masks(raw_masks, max_area_ratio=max_area_ratio, border_touch_threshold=border_touch_threshold)
	masks = merge_overlapping_masks(masks, thresh=merge_thresh)
	masks = merge_by_proximity(masks, frame_bgr=frame_bgr, gap_px=proximity_gap, color_thresh=proximity_color_thresh)
	if split_components:
		masks = split_disconnected(masks, min_component_area=min_mask_area)

	masks = [m.astype(bool) for m in masks if int(m.sum()) >= min_mask_area]
	return masks, {"raw_mask_count": len(raw_masks), "final_mask_count": len(masks)}


def generate_colors(n: int) -> List[Tuple[int, int, int]]:
	"""Generate stable visually distinct BGR colors."""
	colors: List[Tuple[int, int, int]] = []
	for i in range(n):
		hue = int(180 * i / max(n, 1))
		color = cv2.cvtColor(np.array([[[hue, 200, 255]]], dtype=np.uint8), cv2.COLOR_HSV2BGR)[0][0]
		colors.append(tuple(int(c) for c in color))
	return colors


def draw_mask_overlay(image: np.ndarray, masks: List[np.ndarray], alpha: float = 0.5) -> np.ndarray:
	"""Overlay colored masks on top of the input frame."""
	overlay = image.copy()
	colors = generate_colors(len(masks))
	for mask, color in zip(masks, colors):
		overlay[mask] = color
	return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)


def save_frame_masks(path: Path, masks: List[np.ndarray]):
	"""Persist masks as compressed NPZ in dense boolean format."""
	if masks:
		stacked = np.stack([m.astype(np.bool_) for m in masks], axis=0)
	else:
		stacked = np.zeros((0, 0, 0), dtype=np.bool_)
	np.savez_compressed(str(path), masks=stacked)


def run_pipeline(args):
	"""Run segmentation pipeline over all frames in input_dir."""
	input_dir = Path(args.input_dir)
	output_dir = Path(args.output_dir)
	mask_dir = output_dir / "masks"
	vis_dir = output_dir / "visible_masks"

	if not input_dir.exists():
		sys.exit(logger.error(f"Input directory missing: {input_dir}"))

	mask_dir.mkdir(parents=True, exist_ok=True)
	vis_dir.mkdir(parents=True, exist_ok=True)

	segmenter = SAM3TextSegmenter(
		checkpoint=args.sam_ckpt,
		device=args.device,
		text_prompts=args.text_prompts,
		min_mask_area=args.min_mask_area,
		ultralytics_home=args.ultralytics_home,
	)

	image_paths = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg"]])
	if not image_paths:
		sys.exit(logger.error(f"No images found in {input_dir}"))

	logger.info(f"Processing {len(image_paths)} frames with SAM3 text prompting + postprocessing...")

	for frame_idx, img_path in enumerate(image_paths):
		frame = cv2.imread(str(img_path))
		if frame is None:
			logger.warning(f"Could not read image: {img_path}")
			continue

		raw_masks = segmenter.predict_raw_masks(frame, text_prompts=args.text_prompts)

		final_masks, debug = postprocess_masks(
			raw_masks=raw_masks,
			frame_bgr=frame,
			min_mask_area=args.min_mask_area,
			max_area_ratio=args.max_area_ratio,
			border_touch_threshold=args.border_touch_threshold,
			merge_thresh=args.merge_thresh,
			proximity_gap=args.proximity_gap,
			proximity_color_thresh=args.proximity_color_thresh,
			split_components=not args.no_split_disconnected,
		)

		save_frame_masks(mask_dir / f"masks_{frame_idx:05d}.npz", final_masks)

		vis = draw_mask_overlay(frame, final_masks)
		cv2.putText(
			vis,
			f"Frame {frame_idx:05d} | Raw={debug['raw_mask_count']} Final={debug['final_mask_count']}",
			(10, 30),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.7,
			(255, 255, 255),
			2,
		)
		cv2.imwrite(str(vis_dir / f"vis_{frame_idx:05d}.png"), vis)

		logger.info(
			f"Frame {frame_idx:05d} ({img_path.name}) | "
			f"Raw={debug['raw_mask_count']} Final={debug['final_mask_count']}"
		)

	logger.info(f"Done. Masks: {mask_dir}, Visualizations: {vis_dir}")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="VRoom SAM3 Text-Prompt Mask Processor")
	parser.add_argument("--input_dir", required=True, help="Path to input images directory")
	parser.add_argument("--output_dir", required=True, help="Path to save masks and visualizations")
	parser.add_argument("--sam_ckpt", default="Module-1/models/sam3.pt", help="Ultralytics SAM3 checkpoint name or .pt path")
	parser.add_argument("--device", default="cuda", help="Device ('cuda' or 'cpu')")
	parser.add_argument("--ultralytics_home", default="", help="Directory for Ultralytics checkpoints/cache")
	parser.add_argument("--text_prompts", nargs="+", default=["object"], help="Open-vocabulary text prompts")
	parser.add_argument("--min_mask_area", type=int, default=120, help="Minimum kept mask area in pixels")
	parser.add_argument("--max_area_ratio", type=float, default=0.50, help="Drop masks larger than this image area ratio")
	parser.add_argument("--border_touch_threshold", type=float, default=0.35, help="Drop masks with high border-touch ratio")
	parser.add_argument("--merge_thresh", type=float, default=0.78, help="Containment threshold for overlap merge")
	parser.add_argument("--proximity_gap", type=int, default=20, help="Pixel gap threshold for proximity merge") 
	parser.add_argument("--proximity_color_thresh", type=float, default=0.32, help="HSV distance threshold for proximity merge") 
	parser.add_argument("--no_split_disconnected", action="store_true", help="Disable splitting disconnected components")

	run_pipeline(parser.parse_args())