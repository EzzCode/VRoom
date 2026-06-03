import logging
import argparse
from pathlib import Path

import cv2
import numpy as np

from masks_and_tracking.helpers import draw_mask_overlay, save_frame_masks
from masks_and_tracking.sam_inference import SAM3TextSegmenter

logger = logging.getLogger(__name__)



def _bounding_box(mask):
	"""[x1, y1, x2, y2] bounding box for a boolean mask."""
	ys, xs = np.where(mask)
	if xs.size == 0:
		return None
	return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def _bbox_gap_px(a, b):
	"""Edge Euclidean gap between two boxes."""
	ax1, ay1, ax2, ay2 = a
	bx1, by1, bx2, by2 = b
	dx = max(0, max(ax1 - bx2, bx1 - ax2))
	dy = max(0, max(ay1 - by2, by1 - ay2))
	return float(np.hypot(dx, dy))


def _mean_hsv(frame_hsv, mask):
	"""Mean HSV color of masked pixels."""
	px = frame_hsv[mask]
	if px.size == 0:
		return np.array([0.0, 0.0, 0.0], dtype=np.float32)
	return px.mean(axis=0).astype(np.float32)


def filter_background(
	masks,
	max_area_ratio=0.50,
	border_threshold=0.35,
):
	"""background masks: too large or touching too much border."""
	if not masks:
		return []

	stack = np.stack(masks, axis=0).astype(bool, copy=False)
	n, height, width = stack.shape

	img_area = float(height * width)
	areas = stack.reshape(n, -1).sum(axis=1).astype(np.float32)
	border_len = float((width * 2) + (height * 2) - 4)
	top_hits    = stack[:,0,:].sum(axis=1)
	bottom_hits = stack[:,height-1,:].sum(axis=1)
	if height > 2:
		left_hits  = stack[:,1:height-1,0].sum(axis=1)
		right_hits = stack[:,1:height-1,width-1].sum(axis=1)
	else:
		left_hits = right_hits = np.zeros(n, dtype=np.int64)
	border_ratio = (top_hits + bottom_hits + left_hits + right_hits) / max(1.0, border_len)

	valid_border = border_ratio <= border_threshold
	valid_area = (areas / max(1.0, img_area)) <= max_area_ratio
	
	return [stack[i] for i in np.flatnonzero(valid_area & valid_border)]


def merge_overlapping(masks, thresh=0.78):
	"""Merge masks when one is strongly contained in another (containment >= thresh)."""
	if not masks:
		return []

	merged = [m.copy() for m in masks]
	changed = True
	while changed:
		changed = False
		out = []
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
				if float(inter) / min_area >= thresh:
					base = np.logical_or(base, other)
					used[j] = True
					changed = True
					area_base = float(base.sum())

			out.append(base)
		merged = out

	return merged


def proximity_merge(
	masks,
	frame_bgr,
	gap_px=20,
	color_thresh=0.32,
):
	"""Merge nearby masks if mean HSV color distance is below threshold."""
	if len(masks) <= 1:
		return masks

	frame_hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
	boxes = [_bounding_box(m) for m in masks]
	means = [_mean_hsv(frame_hsv, m) for m in masks]
	hsv_norm = float(np.linalg.norm(np.array([180.0, 255.0, 255.0], dtype=np.float32)))

	merged = [m.copy() for m in masks]
	changed = True
	while changed:
		changed = False
		out = []
		used = [False] * len(merged)

		for i in range(len(merged)):
			if used[i]:
				continue
			base = merged[i].copy()
			used[i] = True

			for j in range(i + 1, len(merged)):
				if used[j]:
					continue
				box_i, box_j = boxes[i], boxes[j]
				if box_i is None or box_j is None:
					continue
				if _bbox_gap_px(box_i, box_j) > float(gap_px):
					continue
				if np.linalg.norm(means[i] - means[j]) / hsv_norm > color_thresh:
					continue
				base = np.logical_or(base, merged[j])
				used[j] = True
				changed = True

			out.append(base)

		merged = out
		boxes = [_bounding_box(m) for m in merged]
		means = [_mean_hsv(frame_hsv, m) for m in merged]

	return merged


def split_disconnected(masks, min_component_area=1):
	"""Split each mask into connected components, keeping those >= min_component_area."""
	if not masks:
		return []

	out = []
	for mask in masks:
		num_labels, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
		if num_labels <= 2:
			out.append(mask)
			continue
		for label in range(1, num_labels):
			comp = labels == label
			if int(comp.sum()) >= min_component_area:
				out.append(comp)

	return out


def postprocess_masks(
	raw_masks,
	frame_bgr,
	min_mask_area=120,
	max_area_ratio=0.50,
	border_threshold=0.35,
	merge_thresh=0.78,
	proximity_gap=20,
	proximity_color_thresh=0.32,
	split_components=True,
):
	"""Full spatial post-processing pipeline: filter → merge → split → gate.
	"""
	masks = filter_background(raw_masks, max_area_ratio=max_area_ratio, border_threshold=border_threshold)
	masks = merge_overlapping(masks, thresh=merge_thresh)
	masks = proximity_merge(masks, frame_bgr=frame_bgr, gap_px=proximity_gap, color_thresh=proximity_color_thresh)
	if split_components:
		masks = split_disconnected(masks, min_component_area=min_mask_area)

	masks = [m.astype(bool) for m in masks if int(m.sum()) >= min_mask_area]
	return masks, {"raw_mask_count": len(raw_masks), "final_mask_count": len(masks)}





# ── CLI pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(args):
	"""Run segmentation + postprocessing over all frames in input_dir."""
	input_dir = Path(args.input_dir)
	output_dir = Path(args.output_dir)
	mask_dir = output_dir / "masks"
	vis_dir  = output_dir / "visible_masks"

	if not input_dir.exists():
		raise FileNotFoundError(f"Input directory missing: {input_dir}")

	mask_dir.mkdir(parents=True, exist_ok=True)
	vis_dir.mkdir(parents=True, exist_ok=True)

	segmenter = SAM3TextSegmenter(
		checkpoint=args.sam_ckpt,
		device=args.device,
		text_prompts=args.text_prompts,
		min_mask_area=args.min_mask_area,
		ultralytics_home=args.ultralytics_home,
	)

	image_paths = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
	if not image_paths:
		raise FileNotFoundError(f"No images found in {input_dir}")

	logger.info("Processing %d frames with SAM3 text prompting + postprocessing...", len(image_paths))

	for frame_idx, img_path in enumerate(image_paths):
		frame = cv2.imread(str(img_path))
		if frame is None:
			logger.warning("Could not read image: %s", img_path)
			continue

		raw_masks = segmenter.predict_raw_masks(frame, text_prompts=args.text_prompts)
		final_masks, debug = postprocess_masks(
			raw_masks=raw_masks,
			frame_bgr=frame,
			min_mask_area=args.min_mask_area,
			max_area_ratio=args.max_area_ratio,
			border_threshold=args.border_threshold,
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
			(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
		)
		cv2.imwrite(str(vis_dir / f"vis_{frame_idx:05d}.png"), vis)

		logger.info(
			"Frame %05d | Raw=%d Final=%d",
			frame_idx, debug["raw_mask_count"], debug["final_mask_count"],
		)

	logger.info("Done. Masks: %s, Visualizations: %s", mask_dir, vis_dir)


if __name__ == "__main__":
	logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S")
	parser = argparse.ArgumentParser(description="VRoom SAM3 Text-Prompt Mask Processor")
	parser.add_argument("--input_dir", required=True, help="Path to input images directory")
	parser.add_argument("--output_dir", required=True, help="Path to save masks and visualizations")
	parser.add_argument("--sam_ckpt", default="masks_and_tracking/models/sam3.pt", help="Ultralytics SAM3 checkpoint name or .pt path")
	parser.add_argument("--device", default="cuda", help="Device ('cuda' or 'cpu')")
	parser.add_argument("--ultralytics_home", default="", help="Directory for Ultralytics checkpoints/cache")
	parser.add_argument("--text_prompts", nargs="+", default=["desk", "table", "chair", "couch", "sofa", "cabinet"], help="Open-vocabulary text prompts")
	parser.add_argument("--min_mask_area", type=int, default=120, help="Minimum kept mask area in pixels")
	parser.add_argument("--max_area_ratio", type=float, default=0.50, help="Drop masks larger than this image area ratio")
	parser.add_argument("--border_threshold", type=float, default=0.35, help="Drop masks with high border-touch ratio")
	parser.add_argument("--merge_thresh", type=float, default=0.78, help="Containment threshold for overlap merge")
	parser.add_argument("--proximity_gap", type=int, default=20, help="Pixel gap threshold for proximity merge")
	parser.add_argument("--proximity_color_thresh", type=float, default=0.32, help="HSV distance threshold for proximity merge")
	parser.add_argument("--no_split_disconnected", action="store_true", help="Disable splitting disconnected components")
	run_pipeline(parser.parse_args())