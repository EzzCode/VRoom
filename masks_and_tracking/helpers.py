import colorsys
from functools import lru_cache

import cv2
import numpy as np


# ── Visualization helpers ────────────────────────────────────────────────────

def generate_colors(n):
	"""Generate n visually distinct BGR colors via HSV spacing."""
	colors = []
	for i in range(n):
		hue = int(180 * i / max(n, 1))
		color = cv2.cvtColor(np.array([[[hue, 200, 255]]], dtype=np.uint8), cv2.COLOR_HSV2BGR)[0][0]
		colors.append((int(color[0]), int(color[1]), int(color[2])))
	return colors


def draw_mask_overlay(image, masks, alpha=0.5):
	"""Overlay colored masks on top of the input frame."""
	overlay = image.copy()
	for mask, color in zip(masks, generate_colors(len(masks))):
		overlay[mask] = color
	return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)


def save_frame_masks(path, masks):
	"""Persist masks as compressed NPZ in dense boolean format."""
	stacked = np.stack([m.astype(np.bool_) for m in masks], axis=0) if masks else np.zeros((0, 0, 0), dtype=np.bool_)
	np.savez_compressed(str(path), masks=stacked)
	

@lru_cache(maxsize=8192)
def get_id_color(obj_id):
	"""Deterministic, cached color for a track ID."""
	rng = np.random.RandomState(int(obj_id) * 31)
	return tuple(int(c) for c in rng.randint(60, 255, size=3))


def label_to_color(lbl):
	"""Deterministic HSV color for a point-cloud label ID."""
	if lbl == 0:
		return (150, 150, 150)
	hue = ((lbl * 137.508) % 360) / 360.0
	r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 0.9)
	return (int(r * 255), int(g * 255), int(b * 255))


def draw_tracked_overlay(image, tracked_objects, alpha=0.5):
	"""Render ID-colored masks and centroid labels for visualization."""
	overlay = image.copy()
	for obj_id, mask in tracked_objects.items():
		overlay[mask] = get_id_color(obj_id)
	blended = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
	for obj_id, mask in tracked_objects.items():
		ys, xs = np.where(mask)
		if len(xs) > 0:
			cx, cy = int(xs.mean()), int(ys.mean())
			color = get_id_color(obj_id)
			cv2.putText(blended, str(obj_id), (cx - 10, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
			cv2.putText(blended, str(obj_id), (cx - 10, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
	return blended
