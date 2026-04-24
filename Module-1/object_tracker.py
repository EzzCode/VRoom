"""
VRoom Multi-Modal Tracking Pipeline

Loads pre-computed masks from mask_processor.py and performs Kalman + Hungarian
tracking to generate ID-consistent masks for downstream reconstruction.

Usage:
	python object_tracker.py --input_dir data/images --mask_dir data/sam_output/masks --output_dir Tracked

Outputs:
	<output_dir>/id_maps/       — per-frame 16-bit PNG ID maps (named after source image)
	<output_dir>/tracked_vis/   — overlay PNGs with ID-consistent colors and labels
"""

import sys
import argparse
import logging
import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Sequence

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def load_frame_masks(npz_path: Path) -> List[np.ndarray]:
	"""Load boolean masks for one frame from compressed NPZ."""
	if not npz_path.exists():
		return []
	with np.load(str(npz_path)) as data:
		arr = data.get("masks")
	if arr is None or arr.ndim != 3:
		return []
	return [arr[i].astype(bool) for i in range(arr.shape[0])]


def mask_centroid(mask):
	"""Return centroid (x, y) of a boolean mask."""
	ys, xs = np.where(mask)
	if len(xs) == 0:
		return 0.0, 0.0
	return float(xs.mean()), float(ys.mean())


def mask_bbox(mask):
	"""Return XYXY bounding box for a boolean mask."""
	ys, xs = np.where(mask)
	if len(xs) == 0:
		return [0.0, 0.0, 0.0, 0.0]
	return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def box_iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
	"""Compute IoU between two XYXY boxes."""
	ax1, ay1, ax2, ay2 = a
	bx1, by1, bx2, by2 = b
	ix1 = max(ax1, bx1)
	iy1 = max(ay1, by1)
	ix2 = min(ax2, bx2)
	iy2 = min(ay2, by2)
	iw = max(0.0, ix2 - ix1)
	ih = max(0.0, iy2 - iy1)
	inter = iw * ih
	if inter <= 0.0:
		return 0.0
	aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
	ab = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
	union = aa + ab - inter
	return 0.0 if union <= 0.0 else float(inter / union)


def extract_lbp_hist(gray_u8, mask):
	"""Extract normalized 8-neighbor LBP histogram over masked pixels."""
	if gray_u8.shape != mask.shape:
		raise ValueError("Gray image and mask must have identical shape for LBP extraction")
	if gray_u8.shape[0] < 3 or gray_u8.shape[1] < 3:
		return np.zeros((256, 1), dtype=np.float32)

	center = gray_u8[1:-1, 1:-1]
	lbp = np.zeros_like(center, dtype=np.uint8)
	neighbors = [(-1, -1, 1), (-1, 0, 2), (-1, 1, 4), (0, 1, 8), (1, 1, 16), (1, 0, 32), (1, -1, 64), (0, -1, 128)]
	for dy, dx, bit in neighbors:
		neighbor = gray_u8[1 + dy:gray_u8.shape[0] - 1 + dy, 1 + dx:gray_u8.shape[1] - 1 + dx]
		lbp |= ((neighbor >= center).astype(np.uint8) * bit)

	inner_mask = mask[1:-1, 1:-1]
	values = lbp[inner_mask]
	hist = np.bincount(values, minlength=256).astype(np.float32).reshape(-1, 1)
	total = float(hist.sum())
	if total > 0.0:
		hist /= total
	return hist


def create_kalman_filter(cx, cy):
	"""Create a constant-velocity Kalman filter for 2D centroid motion."""
	kf = cv2.KalmanFilter(4, 2)
	kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
	kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
	kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
	kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
	kf.errorCovPost = np.eye(4, dtype=np.float32)
	kf.statePost = np.array([[cx], [cy], [0.0], [0.0]], dtype=np.float32)
	return kf


def identity_affine() -> np.ndarray:
	"""Return 2x3 identity affine matrix."""
	return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)


def estimate_camera_motion(prev_bgr, curr_bgr, curr_masks, max_corners=1200):
	"""Estimate global camera motion from background optical flow."""
	if prev_bgr is None or curr_bgr is None:
		return identity_affine()

	prev_gray = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
	curr_gray = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2GRAY)
	h, w = curr_gray.shape[:2]

	fg = np.zeros((h, w), dtype=np.uint8)
	for m in curr_masks:
		if m.shape == fg.shape:
			fg[m] = 255

	if np.any(fg):
		kernel = np.ones((5, 5), dtype=np.uint8)
		fg = cv2.dilate(fg, kernel, iterations=2)

	bg = cv2.bitwise_not(fg)
	if int(np.count_nonzero(bg)) < 200:
		return identity_affine()

	p0 = cv2.goodFeaturesToTrack(prev_gray, mask=bg, maxCorners=max_corners, qualityLevel=0.01, minDistance=7, blockSize=7)
	if p0 is None or len(p0) < 12:
		return identity_affine()

	p1, st, _ = cv2.calcOpticalFlowPyrLK(
		prev_gray,
		curr_gray,
		p0,
		None,
		winSize=(21, 21),
		maxLevel=3,
		criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
	)
	if p1 is None or st is None:
		return identity_affine()

	good_prev = p0[st.flatten() == 1]
	good_curr = p1[st.flatten() == 1]
	if len(good_prev) < 8:
		return identity_affine()

	affine, _ = cv2.estimateAffinePartial2D(good_prev, good_curr, method=cv2.RANSAC, ransacReprojThreshold=3.0)
	if affine is None:
		return identity_affine()
	return affine.astype(np.float32)


def apply_affine_to_kalman(kf, affine):
	"""Warp Kalman position and velocity using affine transform."""
	if kf is None:
		return

	linear = affine[:, :2].astype(np.float32)
	trans = affine[:, 2].astype(np.float32)

	def _warp_state(state_vec):
		flat = state_vec.reshape(-1).astype(np.float32)
		pos = linear @ flat[:2] + trans
		vel = linear @ flat[2:4]
		flat[0], flat[1], flat[2], flat[3] = float(pos[0]), float(pos[1]), float(vel[0]), float(vel[1])
		return flat.reshape(4, 1)

	kf.statePost = _warp_state(kf.statePost)
	if getattr(kf, "statePre", None) is not None:
		kf.statePre = _warp_state(kf.statePre)


def apply_affine_to_bbox(bbox, affine, frame_shape=None):
	"""Warp XYXY bbox corners through affine and rebuild axis-aligned box."""
	x1, y1, x2, y2 = [float(v) for v in bbox]
	pts = np.array([[x1, y1], [x2, y1], [x1, y2], [x2, y2]], dtype=np.float32)
	warped = (affine[:, :2] @ pts.T).T + affine[:, 2]
	wx1, wy1 = np.min(warped[:, 0]), np.min(warped[:, 1])
	wx2, wy2 = np.max(warped[:, 0]), np.max(warped[:, 1])

	if frame_shape is not None:
		h, w = frame_shape[:2]
		wx1 = max(0.0, min(wx1, w - 1.0))
		wy1 = max(0.0, min(wy1, h - 1.0))
		wx2 = max(0.0, min(wx2, w - 1.0))
		wy2 = max(0.0, min(wy2, h - 1.0))

	return [float(wx1), float(wy1), float(wx2), float(wy2)]


def compute_iou_vote(track_id, det_mask, frame_history, window_size):
	"""Average IoU vote for a candidate detection against track history."""
	recent = frame_history[-window_size:] if window_size > 0 else frame_history
	scores = []
	for entry in recent:
		hist_mask = entry.get("masks", {}).get(track_id)
		if hist_mask is None:
			continue
		scores.append(mask_iou(hist_mask, det_mask))
	if not scores:
		return 0.0
	return float(np.mean(scores))


def appearance_distance(track_data, det_feat):
	"""Compute appearance distance for tie-breaks."""
	dist_color = cv2.compareHist(track_data["hist"], det_feat["hist"], cv2.HISTCMP_BHATTACHARYYA)
	dist_texture = cv2.compareHist(track_data["lbp"], det_feat["lbp"], cv2.HISTCMP_BHATTACHARYYA)
	return 0.60 * dist_color + 0.40 * dist_texture


def compute_appearance_tiebreak(tracks, tid_a, tid_b, det_feat):
	"""Resolve near-equal IoU vote ties using appearance similarity."""
	a_cost = appearance_distance(tracks[tid_a], det_feat)
	b_cost = appearance_distance(tracks[tid_b], det_feat)
	return tid_a if a_cost <= b_cost else tid_b


def update_frame_history(state, tracks, max_window):
	"""Append current active track masks to sliding frame history."""
	entry = {"masks": {tid: trk["mask"] for tid, trk in tracks.items()}}
	history = state.setdefault("frame_history", [])
	history.append(entry)
	if len(history) > max_window:
		del history[:-max_window]


def match_with_consensus(
	tracks,
	next_id,
	cost_matrix,
	active_ids,
	new_feats,
	match_threshold,
	ema=0.7,
	graveyard=None,
	reid_threshold=0.50,
	frame_history=None,
	consensus_window=8,
	tie_margin=0.05,
):
	"""Run Hungarian matching, then refine assignment by temporal IoU consensus."""
	if graveyard is None:
		graveyard = {}
	if frame_history is None:
		frame_history = []

	row_ind, col_ind = linear_sum_assignment(cost_matrix)
	row_by_tid = {tid: idx for idx, tid in enumerate(active_ids)}

	proposals = []
	for r, c in zip(row_ind, col_ind):
		base_cost = float(cost_matrix[r, c])
		if base_cost >= match_threshold:
			continue

		hung_tid = active_ids[r]
		hung_vote = compute_iou_vote(hung_tid, new_feats[c]["mask"], frame_history, consensus_window)

		best_tid = hung_tid
		best_vote = hung_vote
		for tid in active_ids:
			vote = compute_iou_vote(tid, new_feats[c]["mask"], frame_history, consensus_window)
			if vote > best_vote:
				best_tid = tid
				best_vote = vote

		chosen_tid = hung_tid
		chosen_vote = hung_vote
		if best_tid != hung_tid:
			margin = abs(best_vote - hung_vote)
			if margin < tie_margin:
				chosen_tid = compute_appearance_tiebreak(tracks, best_tid, hung_tid, new_feats[c])
				chosen_vote = best_vote if chosen_tid == best_tid else hung_vote
			else:
				chosen_tid = best_tid
				chosen_vote = best_vote

		chosen_row = row_by_tid[chosen_tid]
		chosen_cost = float(cost_matrix[chosen_row, c])
		proposals.append((chosen_tid, chosen_row, c, chosen_vote, chosen_cost))

	used_rows = set()
	used_cols = set()
	accepted_pairs = []
	for tid, r, c, vote_score, pair_cost in sorted(proposals, key=lambda x: (-x[3], x[4])):
		if r in used_rows or c in used_cols:
			continue
		if pair_cost >= match_threshold:
			continue
		accepted_pairs.append((r, c))
		used_rows.add(r)
		used_cols.add(c)

	current_output = {}
	assigned_new = set()

	for r, c in accepted_pairs:
		tid = active_ids[r]
		tracks[tid]["hist"] = ema * new_feats[c]["hist"] + (1 - ema) * tracks[tid]["hist"]
		tracks[tid]["lbp"] = ema * new_feats[c]["lbp"] + (1 - ema) * tracks[tid]["lbp"]
		tracks[tid]["mask"] = new_feats[c]["mask"]
		tracks[tid]["centroid"] = new_feats[c]["centroid"]
		tracks[tid]["bbox"] = new_feats[c]["bbox"]
		tracks[tid]["bbox_wh"] = (max(1.0, new_feats[c]["bbox"][2] - new_feats[c]["bbox"][0]), max(1.0, new_feats[c]["bbox"][3] - new_feats[c]["bbox"][1]))
		if tracks[tid].get("kalman") is None:
			tracks[tid]["kalman"] = create_kalman_filter(*new_feats[c]["centroid"])
		measurement = np.array([[new_feats[c]["centroid"][0]], [new_feats[c]["centroid"][1]]], dtype=np.float32)
		tracks[tid]["kalman"].correct(measurement)
		tracks[tid]["lost"] = 0
		current_output[tid] = new_feats[c]["mask"]
		assigned_new.add(c)

	for r in set(range(len(active_ids))) - {r for r, _ in accepted_pairs}:
		tracks[active_ids[r]]["lost"] += 1

	for c in set(range(len(new_feats))) - assigned_new:
		best_gid, best_cost = None, reid_threshold
		for gid, gdata in graveyard.items():
			dist_color = cv2.compareHist(gdata["hist"], new_feats[c]["hist"], cv2.HISTCMP_BHATTACHARYYA)
			dist_texture = cv2.compareHist(gdata["lbp"], new_feats[c]["lbp"], cv2.HISTCMP_BHATTACHARYYA)
			dist_bbox = 1.0 - box_iou_xyxy(gdata.get("bbox", [0, 0, 0, 0]), new_feats[c]["bbox"])
			cost = 0.45 * dist_color + 0.35 * dist_texture + 0.20 * dist_bbox
			if cost < best_cost:
				best_gid, best_cost = gid, cost

		feat_entry = {
			"mask": new_feats[c]["mask"],
			"hist": new_feats[c]["hist"],
			"lbp": new_feats[c]["lbp"],
			"centroid": new_feats[c]["centroid"],
			"bbox": new_feats[c]["bbox"],
			"bbox_wh": (max(1.0, new_feats[c]["bbox"][2] - new_feats[c]["bbox"][0]), max(1.0, new_feats[c]["bbox"][3] - new_feats[c]["bbox"][1])),
			"kalman": create_kalman_filter(*new_feats[c]["centroid"]),
			"predicted_centroid": new_feats[c]["centroid"],
			"predicted_bbox": new_feats[c]["bbox"],
			"lost": 0,
		}
		if best_gid:
			tracks[best_gid] = feat_entry
			current_output[best_gid] = new_feats[c]["mask"]
			del graveyard[best_gid]
		else:
			tracks[next_id] = feat_entry
			current_output[next_id] = new_feats[c]["mask"]
			next_id += 1

	return current_output, next_id


def extract_features(mask, frame_hsv, frame_gray):
	"""Extract HSV histogram, LBP histogram, and centroid for one mask."""
	mask_uint8 = mask.astype(np.uint8)
	hist = cv2.calcHist([frame_hsv], [0, 1], mask_uint8, [32, 32], [0, 180, 0, 256])
	cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
	lbp_hist = extract_lbp_hist(frame_gray, mask)
	centroid = mask_centroid(mask)
	return hist, lbp_hist, centroid


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
	"""Compute IoU between two binary masks."""
	inter = np.logical_and(a, b).sum()
	union = np.logical_or(a, b).sum()
	return 0.0 if union == 0 else float(inter / union)


def compute_cost_matrix(tracks, active_ids, new_feats, alpha, beta, gamma, delta, img_diag):
	"""Build assignment cost matrix for Hungarian matching.

	The implementation keeps matching logic unchanged while reducing repeated
	computation in the inner loop through precomputed detection arrays.
	"""
	cost = np.zeros((len(active_ids), len(new_feats)))
	if not active_ids or not new_feats:
		return cost

	det_centroids = np.array([f["centroid"] for f in new_feats], dtype=np.float32)
	det_bboxes = np.array([f["bbox"] for f in new_feats], dtype=np.float32)
	for i, tid in enumerate(active_ids):
		trk = tracks[tid]
		cx_k, cy_k = trk["predicted_centroid"]
		pred_box = trk["predicted_bbox"]
		dist_centroids = np.sqrt((det_centroids[:, 0] - cx_k) ** 2 + (det_centroids[:, 1] - cy_k) ** 2) / img_diag
		dist_bboxes = 1.0 - np.array([box_iou_xyxy(pred_box, det_box) for det_box in det_bboxes], dtype=np.float32)

		for j, det in enumerate(new_feats):
			dist_iou = 1.0 - mask_iou(trk["mask"], det["mask"])
			dist_color = cv2.compareHist(trk["hist"], det["hist"], cv2.HISTCMP_BHATTACHARYYA)
			dist_texture = cv2.compareHist(trk["lbp"], det["lbp"], cv2.HISTCMP_BHATTACHARYYA)
			dist_centroid = float(dist_centroids[j])
			dist_bbox = float(dist_bboxes[j])

			cost[i, j] = alpha * dist_iou + beta * dist_color + gamma * dist_texture + delta * (0.6 * dist_centroid + 0.4 * dist_bbox)
	return cost


def init_tracks(tracks, next_id, masks, frame_hsv, frame_gray):
	"""Initialize tracker state from first-frame detections."""
	for mask in masks:
		hist, lbp, centroid = extract_features(mask, frame_hsv, frame_gray)
		bbox = mask_bbox(mask)
		tracks[next_id] = {
			"mask": mask,
			"hist": hist,
			"lbp": lbp,
			"centroid": centroid,
			"bbox": bbox,
			"bbox_wh": (max(1.0, bbox[2] - bbox[0]), max(1.0, bbox[3] - bbox[1])),
			"kalman": create_kalman_filter(*centroid),
			"predicted_centroid": centroid,
			"predicted_bbox": bbox,
			"lost": 0,
		}
		next_id += 1
	return next_id


def predict_tracks(tracks, frame_shape):
	"""Run Kalman predict step and refresh predicted XYXY priors."""
	h, w = frame_shape[:2]
	for trk in tracks.values():
		if trk.get("kalman") is not None:
			pred = trk["kalman"].predict()
			cx = float(pred[0, 0])
			cy = float(pred[1, 0])
		else:
			cx, cy = trk["centroid"]

		bw, bh = trk.get("bbox_wh", (20.0, 20.0))
		x1 = max(0.0, min(cx - bw * 0.5, w - 1.0))
		y1 = max(0.0, min(cy - bh * 0.5, h - 1.0))
		x2 = max(0.0, min(cx + bw * 0.5, w - 1.0))
		y2 = max(0.0, min(cy + bh * 0.5, h - 1.0))
		trk["predicted_centroid"] = (cx, cy)
		trk["predicted_bbox"] = [x1, y1, x2, y2]


def match_and_update(tracks, next_id, cost_matrix, active_ids, new_feats, match_threshold, ema=0.7, graveyard=None, reid_threshold=0.50):
	"""Run Hungarian assignment, update matched tracks, and spawn/re-ID unmatched detections."""
	if graveyard is None:
		graveyard = {}

	row_ind, col_ind = linear_sum_assignment(cost_matrix)
	current_output = {}
	assigned_new = set()

	for r, c in zip(row_ind, col_ind):
		if cost_matrix[r, c] < match_threshold:
			tid = active_ids[r]
			tracks[tid]["hist"] = ema * new_feats[c]["hist"] + (1 - ema) * tracks[tid]["hist"]
			tracks[tid]["lbp"] = ema * new_feats[c]["lbp"] + (1 - ema) * tracks[tid]["lbp"]
			tracks[tid]["mask"] = new_feats[c]["mask"]
			tracks[tid]["centroid"] = new_feats[c]["centroid"]
			tracks[tid]["bbox"] = new_feats[c]["bbox"]
			tracks[tid]["bbox_wh"] = (max(1.0, new_feats[c]["bbox"][2] - new_feats[c]["bbox"][0]), max(1.0, new_feats[c]["bbox"][3] - new_feats[c]["bbox"][1]))
			if tracks[tid].get("kalman") is None:
				tracks[tid]["kalman"] = create_kalman_filter(*new_feats[c]["centroid"])
			measurement = np.array([[new_feats[c]["centroid"][0]], [new_feats[c]["centroid"][1]]], dtype=np.float32)
			tracks[tid]["kalman"].correct(measurement)
			tracks[tid]["lost"] = 0
			current_output[tid] = new_feats[c]["mask"]
			assigned_new.add(c)
		else:
			tracks[active_ids[r]]["lost"] += 1

	for r in set(range(len(active_ids))) - set(row_ind):
		tracks[active_ids[r]]["lost"] += 1

	for c in set(range(len(new_feats))) - assigned_new:
		best_gid, best_cost = None, reid_threshold
		for gid, gdata in graveyard.items():
			dist_color = cv2.compareHist(gdata["hist"], new_feats[c]["hist"], cv2.HISTCMP_BHATTACHARYYA)
			dist_texture = cv2.compareHist(gdata["lbp"], new_feats[c]["lbp"], cv2.HISTCMP_BHATTACHARYYA)
			dist_bbox = 1.0 - box_iou_xyxy(gdata.get("bbox", [0, 0, 0, 0]), new_feats[c]["bbox"])
			cost = 0.45 * dist_color + 0.35 * dist_texture + 0.20 * dist_bbox
			if cost < best_cost:
				best_gid, best_cost = gid, cost

		feat_entry = {
			"mask": new_feats[c]["mask"],
			"hist": new_feats[c]["hist"],
			"lbp": new_feats[c]["lbp"],
			"centroid": new_feats[c]["centroid"],
			"bbox": new_feats[c]["bbox"],
			"bbox_wh": (max(1.0, new_feats[c]["bbox"][2] - new_feats[c]["bbox"][0]), max(1.0, new_feats[c]["bbox"][3] - new_feats[c]["bbox"][1])),
			"kalman": create_kalman_filter(*new_feats[c]["centroid"]),
			"predicted_centroid": new_feats[c]["centroid"],
			"predicted_bbox": new_feats[c]["bbox"],
			"lost": 0,
		}
		if best_gid:
			tracks[best_gid] = feat_entry
			current_output[best_gid] = new_feats[c]["mask"]
			del graveyard[best_gid]
		else:
			tracks[next_id] = feat_entry
			current_output[next_id] = new_feats[c]["mask"]
			next_id += 1

	return current_output, next_id


def prune_lost_tracks(tracks, graveyard, patience):
	"""Move expired tracks to graveyard and keep active tracks only."""
	alive, dead = {}, {}
	for tid, data in tracks.items():
		if data["lost"] <= patience:
			alive[tid] = data
		else:
			dead[tid] = data
	graveyard.update(dead)
	return alive


def track_frame(
	state,
	frame_bgr,
	new_masks,
	alpha,
	beta,
	gamma,
	delta,
	match_threshold,
	patience,
	ema,
	reid_threshold,
	enable_motion_comp=True,
	enable_consensus=True,
	consensus_window=8,
	consensus_tie_margin=0.05,
):
	"""Process one frame through predict, compensate, associate, update, and prune."""
	h, w = frame_bgr.shape[:2]
	img_diag = np.sqrt(h ** 2 + w ** 2)
	tracks = state["tracks"]
	next_id = state["next_id"]
	curr_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
	curr_hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

	if tracks:
		predict_tracks(tracks, frame_bgr.shape)
		if enable_motion_comp and state.get("prev_bgr") is not None:
			affine = estimate_camera_motion(state["prev_bgr"], frame_bgr, new_masks)
			for trk in tracks.values():
				apply_affine_to_kalman(trk.get("kalman"), affine)
				pcx, pcy = trk["predicted_centroid"]
				warped_pt = (affine[:, :2] @ np.array([pcx, pcy], dtype=np.float32)) + affine[:, 2]
				trk["predicted_centroid"] = (float(warped_pt[0]), float(warped_pt[1]))
				trk["predicted_bbox"] = apply_affine_to_bbox(trk["predicted_bbox"], affine, frame_shape=frame_bgr.shape)

	if not tracks and new_masks:
		next_id = init_tracks(tracks, next_id, new_masks, curr_hsv, curr_gray)
		state.update({"next_id": next_id, "prev_gray": curr_gray, "prev_bgr": frame_bgr.copy()})
		update_frame_history(state, tracks, consensus_window)
		return {tid: data["mask"] for tid, data in tracks.items()}

	if not new_masks:
		if not tracks:
			state["prev_gray"] = curr_gray
			state["prev_bgr"] = frame_bgr.copy()
			return {}
		for tid in list(tracks.keys()):
			tracks[tid]["lost"] += 1
		state["tracks"] = prune_lost_tracks(tracks, state["graveyard"], patience)
		state["next_id"] = next_id
		state["prev_gray"] = curr_gray
		state["prev_bgr"] = frame_bgr.copy()
		update_frame_history(state, state["tracks"], consensus_window)
		return {}

	new_feats = []
	for m in new_masks:
		hist, lbp, centroid = extract_features(m, curr_hsv, curr_gray)
		new_feats.append({"mask": m, "hist": hist, "lbp": lbp, "centroid": centroid, "bbox": mask_bbox(m)})

	active_ids = list(tracks.keys())
	cost = compute_cost_matrix(tracks, active_ids, new_feats, alpha, beta, gamma, delta, img_diag)
	if enable_consensus:
		output, next_id = match_with_consensus(
			tracks,
			next_id,
			cost,
			active_ids,
			new_feats,
			match_threshold,
			ema=ema,
			graveyard=state["graveyard"],
			reid_threshold=reid_threshold,
			frame_history=state.get("frame_history", []),
			consensus_window=consensus_window,
			tie_margin=consensus_tie_margin,
		)
	else:
		output, next_id = match_and_update(tracks, next_id, cost, active_ids, new_feats, match_threshold, ema=ema, graveyard=state["graveyard"], reid_threshold=reid_threshold)

	state["tracks"] = prune_lost_tracks(tracks, state["graveyard"], patience)
	state["next_id"] = next_id
	state["prev_gray"] = curr_gray
	state["prev_bgr"] = frame_bgr.copy()
	update_frame_history(state, state["tracks"], consensus_window)
	return output


@lru_cache(maxsize=8192)
def get_id_color(obj_id):
	"""Return deterministic, cached color for a track ID."""
	rng = np.random.RandomState(int(obj_id) * 31)
	return tuple(int(c) for c in rng.randint(60, 255, size=3))


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


def run_pipeline(args):
	"""Run tracker over all images and aligned NPZ masks."""
	input_dir = Path(args.input_dir)
	mask_dir = Path(args.mask_dir)
	output_dir = Path(args.output_dir)

	if not input_dir.exists():
		sys.exit(logger.error(f"Input directory missing: {input_dir}"))
	if not mask_dir.exists():
		sys.exit(logger.error(f"Mask directory missing: {mask_dir}"))

	id_map_dir = output_dir / "id_maps"
	vis_dir = output_dir / "tracked_vis"
	id_map_dir.mkdir(parents=True, exist_ok=True)
	vis_dir.mkdir(parents=True, exist_ok=True)

	meta_path = output_dir / "id_map_meta.json"
	with open(meta_path, "w", encoding="utf-8") as f:
		json.dump({"format": "png", "bit_depth": 16, "dtype": "uint16", "background_id": 0, "id_range": [0, int(np.iinfo(np.uint16).max)]}, f, indent=2)

	state = {"tracks": {}, "next_id": 1, "prev_gray": None, "prev_bgr": None, "graveyard": {}, "frame_history": []}
	image_paths = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg"]])
	if not image_paths:
		sys.exit(logger.error(f"No images found in {input_dir}"))

	logger.info(f"Tracking {len(image_paths)} frames in vanilla NPZ-mask mode...")

	for frame_idx, img_path in enumerate(image_paths):
		frame = cv2.imread(str(img_path))
		if frame is None:
			logger.warning(f"Could not read image: {img_path}")
			continue

		seg_masks = load_frame_masks(mask_dir / f"masks_{frame_idx:05d}.npz")
		tracked = track_frame(
			state,
			frame,
			seg_masks,
			args.alpha,
			args.beta,
			args.gamma,
			args.delta,
			args.match_threshold,
			args.patience,
			args.ema,
			args.reid_threshold,
			enable_motion_comp=not args.disable_motion_comp,
			enable_consensus=not args.disable_consensus,
			consensus_window=args.consensus_window,
			consensus_tie_margin=args.consensus_tie_margin,
		)

		frame_h, frame_w = frame.shape[:2]
		id_map = np.zeros((frame_h, frame_w), dtype=np.uint16)
		empty = np.ones((frame_h, frame_w), dtype=bool)
		for obj_id, mask in tracked.items():
			if obj_id > np.iinfo(np.uint16).max:
				raise ValueError(f"Track ID {obj_id} exceeds uint16 range")
			fill = empty & mask
			id_map[fill] = np.uint16(obj_id)
			empty[fill] = False

		cv2.imwrite(str(id_map_dir / (img_path.stem + ".png")), id_map)
		vis = draw_tracked_overlay(frame, tracked)
		cv2.putText(vis, f"Frame {frame_idx:05d} | Seg={len(seg_masks)} Track={len(tracked)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
		cv2.imwrite(str(vis_dir / f"tracked_{frame_idx:05d}.png"), vis)
		logger.info(f"Frame {frame_idx:05d} | Seg={len(seg_masks)} Track={len(tracked)}")

	logger.info(f"Done. ID maps: {id_map_dir}, Visualizations: {vis_dir}")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="VRoom Vanilla Object Tracker")
	parser.add_argument("--input_dir", required=True, help="Path to input images directory")
	parser.add_argument("--mask_dir", required=True, help="Path to precomputed NPZ masks directory")
	parser.add_argument("--output_dir", required=True, help="Path to save tracking output")
	parser.add_argument("--alpha", type=float, default=0.68, help="Weight for mask IoU distance")
	parser.add_argument("--beta", type=float, default=0.25, help="Weight for color match")
	parser.add_argument("--gamma", type=float, default=0.15, help="Weight for texture match")
	parser.add_argument("--delta", type=float, default=0.12, help="Weight for centroid/bbox prior")
	parser.add_argument("--match_threshold", type=float, default=0.7, help="Cost cutoff for Hungarian match")
	parser.add_argument("--patience", type=int, default=28, help="Frames to remember occluded IDs")
	parser.add_argument("--ema", type=float, default=0.7, help="EMA weight for feature smoothing")
	parser.add_argument("--reid_threshold", type=float, default=0.5, help="Max appearance distance for graveyard re-ID")
	parser.add_argument("--disable_motion_comp", action="store_true", help="Disable global camera-motion compensation")
	parser.add_argument("--disable_consensus", action="store_true", help="Disable in-clip consensus refinement")
	parser.add_argument("--consensus_window", type=int, default=8, help="Temporal window length for consensus voting")
	parser.add_argument("--consensus_tie_margin", type=float, default=0.05, help="IoU vote margin to trigger appearance tie-break")
	run_pipeline(parser.parse_args())
