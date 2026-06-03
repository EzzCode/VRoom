import argparse
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from masks_and_tracking.helpers import draw_tracked_overlay
from masks_and_tracking.tracker_defaults import TRACKING_DEFAULTS

try:
	from masks_and_tracking import opencv_vroom
except ImportError:
	import importlib
	opencv_vroom = importlib.import_module("opencv_vroom")

cv = opencv_vroom

logger = logging.getLogger(__name__)


def load_masks(npz_path):
	"""Load boolean masks from a compressed NPZ file."""
	path = npz_path
	if not path.exists():
		return []

	with np.load(str(path)) as data:
		arr = data.get("masks")

	if arr is None or arr.ndim != 3:
		return []
	return [arr[i].astype(bool) for i in range(arr.shape[0])]


def mask_bbox(mask):
	ys, xs = np.where(mask)
	if len(xs) == 0:
		return [0.0, 0.0, 0.0, 0.0]
	return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def box_iou(a, b):
	ax1, ay1, ax2, ay2 = a
	bx1, by1, bx2, by2 = b
	ix1, iy1 = max(ax1, bx1), max(ay1, by1)
	ix2, iy2 = min(ax2, bx2), min(ay2, by2)
	inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
	if inter <= 0.0:
		return 0.0
	union = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1) + max(0.0, bx2 - bx1) * max(0.0, by2 - by1) - inter
	return 0.0 if union <= 0.0 else inter / union


def extract_lbp(gray, mask):
	"""normalized 8 neighbor LBP histogram"""
	if gray.shape != mask.shape:
		raise ValueError("image and mask must have identical shape for LBP extraction")

	ys, xs = np.where(mask)
	if len(xs) == 0:
		return np.zeros((256, 1), dtype=np.float32)

	y_min = max(0, int(ys.min()) - 1)
	y_max = min(gray.shape[0], int(ys.max()) + 2)
	x_min = max(0, int(xs.min()) - 1)
	x_max = min(gray.shape[1], int(xs.max()) + 2)

	crop_gray = gray[y_min:y_max, x_min:x_max]
	crop_mask = mask[y_min:y_max, x_min:x_max]

	if crop_gray.shape[0] < 3 or crop_gray.shape[1] < 3:
		return np.zeros((256, 1), dtype=np.float32)

	center = crop_gray[1:-1, 1:-1]
	lbp = np.zeros_like(center, dtype=np.uint8)

	for dy, dx, bit in [(-1,-1,1),(-1,0,2),(-1,1,4),(0,1,8),(1,1,16),(1,0,32),(1,-1,64),(0,-1,128)]:
		neighbor = crop_gray[1+dy:crop_gray.shape[0]-1+dy, 1+dx:crop_gray.shape[1]-1+dx]
		lbp |= (neighbor >= center).astype(np.uint8) * bit

	values = lbp[crop_mask[1:-1, 1:-1]]
	hist = np.bincount(values, minlength=256).astype(np.float32).reshape(-1, 1)
	total = float(hist.sum())
	if total > 0.0:
		hist /= total
	return hist


def create_kalman_filter(cx, cy, w, h):
	"""Constant-velocity Kalman filter for 2D centroid and bbox size. State: [cx, cy, w, h, vx, vy, vw, vh]."""
	kf = cv.KalmanFilter(8, 4)
	kf.transitionMatrix = np.array([
		[1, 0, 0, 0, 1, 0, 0, 0],
		[0, 1, 0, 0, 0, 1, 0, 0],
		[0, 0, 1, 0, 0, 0, 1, 0],
		[0, 0, 0, 1, 0, 0, 0, 1],
		[0, 0, 0, 0, 1, 0, 0, 0],
		[0, 0, 0, 0, 0, 1, 0, 0],
		[0, 0, 0, 0, 0, 0, 1, 0],
		[0, 0, 0, 0, 0, 0, 0, 1],
	], dtype=np.float32)
	kf.measurementMatrix  = np.array([
		[1,0,0,0,0,0,0,0],
		[0,1,0,0,0,0,0,0],
		[0,0,1,0,0,0,0,0],
		[0,0,0,1,0,0,0,0]
	], dtype=np.float32)
	kf.processNoiseCov = np.eye(8, dtype=np.float32) * 1e-2
	kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1e-1
	kf.errorCovPost = np.eye(8, dtype=np.float32)
	kf.statePost = np.array([[cx],[cy],[w],[h],[0.0],[0.0],[0.0],[0.0]], dtype=np.float32)
	
	return kf

def estimate_camera_motion(prev_bgr, curr_bgr, curr_masks, max_corners=1200):
	_MIN_FEATURES = 12
	_MIN_TRACKED  = 8
	if prev_bgr is None or curr_bgr is None:
		return np.array([[1.0, 0.0, 0.0],[0.0, 1.0, 0.0]], dtype=np.float32)

	prev_gray = cv.cvtColor(prev_bgr, cv.COLOR_BGR2GRAY)
	curr_gray = cv.cvtColor(curr_bgr, cv.COLOR_BGR2GRAY)
	h, w = curr_gray.shape[:2]

	fg = np.zeros((h, w), dtype=np.uint8)
	for m in curr_masks:
		if m.shape == fg.shape:
			fg[m] = 255
	if np.any(fg):
		fg = cv.dilate(fg, np.ones((5, 5), dtype=np.uint8), iterations=2)

	bg = cv.bitwise_not(fg)
	if np.count_nonzero(bg) < 200:
		return np.array([[1.0, 0.0, 0.0],[0.0, 1.0, 0.0]], dtype=np.float32)

	p0 = cv.goodFeaturesToTrack(prev_gray, mask=bg, maxCorners=max_corners, qualityLevel=0.01, minDistance=7, blockSize=7)
	if p0 is None or len(p0) < _MIN_FEATURES:
		return np.array([[1.0, 0.0, 0.0],[0.0, 1.0, 0.0]], dtype=np.float32)

	p1, st, _ = cv.calcOpticalFlowPyrLK(
		prev_gray, curr_gray, p0, None,
		winSize=(21, 21), maxLevel=3,
		criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.01),
	)
	if p1 is None or st is None:
		return np.array([[1.0, 0.0, 0.0],[0.0, 1.0, 0.0]], dtype=np.float32)

	good_prev = p0[st.flatten() == 1]
	good_curr = p1[st.flatten() == 1]
	if len(good_prev) < _MIN_TRACKED:
		return np.array([[1.0, 0.0, 0.0],[0.0, 1.0, 0.0]], dtype=np.float32)

	affine, _ = cv.estimateAffinePartial2D(good_prev, good_curr, method=cv.RANSAC, ransacReprojThreshold=3.0)
	
	if affine is None:
		return np.array([[1.0, 0.0, 0.0],[0.0, 1.0, 0.0]], dtype=np.float32)
	return affine.astype(np.float32)


def apply_affine_to_kalman(kf, affine):
	if kf is None:
		return
	linear = affine[:, :2].astype(np.float32)
	trans = affine[:, 2].astype(np.float32)

	def _warp_state(state_vec):
		flat = state_vec.reshape(-1).astype(np.float32)
		pos = linear @ flat[:2] + trans
		vel = linear @ flat[4:6]
		flat[0], flat[1], flat[4], flat[5] = float(pos[0]), float(pos[1]), float(vel[0]), float(vel[1])
		return flat.reshape(8, 1)

	kf.statePost = _warp_state(kf.statePost)
	if getattr(kf, "statePre", None) is not None:
		kf.statePre = _warp_state(kf.statePre)


def apply_affine_to_bbox(bbox, affine, frame_shape=None):
	x1, y1, x2, y2 = [float(v) for v in bbox]
	pts    = np.array([[x1,y1],[x2,y1],[x1,y2],[x2,y2]], dtype=np.float32)
	warped = (affine[:, :2] @ pts.T).T + affine[:, 2]
	wx1, wy1 = float(np.min(warped[:, 0])), float(np.min(warped[:, 1]))
	wx2, wy2 = float(np.max(warped[:, 0])), float(np.max(warped[:, 1]))
	if frame_shape is not None:
		h, w = frame_shape[:2]
		wx1 = max(0.0, min(wx1, w - 1.0))
		wy1 = max(0.0, min(wy1, h - 1.0))
		wx2 = max(0.0, min(wx2, w - 1.0))
		wy2 = max(0.0, min(wy2, h - 1.0))
	return [wx1, wy1, wx2, wy2]


# ── Temporal consensus matching ──────────────────────────────────────────────

def compute_iou_vote(track_id, det_mask, frame_history, window_size):
	if window_size > 0:
		history = frame_history[-window_size:]
	else:
		history = frame_history

	scores = []
	for entry in history:
		masks = entry.get("masks", {})
		if track_id in masks:
			scores.append(mask_iou(masks[track_id], det_mask))
	
	return float(np.mean(scores)) if scores else 0.0


def appearance_distance(track_data, det_feat):
	dist_color   = cv.compareHist(track_data["hist"], det_feat["hist"],  cv.HISTCMP_BHATTACHARYYA)
	dist_texture = cv.compareHist(track_data["lbp"],  det_feat["lbp"],   cv.HISTCMP_BHATTACHARYYA)
	return 0.60 * dist_color + 0.40 * dist_texture


def compute_tiebreak(tracks, tid_a, tid_b, det_feat):
	def score(tid):
		dist_app = appearance_distance(tracks[tid], det_feat)
		cx_k, cy_k = tracks[tid].get("predicted_centroid", tracks[tid]["centroid"])
		cx_d, cy_d = det_feat["centroid"]
		dist_geom = np.sqrt((cx_k - cx_d)**2 + (cy_k - cy_d)**2) / 1000.0
		return dist_app + 0.5 * dist_geom
	
	if score(tid_a) <= score(tid_b):
		return tid_a
	return tid_b


def update_frame_history(state, tracks, max_window):
	"""Append current active track masks to sliding frame history."""
	history = state.setdefault("frame_history", [])
	history.append({"masks": {tid: trk["mask"] for tid, trk in tracks.items()}})
	if len(history) > max_window:
		del history[:-max_window]


def match_with_consensus(
	tracks,
	next_id,
	cost_matrix,
	active_ids,
	new_feats,
	match_threshold,
	smoothing_factor=0.7,
	graveyard=None,
	reid_threshold=0.50,
	frame_history=None,
	consensus_window=8,
	tie_margin=0.05,
):
	"""Hungarian matching refined by temporal IoU consensus.

	1. Hungarian on cost_matrix.
	2. For each match, check if another track has a better temporal IoU consensus.
	3. Reassign based on IoU or appearance tie-break.
	4. Handle lost tracks and re-identify from graveyard.
	"""
	if graveyard is None:
		graveyard = {}
	if frame_history is None:
		frame_history = []

	row_ind, col_ind = linear_sum_assignment(cost_matrix)
	row_by_tid = {tid: idx for idx, tid in enumerate(active_ids)}

	proposals = []
	for r, c in zip(row_ind, col_ind):
		if float(cost_matrix[r, c]) >= match_threshold:
			continue

		# Bi-directional match (Mutual Nearest Neighbor) check
		if np.argmin(cost_matrix[r, :]) != c or np.argmin(cost_matrix[:, c]) != r:
			continue

		hung_tid  = active_ids[r]
		hung_vote = compute_iou_vote(hung_tid, new_feats[c]["mask"], frame_history, consensus_window)

		best_tid, best_vote = hung_tid, hung_vote
		for tid in active_ids:
			vote = compute_iou_vote(tid, new_feats[c]["mask"], frame_history, consensus_window)
			if vote > best_vote:
				best_tid, best_vote = tid, vote

		chosen_tid, chosen_vote = hung_tid, hung_vote
		if best_tid != hung_tid:
			margin = abs(best_vote - hung_vote)
			if margin < tie_margin:
				chosen_tid  = compute_tiebreak(tracks, best_tid, hung_tid, new_feats[c])
				chosen_vote = best_vote if chosen_tid == best_tid else hung_vote
			else:
				chosen_tid, chosen_vote = best_tid, best_vote

		chosen_row  = row_by_tid[chosen_tid]
		chosen_cost = float(cost_matrix[chosen_row, c])
		proposals.append((chosen_row, c, chosen_vote, chosen_cost))

	used_rows, used_cols = set(), set()
	accepted_pairs = []
	for r, c, _, pair_cost in sorted(proposals, key=lambda x: (-x[2], x[3])):
		if r in used_rows or c in used_cols:
			continue
		if pair_cost >= match_threshold:
			continue
		accepted_pairs.append((r, c))
		used_rows.add(r)
		used_cols.add(c)

	current_output = {}
	assigned_new   = set()

	for r, c in accepted_pairs:
		tid = active_ids[r]
		tracks[tid]["hist"]     = smoothing_factor * new_feats[c]["hist"] + (1 - smoothing_factor) * tracks[tid]["hist"]
		tracks[tid]["lbp"]      = smoothing_factor * new_feats[c]["lbp"]  + (1 - smoothing_factor) * tracks[tid]["lbp"]
		tracks[tid]["mask"]     = new_feats[c]["mask"]
		tracks[tid]["centroid"] = new_feats[c]["centroid"]
		tracks[tid]["bbox"]     = new_feats[c]["bbox"]
		tracks[tid]["bbox_wh"]  = (max(1.0, new_feats[c]["bbox"][2] - new_feats[c]["bbox"][0]),
		                           max(1.0, new_feats[c]["bbox"][3] - new_feats[c]["bbox"][1]))
		if tracks[tid].get("kalman") is None:
			tracks[tid]["kalman"] = create_kalman_filter(new_feats[c]["centroid"][0], new_feats[c]["centroid"][1], new_feats[c]["bbox_wh"][0], new_feats[c]["bbox_wh"][1])
		measurement = np.array([[new_feats[c]["centroid"][0]], [new_feats[c]["centroid"][1]], [new_feats[c]["bbox_wh"][0]], [new_feats[c]["bbox_wh"][1]]], dtype=np.float32)
		tracks[tid]["kalman"].correct(measurement)
		tracks[tid]["lost"]    = 0
		current_output[tid]    = new_feats[c]["mask"]
		assigned_new.add(c)

	matched_rows   = {r for r, _ in accepted_pairs}
	unmatched_rows = set(range(len(active_ids))) - matched_rows
	for r in unmatched_rows:
		tracks[active_ids[r]]["lost"] += 1

	for c in set(range(len(new_feats))) - assigned_new:
		# Try re-id against unmatched active tracks first
		best_row, best_cost = None, reid_threshold
		for r in list(unmatched_rows):
			tid = active_ids[r]
			dist_color   = cv.compareHist(tracks[tid]["hist"], new_feats[c]["hist"], cv.HISTCMP_BHATTACHARYYA)
			dist_texture = cv.compareHist(tracks[tid]["lbp"],  new_feats[c]["lbp"],  cv.HISTCMP_BHATTACHARYYA)
			track_bbox   = tracks[tid].get("predicted_bbox", tracks[tid].get("bbox", [0,0,0,0]))
			track_centroid = tracks[tid].get("predicted_centroid", tracks[tid].get("centroid", (0, 0)))
			dist_bbox    = 1.0 - box_iou(track_bbox, new_feats[c]["bbox"])
			dist_centroid = np.sqrt((track_centroid[0] - new_feats[c]["centroid"][0])**2 + (track_centroid[1] - new_feats[c]["centroid"][1])**2) / 1000.0
			dist_geom    = 0.5 * dist_bbox + 0.5 * min(1.0, dist_centroid)
			cost = 0.45 * dist_color + 0.35 * dist_texture + 0.20 * dist_geom
			if cost < best_cost:
				best_row, best_cost = r, cost

		if best_row is not None:
			tid = active_ids[best_row]
			tracks[tid]["hist"]     = smoothing_factor * new_feats[c]["hist"] + (1 - smoothing_factor) * tracks[tid]["hist"]
			tracks[tid]["lbp"]      = smoothing_factor * new_feats[c]["lbp"]  + (1 - smoothing_factor) * tracks[tid]["lbp"]
			tracks[tid]["mask"]     = new_feats[c]["mask"]
			tracks[tid]["centroid"] = new_feats[c]["centroid"]
			tracks[tid]["bbox"]     = new_feats[c]["bbox"]
			tracks[tid]["bbox_wh"]  = (max(1.0, new_feats[c]["bbox"][2] - new_feats[c]["bbox"][0]),
			                           max(1.0, new_feats[c]["bbox"][3] - new_feats[c]["bbox"][1]))
			if tracks[tid].get("kalman") is None:
				tracks[tid]["kalman"] = create_kalman_filter(new_feats[c]["centroid"][0], new_feats[c]["centroid"][1], new_feats[c]["bbox_wh"][0], new_feats[c]["bbox_wh"][1])
			tracks[tid]["kalman"].correct(np.array([[new_feats[c]["centroid"][0]], [new_feats[c]["centroid"][1]], [new_feats[c]["bbox_wh"][0]], [new_feats[c]["bbox_wh"][1]]], dtype=np.float32))
			tracks[tid]["lost"]  = 0
			current_output[tid]  = new_feats[c]["mask"]
			assigned_new.add(c)
			unmatched_rows.remove(best_row)
			continue

		# Try re-id against graveyard
		best_gid, best_cost = None, reid_threshold
		for gid, gdata in graveyard.items():
			dist_color   = cv.compareHist(gdata["hist"], new_feats[c]["hist"], cv.HISTCMP_BHATTACHARYYA)
			dist_texture = cv.compareHist(gdata["lbp"],  new_feats[c]["lbp"],  cv.HISTCMP_BHATTACHARYYA)
			track_bbox   = gdata.get("predicted_bbox", gdata.get("bbox", [0,0,0,0]))
			track_centroid = gdata.get("predicted_centroid", gdata.get("centroid", (0, 0)))
			dist_bbox    = 1.0 - box_iou(track_bbox, new_feats[c]["bbox"])
			dist_centroid = np.sqrt((track_centroid[0] - new_feats[c]["centroid"][0])**2 + (track_centroid[1] - new_feats[c]["centroid"][1])**2) / 1000.0
			dist_geom    = 0.5 * dist_bbox + 0.5 * min(1.0, dist_centroid)
			cost = 0.45 * dist_color + 0.35 * dist_texture + 0.20 * dist_geom
			if cost < best_cost:
				best_gid, best_cost = gid, cost

		feat_entry = {
			"mask":               new_feats[c]["mask"],
			"hist":               new_feats[c]["hist"],
			"lbp":                new_feats[c]["lbp"],
			"centroid":           new_feats[c]["centroid"],
			"bbox":               new_feats[c]["bbox"],
			"bbox_wh":            (max(1.0, new_feats[c]["bbox"][2] - new_feats[c]["bbox"][0]),
			                       max(1.0, new_feats[c]["bbox"][3] - new_feats[c]["bbox"][1])),
			"kalman":             create_kalman_filter(new_feats[c]["centroid"][0], new_feats[c]["centroid"][1], max(1.0, new_feats[c]["bbox"][2] - new_feats[c]["bbox"][0]), max(1.0, new_feats[c]["bbox"][3] - new_feats[c]["bbox"][1])),
			"predicted_centroid": new_feats[c]["centroid"],
			"predicted_bbox":     new_feats[c]["bbox"],
			"lost":               0,
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
	mask_uint8 = mask.astype(np.uint8)
	hist = cv.calcHist([frame_hsv], [0, 1], mask_uint8, [32, 32], [0, 180, 0, 256])
	cv.normalize(hist, hist, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)
	lbp = extract_lbp(frame_gray, mask)
	ys, xs = np.where(mask)
	if len(xs) == 0:
		centroid = (0.0, 0.0)
	else:
		centroid = (float(xs.mean()), float(ys.mean()))
	return hist, lbp, centroid


# ── Cost matrix ──────────────────────────────────────────────────────────────

def mask_iou(a, b):
	inter = np.logical_and(a, b).sum()
	union = np.logical_or(a, b).sum()
	return 0.0 if union == 0 else float(inter / union)


def compute_cost_matrix(tracks, active_ids, new_feats, iou_w, color_w, texture_w, bbox_w, img_diag):
	cost = np.zeros((len(active_ids), len(new_feats)))
	if not active_ids or not new_feats:
		return cost

	det_centroids = np.array([f["centroid"] for f in new_feats], dtype=np.float32)
	det_bboxes    = np.array([f["bbox"]     for f in new_feats], dtype=np.float32)
	for i, tid in enumerate(active_ids):
		trk          = tracks[tid]
		cx_k, cy_k   = trk["predicted_centroid"]
		pred_box     = trk["predicted_bbox"]
		dist_centroids = np.sqrt((det_centroids[:, 0] - cx_k) ** 2 + (det_centroids[:, 1] - cy_k) ** 2) / img_diag
		dist_bboxes    = 1.0 - np.array([box_iou(pred_box, det_box) for det_box in det_bboxes], dtype=np.float32)

		for j, det in enumerate(new_feats):
			dist_iou     = 1.0 - mask_iou(trk["mask"], det["mask"])
			dist_color   = cv.compareHist(trk["hist"], det["hist"], cv.HISTCMP_BHATTACHARYYA)
			dist_texture = cv.compareHist(trk["lbp"],  det["lbp"],  cv.HISTCMP_BHATTACHARYYA)
			cost[i, j]   = iou_w * dist_iou + color_w * dist_color + texture_w * dist_texture + bbox_w * (0.6 * float(dist_centroids[j]) + 0.4 * float(dist_bboxes[j]))

	return cost


# ── Track lifecycle ──────────────────────────────────────────────────────────

def init_tracks(tracks, next_id, masks, frame_hsv, frame_gray):
	"""Initialize tracker state from first-frame detections."""
	for mask in masks:
		hist, lbp, centroid = extract_features(mask, frame_hsv, frame_gray)
		bbox = mask_bbox(mask)
		tracks[next_id] = {
			"mask": mask,
			"hist": hist,
			"lbp":  lbp,
			"centroid": centroid,
			"bbox": bbox,
			"bbox_wh": (max(1.0, bbox[2] - bbox[0]), max(1.0, bbox[3] - bbox[1])),
			"kalman": create_kalman_filter(centroid[0], centroid[1], max(1.0, bbox[2] - bbox[0]), max(1.0, bbox[3] - bbox[1])),
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
			cx, cy = float(pred[0, 0]), float(pred[1, 0])
			bw, bh = max(1.0, float(pred[2, 0])), max(1.0, float(pred[3, 0]))
		else:
			cx, cy = trk["centroid"]
			bw, bh = trk.get("bbox_wh", (20.0, 20.0))
		x1 = max(0.0, min(cx - bw * 0.5, w - 1.0))
		y1 = max(0.0, min(cy - bh * 0.5, h - 1.0))
		x2 = max(0.0, min(cx + bw * 0.5, w - 1.0))
		y2 = max(0.0, min(cy + bh * 0.5, h - 1.0))
		trk["predicted_centroid"] = (cx, cy)
		trk["predicted_bbox"]     = [x1, y1, x2, y2]


def prune_lost_tracks(tracks, graveyard, patience):
	"""Move expired tracks to graveyard and return only alive tracks."""
	alive, dead = {}, {}
	for tid, data in tracks.items():
		(alive if data["lost"] <= patience else dead)[tid] = data
	graveyard.update(dead)
	return alive


# ── Per-frame pipeline ───────────────────────────────────────────────────────

def track(
	state,
	frame_bgr,
	new_masks,
	iou_w,
	color_w,
	texture_w,
	bbox_w,
	match_threshold,
	patience,
	smoothing_factor,
	reid_threshold,
	enable_motion_comp=True,
	consensus_window=8,
	consensus_tie_margin=0.05,
):
	"""Process one frame: predict → compensate → associate → update → prune."""
	h, w     = frame_bgr.shape[:2]
	img_diag = np.sqrt(h ** 2 + w ** 2)
	tracks   = state["tracks"]
	next_id  = state["next_id"]
	curr_gray = cv.cvtColor(frame_bgr, cv.COLOR_BGR2GRAY)
	curr_hsv  = cv.cvtColor(frame_bgr, cv.COLOR_BGR2HSV)

	if tracks:
		predict_tracks(tracks, frame_bgr.shape)
		if enable_motion_comp and state.get("prev_bgr") is not None:
			affine = estimate_camera_motion(state["prev_bgr"], frame_bgr, new_masks)
			for trk in tracks.values():
				apply_affine_to_kalman(trk.get("kalman"), affine)
				pcx, pcy = trk["predicted_centroid"]
				warped_pt = (affine[:, :2] @ np.array([pcx, pcy], dtype=np.float32)) + affine[:, 2]
				trk["predicted_centroid"] = (float(warped_pt[0]), float(warped_pt[1]))
				trk["predicted_bbox"]     = apply_affine_to_bbox(trk["predicted_bbox"], affine, frame_shape=frame_bgr.shape)
			
			for gdata in state.get("graveyard", {}).values():
				apply_affine_to_kalman(gdata.get("kalman"), affine)
				pcx, pcy = gdata.get("predicted_centroid", gdata["centroid"])
				warped_pt = (affine[:, :2] @ np.array([pcx, pcy], dtype=np.float32)) + affine[:, 2]
				gdata["predicted_centroid"] = (float(warped_pt[0]), float(warped_pt[1]))
				gdata["predicted_bbox"]     = apply_affine_to_bbox(gdata.get("predicted_bbox", gdata["bbox"]), affine, frame_shape=frame_bgr.shape)

	if not tracks and new_masks:
		next_id = init_tracks(tracks, next_id, new_masks, curr_hsv, curr_gray)
		state.update({"next_id": next_id, "prev_bgr": frame_bgr.copy()})
		update_frame_history(state, tracks, consensus_window)
		return {tid: data["mask"] for tid, data in tracks.items()}

	if not new_masks:
		if not tracks:
			state["prev_bgr"] = frame_bgr.copy()
			return {}
		for tid in list(tracks.keys()):
			tracks[tid]["lost"] += 1
		state["tracks"]   = prune_lost_tracks(tracks, state["graveyard"], patience)
		state["next_id"]  = next_id
		state["prev_bgr"] = frame_bgr.copy()
		update_frame_history(state, state["tracks"], consensus_window)
		return {}

	new_feats = []
	for m in new_masks:
		h_, l_, c_ = extract_features(m, curr_hsv, curr_gray)
		bbox = mask_bbox(m)
		new_feats.append({
			"mask": m, "hist": h_, "lbp": l_, "centroid": c_, "bbox": bbox,
			"bbox_wh": (max(1.0, bbox[2] - bbox[0]), max(1.0, bbox[3] - bbox[1]))
		})
	active_ids = list(tracks.keys())
	cost       = compute_cost_matrix(tracks, active_ids, new_feats, iou_w, color_w, texture_w, bbox_w, img_diag)
	output, next_id = match_with_consensus(
		tracks, next_id, cost, active_ids, new_feats, match_threshold,
		smoothing_factor=smoothing_factor, graveyard=state["graveyard"], reid_threshold=reid_threshold,
		frame_history=state.get("frame_history", []), consensus_window=consensus_window,
		tie_margin=consensus_tie_margin,
	)

	state["tracks"]   = prune_lost_tracks(tracks, state["graveyard"], patience)
	state["next_id"]  = next_id
	state["prev_bgr"] = frame_bgr.copy()
	update_frame_history(state, state["tracks"], consensus_window)
	return output




# ── CLI pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(args):
	"""Run tracker over all images and aligned NPZ masks."""
	global cv
	if args.use_opencv:
		logger.info("Using standard OpenCV library.")
		cv = cv2
	else:
		logger.info("Using custom OpenCV replacement.")

	input_dir = Path(args.input_dir)
	mask_dir = Path(args.mask_dir)
	output_dir = Path(args.output_dir)

	id_map_dir = output_dir / "id_maps"
	vis_dir = output_dir / "tracked_vis"
	id_map_dir.mkdir(parents=True, exist_ok=True)
	vis_dir.mkdir(parents=True, exist_ok=True)

	# Configure unified logging to write both to console and to scene-level log file
	log_file = output_dir.resolve().parent / "masks_and_tracking.log"
	logging.basicConfig(
		level=logging.INFO,
		format="%(asctime)s - %(levelname)s - %(message)s",
		datefmt="%H:%M:%S",
		handlers=[
			logging.StreamHandler(sys.stderr),
			logging.FileHandler(log_file, encoding="utf-8")
		],
		force=True
	)

	if not input_dir.exists():
		logger.error(f"Input directory missing: {input_dir}")
		sys.exit(1)
	if not mask_dir.exists():
		logger.error(f"Mask directory missing: {mask_dir}")
		sys.exit(1)

	weights_path = output_dir / "tracker_weights.json"
	weights_log = {
		"iou_w": args.iou_w,
		"color_w": args.color_w,
		"texture_w": args.texture_w,
		"bbox_w": args.bbox_w,
		"match_threshold": args.match_threshold,
		"patience": args.patience,
		"smoothing_factor": args.smoothing_factor,
		"reid_threshold": args.reid_threshold,
		"consensus_window": args.consensus_window,
		"consensus_tie_margin": args.consensus_tie_margin,
		"use_opencv": args.use_opencv,
		"disable_motion_comp": args.disable_motion_comp,
	}
	with open(weights_path, "w", encoding="utf-8") as f:
		json.dump(weights_log, f, indent=2)

	state = {"tracks": {}, "next_id": 1, "prev_bgr": None, "graveyard": {}, "frame_history": []}
	image_paths = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
	if not image_paths:
		logger.error(f"No images found in {input_dir}")
		sys.exit(1)

	logger.info("Tracking %d frames in vanilla NPZ-mask mode...", len(image_paths))

	for frame_idx, img_path in enumerate(image_paths):
		frame = cv.imread(str(img_path))
		if frame is None:
			logger.warning("Could not read image: %s", img_path)
			continue

		seg_masks = load_masks(mask_dir / f"masks_{frame_idx:05d}.npz")
		tracked   = track(
			state, frame, seg_masks,
			args.iou_w, args.color_w, args.texture_w, args.bbox_w,
			args.match_threshold, args.patience, args.smoothing_factor, args.reid_threshold,
			enable_motion_comp=not args.disable_motion_comp,
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

		cv.imwrite(str(id_map_dir / (img_path.stem + ".png")), id_map)
		vis = draw_tracked_overlay(frame, tracked)
		cv.putText(vis, f"Frame {frame_idx:05d} | Seg={len(seg_masks)} Track={len(tracked)}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
		cv.imwrite(str(vis_dir / f"tracked_{frame_idx:05d}.png"), vis)
		logger.info("Frame %05d | Seg=%d Track=%d", frame_idx, len(seg_masks), len(tracked))

	logger.info("Done. ID maps: %s, Visualizations: %s", id_map_dir, vis_dir)


if __name__ == "__main__":
	logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S")

	parser = argparse.ArgumentParser(description="VRoom Vanilla Object Tracker")
	parser.add_argument("--input_dir", required=True, help="Path to input images directory")
	parser.add_argument("--mask_dir", required=True, help="Path to precomputed NPZ masks directory")
	parser.add_argument("--output_dir", required=True, help="Path to save tracking output")
	parser.add_argument("--iou_w", type=float, default=TRACKING_DEFAULTS["iou_w"], help="Weight for mask IoU distance")
	parser.add_argument("--color_w", type=float, default=TRACKING_DEFAULTS["color_w"], help="Weight for color match")
	parser.add_argument("--texture_w", type=float, default=TRACKING_DEFAULTS["texture_w"], help="Weight for texture match")
	parser.add_argument("--bbox_w", type=float, default=TRACKING_DEFAULTS["bbox_w"], help="Weight for centroid/bbox prior")
	parser.add_argument("--match_threshold", type=float, default=TRACKING_DEFAULTS["match_threshold"], help="Cost cutoff for Hungarian match")
	parser.add_argument("--patience", type=int, default=TRACKING_DEFAULTS["patience"], help="Frames to remember occluded IDs")
	parser.add_argument("--smoothing_factor", type=float, default=TRACKING_DEFAULTS["smoothing_factor"], help="Exponential moving average factor for feature smoothing")
	parser.add_argument("--reid_threshold", type=float, default=TRACKING_DEFAULTS["reid_threshold"], help="Max appearance distance for graveyard re-ID")
	parser.add_argument("--disable_motion_comp", action="store_true", help="Disable global camera-motion compensation")
	parser.add_argument("--consensus_window", type=int, default=TRACKING_DEFAULTS["consensus_window"], help="Temporal window length for consensus voting")
	parser.add_argument("--consensus_tie_margin", type=float, default=TRACKING_DEFAULTS["consensus_tie_margin"], help="IoU vote margin to trigger appearance tie-break")
	parser.add_argument("--use_opencv", action="store_true", help="Use standard OpenCV (cv2) instead of the default from-scratch implementation")
	run_pipeline(parser.parse_args())
