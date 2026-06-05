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
	DEFAULT_BBOX = [0.0, 0.0, 0.0, 0.0]
	COORD_OFFSET = 1
	ys, xs = np.where(mask)
	if len(xs) == 0:
		return DEFAULT_BBOX
	return [float(xs.min()), float(ys.min()), float(xs.max() + COORD_OFFSET), float(ys.max() + COORD_OFFSET)]


def box_iou(a, b):
	MIN_AREA = 0.0
	MIN_IOU = 0.0
	ax1, ay1, ax2, ay2 = a
	bx1, by1, bx2, by2 = b
	ix1, iy1 = max(ax1, bx1), max(ay1, by1)
	ix2, iy2 = min(ax2, bx2), min(ay2, by2)
	inter = max(MIN_AREA, ix2 - ix1) * max(MIN_AREA, iy2 - iy1)
	if inter <= MIN_AREA:
		return MIN_IOU
	union = max(MIN_AREA, ax2 - ax1) * max(MIN_AREA, ay2 - ay1) + max(MIN_AREA, bx2 - bx1) * max(MIN_AREA, by2 - by1) - inter
	return MIN_IOU if union <= MIN_AREA else inter / union


def extract_lbp(gray, mask):
	""" 8 neighbor LBP histogram"""
	HIST_BINS = 256
	START_OFFSET = 1
	END_OFFSET = 2
	MIN_CROP_DIM = 3

	if gray.shape != mask.shape:
		raise ValueError("image and mask must have identical shape for LBP extraction")

	ys, xs = np.where(mask)
	if len(xs) == 0:
		return np.zeros((HIST_BINS, 1), dtype=np.float32)

	y_min = max(0, int(ys.min()) - START_OFFSET)
	y_max = min(gray.shape[0], int(ys.max()) + END_OFFSET)
	x_min = max(0, int(xs.min()) - START_OFFSET)
	x_max = min(gray.shape[1], int(xs.max()) + END_OFFSET)

	crop_gray = gray[y_min:y_max, x_min:x_max]
	crop_mask = mask[y_min:y_max, x_min:x_max]

	if crop_gray.shape[0] < MIN_CROP_DIM or crop_gray.shape[1] < MIN_CROP_DIM:
		return np.zeros((HIST_BINS, 1), dtype=np.float32)

	center = crop_gray[START_OFFSET:-START_OFFSET, START_OFFSET:-START_OFFSET]
	lbp = np.zeros_like(center, dtype=np.uint8)

	# LBP neighbor patterns: (dy, dx, bit_value)
	LBP_NEIGHBORS = [
		(-1, -1, 1),
		(-1, 0, 2),
		(-1, 1, 4),
		(0, 1, 8),
		(1, 1, 16),
		(1, 0, 32),
		(1, -1, 64),
		(0, -1, 128)
	]
	for dy, dx, bit in LBP_NEIGHBORS:
		neighbor = crop_gray[START_OFFSET+dy:crop_gray.shape[0]-START_OFFSET+dy, START_OFFSET+dx:crop_gray.shape[1]-START_OFFSET+dx]
		lbp |= (neighbor >= center).astype(np.uint8) * bit

	values = lbp[crop_mask[START_OFFSET:-START_OFFSET, START_OFFSET:-START_OFFSET]]
	hist = np.bincount(values, minlength=HIST_BINS).astype(np.float32).reshape(-1, 1)
	total = float(hist.sum())
	if total > 0.0:
		hist /= total
	return hist


def create_kalman_filter(cx, cy, w, h):
	"""const velocity Kalman filter for 2D centroid and bbox size [cx, cy, w, h, vx, vy, vw, vh]"""
	STATE_DIM = 8
	MEASUREMENT_DIM = 4
	PROCESS_NOISE_COV = 1e-2
	MEASUREMENT_NOISE_COV = 1e-1

	kf = cv.KalmanFilter(STATE_DIM, MEASUREMENT_DIM)
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
	
	kf.processNoiseCov = np.eye(STATE_DIM, dtype=np.float32) * PROCESS_NOISE_COV
	kf.measurementNoiseCov = np.eye(MEASUREMENT_DIM, dtype=np.float32) * MEASUREMENT_NOISE_COV
	kf.errorCovPost = np.eye(STATE_DIM, dtype=np.float32)
	kf.statePost = np.array([[cx],[cy],[w],[h],
						  [0.0],[0.0],[0.0],[0.0]], dtype=np.float32)
	
	return kf

def estimate_camera_motion(prev_img, curr_img, curr_masks, max_corners=1200):
	"""affine transform from prev_img to curr_img """
	_MIN_FEATURES = 12
	_MIN_TRACKED  = 8
	_IDENTITY_AFFINE = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
	DILATION_KERNEL_SIZE = (5, 5)
	DILATION_ITERATIONS = 2
	MIN_BG_NONZERO = 200
	CORNER_QUALITY = 0.01
	CORNER_MIN_DIST = 7
	CORNER_BLOCK_SIZE = 7
	LK_WIN_SIZE = (21, 21)
	LK_MAX_LEVEL = 3
	LK_CRITERIA_MAX_COUNT = 30
	LK_CRITERIA_EPS = 0.01
	FLOW_SUCCESS_STATUS = 1
	RANSAC_THRESHOLD = 3.0

	if prev_img is None or curr_img is None:
		return _IDENTITY_AFFINE.copy()

	prev_gray = cv.cvtColor(prev_img, cv.COLOR_BGR2GRAY)
	curr_gray = cv.cvtColor(curr_img, cv.COLOR_BGR2GRAY)
	h, w = curr_gray.shape[:2]

	fg = np.zeros((h, w), dtype=np.uint8)
	for m in curr_masks:
		if m.shape == fg.shape:
			fg[m] = 255
	if np.any(fg):
		fg = cv.dilate(fg, np.ones(DILATION_KERNEL_SIZE, dtype=np.uint8), iterations=DILATION_ITERATIONS)

	bg = cv.bitwise_not(fg)
	if np.count_nonzero(bg) < MIN_BG_NONZERO:
		return _IDENTITY_AFFINE.copy()

	#shi-Tomasi corner detection + pyramidal Lucas-Kanade optical flow
	p0 = cv.goodFeaturesToTrack(prev_gray, mask=bg, maxCorners=max_corners, qualityLevel=CORNER_QUALITY, minDistance=CORNER_MIN_DIST, blockSize=CORNER_BLOCK_SIZE)
	if p0 is None or len(p0) < _MIN_FEATURES:
		return _IDENTITY_AFFINE.copy()

	next_point, output_status, _ = cv.calcOpticalFlowPyrLK(
		prev_gray, curr_gray, p0, None,
		winSize=LK_WIN_SIZE, maxLevel=LK_MAX_LEVEL,
		criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, LK_CRITERIA_MAX_COUNT, LK_CRITERIA_EPS),
	)
	if next_point is None or output_status is None:
		return _IDENTITY_AFFINE.copy()

	good_prev = p0[output_status.flatten() == FLOW_SUCCESS_STATUS]
	good_curr = next_point[output_status.flatten() == FLOW_SUCCESS_STATUS]
	if len(good_prev) < _MIN_TRACKED:
		return _IDENTITY_AFFINE.copy()

	affine, _ = cv.estimateAffinePartial2D(good_prev, good_curr, method=cv.RANSAC, ransacReprojThreshold=RANSAC_THRESHOLD)
	
	if affine is None:
		return _IDENTITY_AFFINE.copy()
	return affine.astype(np.float32)


def apply_affine_to_kalman(kf, affine):
	STATE_DIM = 8
	if kf is None:
		return
	linear = affine[:, :2].astype(np.float32)
	trans = affine[:, 2].astype(np.float32)

	def _warp_state(state_vec):
		flat = state_vec.reshape(-1).astype(np.float32)
		pos = linear @ flat[:2] + trans
		vel = linear @ flat[4:6]
		flat[0], flat[1], flat[4], flat[5] = float(pos[0]), float(pos[1]), float(vel[0]), float(vel[1])
		return flat.reshape(STATE_DIM, 1)

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


#-------------------------------------------------------------------------

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


def compute_tiebreak(tracks, tid_a, tid_b, det_feat):
	COLOR_WEIGHT = 0.60
	TEXTURE_WEIGHT = 0.40
	GEOMETRY_NORM_FACTOR = 1000.0
	GEOMETRY_WEIGHT = 0.5

	def score(tid):
		dist_color   = cv.compareHist(tracks[tid]["hist"], det_feat["hist"], cv.HISTCMP_BHATTACHARYYA)
		dist_texture = cv.compareHist(tracks[tid]["lbp"],  det_feat["lbp"],  cv.HISTCMP_BHATTACHARYYA)
		dist_appearance = COLOR_WEIGHT * dist_color + TEXTURE_WEIGHT * dist_texture
		cx_k, cy_k = tracks[tid].get("predicted_centroid", tracks[tid]["centroid"])
		cx_d, cy_d = det_feat["centroid"]
		dist_geometry = np.sqrt((cx_k - cx_d)**2 + (cy_k - cy_d)**2) / GEOMETRY_NORM_FACTOR
		return dist_appearance + GEOMETRY_WEIGHT * dist_geometry

	if score(tid_a) <= score(tid_b):
		return tid_a
	return tid_b


def update_frame_history(state, tracks, max_window):
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
	smoothing_factor=TRACKING_DEFAULTS["smoothing_factor"],
	graveyard=None,
	reid_threshold=TRACKING_DEFAULTS["reid_threshold"],
	frame_history=None,
	consensus_window=TRACKING_DEFAULTS["consensus_window"],
	tie_margin=TRACKING_DEFAULTS["consensus_tie_margin"],
):
	SMOOTHING_BASE = 1.0
	MIN_BBOX_DIM = 1.0
	RESET_LOST_COUNT = 0
	DEFAULT_BBOX = [0, 0, 0, 0]
	DEFAULT_CENTROID = (0, 0)
	
	LOST_INCREMENT = 1
	MAX_DIST_VAL = 1.0
	GEOMETRY_NORM_FACTOR = 1000.0
	GEOM_BBOX_WEIGHT = 0.5
	GEOM_CENTROID_WEIGHT = 0.5
	
	COST_COLOR_WEIGHT = 0.45
	COST_TEXTURE_WEIGHT = 0.35
	COST_GEOM_WEIGHT = 0.20

	if graveyard is None:
		graveyard = {}
	if frame_history is None:
		frame_history = []

	row_idx, col_idx = linear_sum_assignment(cost_matrix)
	row_by_id = {id: idx for idx, id in enumerate(active_ids)}

	proposals = []
	for r, c in zip(row_idx, col_idx):
		if float(cost_matrix[r, c]) >= match_threshold:
			continue

		# bidirectional match nearest neighbor check
		if np.argmin(cost_matrix[r, :]) != c or np.argmin(cost_matrix[:, c]) != r:
			continue

		hungarian_id  = active_ids[r]
		hungarian_vote = compute_iou_vote(hungarian_id, new_feats[c]["mask"], frame_history, consensus_window)

		best_id, best_vote = hungarian_id, hungarian_vote
		for id in active_ids:
			vote = compute_iou_vote(id, new_feats[c]["mask"], frame_history, consensus_window)
			if vote > best_vote:
				best_id, best_vote = id, vote

		chosen_id, chosen_vote = hungarian_id, hungarian_vote
		if best_id != hungarian_id:
			margin = abs(best_vote - hungarian_vote)
			if margin < tie_margin:
				chosen_id  = compute_tiebreak(tracks, best_id, hungarian_id, new_feats[c])
				chosen_vote = best_vote if chosen_id == best_id else hungarian_vote
			else:
				chosen_id = best_id
				chosen_vote = best_vote

		chosen_row  = row_by_id[chosen_id]
		chosen_cost = float(cost_matrix[chosen_row, c])
		proposals.append((chosen_row, c, chosen_vote, chosen_cost))

	used_rows, used_cols = set(), set()
	accepted = []
	for r, c, _, pair_cost in sorted(proposals, key=lambda x: (-x[2], x[3])):
		if r in used_rows or c in used_cols:
			continue
		if pair_cost >= match_threshold:
			continue
		accepted.append((r, c))
		used_rows.add(r)
		used_cols.add(c)

	current_output = {}
	assigned_new   = set()

	for r, c in accepted:
		tid = active_ids[r]
		
		hist = smoothing_factor * new_feats[c]["hist"] + (SMOOTHING_BASE - smoothing_factor) * tracks[tid]["hist"]
		tracks[tid]["hist"]     = hist
		
		lbp = smoothing_factor * new_feats[c]["lbp"]  + (SMOOTHING_BASE - smoothing_factor) * tracks[tid]["lbp"]
		tracks[tid]["lbp"]      = lbp

		tracks[tid]["mask"]     = new_feats[c]["mask"]
		tracks[tid]["centroid"] = new_feats[c]["centroid"]
		tracks[tid]["bbox"]     = new_feats[c]["bbox"]
		
		bbox_wh = (max(MIN_BBOX_DIM, new_feats[c]["bbox"][2] - new_feats[c]["bbox"][0]),
		           max(MIN_BBOX_DIM, new_feats[c]["bbox"][3] - new_feats[c]["bbox"][1]))
		tracks[tid]["bbox_wh"]  = bbox_wh
		if tracks[tid].get("kalman") is None:
			tracks[tid]["kalman"] = create_kalman_filter(new_feats[c]["centroid"][0], 
														 new_feats[c]["centroid"][1],
														 bbox_wh[0], bbox_wh[1])

		measurement = np.array([
								[new_feats[c]["centroid"][0]],
								[new_feats[c]["centroid"][1]],
							    [bbox_wh[0]], [bbox_wh[1]]], dtype=np.float32)
		
		tracks[tid]["kalman"].correct(measurement)
		tracks[tid]["lost"]    = RESET_LOST_COUNT
		current_output[tid]    = new_feats[c]["mask"]
		assigned_new.add(c)

	matched_rows   = {r for r, _ in accepted}
	unmatched_rows = set(range(len(active_ids))) - matched_rows
	for r in unmatched_rows:
		tracks[active_ids[r]]["lost"] += LOST_INCREMENT

	for c in set(range(len(new_feats))) - assigned_new:
		# re-id against unmatched active tracks first
		best_row, best_cost = None, reid_threshold
		for r in list(unmatched_rows):
			tid = active_ids[r]
			dist_color   = cv.compareHist(tracks[tid]["hist"], new_feats[c]["hist"], cv.HISTCMP_BHATTACHARYYA)
			dist_texture = cv.compareHist(tracks[tid]["lbp"],  new_feats[c]["lbp"],  cv.HISTCMP_BHATTACHARYYA)
			track_bbox   = tracks[tid].get("predicted_bbox", tracks[tid].get("bbox", DEFAULT_BBOX))
			track_centroid = tracks[tid].get("predicted_centroid", tracks[tid].get("centroid", DEFAULT_CENTROID))
			dist_bbox    = MAX_DIST_VAL - box_iou(track_bbox, new_feats[c]["bbox"])
			dist_centroid = np.sqrt((track_centroid[0] - new_feats[c]["centroid"][0])**2 + (track_centroid[1] - new_feats[c]["centroid"][1])**2) / GEOMETRY_NORM_FACTOR
			dist_geometry    = GEOM_BBOX_WEIGHT * dist_bbox + GEOM_CENTROID_WEIGHT * min(MAX_DIST_VAL, dist_centroid)
			cost = COST_COLOR_WEIGHT * dist_color + COST_TEXTURE_WEIGHT * dist_texture + COST_GEOM_WEIGHT * dist_geometry
			if cost < best_cost:
				best_row, best_cost = r, cost

		if best_row is not None:
			tid = active_ids[best_row]

			hist = smoothing_factor * new_feats[c]["hist"] + (SMOOTHING_BASE - smoothing_factor) * tracks[tid]["hist"]
			tracks[tid]["hist"] = hist
			lbp = smoothing_factor * new_feats[c]["lbp"]  + (SMOOTHING_BASE - smoothing_factor) * tracks[tid]["lbp"]
			tracks[tid]["lbp"] = lbp
			tracks[tid]["mask"] = new_feats[c]["mask"]
			tracks[tid]["centroid"] = new_feats[c]["centroid"]
			tracks[tid]["bbox"] = new_feats[c]["bbox"]
			bbox_wh = (max(MIN_BBOX_DIM, new_feats[c]["bbox"][2] - new_feats[c]["bbox"][0]),
			           max(MIN_BBOX_DIM, new_feats[c]["bbox"][3] - new_feats[c]["bbox"][1]))
			tracks[tid]["bbox_wh"]  = bbox_wh
			if tracks[tid].get("kalman") is None:
				tracks[tid]["kalman"] = create_kalman_filter(new_feats[c]["centroid"][0],
												 			 new_feats[c]["centroid"][1], 
															 bbox_wh[0], bbox_wh[1])
			
			tracks[tid]["kalman"].correct(np.array([[new_feats[c]["centroid"][0]], 
										   			[new_feats[c]["centroid"][1]],
													[bbox_wh[0]], [bbox_wh[1]]], dtype=np.float32))
			tracks[tid]["lost"]  = RESET_LOST_COUNT
			current_output[tid]  = new_feats[c]["mask"]
			assigned_new.add(c)
			unmatched_rows.remove(best_row)
			continue

		# re-id against graveyard
		best_gid, best_cost = None, reid_threshold
		for id, data in graveyard.items():
			dist_color   = cv.compareHist(data["hist"], new_feats[c]["hist"], cv.HISTCMP_BHATTACHARYYA)
			dist_texture = cv.compareHist(data["lbp"],  new_feats[c]["lbp"],  cv.HISTCMP_BHATTACHARYYA)

			track_bbox   = data.get("predicted_bbox", data.get("bbox", DEFAULT_BBOX))
			track_centroid = data.get("predicted_centroid", data.get("centroid", DEFAULT_CENTROID))
			
			dist_bbox    = MAX_DIST_VAL - box_iou(track_bbox, new_feats[c]["bbox"])
			dist_centroid = np.sqrt((track_centroid[0] - new_feats[c]["centroid"][0])**2 + (track_centroid[1] - new_feats[c]["centroid"][1])**2) / GEOMETRY_NORM_FACTOR
			dist_geometry    = GEOM_BBOX_WEIGHT * dist_bbox + GEOM_CENTROID_WEIGHT * min(MAX_DIST_VAL, dist_centroid)
			
			cost = COST_COLOR_WEIGHT * dist_color + COST_TEXTURE_WEIGHT * dist_texture + COST_GEOM_WEIGHT * dist_geometry
			if cost < best_cost:
				best_gid, best_cost = id, cost

		bbox_wh = (max(MIN_BBOX_DIM, new_feats[c]["bbox"][2] - new_feats[c]["bbox"][0]),
		           max(MIN_BBOX_DIM, new_feats[c]["bbox"][3] - new_feats[c]["bbox"][1]))
		kalman = create_kalman_filter(new_feats[c]["centroid"][0], new_feats[c]["centroid"][1], bbox_wh[0], bbox_wh[1])
		feat_entry = {
			"mask": new_feats[c]["mask"],
			"hist": new_feats[c]["hist"],
			"lbp": new_feats[c]["lbp"],
			"centroid": new_feats[c]["centroid"],
			"bbox": new_feats[c]["bbox"],
			"bbox_wh": bbox_wh,
			"kalman": kalman,
			"predicted_centroid": new_feats[c]["centroid"],
			"predicted_bbox": new_feats[c]["bbox"],
			"lost": RESET_LOST_COUNT,
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
	HIST_CHANNELS = [0, 1]
	HIST_BINS = [32, 32]
	HIST_RANGES = [0, 180, 0, 256]
	NORM_ALPHA = 0
	NORM_BETA = 1
	DEFAULT_CENTROID = (0.0, 0.0)

	mask_uint8 = mask.astype(np.uint8)
	hist = cv.calcHist([frame_hsv], HIST_CHANNELS, mask_uint8, HIST_BINS, HIST_RANGES)
	cv.normalize(hist, hist, alpha=NORM_ALPHA, beta=NORM_BETA, norm_type=cv.NORM_MINMAX)
	lbp = extract_lbp(frame_gray, mask)
	ys, xs = np.where(mask)
	if len(xs) == 0:
		centroid = DEFAULT_CENTROID
	else:
		centroid = (float(xs.mean()), float(ys.mean()))
	return hist, lbp, centroid


# cost -------------------- 

def mask_iou(a, b):
	MIN_UNION_SIZE = 0
	DEFAULT_IOU = 0.0
	inter = np.logical_and(a, b).sum()
	union = np.logical_or(a, b).sum()
	return DEFAULT_IOU if union == MIN_UNION_SIZE else float(inter / union)


def compute_cost_matrix(tracks, active_ids, new_feats, iou_w, color_w, texture_w, bbox_w, img_diag):
	MAX_DIST_VAL = 1.0
	CENTROID_DIST_WEIGHT = 0.6
	BBOX_DIST_WEIGHT = 0.4

	cost = np.zeros((len(active_ids), len(new_feats)))
	if not active_ids or not new_feats:
		return cost

	det_centroids = np.array([f["centroid"] for f in new_feats], dtype=np.float32)
	det_bboxes    = np.array([f["bbox"]     for f in new_feats], dtype=np.float32)
	for i, tid in enumerate(active_ids):
		track = tracks[tid]
		cx_k, cy_k = track["predicted_centroid"]
		pred_box = track["predicted_bbox"]
		dist_centroids = np.sqrt((det_centroids[:, 0] - cx_k) ** 2 + (det_centroids[:, 1] - cy_k) ** 2) / img_diag
		dist_bboxes = MAX_DIST_VAL - np.array([box_iou(pred_box, det_box) for det_box in det_bboxes], dtype=np.float32)

		for j, det in enumerate(new_feats):
			dist_iou = MAX_DIST_VAL - mask_iou(track["mask"], det["mask"])
			dist_color = cv.compareHist(track["hist"], det["hist"], cv.HISTCMP_BHATTACHARYYA)
			dist_texture = cv.compareHist(track["lbp"],  det["lbp"],  cv.HISTCMP_BHATTACHARYYA)
			cost[i, j] = iou_w * dist_iou + color_w * dist_color + texture_w * dist_texture + bbox_w * (CENTROID_DIST_WEIGHT * float(dist_centroids[j]) + BBOX_DIST_WEIGHT * float(dist_bboxes[j]))

	return cost


# Track lifecycle

def init_tracks(tracks, next_id, masks, frame_hsv, frame_gray):
	MIN_BBOX_DIM = 1.0
	INIT_LOST_COUNT = 0
	for mask in masks:
		hist, lbp, centroid = extract_features(mask, frame_hsv, frame_gray)
		bbox = mask_bbox(mask)
		tracks[next_id] = {
			"mask": mask,
			"hist": hist,
			"lbp":  lbp,
			"centroid": centroid,
			"bbox": bbox,
			"bbox_wh": (max(MIN_BBOX_DIM, bbox[2] - bbox[0]), max(MIN_BBOX_DIM, bbox[3] - bbox[1])),
			"kalman": create_kalman_filter(centroid[0], centroid[1], max(MIN_BBOX_DIM, bbox[2] - bbox[0]), max(MIN_BBOX_DIM, bbox[3] - bbox[1])),
			"predicted_centroid": centroid,
			"predicted_bbox": bbox,
			"lost": INIT_LOST_COUNT,
		}
		next_id += 1
	return next_id


def predict_tracks(tracks, frame_shape):
	"""Run Kalman predict step and refresh predicted XYXY priors."""
	MIN_BBOX_DIM = 1.0
	DEFAULT_BBOX_WH = (20.0, 20.0)
	MIN_BOUND = 0.0
	HALF_FACTOR = 0.5
	COORD_PADDING = 1.0

	h, w = frame_shape[:2]
	for trk in tracks.values():
		if trk.get("kalman") is not None:
			pred = trk["kalman"].predict()
			cx, cy = float(pred[0, 0]), float(pred[1, 0])
			bw, bh = max(MIN_BBOX_DIM, float(pred[2, 0])), max(MIN_BBOX_DIM, float(pred[3, 0]))
		else:
			cx, cy = trk["centroid"]
			bw, bh = trk.get("bbox_wh", DEFAULT_BBOX_WH)
		x1 = max(MIN_BOUND, min(cx - bw * HALF_FACTOR, w - COORD_PADDING))
		y1 = max(MIN_BOUND, min(cy - bh * HALF_FACTOR, h - COORD_PADDING))
		x2 = max(MIN_BOUND, min(cx + bw * HALF_FACTOR, w - COORD_PADDING))
		y2 = max(MIN_BOUND, min(cy + bh * HALF_FACTOR, h - COORD_PADDING))
		trk["predicted_centroid"] = (cx, cy)
		trk["predicted_bbox"]     = [x1, y1, x2, y2]


def prune_lost_tracks(tracks, graveyard, patience):
	"""Move expired tracks to graveyard and return only alive tracks."""
	alive, dead = {}, {}
	for tid, data in tracks.items():
		(alive if data["lost"] <= patience else dead)[tid] = data
	graveyard.update(dead)
	return alive


def track(
	state,
	frame_bgr,
	new_masks,
	iou_w=TRACKING_DEFAULTS["iou_w"],
	color_w=TRACKING_DEFAULTS["color_w"],
	texture_w=TRACKING_DEFAULTS["texture_w"],
	bbox_w=TRACKING_DEFAULTS["bbox_w"],
	match_threshold=TRACKING_DEFAULTS["match_threshold"],
	patience=TRACKING_DEFAULTS["patience"],
	smoothing_factor=TRACKING_DEFAULTS["smoothing_factor"],
	reid_threshold=TRACKING_DEFAULTS["reid_threshold"],
	enable_motion_comp=True,
	consensus_window=TRACKING_DEFAULTS["consensus_window"],
	consensus_tie_margin=TRACKING_DEFAULTS["consensus_tie_margin"],
):
	height, width = frame_bgr.shape[:2]
	img_diag = np.sqrt(height ** 2 + width ** 2)
	tracks = state["tracks"]
	next_id = state["next_id"]
	curr_gray = cv.cvtColor(frame_bgr, cv.COLOR_BGR2GRAY)
	curr_hsv = cv.cvtColor(frame_bgr, cv.COLOR_BGR2HSV)

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
			
			for gdata in state.get("graveyard", {}).values():
				apply_affine_to_kalman(gdata.get("kalman"), affine)
				pcx, pcy = gdata.get("predicted_centroid", gdata["centroid"])
				warped_pt = (affine[:, :2] @ np.array([pcx, pcy], dtype=np.float32)) + affine[:, 2]
				gdata["predicted_centroid"] = (float(warped_pt[0]), float(warped_pt[1]))
				gdata["predicted_bbox"] = apply_affine_to_bbox(gdata.get("predicted_bbox", gdata["bbox"]), affine, frame_shape=frame_bgr.shape)

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
		state["tracks"] = prune_lost_tracks(tracks, state["graveyard"], patience)
		state["next_id"] = next_id
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

	state["tracks"] = prune_lost_tracks(tracks, state["graveyard"], patience)
	state["next_id"] = next_id
	state["prev_bgr"] = frame_bgr.copy()
	update_frame_history(state, state["tracks"], consensus_window)
	return output
#----------------------------------------------------------------------------------------------------------------
def run_pipeline(args):
	"""Run tracker over all images and aligned NPZ masks."""
	global cv
	INITIAL_TRACK_ID = 1
	TEXT_POSITION = (10, 30)
	TEXT_SCALE = 0.8
	TEXT_COLOR = (255, 255, 255)
	TEXT_THICKNESS = 2

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

	state = {"tracks": {}, "next_id": INITIAL_TRACK_ID, "prev_bgr": None, "graveyard": {}, "frame_history": []}
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
		id_map = np.full((frame_h, frame_w), 255, dtype=np.uint16)
		empty = np.ones((frame_h, frame_w), dtype=bool)
		for obj_id, mask in tracked.items():
			if obj_id > np.iinfo(np.uint16).max:
				raise ValueError(f"Track ID {obj_id} exceeds uint16 range")
			fill = empty & mask
			id_map[fill] = np.uint16(obj_id)
			empty[fill] = False

		cv.imwrite(str(id_map_dir / (img_path.stem + ".png")), id_map)
		vis = draw_tracked_overlay(frame, tracked)
		cv.putText(vis, f"Frame {frame_idx:05d} | Seg={len(seg_masks)} Track={len(tracked)}", TEXT_POSITION, cv.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, TEXT_COLOR, TEXT_THICKNESS)
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
