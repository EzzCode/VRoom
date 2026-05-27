"""
VRoom Mask-Object Tracking Pipeline

Loads pre-computed masks from mask_processor.py
and performs tracking to generate ID-consistent masks
for downstream vote.py 3D voting.

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
from typing import List, Sequence

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

try:
	from . import opencv_scratch
except ImportError:
	import opencv_scratch

cv = opencv_scratch


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


# ── Mask I/O ─────────────────────────────────────────────────────────────────


def load_frame_masks(npz_path: Path) -> List[np.ndarray]:
	"""Load boolean masks for a frame from compressed NPZ or uncompressed NPY."""
	path = npz_path
	if not path.exists():
		npy_path = npz_path.with_suffix(".npy")
		if npy_path.exists():
			path = npy_path
		else:
			return []
	
	if path.suffix == ".npy":
		arr = np.load(str(path), allow_pickle=True)
	else:
		with np.load(str(path)) as data:
			arr = data.get("masks")
			
	if arr is None or arr.ndim != 3:
		return []
	return [arr[i].astype(bool) for i in range(arr.shape[0])]


# ── Geometry helpers ─────────────────────────────────────────────────────────

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
	# Unpack boxs:
	# (x1,y1) is top-left
	# (x2,y2) is bottom-right
	ax1, ay1, ax2, ay2 = a 
	bx1, by1, bx2, by2 = b
	
	# intersection box coordinates
	ix1 = max(ax1, bx1)  # Intersection left edge (rightmost of two left edges)
	iy1 = max(ay1, by1)  # Intersection top edge (bottommost of two top edges)
	ix2 = min(ax2, bx2)  # Intersection right edge (leftmost of two right edges)
	iy2 = min(ay2, by2)  # Intersection bottom edge (topmost of two bottom edges)
	
	# intersection area
	iw = max(0.0, ix2 - ix1)  # Intersection width (0 if boxes don't overlap horizontally)
	ih = max(0.0, iy2 - iy1)  # Intersection height (0 if boxes don't overlap vertically)
	inter = iw * ih  # Intersection area
	
	# if no intersection, IoU is 0
	if inter <= 0.0:
		return 0.0
	
	# Calculate areas of both boxes
	areaA = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)  
	areaB = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
	
	# Calculate union area: sum of both areas minus intersection
	union = areaA + areaB - inter
	
	# Return IoU = intersection / union
	return 0.0 if union <= 0.0 else float(inter / union)


def extract_lbp_hist(gray, mask):
	"""
	Extract normalized 8-neighbor LBP histogram over masked pixels.

	Parameters:
		gray (np.ndarray): Grayscale image (2D array).
		mask (np.ndarray): Binary mask (same shape as gray), selects pixels to include.

	Returns:
		hist (np.ndarray): 256x1 normalized histogram of LBP codes for masked pixels.
	"""
	# Ensure input image and mask have the same shape
	if gray.shape != mask.shape:
		raise ValueError("Gray image and mask must have identical shape for LBP extraction")

	# If image is too small for LBP (needs at least 3x3), return zero histogram
	if gray.shape[0] < 3 or gray.shape[1] < 3:
		return np.zeros((256, 1), dtype=np.float32)

	# Center pixels (exclude border, since LBP needs 8 neighbors)
	center = gray[1:-1, 1:-1]
	# Prepare output array for LBP 
	lbp = np.zeros_like(center, dtype=np.uint8)

	# Define 8 neighbors: (dy, dx, bit value)
	neighbors = [
		(-1, -1, 1),   # top-left
		(-1,  0, 2),   # top
		(-1,  1, 4),   # top-right
		( 0,  1, 8),   # right
		( 1,  1, 16),  # bottom-right
		( 1,  0, 32),  # bottom
		( 1, -1, 64),  # bottom-left
		( 0, -1, 128)  # left
	]

	# Compute LBP code for each center pixel
	for dy, dx, bit in neighbors:
		# Shifted neighbor region
		neighbor = gray[1 + dy:gray.shape[0] - 1 + dy, 1 + dx:gray.shape[1] - 1 + dx]
		# If neighbor >= center, set corresponding bit
		lbp |= ((neighbor >= center).astype(np.uint8) * bit)

	# Mask out border pixels (LBP is not defined there)
	inner_mask = mask[1:-1, 1:-1]
	# Get LBP values for masked pixels
	values = lbp[inner_mask]

	# Compute histogram of LBP codes (0 to 255)
	hist = np.bincount(values, minlength=256).astype(np.float32).reshape(-1, 1)

	# Normalize histogram to sum to 1 (if any pixels)
	total = float(hist.sum())
	if total > 0.0:
		hist /= total
	return hist


# ── Kalman filter ────────────────────────────────────────────────────────────

def create_kalman_filter(cx, cy):
	"""
	Create a constant-velocity Kalman filter for 2D centroid motion.
	State vector: [x, y, vx, vy] (position and velocity)
	Measurement: [x, y] (position only)
	Args:
		cx (float): Initial x position (centroid x)
		cy (float): Initial y position (centroid y)
	Returns:
		cv.KalmanFilter: Configured Kalman filter instance
	"""
	# Create Kalman filter with 4 dynamic params (x, y, vx, vy) and 2 measured params (x, y)
	kf = cv.KalmanFilter(4, 2)

	# State transition matrix (models constant velocity)
	kf.transitionMatrix = np.array([
		[1, 0, 1, 0], # x' = x + vx
		[0, 1, 0, 1], # y' = y + vy
		[0, 0, 1, 0], # vx' = vx
		[0, 0, 0, 1]  # vy' = vy
	], dtype=np.float32)

	# Measurement matrix
	kf.measurementMatrix = np.array([
		[1, 0, 0, 0], # measure x
		[0, 1, 0, 0]  # measure y
	], dtype=np.float32)

	# Process noise covariance (model uncertainty)
	# Small values: assume nearly constant velocity
	kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2 	# more/less smoothing (higher = trust in measurements, less smoothing)

	# Measurement noise covariance (sensor uncertainty)
	# Larger value: measurements are noisy
	kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1 # more/less responsive to measurements (higher = less trust in measurements, more smoothing)

	# Posterior error covariance (initial state uncertainty) 
	# start with uncertainty we only know initial position, not velocity
	kf.errorCovPost = np.eye(4, dtype=np.float32)

	# Initial state: [x, y, vx, vy] (start at given position, zero velocity)
	kf.statePost = np.array([
		[cx],
		[cy],
		[0.0],
		[0.0]
	], dtype=np.float32)

	return kf


# ── Camera motion compensation ───────────────────────────────────────────────

def identity_affine() -> np.ndarray:
	"""Return 2x3 identity affine matrix."""
	return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)


def estimate_camera_motion(prev_bgr, curr_bgr, curr_masks, max_corners=1200):
	"""Estimate global camera motion from background optical flow."""
	_MIN_FEATURES = 12  # Minimum features required to estimate motion; otherwise return identity
	_MIN_TRACKED = 8     # Minimum successfully tracked features required; otherwise return identity
	# If previous or current frame is missing, return identity (no motion)
	if prev_bgr is None or curr_bgr is None:
		return identity_affine()

	# Convert both frames to grayscale for feature detection and tracking
	prev_gray = cv.cvtColor(prev_bgr, cv.COLOR_BGR2GRAY)
	curr_gray = cv.cvtColor(curr_bgr, cv.COLOR_BGR2GRAY)
	h, w = curr_gray.shape[:2]

	# Create a foreground mask (fg) of zeros 
	fg = np.zeros((h, w), dtype=np.uint8)
	# Mark foreground regions (moving objects) as 255 in the mask
	for m in curr_masks:
		if m.shape == fg.shape:
			fg[m] = 255

	# Dilate the foreground mask to cover more area around objects
	if np.any(fg):
		kernel = np.ones((5, 5), dtype=np.uint8)
		fg = cv.dilate(fg, kernel, iterations=2)

	# Invert the foreground mask to get the background mask (bg)
	bg = cv.bitwise_not(fg)
	# If not enough background pixels, return identity
	if int(np.count_nonzero(bg)) < 200:
		return identity_affine()

	# Detect good features to track in the background regions of the previous frame
	# Shi-Tomasi corner detection
	p0 = cv.goodFeaturesToTrack( 
		prev_gray,
		mask=bg,
		maxCorners=max_corners,
		qualityLevel=0.01,
		minDistance=7,
		blockSize=7
	)
	# If not enough features found, return identity
	if p0 is None or len(p0) < _MIN_FEATURES:
		return identity_affine()

	# Track the detected features from previous to current frame using optical flow
	# Lucas-Kanade optical flow
	p1, st, _ = cv.calcOpticalFlowPyrLK(
		prev_gray,
		curr_gray,
		p0,
		None,
		winSize=(21, 21),
		maxLevel=3,
		criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.01),
	)
	# If tracking failed, return identity
	if p1 is None or st is None:
		return identity_affine()

	# Select only successfully tracked points
	good_prev = p0[st.flatten() == 1]
	good_curr = p1[st.flatten() == 1]
	# If not enough tracked points, return identity
	if len(good_prev) < _MIN_TRACKED:
		return identity_affine()

	# Estimate affine transform (translation, rotation, scale, shear) using RANSAC
	affine, _ = cv.estimateAffinePartial2D(
		good_prev,
		good_curr,
		method=cv.RANSAC,
		ransacReprojThreshold=3.0
	)
	# If estimation failed, return identity
	if affine is None:
		return identity_affine()
	# Return the estimated affine matrix as float32
	return affine.astype(np.float32)


def apply_affine_to_kalman(kf, affine):
	"""
	Warp Kalman filter's position and velocity using an affine transform.
	This updates both the posterior (statePost) and, if present, the prior (statePre) state vectors.
	"""
	# If Kalman filter object is None
	if kf is None:
		return

	# Extract the linear (2x2) and translation (2,) parts of the affine matrix
	linear = affine[:, :2].astype(np.float32)  # 2x2 matrix for rotation, scale, shear
	trans = affine[:, 2].astype(np.float32)    # 2-vector for translation

	def _warp_state(state_vec):
		# Flatten the state vector to 1D
		flat = state_vec.reshape(-1).astype(np.float32)
		# Apply affine transform to position
		pos = linear @ flat[:2] + trans
		# Apply only the linear part to velocity (no translation)
		vel = linear @ flat[2:4]
		# Update the state vector with the warped position and velocity
		flat[0], flat[1], flat[2], flat[3] = float(pos[0]), float(pos[1]), float(vel[0]), float(vel[1])
		# Reshape back to column vector (4, 1)
		return flat.reshape(4, 1)

	# Warp the posterior state
	kf.statePost = _warp_state(kf.statePost)
	# If the prior state exists, warp it as well
	if getattr(kf, "statePre", None) is not None:
		kf.statePre = _warp_state(kf.statePre)


def apply_affine_to_bbox(bbox, affine, frame_shape=None):
	"""
	Warp XYXY bbox corners through affine transformation and rebuild box.
	Args:
		bbox: [x1, y1, x2, y2] coordinates (top-left and bottom-right corners).
		affine: 2x3 affine transformation matrix.
		frame_shape: (height, width) tuple to clip the output box to frame boundaries.
	Returns:
		List of [wx1, wy1, wx2, wy2]: warped bounding box coordinates.
	"""
	x1, y1, x2, y2 = [float(v) for v in bbox]

	# top-left, top-right, bottom-left, bottom-right
	pts = np.array([[x1, y1], [x2, y1], [x1, y2], [x2, y2]], dtype=np.float32)
	# Apply affine transformation to each corner
	# affine[:, :2] is the 2x2 linear part, affine[:, 2] is the translation
	warped = (affine[:, :2] @ pts.T).T + affine[:, 2]
	# Find the new min/max x and y from the warped corners
	wx1, wy1 = np.min(warped[:, 0]), np.min(warped[:, 1])
	wx2, wy2 = np.max(warped[:, 0]), np.max(warped[:, 1])

	# clip the coordinates to the frame boundaries
	if frame_shape is not None:
		h, w = frame_shape[:2]
		wx1 = max(0.0, min(wx1, w - 1.0))  # wx1 to [0, w-1]
		wy1 = max(0.0, min(wy1, h - 1.0))  # wy1 to [0, h-1]
		wx2 = max(0.0, min(wx2, w - 1.0))  # wx2 to [0, w-1]
		wy2 = max(0.0, min(wy2, h - 1.0))  # wy2 to [0, h-1]

	# Return the warped bounding box
	return [float(wx1), float(wy1), float(wx2), float(wy2)]


# ── Temporal consensus matching ──────────────────────────────────────────────

def compute_iou_vote(track_id, det_mask, frame_history, window_size):
	"""Average IoU vote for a candidate detection against track history.
	For a given track_id and detection mask, this function computes the average IoU
	between the detection mask and the masks of the track in the recent frame history (up to window_size frames).
	"""
	# Select the most recent window_size frames from the history (or all if window_size==0)
	recent = frame_history[-window_size:] if window_size > 0 else frame_history
	scores = []
	for entry in recent:
		# Get the mask for this track_id in the historical entry
		hist_mask = entry.get("masks", {}).get(track_id)
		if hist_mask is None:
			continue
		# Compute IoU between the historical mask and the candidate detection mask
		scores.append(mask_iou(hist_mask, det_mask))
	if not scores:
		# If there are no valid scores, return 0
		return 0.0
	# Return the average IoU score
	return float(np.mean(scores))


def appearance_distance(track_data, det_feat):
	"""Compute appearance distance for tie-breaks.
	Uses a weighted sum of color histogram and texture (LBP) histogram distances.
	Lower mean more similar
	"""
	# Compute Bhattacharyya distance between color histograms and LBP histograms
	dist_color = cv.compareHist(track_data["hist"], det_feat["hist"], cv.HISTCMP_BHATTACHARYYA)
	dist_texture = cv.compareHist(track_data["lbp"], det_feat["lbp"], cv.HISTCMP_BHATTACHARYYA)
	# color is weighted more than texture 
	return 0.60 * dist_color + 0.40 * dist_texture


def compute_appearance_tiebreak(tracks, tid_a, tid_b, det_feat):
	"""Resolve near-equal IoU vote ties using appearance similarity.
	Compares two tracks (tid_a, tid_b) returns the track with the closest appearance.
	"""
	a_cost = appearance_distance(tracks[tid_a], det_feat)
	b_cost = appearance_distance(tracks[tid_b], det_feat)
	# Return the track id with the lower distance
	return tid_a if a_cost <= b_cost else tid_b


def update_frame_history(state, tracks, max_window):
	"""Append current active track masks to sliding frame history.
	Maintains a sliding window of the most recent max_window frames in the state dict.
	"""
	# Create a new entry with all current track masks
	entry = {"masks": {tid: trk["mask"] for tid, trk in tracks.items()}}
	# Get or create the frame_history list in the state
	history = state.setdefault("frame_history", [])
	# Append the new entry
	history.append(entry)
	# Keep only the most recent max_window entries
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
	"""Run Hungarian matching, then refine assignment by temporal IoU consensus.
	This function performs object association between existing tracks and new detections.
	1. Runs Hungarian matching on the cost matrix.
	2. For each match, checks if another track has a better temporal IoU consensus with the detection.
	3. If so, may reassign based on IoU or appearance tie-break.
	4. Updates tracks, handles lost tracks, and re-identifies from the graveyard if possible.
	Returns a dict of current output masks and the next available track id.
	"""
	if graveyard is None:
		graveyard = {}
	if frame_history is None:
		frame_history = []

	# Step 1: Hungarian matching 
	# Guarantees the lowest total assignment cost across all objects, preventing two tracks from claiming the same detection
	row_ind, col_ind = linear_sum_assignment(cost_matrix)
	# Map track id to row index in cost matrix
	row_by_tid = {tid: idx for idx, tid in enumerate(active_ids)}

	proposals = []
	# Step 2: For each matched pair, check for better consensus
	for r, c in zip(row_ind, col_ind):
		base_cost = float(cost_matrix[r, c])
		if base_cost >= match_threshold:
			continue

		hung_tid = active_ids[r]
		# Compute IoU vote for the Hungarian-assigned track
		hung_vote = compute_iou_vote(hung_tid, new_feats[c]["mask"], frame_history, consensus_window)

		# Search for any other track with a better IoU vote for this detection
		#looks at the frame_history. It checks if the proposed match has a high IoU with the track over the last window frames 
		best_tid = hung_tid
		best_vote = hung_vote
		for tid in active_ids:
			vote = compute_iou_vote(tid, new_feats[c]["mask"], frame_history, consensus_window)
			if vote > best_vote:
				best_tid = tid
				best_vote = vote

		# Decide which track to assign: Hungarian or best by consensus
		chosen_tid = hung_tid
		chosen_vote = hung_vote
		if best_tid != hung_tid:
			margin = abs(best_vote - hung_vote)
			if margin < tie_margin:
				# If votes are close, use appearance to break the tie
				chosen_tid = compute_appearance_tiebreak(tracks, best_tid, hung_tid, new_feats[c])
				chosen_vote = best_vote if chosen_tid == best_tid else hung_vote
			else:
				# Otherwise, pick the best by consensus
				chosen_tid = best_tid
				chosen_vote = best_vote

		# Get the row index for the chosen track
		chosen_row = row_by_tid[chosen_tid]
		chosen_cost = float(cost_matrix[chosen_row, c])
		proposals.append((chosen_tid, chosen_row, c, chosen_vote, chosen_cost))

	# Step 3: Accept pairs by best vote, lowest cost, no duplicates to solve conflicts
	used_rows = set()
	used_cols = set()
	accepted_pairs = []
	# Sort proposals by vote (desc) then cost (asc) to prioritize better matches
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

	# Step 4: Update matched tracks with new features and reset lost counter
	for r, c in accepted_pairs:
		tid = active_ids[r]
		# Exponential moving average update for histograms
		tracks[tid]["hist"] = ema * new_feats[c]["hist"] + (1 - ema) * tracks[tid]["hist"]
		tracks[tid]["lbp"] = ema * new_feats[c]["lbp"] + (1 - ema) * tracks[tid]["lbp"]
		# Update mask, centroid, bbox, and bbox_wh
		tracks[tid]["mask"] = new_feats[c]["mask"]
		tracks[tid]["centroid"] = new_feats[c]["centroid"]
		tracks[tid]["bbox"] = new_feats[c]["bbox"]
		tracks[tid]["bbox_wh"] = (max(1.0, new_feats[c]["bbox"][2] - new_feats[c]["bbox"][0]), max(1.0, new_feats[c]["bbox"][3] - new_feats[c]["bbox"][1]))
		# Initialize Kalman filter if not present
		if tracks[tid].get("kalman") is None:
			tracks[tid]["kalman"] = create_kalman_filter(*new_feats[c]["centroid"])
		# Correct Kalman filter with new measurement
		measurement = np.array([[new_feats[c]["centroid"][0]], [new_feats[c]["centroid"][1]]], dtype=np.float32)
		tracks[tid]["kalman"].correct(measurement)
		# Reset lost counter
		tracks[tid]["lost"] = 0
		current_output[tid] = new_feats[c]["mask"]
		assigned_new.add(c)

	# Step 5: Increment lost counter for unmatched tracks
	for r in set(range(len(active_ids))) - {r for r, _ in accepted_pairs}:
		tracks[active_ids[r]]["lost"] += 1

	# Step 6: Try to re-identify unmatched detections from graveyard, or create new tracks
	for c in set(range(len(new_feats))) - assigned_new:
		best_gid, best_cost = None, reid_threshold
		for gid, gdata in graveyard.items():
			# Compute appearance and bbox distance to graveyard tracks
			dist_color = cv.compareHist(gdata["hist"], new_feats[c]["hist"], cv.HISTCMP_BHATTACHARYYA)
			dist_texture = cv.compareHist(gdata["lbp"], new_feats[c]["lbp"], cv.HISTCMP_BHATTACHARYYA)
			dist_bbox = 1.0 - box_iou_xyxy(gdata.get("bbox", [0, 0, 0, 0]), new_feats[c]["bbox"])
			cost = 0.45 * dist_color + 0.35 * dist_texture + 0.20 * dist_bbox
			if cost < best_cost:
				best_gid, best_cost = gid, cost

		# Prepare new track entry
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
			# Re-identify: assign graveyard id to this detection
			tracks[best_gid] = feat_entry
			current_output[best_gid] = new_feats[c]["mask"]
			del graveyard[best_gid]
		else:
			# Create a new track
			tracks[next_id] = feat_entry
			current_output[next_id] = new_feats[c]["mask"]
			next_id += 1

	return current_output, next_id


def extract_features(mask, frame_hsv, frame_gray):
	"""Extract HSV histogram, LBP histogram, and centroid for one mask.
	- HSV histogram: color descriptor in masked region
	- LBP histogram: texture descriptor in masked region
	- Centroid: spatial center of the mask
	"""
	# Convert mask to uint8 for OpenCV functions
	mask_uint8 = mask.astype(np.uint8)
	# Compute 2D HSV histogram for masked region
	hist = cv.calcHist([frame_hsv], [0, 1], mask_uint8, [32, 32], [0, 180, 0, 256])
	# Normalize histogram to [0, 1]
	cv.normalize(hist, hist, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)
	# Compute LBP histogram for masked region
	lbp_hist = extract_lbp_hist(frame_gray, mask)
	# Compute centroid of the mask
	centroid = mask_centroid(mask)
	return hist, lbp_hist, centroid



# ── Cost matrix ──────────────────────────────────────────────────────────────

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
			dist_color = cv.compareHist(trk["hist"], det["hist"], cv.HISTCMP_BHATTACHARYYA)
			dist_texture = cv.compareHist(trk["lbp"], det["lbp"], cv.HISTCMP_BHATTACHARYYA)
			dist_centroid = float(dist_centroids[j])
			dist_bbox = float(dist_bboxes[j])

			cost[i, j] = alpha * dist_iou + beta * dist_color + gamma * dist_texture + delta * (0.6 * dist_centroid + 0.4 * dist_bbox)
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



# ── Per-frame pipeline ───────────────────────────────────────────────────────

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
	consensus_window=8,
	consensus_tie_margin=0.05,
):
	"""Process one frame: predict → compensate → associate → update → prune."""
	h, w = frame_bgr.shape[:2]
	img_diag = np.sqrt(h ** 2 + w ** 2)
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

	state["tracks"] = prune_lost_tracks(tracks, state["graveyard"], patience)
	state["next_id"] = next_id
	state["prev_gray"] = curr_gray
	state["prev_bgr"] = frame_bgr.copy()
	update_frame_history(state, state["tracks"], consensus_window)
	return output


# ── Visualization ────────────────────────────────────────────────────────────

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
	blended = cv.addWeighted(overlay, alpha, image, 1 - alpha, 0)
	for obj_id, mask in tracked_objects.items():
		ys, xs = np.where(mask)
		if len(xs) > 0:
			cx, cy = int(xs.mean()), int(ys.mean())
			color = get_id_color(obj_id)
			cv.putText(blended, str(obj_id), (cx - 10, cy + 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
			cv.putText(blended, str(obj_id), (cx - 10, cy + 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
	return blended


# ── CLI pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(args):
	"""Run tracker over all images and aligned NPZ masks."""
	global cv
	if args.use_opencv:
		logger.info("Using standard OpenCV (cv2) library as requested.")
		cv = cv2
	else:
		logger.info("Using custom from-scratch OpenCV replacement (opencv_scratch) by default.")

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
		frame = cv.imread(str(img_path))
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
	parser.add_argument("--consensus_window", type=int, default=8, help="Temporal window length for consensus voting")
	parser.add_argument("--consensus_tie_margin", type=float, default=0.05, help="IoU vote margin to trigger appearance tie-break")
	parser.add_argument("--use_opencv", action="store_true", help="Use standard OpenCV (cv2) instead of the default from-scratch implementation")
	run_pipeline(parser.parse_args())
