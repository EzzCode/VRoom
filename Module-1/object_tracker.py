"""
VRoom Multi-Modal Tracking Pipeline

Loads pre-computed SAM masks (from mask_processor.py) and applies multi-modal
tracking to generate consistent ID masks for Gaussian Splatting.

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
from pathlib import Path
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)


# ── Feature Extraction ────────────────────────────────────────────────────────

def mask_centroid(mask):
    """Return (cx, cy) centroid of a boolean mask."""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return 0.0, 0.0
    return float(xs.mean()), float(ys.mean())


def extract_features(mask, frame_hsv):
    """Extract normalized HSV histogram, log Hu Moments, and centroid."""
    mask_uint8 = mask.astype(np.uint8)

    hist = cv2.calcHist([frame_hsv], [0, 1], mask_uint8, [32, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    hu = cv2.HuMoments(cv2.moments(mask_uint8)).flatten()
    for i in range(7):
        if hu[i] != 0:
            hu[i] = -1 * np.copysign(1.0, hu[i]) * np.log10(np.abs(hu[i]))

    centroid = mask_centroid(mask)
    return hist, hu, centroid


# ── Motion Prediction ─────────────────────────────────────────────────────────

def warp_mask(mask, flow):
    """Predict mask location using Dense Optical Flow."""
    h, w = mask.shape
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (x + flow[..., 0]).astype(np.float32)
    map_y = (y + flow[..., 1]).astype(np.float32)
    return cv2.remap(
        mask.astype(np.uint8), map_x, map_y,
        interpolation=cv2.INTER_NEAREST
    ).astype(bool)


def infer_prev_assignments(prev_output, new_masks, min_iou=0.08):
    """Infer likely previous IDs for current detections based on IoU overlap."""
    if not prev_output or not new_masks:
        return {}

    assigned = {}
    for j, det_mask in enumerate(new_masks):
        best_tid = None
        best_iou = 0.0
        for tid, prev_mask in prev_output.items():
            inter = np.logical_and(prev_mask, det_mask).sum()
            union = np.logical_or(prev_mask, det_mask).sum()
            iou = 0.0 if union == 0 else inter / union
            if iou > best_iou:
                best_iou = iou
                best_tid = tid
        if best_tid is not None and best_iou >= min_iou:
            assigned[j] = best_tid
    return assigned


def compute_cost_matrix(
    tracks,
    active_ids,
    new_feats,
    flow,
    alpha,
    beta,
    gamma,
    delta,
    img_diag,
    prev_assignment=None,
    stickiness=0.90,
    flow_reliability_threshold=0.25,
):
    """Build the cost matrix between existing tracks and new detections.

    Combines four distance metrics:
        - Motion   (Warped IoU via optical flow)
        - Color    (Bhattacharyya distance on HSV histograms)
        - Shape    (Exponential decay on Hu Moment distance)
        - Centroid (Euclidean distance normalised by image diagonal)
    """
    cost = np.zeros((len(active_ids), len(new_feats)))

    if prev_assignment is None:
        prev_assignment = {}

    for i, tid in enumerate(active_ids):
        trk = tracks[tid]
        warped = warp_mask(trk['mask'], flow)
        
        # Calculate predicted centroid based on optical flow
        cx_w, cy_w = mask_centroid(warped)
        if np.sum(warped) == 0:
            cx_w, cy_w = trk['centroid']

        for j, det in enumerate(new_feats):
            # Motion cost
            intersection = np.logical_and(warped, det['mask']).sum()
            union = np.logical_or(warped, det['mask']).sum()

            dist_motion_raw = 1.0 if union == 0 else 1.0 - (intersection / union)

            # Gate motion when flow is weak/noisy over the warped region.
            motion_region = warped
            if np.any(motion_region):
                region_flow_mag = np.sqrt(
                    flow[..., 0][motion_region] ** 2 + flow[..., 1][motion_region] ** 2
                )
                motion_reliability = float(np.mean(region_flow_mag))
            else:
                motion_reliability = 0.0
            dist_motion = dist_motion_raw if motion_reliability >= flow_reliability_threshold else 0.5

            # Appearance costs
            dist_color = cv2.compareHist(trk['hist'], det['hist'], cv2.HISTCMP_BHATTACHARYYA)
            dist_shape = 1.0 - np.exp(-0.1 * np.linalg.norm(trk['hu'] - det['hu']))

            # Centroid distance using predicted location (robust to mask shape changes)
            cx_d, cy_d = det['centroid']
            dist_centroid = np.sqrt((cx_w - cx_d)**2 + (cy_w - cy_d)**2) / img_diag

            score = (
                alpha * dist_motion
                + beta * dist_color
                + gamma * dist_shape
                + delta * dist_centroid
            )

            # Prefer continuity when a detection strongly overlaps a previous ID.
            if prev_assignment.get(j) == tid:
                score *= stickiness

            cost[i, j] = score

    return cost


# ── Track Management ──────────────────────────────────────────────────────────

def init_tracks(tracks, next_id, masks, frame_hsv):
    """Initialise new tracks from a set of masks (used on the first frame)."""
    for mask in masks:
        hist, hu, centroid = extract_features(mask, frame_hsv)
        tracks[next_id] = {'mask': mask, 'hist': hist, 'hu': hu, 'centroid': centroid, 'lost': 0}
        next_id += 1
    return next_id


def match_and_update(
    tracks,
    next_id,
    cost_matrix,
    active_ids,
    new_feats,
    match_threshold,
    ema=0.7,
    graveyard=None,
    reid_threshold=0.50,
):
    """Run Hungarian matching, update track states with EMA feature smoothing.

    Args:
        ema: Exponential moving average weight for new observations (0-1).
             Higher = trust new frame more. Lower = more temporal smoothing.

    Returns:
        current_output: dict {track_id: mask} of successfully matched tracks.
        next_id: updated next available ID.
    """
    if graveyard is None:
        graveyard = {}

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    current_output = {}
    assigned_new = set()

    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] < match_threshold:
            tid = active_ids[r]
            # EMA feature smoothing — prevents cascading mismatches from one noisy frame
            tracks[tid]['hist'] = ema * new_feats[c]['hist'] + (1 - ema) * tracks[tid]['hist']
            tracks[tid]['hu']   = ema * new_feats[c]['hu']   + (1 - ema) * tracks[tid]['hu']
            tracks[tid]['mask'] = new_feats[c]['mask']
            tracks[tid]['centroid'] = new_feats[c]['centroid']
            tracks[tid]['lost'] = 0
            current_output[tid] = new_feats[c]['mask']
            assigned_new.add(c)
        else:
            tracks[active_ids[r]]['lost'] += 1

    # Increment 'lost' for completely unmatched tracks
    for r in set(range(len(active_ids))) - set(row_ind):
        tracks[active_ids[r]]['lost'] += 1

    # Register unmatched detections as new tracks (with re-ID from graveyard)
    for c in set(range(len(new_feats))) - assigned_new:
        best_gid, best_cost = None, reid_threshold
        for gid, gdata in graveyard.items():
            dist_color = cv2.compareHist(gdata['hist'], new_feats[c]['hist'], cv2.HISTCMP_BHATTACHARYYA)
            dist_shape = 1.0 - np.exp(-0.1 * np.linalg.norm(gdata['hu'] - new_feats[c]['hu']))
            cost = 0.5 * dist_color + 0.5 * dist_shape
            if cost < best_cost:
                best_gid, best_cost = gid, cost

        feat_entry = {
            'mask': new_feats[c]['mask'],
            'hist': new_feats[c]['hist'],
            'hu': new_feats[c]['hu'],
            'centroid': new_feats[c]['centroid'],
            'lost': 0,
        }
        if best_gid:
            tracks[best_gid] = feat_entry
            current_output[best_gid] = new_feats[c]['mask']
            del graveyard[best_gid]
        else:
            tracks[next_id] = feat_entry
            current_output[next_id] = new_feats[c]['mask']
            next_id += 1

    return current_output, next_id


def prune_lost_tracks(tracks, graveyard, patience):
    """Remove tracks that have been lost for too many frames, but keep them in a graveyard for potential re-identification."""
    alive, dead = {}, {}
    for tid, data in tracks.items():
        if data['lost'] <= patience:
            alive[tid] = data
        else:
            dead[tid] = data  # keep the appearance model
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
    flow_reliability_threshold,
    reid_threshold,
):
    """Process a single frame through the full tracking pipeline.

    Args:
        state: dict with keys 'tracks', 'next_id', 'prev_gray', 'graveyard'.
        frame_bgr: current frame in BGR.
        new_masks: list of boolean masks from the segmenter.
        alpha, beta, gamma, delta: cost weights (motion, color, shape, centroid).
        match_threshold: max cost to accept a match.
        patience: frames before a lost track is pruned.
        ema: EMA weight for feature smoothing (0-1).

    Returns:
        dict {track_id: mask} for the current frame.
    """
    h, w = frame_bgr.shape[:2]
    img_diag = np.sqrt(h**2 + w**2)


    tracks = state['tracks']
    next_id = state['next_id']
    prev_gray = state['prev_gray']

    curr_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    curr_hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    # First frame — just initialise
    if not tracks and new_masks:
        next_id = init_tracks(tracks, next_id, new_masks, curr_hsv)
        state.update({'next_id': next_id, 'prev_gray': curr_gray})
        return {tid: data['mask'] for tid, data in tracks.items()}

    if not new_masks:
        if prev_gray is None or not tracks:
            state['prev_gray'] = curr_gray
            state['prev_output'] = {}
            return {}

        # Segmentation dropout fallback: propagate previous masks by optical flow.
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None,
            0.5, 3, 15, 3, 5, 1.2, 0
        )
        propagated = {}
        for tid in list(tracks.keys()):
            tracks[tid]['mask'] = warp_mask(tracks[tid]['mask'], flow)
            tracks[tid]['centroid'] = mask_centroid(tracks[tid]['mask'])
            tracks[tid]['lost'] += 1
            if tracks[tid]['lost'] <= patience:
                propagated[tid] = tracks[tid]['mask']

        state['tracks'] = prune_lost_tracks(tracks, state['graveyard'], patience)
        state['prev_gray'] = curr_gray
        state['prev_output'] = propagated
        return propagated

    if prev_gray is None:
        state['prev_gray'] = curr_gray
        state['prev_output'] = {}
        return {}

    # Optical flow
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        0.5, 3, 15, 3, 5, 1.2, 0
    )

    # Extract features for new detections (now includes centroid)
    new_feats = []
    for m in new_masks:
        hist, hu, centroid = extract_features(m, curr_hsv)
        new_feats.append({'mask': m, 'hist': hist, 'hu': hu, 'centroid': centroid})

    active_ids = list(tracks.keys())

    # Cost matrix → Hungarian matching → state update
    prev_assignment = infer_prev_assignments(state.get('prev_output', {}), [f['mask'] for f in new_feats])
    cost = compute_cost_matrix(
        tracks,
        active_ids,
        new_feats,
        flow,
        alpha,
        beta,
        gamma,
        delta,
        img_diag,
        prev_assignment=prev_assignment,
        flow_reliability_threshold=flow_reliability_threshold,
    )
    output, next_id = match_and_update(tracks, next_id, cost, active_ids, new_feats,
                                       match_threshold, ema=ema,
                                       graveyard=state['graveyard'],
                                       reid_threshold=reid_threshold)

    # Cleanup
    state['tracks'] = prune_lost_tracks(tracks, state['graveyard'], patience)
    state['next_id'] = next_id
    state['prev_gray'] = curr_gray
    state['prev_output'] = output

    return output


# ── Visualization ─────────────────────────────────────────────────────────────

def get_id_color(obj_id):
    """Generate a deterministic color for a track ID (consistent across frames)."""
    np.random.seed(obj_id * 31)
    return tuple(int(c) for c in np.random.randint(60, 255, size=3))


def draw_tracked_overlay(image, tracked_objects, alpha=0.5):
    """Draw tracked objects with ID-consistent colors and centroid labels."""
    overlay = image.copy()
    for obj_id, mask in tracked_objects.items():
        overlay[mask] = get_id_color(obj_id)
    blended = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    # ID labels at mask centroids
    for obj_id, mask in tracked_objects.items():
        ys, xs = np.where(mask)
        if len(xs) > 0:
            cx, cy = int(xs.mean()), int(ys.mean())
            color = get_id_color(obj_id)
            cv2.putText(blended, str(obj_id), (cx - 10, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(blended, str(obj_id), (cx - 10, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return blended


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_pipeline(args):
    """Full pipeline: load masks → track → encode ID maps + debug visualization."""
    input_dir = Path(args.input_dir)
    mask_dir = Path(args.mask_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        sys.exit(logger.error(f"Input directory missing: {input_dir}"))
    if not mask_dir.exists():
        sys.exit(logger.error(f"Mask directory missing: {mask_dir}. Run mask_processor.py first."))

    id_map_dir = output_dir / "id_maps"
    vis_dir = output_dir / "tracked_vis"
    id_map_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    # Persist ID-map format metadata for downstream consumers.
    meta_path = output_dir / "id_map_meta.json"
    meta = {
        "format": "png",
        "bit_depth": 16,
        "dtype": "uint16",
        "background_id": 0,
        "id_range": [0, int(np.iinfo(np.uint16).max)],
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Tracker state
    state = {'tracks': {}, 'next_id': 1, 'prev_gray': None, 'graveyard': {}, 'prev_output': {}}

    image_paths = sorted([
        p for p in input_dir.iterdir()
        if p.suffix.lower() in ['.png', '.jpg', '.jpeg']
    ])
    mask_paths = sorted(mask_dir.glob("masks_*.npz"))

    if not image_paths:
        sys.exit(logger.error(f"No images found in {input_dir}"))
    if not mask_paths:
        sys.exit(logger.error(f"No .npz masks found in {mask_dir}. Run mask_processor.py first."))
    if len(image_paths) != len(mask_paths):
        logger.warning(
            f"Image count ({len(image_paths)}) != mask count ({len(mask_paths)}). "
            "Processing min of both."
        )

    logger.info(f"Tracking {min(len(image_paths), len(mask_paths))} frames...")

    for frame_idx, (img_path, msk_path) in enumerate(zip(image_paths, mask_paths)):
        frame = cv2.imread(str(img_path))
        if frame is None:
            logger.warning(f"Could not read image: {img_path}")
            continue

        # Load pre-computed masks from compressed .npz archives.
        with np.load(str(msk_path), allow_pickle=False) as loaded:
            fg_stack = loaded["masks"].astype(bool)
        fg_masks = [fg_stack[i] for i in range(fg_stack.shape[0])]

        frame_h, frame_w = frame.shape[:2]
        tracked = track_frame(
            state, frame, fg_masks,
            args.alpha, args.beta, args.gamma, args.delta,
            args.match_threshold, args.patience, args.ema,
            args.flow_reliability_threshold,
            args.reid_threshold,
        )

        # Save 16-bit ID map (named after source image for vote.py compatibility)
        id_map = np.zeros((frame_h, frame_w), dtype=np.uint16)
        for obj_id, mask in tracked.items():
            if obj_id > np.iinfo(np.uint16).max:
                raise ValueError(f"Track ID {obj_id} exceeds uint16 range")
            id_layer = mask.astype(np.uint16) * np.uint16(obj_id)
            empty = (id_map == 0)
            id_map[empty] = id_layer[empty]

        cv2.imwrite(str(id_map_dir / (img_path.stem + ".png")), id_map)

        # Save tracking visualization
        vis = draw_tracked_overlay(frame, tracked)
        cv2.putText(
            vis, f"Frame {frame_idx:05d} | {len(tracked)} tracked",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
        )
        cv2.imwrite(str(vis_dir / f"tracked_{frame_idx:05d}.png"), vis)

        logger.info(f"Frame {frame_idx:05d} | Tracking {len(tracked)} objects")

    logger.info(f"Done. ID maps: {id_map_dir}, Visualizations: {vis_dir}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VRoom Multi-Modal Tracking Pipeline")
    parser.add_argument("--input_dir", required=True, help="Path to input images directory")
    parser.add_argument("--mask_dir", required=True, help="Path to pre-computed masks (.npz from mask_processor.py)")
    parser.add_argument("--output_dir", required=True, help="Path to save tracking output")

    # Tracker weights (should sum to 1.0)
    parser.add_argument("--alpha", type=float, default=0.4, help="Weight for Optical Flow Motion (Warped IoU)")
    parser.add_argument("--beta", type=float, default=0.25, help="Weight for Color Match (Bhattacharyya)")
    parser.add_argument("--gamma", type=float, default=0.15, help="Weight for Shape Match (Hu Moments)")
    parser.add_argument("--delta", type=float, default=0.2, help="Weight for Centroid Distance")

    # Tracker logic
    parser.add_argument("--match_threshold", type=float, default=0.7, help="Cost cutoff for Hungarian Match")
    parser.add_argument("--patience", type=int, default=28, help="Frames to remember occluded IDs")
    parser.add_argument("--ema", type=float, default=0.7, help="EMA weight for feature smoothing (0-1, higher=trust new frame more)")
    parser.add_argument("--flow_reliability_threshold", type=float, default=0.25, help="Minimum mean flow magnitude inside a mask to trust motion cost")
    parser.add_argument("--reid_threshold", type=float, default=0.5, help="Maximum appearance distance to re-use a graveyard ID")

    run_pipeline(parser.parse_args())