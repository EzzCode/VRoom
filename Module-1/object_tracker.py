"""
VRoom Multi-Modal Tracking Pipeline

Loads pre-computed SAM masks (from mask_processor.py) and applies multi-modal
tracking to generate consistent ID masks for Gaussian Splatting.

Usage:
    python object_tracker.py --input_dir data/images --mask_dir data/sam_output/masks --output_dir Tracked

Outputs:
    <output_dir>/id_maps/       — per-frame 8-bit PNG ID maps (named after source image)
    <output_dir>/tracked_vis/   — overlay PNGs with ID-consistent colors and labels
"""

import sys
import argparse
import logging
from pathlib import Path
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)


# ── Feature Extraction ────────────────────────────────────────────────────────

def extract_features(mask, frame_hsv):
    """Extract normalized HSV histogram and log-transformed Hu Moments from a mask."""
    mask_uint8 = mask.astype(np.uint8)

    hist = cv2.calcHist([frame_hsv], [0, 1], mask_uint8, [32, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    hu = cv2.HuMoments(cv2.moments(mask_uint8)).flatten()
    for i in range(7):
        if hu[i] != 0:
            hu[i] = -1 * np.copysign(1.0, hu[i]) * np.log10(np.abs(hu[i]))

    return hist, hu


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


def compute_cost_matrix(tracks, active_ids, new_feats, flow, alpha, beta, gamma):
    """Build the cost matrix between existing tracks and new detections.

    Combines three distance metrics:
        - Motion (Warped IoU via optical flow)
        - Color  (Bhattacharyya distance on HSV histograms)
        - Shape  (Exponential decay on Hu Moment distance)
    """
    cost = np.zeros((len(active_ids), len(new_feats)))

    for i, tid in enumerate(active_ids):
        trk = tracks[tid]
        warped = warp_mask(trk['mask'], flow)

        for j, det in enumerate(new_feats):
            # Motion cost
            intersection = np.logical_and(warped, det['mask']).sum()
            union = np.logical_or(warped, det['mask']).sum()
            dist_motion = 1.0 - (intersection / union if union > 0 else 0)

            # Appearance costs
            dist_color = cv2.compareHist(trk['hist'], det['hist'], cv2.HISTCMP_BHATTACHARYYA)
            dist_shape = 1.0 - np.exp(-0.1 * np.linalg.norm(trk['hu'] - det['hu']))

            cost[i, j] = alpha * dist_motion + beta * dist_color + gamma * dist_shape

    return cost


# ── Track Management ──────────────────────────────────────────────────────────

def init_tracks(tracks, next_id, masks, frame_hsv):
    """Initialise new tracks from a set of masks (used on the first frame)."""
    for mask in masks:
        hist, hu = extract_features(mask, frame_hsv)
        tracks[next_id] = {'mask': mask, 'hist': hist, 'hu': hu, 'lost': 0}
        next_id += 1
    return next_id


def match_and_update(tracks, next_id, cost_matrix, active_ids, new_feats, match_threshold):
    """Run Hungarian matching and update track states.

    Returns:
        current_output: dict {track_id: mask} of successfully matched tracks.
        next_id: updated next available ID.
    """
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    current_output = {}
    assigned_new = set()

    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] < match_threshold:
            tid = active_ids[r]
            tracks[tid].update({
                'mask': new_feats[c]['mask'],
                'hist': new_feats[c]['hist'],
                'hu':   new_feats[c]['hu'],
                'lost': 0,
            })
            current_output[tid] = new_feats[c]['mask']
            assigned_new.add(c)
        else:
            tracks[active_ids[r]]['lost'] += 1

    # Increment 'lost' for completely unmatched tracks
    for r in set(range(len(active_ids))) - set(row_ind):
        tracks[active_ids[r]]['lost'] += 1

    # Register unmatched detections as new tracks
    for c in set(range(len(new_feats))) - assigned_new:
        tracks[next_id] = {
            'mask': new_feats[c]['mask'],
            'hist': new_feats[c]['hist'],
            'hu':   new_feats[c]['hu'],
            'lost': 0,
        }
        current_output[next_id] = new_feats[c]['mask']
        next_id += 1

    return current_output, next_id


def prune_lost_tracks(tracks, patience):
    """Remove tracks that have been lost for more than `patience` frames."""
    return {tid: data for tid, data in tracks.items() if data['lost'] <= patience}


def track_frame(state, frame_bgr, new_masks, alpha, beta, gamma, match_threshold, patience):
    """Process a single frame through the full tracking pipeline.

    Args:
        state: dict with keys 'tracks', 'next_id', 'prev_gray'.
        frame_bgr: current frame in BGR.
        new_masks: list of boolean masks from the segmenter.
        alpha, beta, gamma: cost weights.
        match_threshold: max cost to accept a match.
        patience: frames before a lost track is pruned.

    Returns:
        dict {track_id: mask} for the current frame.
    """
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

    if not new_masks or prev_gray is None:
        state['prev_gray'] = curr_gray
        return {}

    # Optical flow
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        0.5, 3, 15, 3, 5, 1.2, 0
    )

    # Extract features for new detections
    new_feats = [
        {'mask': m, **dict(zip(('hist', 'hu'), extract_features(m, curr_hsv)))}
        for m in new_masks
    ]
    active_ids = list(tracks.keys())

    # Cost matrix → Hungarian matching → state update
    cost = compute_cost_matrix(tracks, active_ids, new_feats, flow, alpha, beta, gamma)
    output, next_id = match_and_update(tracks, next_id, cost, active_ids, new_feats, match_threshold)

    # Cleanup
    state['tracks'] = prune_lost_tracks(tracks, patience)
    state['next_id'] = next_id
    state['prev_gray'] = curr_gray

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

    # Tracker state
    state = {'tracks': {}, 'next_id': 1, 'prev_gray': None}

    image_paths = sorted([
        p for p in input_dir.iterdir()
        if p.suffix.lower() in ['.png', '.jpg', '.jpeg']
    ])
    mask_paths = sorted(mask_dir.glob("masks_*.npy"))

    if not image_paths:
        sys.exit(logger.error(f"No images found in {input_dir}"))
    if not mask_paths:
        sys.exit(logger.error(f"No .npy masks found in {mask_dir}. Run mask_processor.py first."))
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

        # Load pre-computed masks (cast back to bool — pickle loses dtype)
        fg_masks = [m.astype(bool) for m in np.load(str(msk_path), allow_pickle=True)]

        frame_h, frame_w = frame.shape[:2]
        tracked = track_frame(
            state, frame, fg_masks,
            args.alpha, args.beta, args.gamma,
            args.match_threshold, args.patience,
        )

        # Save 8-bit ID map (named after source image for vote.py compatibility)
        id_map = np.zeros((frame_h, frame_w), dtype=np.uint8)
        for obj_id, mask in tracked.items():
            id_layer = mask.astype(np.uint8) * obj_id
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
    parser.add_argument("--mask_dir", required=True, help="Path to pre-computed masks (.npy from mask_processor.py)")
    parser.add_argument("--output_dir", required=True, help="Path to save tracking output")

    # Tracker weights (should sum to 1.0)
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight for Optical Flow Motion (Warped IoU)")
    parser.add_argument("--beta", type=float, default=0.3, help="Weight for Color Match (Bhattacharyya)")
    parser.add_argument("--gamma", type=float, default=0.2, help="Weight for Shape Match (Hu Moments)")

    # Tracker logic
    parser.add_argument("--match_threshold", type=float, default=0.7, help="Cost cutoff for Hungarian Match")
    parser.add_argument("--patience", type=int, default=15, help="Frames to remember occluded IDs")

    run_pipeline(parser.parse_args())