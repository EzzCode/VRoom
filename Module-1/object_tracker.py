"""
VRoom Multi-Modal Tracking Pipeline

Loads pre-computed SAM masks (from sam_segmenter.py) and applies a Multi-Modal
Tracker (Optical Flow, Color Histograms, Hu Moments) to generate temporally
consistent 16-bit ID masks for 3D Gaussian Splatting.

Usage:
    python object_tracker.py --input_dir data/images --mask_dir data/sam_output/masks --output_dir Tracked
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


class MultiModalTracker:
    def __init__(self, alpha, beta, gamma, match_threshold, patience):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.match_threshold = match_threshold
        self.patience = patience
        
        self.next_id = 1
        self.tracks = {} 
        self.prev_gray_frame = None

    def _get_features(self, mask, frame_hsv):
        """Extracts normalized HSV histogram and log-transformed Hu Moments."""
        mask_uint8 = mask.astype(np.uint8)
        hist = cv2.calcHist([frame_hsv], [0, 1], mask_uint8, [32, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        
        hu = cv2.HuMoments(cv2.moments(mask_uint8)).flatten()
        for i in range(7):
            if hu[i] != 0:
                hu[i] = -1 * np.copysign(1.0, hu[i]) * np.log10(np.abs(hu[i]))
        return hist, hu

    def _warp_mask(self, mask, flow):
        """Predicts mask location using Dense Optical Flow."""
        h, w = mask.shape
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (x + flow[..., 0]).astype(np.float32)
        map_y = (y + flow[..., 1]).astype(np.float32)
        return cv2.remap(mask.astype(np.uint8), map_x, map_y, interpolation=cv2.INTER_NEAREST).astype(bool)

    def update(self, frame_bgr, new_masks):
        """Core tracking logic. Returns dict of {id: mask} for the current frame."""
        curr_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        curr_hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        
        if not self.tracks and new_masks:
            for mask in new_masks:
                hist, hu = self._get_features(mask, curr_hsv)
                self.tracks[self.next_id] = {'mask': mask, 'hist': hist, 'hu': hu, 'lost': 0}
                self.next_id += 1
            self.prev_gray_frame = curr_gray
            return {tid: data['mask'] for tid, data in self.tracks.items()}

        if not new_masks or self.prev_gray_frame is None:
            self.prev_gray_frame = curr_gray
            return {}

        # 1. Motion Prediction
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray_frame, curr_gray, None, 
            0.5, 3, 15, 3, 5, 1.2, 0
        )

        # 2. Feature Extraction
        new_feats = [{'mask': m, **dict(zip(('hist', 'hu'), self._get_features(m, curr_hsv)))} for m in new_masks]
        active_ids = list(self.tracks.keys())
        cost_matrix = np.zeros((len(active_ids), len(new_feats)))
        
        # 3. Cost Matrix Calculation
        for i, tid in enumerate(active_ids):
            trk = self.tracks[tid]
            warped = self._warp_mask(trk['mask'], flow)
            
            for j, new_obj in enumerate(new_feats):
                # Motion Cost
                intersection = np.logical_and(warped, new_obj['mask']).sum()
                union = np.logical_or(warped, new_obj['mask']).sum()
                dist_motion = 1.0 - (intersection / union if union > 0 else 0)
                
                # Appearance Costs
                dist_color = cv2.compareHist(trk['hist'], new_obj['hist'], cv2.HISTCMP_BHATTACHARYYA)
                dist_shape = 1.0 - np.exp(-0.1 * np.linalg.norm(trk['hu'] - new_obj['hu']))

                cost_matrix[i, j] = (self.alpha * dist_motion) + (self.beta * dist_color) + (self.gamma * dist_shape)

        # 4. Bipartite Matching & State Update
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        current_frame_output = {}
        assigned_new = set()
        
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < self.match_threshold:
                matched_id = active_ids[r]
                self.tracks[matched_id].update({'mask': new_feats[c]['mask'], 'hist': new_feats[c]['hist'], 'hu': new_feats[c]['hu'], 'lost': 0})
                current_frame_output[matched_id] = new_feats[c]['mask']
                assigned_new.add(c)
            else:
                # If the cost is too high, we consider the track lost for this frame
                self.tracks[active_ids[r]]['lost'] += 1

        # Handle Lost & New
        # Increment 'lost' for unmatched active tracks
        for r in set(range(len(active_ids))) - set(row_ind):
            self.tracks[active_ids[r]]['lost'] += 1
        # Add new detections that weren't matched    
        for c in set(range(len(new_feats))) - assigned_new:
            self.tracks[self.next_id] = {'mask': new_feats[c]['mask'], 'hist': new_feats[c]['hist'], 'hu': new_feats[c]['hu'], 'lost': 0}
            current_frame_output[self.next_id] = new_feats[c]['mask']
            self.next_id += 1

        # Memory Cleanup
        # if an ID has been lost > patience frames, we remove it from tracking
        self.tracks = {tid: data for tid, data in self.tracks.items() if data['lost'] <= self.patience}
        self.prev_gray_frame = curr_gray
        
        return current_frame_output


def get_id_color(obj_id):
    """Generate a deterministic color for a given track ID (consistent across frames)."""
    np.random.seed(obj_id * 31)
    return tuple(int(c) for c in np.random.randint(60, 255, size=3))


def draw_tracked_overlay(image, tracked_objects, alpha=0.5):
    """Draw tracked objects with ID-consistent colors and labels on the image."""
    overlay = image.copy()
    for obj_id, mask in tracked_objects.items():
        color = get_id_color(obj_id)
        overlay[mask] = color
    blended = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    # Draw ID labels at mask centroids
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


def run_pipeline(args):
    """Main execution loop."""
    input_dir = Path(args.input_dir)
    mask_dir = Path(args.mask_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        sys.exit(logger.error(f"Input directory missing: {input_dir}"))
    if not mask_dir.exists():
        sys.exit(logger.error(f"Mask directory missing: {mask_dir}. Run sam_segmenter.py first."))
    
    id_map_dir = output_dir / "id_maps"
    vis_dir = output_dir / "tracked_vis"
    id_map_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    logger.info("--- Initializing VRoom Tracker ---")
    tracker = MultiModalTracker(args.alpha, args.beta, args.gamma, args.match_threshold, args.patience)
    
    image_paths = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in ['.png', '.jpg', '.jpeg']])
    mask_paths = sorted(mask_dir.glob("masks_*.npy"))

    if not image_paths:
        sys.exit(logger.error(f"No images found in {input_dir}"))
    if not mask_paths:
        sys.exit(logger.error(f"No mask .npy files found in {mask_dir}. Run sam_segmenter.py first."))
    if len(image_paths) != len(mask_paths):
        logger.warning(f"Image count ({len(image_paths)}) != mask count ({len(mask_paths)}). Processing min of both.")

    for frame_idx, (img_path, msk_path) in enumerate(zip(image_paths, mask_paths)):
        frame = cv2.imread(str(img_path))
        if frame is None:
            logger.warning(f"Could not read image: {img_path}")
            continue

        # Load pre-computed foreground masks (cast back to bool — pickle loses dtype)
        fg_masks = [m.astype(bool) for m in np.load(str(msk_path), allow_pickle=True)]
        
        h, w = frame.shape[:2]
        tracked_objects = tracker.update(frame, fg_masks)
        
        # Save 8-bit ID map (named after source image so vote.py can match COLMAP names)
        id_map = np.zeros((h, w), dtype=np.uint8)
        for obj_id, mask in tracked_objects.items():
            id_layer = (mask.astype(np.uint8) * obj_id)
            empty = (id_map == 0)
            id_map[empty] = id_layer[empty]

        mask_name = img_path.stem + ".png"
        cv2.imwrite(str(id_map_dir / mask_name), id_map)

        # Save tracking visualization overlay
        vis = draw_tracked_overlay(frame, tracked_objects)
        cv2.putText(vis, f"Frame {frame_idx:05d} | {len(tracked_objects)} tracked",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imwrite(str(vis_dir / f"tracked_{frame_idx:05d}.png"), vis)

        logger.info(f"Frame {frame_idx:05d} | Tracking {len(tracked_objects)} objects")

    logger.info(f"Pipeline Complete. ID maps: {id_map_dir}, Visualizations: {vis_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VRoom Multi-Modal Tracking Pipeline")
    parser.add_argument("--input_dir", required=True, help="Path to input images directory")
    parser.add_argument("--mask_dir", required=True, help="Path to pre-computed SAM masks (.npy files from sam_segmenter.py)")
    parser.add_argument("--output_dir", required=True, help="Path to save tracking output")
    
    # Tracker Weights (Must sum to 1.0 for optimal tuning)
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight for Optical Flow Motion (Warped IoU)")
    parser.add_argument("--beta", type=float, default=0.3, help="Weight for Color Match (Bhattacharyya)")
    parser.add_argument("--gamma", type=float, default=0.2, help="Weight for Shape Match (Hu Moments)")
    
    # Tracker Logic
    parser.add_argument("--match_threshold", type=float, default=0.7, help="Cost cutoff for Hungarian Match")
    parser.add_argument("--patience", type=int, default=15, help="Frames to remember occluded IDs")
    
    run_pipeline(parser.parse_args())