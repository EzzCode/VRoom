"""
Consistency Verifier — Multi-view consistency check for inpainted results.

Verifies that inpainted content is coherent across views by rendering
the current 3DGS model from the neighbor viewpoint and comparing the masked
inpainted region using LPIPS (Learned Perceptual Image Patch Similarity).
"""

__all__ = ['verify_consistency']

import logging
import numpy as np
import cv2
import torch
from target_replenishment.core.objectgs_bridge import create_virtual_camera, render_view
from target_replenishment.core.metrics import compute_lpips

logger = logging.getLogger(__name__)

def _compute_masked_lpips(img1_np: np.ndarray, img2_np: np.ndarray, mask_np: np.ndarray) -> float:
    """Compute LPIPS only within the masked region bounding box."""
    if mask_np is None or mask_np.sum() == 0:
        return 0.0

    # Ensure mask is binary
    mask = (mask_np > 0).astype(np.uint8)

    # Find bounding box of the mask
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return 0.0

    y0, y1 = max(0, ys.min() - 10), min(img1_np.shape[0], ys.max() + 10)
    x0, x1 = max(0, xs.min() - 10), min(img1_np.shape[1], xs.max() + 10)

    # Crop
    crop1 = img1_np[y0:y1, x0:x1]
    crop2 = img2_np[y0:y1, x0:x1]

    # Resize to 256x256 for consistent LPIPS evaluation
    crop1_resized = cv2.resize(crop1, (256, 256), interpolation=cv2.INTER_AREA)
    crop2_resized = cv2.resize(crop2, (256, 256), interpolation=cv2.INTER_AREA)

    return compute_lpips(crop1_resized, crop2_resized, net='vgg')


def verify_consistency(
    anchor_result: dict,
    neighbor_candidates: list,
    gaussians=None,
    pipe_config=None,
    thresholds: dict = None,
) -> dict:
    """Verify multi-view consistency of inpainted results using LPIPS rendering.

    Args:
        anchor_result: dict with 'rgb_inpainted', 'camera_params', 'mask'.
        neighbor_candidates: list of lists. For each neighbor view, a list of 'm' candidate dicts.
            Candidate dict: 'rgb_inpainted', 'camera_params', 'mask_warped', 'mask_inpainted'.
        gaussians: Loaded GaussianModel.
        pipe_config: Pipeline configuration.
        thresholds: dict. Use 'lpips_max' (default 0.3). Lower LPIPS is better.

    Returns:
        dict:
            'accepted_views' — list of best candidate dicts for each view
            'rejected_views' — list of all rejected candidate dicts
            'scores' — per-view consistency scores
    """
    if thresholds is None:
        thresholds = {'lpips_max': 0.3}

    lpips_max = thresholds.get('lpips_max', 0.3)

    # The anchor view is always accepted
    anchor_rgb = anchor_result['rgb_inpainted']
    H, W = anchor_rgb.shape[:2]
    accepted = [{
        'rgb_inpainted': anchor_rgb,
        'mask': anchor_result.get('mask', np.zeros((H, W), dtype=np.uint8)),
        'camera_params': anchor_result['camera_params'],
    }]
    rejected = []
    scores = {'anchor': {'status': 'accepted', 'score': 0.0}}

    bg_color = torch.zeros(3, dtype=torch.float32, device="cuda")

    for i, candidates in enumerate(neighbor_candidates):
        best_candidate = None
        best_score = float('inf')  # Lower LPIPS is better

        if not candidates:
            continue

        for c_idx, cand in enumerate(candidates):
            cam_p = cand['camera_params']
            cam = create_virtual_camera(cam_p['R'], cam_p['T'], cam_p['K'],
                                        cam_p['width'], cam_p['height'])

            # Render the ORIGINAL CURRENT geometry from this neighbor viewpoint
            with torch.no_grad():
                current_render = render_view(gaussians, cam, pipe_config, bg_color)
            
            # The rendered RGB image [0, 255]
            current_rgb = (current_render['rgb'].permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)

            mask = cand.get('mask_warped', np.ones((cam_p['height'], cam_p['width']), dtype=np.uint8))

            cand_rgb = cand['rgb_inpainted']
            
            # Compute LPIPS in the masked region only
            score = _compute_masked_lpips(cand_rgb, current_rgb, mask)
            cand['_consistency_score'] = score

            if score < best_score:
                best_score = score
                best_candidate = cand

        # Discard non-best candidates
        for cand in candidates:
            if cand is not best_candidate:
                rejected.append(cand)

        passed = best_score <= lpips_max
        scores[f'neighbor_{i}'] = {'score': best_score, 'status': 'accepted' if passed else 'rejected'}

        if passed:
            logger.info(f"Neighbor {i}: Selected best candidate with LPIPS={best_score:.3f} (threshold={lpips_max}) → ACCEPT")
            accepted.append({
                'rgb_inpainted': best_candidate['rgb_inpainted'],
                'mask': best_candidate.get('mask_inpainted', best_candidate.get('mask_warped', np.zeros(1))),
                'camera_params': best_candidate['camera_params'],
            })
        else:
            logger.info(f"Neighbor {i}: Best candidate LPIPS={best_score:.3f} failed threshold ({lpips_max}) → REJECT")
            rejected.append(best_candidate)

    logger.info(f"Consistency verification: {len(accepted) - 1} neighbors accepted, {len(rejected)} candidates rejected")
    return {
        'accepted_views': accepted,
        'rejected_views': rejected,
        'scores': scores,
    }
