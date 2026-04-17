"""
SAM 2 Inference Module

Handles loading the Segment Anything Model 2 (SAM 2) and generating raw masks
from input images. This module is intentionally kept separate from any
post-processing logic.

Usage (as a library):
    from sam_inference import load_sam, generate_masks
    
    mask_generator = load_sam("sam2.1_hiera_l.yaml", "path/to/checkpoint.pt", "cuda")
    raw_masks = generate_masks(mask_generator, image_bgr)
"""

import logging
import cv2
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

logger = logging.getLogger(__name__)

# SAM 2 model configs bundled with the package (sam2/configs/sam2.1/)
# Checkpoint downloads: https://github.com/facebookresearch/sam2#download-checkpoints
MODEL_CONFIGS = {
    "sam2.1_hiera_t": "configs/sam2.1/sam2.1_hiera_t.yaml",
    "sam2.1_hiera_s": "configs/sam2.1/sam2.1_hiera_s.yaml",
    "sam2.1_hiera_b+": "configs/sam2.1/sam2.1_hiera_b+.yaml",
    "sam2.1_hiera_l": "configs/sam2.1/sam2.1_hiera_l.yaml",
}


def load_sam(model_cfg, checkpoint, device,
             points_per_side=16, pred_iou_thresh=0.90,
             stability_score_thresh=0.96, min_mask_region_area=500):
    """Load SAM 2 and return a configured automatic mask generator.
    
    Args:
        model_cfg: Config key (e.g. 'sam2.1_hiera_l') or full YAML path.
        checkpoint: Path to the SAM 2 checkpoint (.pt) file.
        device: Torch device string ('cuda' or 'cpu').
        points_per_side: Grid density for auto mask generation.
        pred_iou_thresh: Minimum predicted IoU to keep a mask.
        stability_score_thresh: Minimum stability score to keep a mask.
        min_mask_region_area: Minimum mask area in pixels.
    
    Returns:
        SAM2AutomaticMaskGenerator instance.
    """
    config = MODEL_CONFIGS.get(model_cfg, model_cfg)
    logger.info(f"Loading SAM 2 ({config}) from {checkpoint}...")
    
    sam2_model = build_sam2(config, checkpoint, device=device, apply_postprocessing=False)
    
    mask_generator = SAM2AutomaticMaskGenerator(
        sam2_model,
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        min_mask_region_area=min_mask_region_area,
    )
    logger.info("SAM 2 model loaded.")
    return mask_generator


def generate_masks(mask_generator, frame_bgr):
    """Run SAM 2 on a single BGR frame and return the list of mask dicts.
    
    Args:
        mask_generator: SAM2AutomaticMaskGenerator instance.
        frame_bgr: Input image in BGR format (OpenCV convention).
    
    Returns:
        List of mask dictionaries, each containing 'segmentation' key
        with a boolean numpy array.
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return mask_generator.generate(frame_rgb)
