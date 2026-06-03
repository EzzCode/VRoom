# VRoom Object Tracker Default Weights and Hyperparameters

# Default tracking weights (optimized for standard spatial matching and consistency)
TRACKING_DEFAULTS = {
    "iou_w": 0.75,                  # Weight for mask IoU distance
    "color_w": 0.25,                 # Weight for color match (HSV histogram)
    "texture_w": 0.15,               # Weight for texture match (LBP histogram)
    "bbox_w": 0.20,                  # Weight for centroid/bbox prior
    "match_threshold": 0.70,        # Cost cutoff for Hungarian matching
    "patience": 28,                 # Frames to remember occluded track IDs
    "smoothing_factor": 0.40,       # Exponential moving average factor for feature smoothing
    "reid_threshold": 0.60,         # Max appearance distance for Re-ID (graveyard / active unmatched)
    "consensus_window": 8,          # Temporal window length for consensus voting
    "consensus_tie_margin": 0.05,   # IoU vote margin to trigger appearance tie-break
}
