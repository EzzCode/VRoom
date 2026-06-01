"""
Shared constants 
"""

ALPHA_THRESH = 0.4          # ObjectGS alpha → binary mask threshold

SV3D_FILL_FRAC = 0.85       # fraction of the SV3D frame the object should occupy
FOV_Y_DEG      = 50.0       # default vertical field-of-view for supervision cameras

SEED_DEPTH_MIN     = 0.1    # minimum positive depth to count a seed point as "in front"
SEED_MIN_IN_FRONT  = 20     # minimum in-front seed points before scale estimation fails
SEED_PERCENTILE_LO = 2      # lower percentile for world-scale bounding box
SEED_PERCENTILE_HI = 98     # upper percentile for world-scale bounding box
WS_CLIP_MIN        = 0.05   # world-scale ratio clamp (lower)
WS_CLIP_MAX        = 2.0    # world-scale ratio clamp (upper)

GAUSSIAN_MODEL_DEFAULTS = {
    "gs_attr": "2D",
    "feat_dim": 32,
    "view_dim": 3,
    "appearance_dim": 0,
    "n_offsets": 10,
    "voxel_size": 0.001,
    "render_mode": "RGB+ED",
    "tile_size_2dgs": 8,
}
