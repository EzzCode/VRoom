"""
Shared constants 
"""
HF_CACHE_DIR = r"A:\hf_cache"

ALPHA_THRESH = 0.4          # model alpha binary mask threshold

SV3D_FILL_FRAC = 0.85       # fraction of the SV3D frame the object should occupy
FOV_Y_DEG      = 50.0       # default vertical fov for dataset cameras

SEED_DEPTH_MIN     = 0.1    # minimum positive depth to count a seed point as in front
SEED_MIN_IN_FRONT  = 20     # minimum in front seed points 
SEED_PERCENTILE_LO = 2      # lower percentile for world bounding box
SEED_PERCENTILE_HI = 98     # upper percentile for world bounding box
WS_CLIP_MIN        = 0.05   # world scale ratio clamp
WS_CLIP_MAX        = 2.0    # world scale ratio clamp 

GAUSSIAN_MODEL_DEFAULTS = {
    "gaussian_type": "2D",
    "feature_dim": 32,
    "gaussians_per_anchor": 10,
    "quantization_size": 0.001,
    "render_mode": "RGB+ED",
    "tile_size_2dgs": 8,
    "knn_k": 4,
    "knn_chunk_size": 2048,
    "min_quantization_size": 1e-6,
}
