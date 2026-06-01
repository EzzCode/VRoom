from dataclasses import dataclass

@dataclass
class ObjectTrainingConfig:
    # Training Loop
    iterations: int = 1200
    lr_scale: float = 1.0
    
    # Loss Weights
    rgb_weight: float = 1.0
    generated_rgb_scale: float = 1.0
    alpha_weight: float = 1.0
    outside_alpha_weight: float = 5.0
    depth_weight: float = 0.1
    depth_start_iter: int = 100
    depth_front_weight: float = 1.0
    depth_back_weight: float = 0.15
    depth_alpha_threshold: float = 0.35
    
    # Point Cloud & Initialization
    max_init_points: int = 20000
    colmap_init_target_points: int = 8000
    
    # Densification & GS Params
    enable_densification: bool = False
    max_anchor_count: int = 20000
    densify_grad_threshold: float = 0.00005
    max_offset_abs: float = 0.45
    
    # Base Optimization Learning Rates (will be scaled by lr_scale)
    offset_lr_init: float = 0.0040
    offset_lr_final: float = 0.00005
    feature_lr: float = 0.0075
    scaling_lr: float = 0.0015
    rotation_lr: float = 0.0020
    mlp_opacity_lr_init: float = 0.0020
    mlp_opacity_lr_final: float = 0.000020
    mlp_cov_lr: float = 0.0040
    mlp_color_lr_init: float = 0.0080
    mlp_color_lr_final: float = 0.000050
    
    # Pipeline specific
    generated_weight: float = 1.0
    real_weight: float = 1.0
    use_cond_cam_up: bool = True
