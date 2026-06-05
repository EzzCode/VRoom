from dataclasses import dataclass
import dataclasses

@dataclass
class ObjectTrainingConfig:
    iterations: int = 1200
    lr_scale: float = 1.0
    
    # weights
    rgb_weight: float = 1.0
    generated_rgb_scale: float = 1.0
    alpha_weight: float = 1.0
    outside_alpha_weight: float = 5.0
    depth_weight: float = 0.1
    depth_start_iter: int = 100
    depth_front_weight: float = 1.0
    depth_back_weight: float = 0.15
    depth_alpha_threshold: float = 0.35
    
    # initialization
    max_init_points: int = 20000
    colmap_init_target_points: int = 8000
    
    # densification params
    enable_densification: bool = False
    max_anchor_count: int = 20000
    densify_grad_threshold: float = 0.00005
    max_offset_abs: float = 0.45
    
    # base optim lrs (scaled by lr_scale)
    gaussian_offset_lr_init: float = 0.0040
    gaussian_offset_lr_final: float = 0.00005
    anchor_feat_lr: float = 0.0075
    anchor_scale_lr: float = 0.0015
    anchor_rot_lr: float = 0.0020
    decoder_opacity_lr_init: float = 0.0020
    decoder_opacity_lr_final: float = 0.000020
    decoder_cov_lr_init: float = 0.0040
    decoder_cov_lr_final: float = 0.0040
    decoder_color_lr_init: float = 0.0080
    decoder_color_lr_final: float = 0.000050
    
    # Pipeline specific
    generated_weight: float = 0.5
    real_weight: float = 1.0
    use_cond_cam_up: bool = True

    #runtime training parasm
    lambda_dreg: float = 0.01
    anchor_pos_lr_init: float = 0.0
    anchor_pos_lr_final: float = 0.0
    anchor_pos_lr_max_steps: int = 0
    anchor_pos_lr_delay_mult: float = 0.0
    gaussian_offset_lr_max_steps: int = 0
    gaussian_offset_lr_delay_mult: float = 0.0
    decoder_opacity_lr_max_steps: int = 0
    decoder_opacity_lr_delay_mult: float = 0.0
    decoder_cov_lr_max_steps: int = 0
    decoder_cov_lr_delay_mult: float = 0.0
    decoder_color_lr_max_steps: int = 0
    decoder_color_lr_delay_mult: float = 0.0
    densification: bool = False
    update_until: int = 0
    start_stat: int = 0
    update_from: int = 0
    update_interval: int = 0

    def get_optim_args(self) -> "ObjectTrainingConfig":
        #dataclasses.replace creates a copy of the object
        opt = dataclasses.replace(self)    
        # freeze anchor positions
        opt.anchor_pos_lr_init = opt.anchor_pos_lr_final = 0.0
        opt.anchor_pos_lr_max_steps = self.iterations
        opt.anchor_pos_lr_delay_mult = 0.01

        opt.gaussian_offset_lr_init = self.gaussian_offset_lr_init * self.lr_scale
        opt.gaussian_offset_lr_final = self.gaussian_offset_lr_final * self.lr_scale
        opt.gaussian_offset_lr_max_steps = self.iterations
        opt.gaussian_offset_lr_delay_mult = 0.01

        opt.anchor_feat_lr = self.anchor_feat_lr * self.lr_scale
        opt.anchor_scale_lr = self.anchor_scale_lr * self.lr_scale
        opt.anchor_rot_lr = self.anchor_rot_lr * self.lr_scale

        opt.decoder_opacity_lr_init = self.decoder_opacity_lr_init * self.lr_scale
        opt.decoder_opacity_lr_final = self.decoder_opacity_lr_final * self.lr_scale
        opt.decoder_opacity_lr_max_steps = self.iterations
        opt.decoder_opacity_lr_delay_mult = 0.01

        opt.decoder_cov_lr_init = self.decoder_cov_lr_init * self.lr_scale
        opt.decoder_cov_lr_final = self.decoder_cov_lr_final * self.lr_scale
        opt.decoder_cov_lr_max_steps = self.iterations
        opt.decoder_cov_lr_delay_mult = 0.01

        opt.decoder_color_lr_init = self.decoder_color_lr_init * self.lr_scale
        opt.decoder_color_lr_final = self.decoder_color_lr_final * self.lr_scale
        opt.decoder_color_lr_max_steps = self.iterations
        opt.decoder_color_lr_delay_mult = 0.01

        opt.densification = self.enable_densification
        opt.update_until = self.iterations if self.enable_densification else 0
        opt.densify_grad_threshold = self.densify_grad_threshold

        opt.start_stat = max(25, min(500, self.iterations // 8))
        opt.update_from = max(50, min(1500, self.iterations // 4))
        opt.update_interval = max(25, min(100, self.iterations // 20))
        
        return opt
