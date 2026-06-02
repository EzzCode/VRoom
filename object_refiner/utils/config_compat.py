"""
config_compat.py

A compatibility layer to adapt old configuration keys (used by the facade/wrapper)
to the modern native gstrain API names.

This can be deleted once the rest of the pipeline is migrated to the new gstrain API.
"""
from typing import Any, Dict

def adapt_legacy_model_config(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Translates model instantiation parameters from older config formats to
    the new gstrain API names.
    """
    mapping = {
        "n_offsets": "gaussians_per_anchor",
        "feat_dim": "feature_dim",
    }
    
    adapted = {}
    for k, v in kwargs.items():
        if k in mapping:
            adapted[mapping[k]] = v
        else:
            adapted[k] = v
            
    return adapted

def adapt_legacy_training_config(opt: Any) -> Any:
    """
    Translates parameters from older training config formats to the modern 
    native gstrain attribute names expected by the Optimizer.
    """
    # Position
    opt.anchor_pos_lr_init = getattr(opt, "position_lr_init", 0.0)
    opt.anchor_pos_lr_final = getattr(opt, "position_lr_final", 0.0)
    opt.anchor_pos_lr_max_steps = getattr(opt, "position_lr_max_steps", getattr(opt, "iterations", 0))
    opt.anchor_pos_lr_delay_mult = getattr(opt, "position_lr_delay_mult", 0.01)

    # Offset
    opt.gaussian_offset_lr_init = getattr(opt, "offset_lr_init", 0.0)
    opt.gaussian_offset_lr_final = getattr(opt, "offset_lr_final", 0.0)
    opt.gaussian_offset_lr_max_steps = getattr(opt, "offset_lr_max_steps", getattr(opt, "iterations", 0))
    opt.gaussian_offset_lr_delay_mult = getattr(opt, "offset_lr_delay_mult", 0.01)

    # Features, scale, rotation
    opt.anchor_feat_lr = getattr(opt, "feature_lr", 0.0)
    opt.anchor_scale_lr = getattr(opt, "scaling_lr", 0.0)
    opt.anchor_rot_lr = getattr(opt, "rotation_lr", 0.0)

    # Decoder Opacity
    opt.decoder_opacity_lr_init = getattr(opt, "mlp_opacity_lr_init", 0.0)
    opt.decoder_opacity_lr_final = getattr(opt, "mlp_opacity_lr_final", 0.0)
    opt.decoder_opacity_lr_max_steps = getattr(opt, "mlp_opacity_lr_max_steps", getattr(opt, "iterations", 0))
    opt.decoder_opacity_lr_delay_mult = getattr(opt, "mlp_opacity_lr_delay_mult", 0.01)

    # Decoder Covariance
    opt.decoder_cov_lr_init = getattr(opt, "mlp_cov_lr_init", getattr(opt, "mlp_cov_lr", 0.0))
    opt.decoder_cov_lr_final = getattr(opt, "mlp_cov_lr_final", getattr(opt, "mlp_cov_lr", 0.0))
    opt.decoder_cov_lr_max_steps = getattr(opt, "mlp_cov_lr_max_steps", getattr(opt, "iterations", 0))
    opt.decoder_cov_lr_delay_mult = getattr(opt, "mlp_cov_lr_delay_mult", 0.01)

    # Decoder Color
    opt.decoder_color_lr_init = getattr(opt, "mlp_color_lr_init", 0.0)
    opt.decoder_color_lr_final = getattr(opt, "mlp_color_lr_final", 0.0)
    opt.decoder_color_lr_max_steps = getattr(opt, "mlp_color_lr_max_steps", getattr(opt, "iterations", 0))
    opt.decoder_color_lr_delay_mult = getattr(opt, "mlp_color_lr_delay_mult", 0.01)

    return opt
