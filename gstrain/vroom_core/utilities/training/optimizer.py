import torch
from gstrain.vroom_core.core.model.density import DensifcationController
from gstrain.vroom_core.utilities.utils.runtime import exponential_lr_schedule


class Optimizer:
    def __init__(self, optimizer_configs, densifier):
        self.opt_configs = optimizer_configs
        self.densifier = densifier

    def setup(self):
        args = self.opt_configs["args"]
        spatial_lr_scale = self.opt_configs["spatial_lr_scale"]
        anchor_cloud = self.opt_configs["anchor_cloud"]
        decoder = self.opt_configs["decoder"]

        groups = [
            {"params": [anchor_cloud.anchors_positions], "lr": args.anchor_pos_lr_init * spatial_lr_scale, "name": "anchors_positions"},
            {"params": [anchor_cloud.gaussians_offsets], "lr": args.gaussian_offset_lr_init * spatial_lr_scale, "name": "gaussians_offsets"},
            {"params": [anchor_cloud.anchor_features], "lr": args.anchor_feat_lr, "name": "anchor_features"},
            {"params": [anchor_cloud.anchors_log_scales], "lr": args.anchor_scale_lr, "name": "anchors_log_scales"},
            {"params": [anchor_cloud.anchors_rotations], "lr": args.anchor_rot_lr, "name": "anchors_rotations"},
            {"params": decoder.opacity_network.parameters(), "lr": args.decoder_opacity_lr_init, "name": "opacity_head"},
            {"params": decoder.covariance_network.parameters(), "lr": args.decoder_cov_lr_init, "name": "covariance_head"},
            {"params": decoder.color_network.parameters(), "lr": args.decoder_color_lr_init, "name": "color_head"},
        ]

        self.optimizer = torch.optim.Adam(groups, lr=0.0, eps=1e-15)

        self._lr_schedulers = {
            "anchors_positions": exponential_lr_schedule(args.anchor_pos_lr_init * spatial_lr_scale, args.anchor_pos_lr_final * spatial_lr_scale, lr_delay_mult=args.anchor_pos_lr_delay_mult, max_steps=args.anchor_pos_lr_max_steps),
            "gaussians_offsets": exponential_lr_schedule(args.gaussian_offset_lr_init * spatial_lr_scale, args.gaussian_offset_lr_final * spatial_lr_scale, lr_delay_mult=args.gaussian_offset_lr_delay_mult, max_steps=args.gaussian_offset_lr_max_steps),
            "opacity_head": exponential_lr_schedule(args.decoder_opacity_lr_init, args.decoder_opacity_lr_final, lr_delay_mult=args.decoder_opacity_lr_delay_mult, max_steps=args.decoder_opacity_lr_max_steps),
            "covariance_head": exponential_lr_schedule(args.decoder_cov_lr_init, args.decoder_cov_lr_final, lr_delay_mult=args.decoder_cov_lr_delay_mult, max_steps=args.decoder_cov_lr_max_steps),
            "color_head": exponential_lr_schedule(args.decoder_color_lr_init, args.decoder_color_lr_final, lr_delay_mult=args.decoder_color_lr_delay_mult, max_steps=args.decoder_color_lr_max_steps),
        }

        self.densifier.reset_state()

    def zero_grad(self):
        self.optimizer.zero_grad(set_to_none=True)

    def step(self):
        self.optimizer.step()

    def step_learning_rate(self, iteration: int):
        for group in self.optimizer.param_groups:
            scheduler = self._lr_schedulers.get(group["name"])
            if scheduler is not None:
                group["lr"] = scheduler(iteration)

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    @property
    def state(self):
        return self.optimizer.state
