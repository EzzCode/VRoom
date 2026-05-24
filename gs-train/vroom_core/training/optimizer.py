import torch
from vroom_core.models.density import DensifcationController
from vroom_core.utils.runtime import exponential_lr_schedule


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
            {"params": [anchor_cloud.anchors_positions], "lr": args.position_lr_init * spatial_lr_scale, "name": "anchor"},
            {"params": [anchor_cloud.gaussians_offsets], "lr": args.offset_lr_init * spatial_lr_scale, "name": "offset"},
            {"params": [anchor_cloud.anchor_features], "lr": args.feature_lr, "name": "feature"},
            {"params": [anchor_cloud.anchors_log_scales], "lr": args.scaling_lr, "name": "scaling"},
            {"params": [anchor_cloud.anchors_rotations], "lr": args.rotation_lr, "name": "rotation"},
            {"params": decoder.opacity_network.parameters(), "lr": args.mlp_opacity_lr_init, "name": "opacity_head"},
            {"params": decoder.covariance_network.parameters(), "lr": args.mlp_cov_lr_init, "name": "covariance_head"},
            {"params": decoder.color_network.parameters(), "lr": args.mlp_color_lr_init, "name": "color_head"},
        ]

        self.optimizer = torch.optim.Adam(groups, lr=0.0, eps=1e-15)

        self._lr_schedulers = {
            "anchor": exponential_lr_schedule(args.position_lr_init * spatial_lr_scale, args.position_lr_final * spatial_lr_scale, lr_delay_mult=args.position_lr_delay_mult, max_steps=args.position_lr_max_steps),
            "offset": exponential_lr_schedule(args.offset_lr_init * spatial_lr_scale, args.offset_lr_final * spatial_lr_scale, lr_delay_mult=args.offset_lr_delay_mult, max_steps=args.offset_lr_max_steps),
            "opacity_head": exponential_lr_schedule(args.mlp_opacity_lr_init, args.mlp_opacity_lr_final, lr_delay_mult=args.mlp_opacity_lr_delay_mult, max_steps=args.mlp_opacity_lr_max_steps),
            "covariance_head": exponential_lr_schedule(args.mlp_cov_lr_init, args.mlp_cov_lr_final, lr_delay_mult=args.mlp_cov_lr_delay_mult, max_steps=args.mlp_cov_lr_max_steps),
            "color_head": exponential_lr_schedule(args.mlp_color_lr_init, args.mlp_color_lr_final, lr_delay_mult=args.mlp_color_lr_delay_mult, max_steps=args.mlp_color_lr_max_steps),
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
