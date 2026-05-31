from __future__ import annotations

import os

import torch
import torch.nn as nn
import torchvision
from tqdm import tqdm

from gstrain.vroom_core.utilities.utils import exponential_lr_schedule

# Optimizer group names that belong to the anchor cloud
_FIELD_GROUPS = {
    "anchors_positions",
    "gaussians_offsets",
    "anchor_features",
    "anchors_log_scales",
    "anchors_rotations",
}


def extend_optimizer(optimizer, anchor_cloud, extension_dict):
    """
    For each anchor cloud optimizer group:
      1. Extend Adam state buffers (exp_avg, exp_avg_sq) with zeros for new entries
      2. Concatenate old param with extension to form a new Parameter
      3. Replace group["params"][0] with the new Parameter
      4. Mirror the new Parameter onto the AnchorCloud attribute
    """
    for group in optimizer.param_groups:
        name = group.get("name")
        if name not in extension_dict:
            continue
        ext = extension_dict[name].to(dtype=group["params"][0].dtype)
        old_param = group["params"][0]
        state = optimizer.state.pop(old_param, {})

        for key in ("exp_avg", "exp_avg_sq"):
            if key in state:
                state[key] = torch.cat([state[key], torch.zeros_like(ext)], dim=0)

        new_param = nn.Parameter(
            torch.cat([old_param.detach(), ext], dim=0),
            requires_grad=(name != "anchors_rotations"),
        )
        group["params"][0] = new_param
        if state:
            optimizer.state[new_param] = state

        setattr(anchor_cloud, name, new_param)


def prune_optimizer(optimizer, anchor_cloud, keep_mask):
    """
    For each anchor cloud optimizer group:
      1. Slice Adam state buffers to surviving rows
      2. Create new Parameter from surviving rows
      3. Replace group["params"][0] and the AnchorCloud attribute
    """
    for group in optimizer.param_groups:
        name = group.get("name")
        if name not in _FIELD_GROUPS:
            continue
        old_param = group["params"][0]
        state = optimizer.state.pop(old_param, {})

        for key in ("exp_avg", "exp_avg_sq"):
            if key in state:
                state[key] = state[key][keep_mask]

        new_param = nn.Parameter(
            old_param[keep_mask].detach().clone(),
            requires_grad=(name != "anchors_rotations"),
        )
        group["params"][0] = new_param
        if state:
            optimizer.state[new_param] = state

        setattr(anchor_cloud, name, new_param)


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


def state_snapshot(anchor_cloud, spatial_lr_scale, optimizer, opacity_network, covariance_network, color_network) -> dict:
    """
    Captures a snapshot of all trainable state for checkpointing.
    """
    return {
        "anchor_postions": anchor_cloud.anchors_positions.detach(),
        "gaussian_offsests": anchor_cloud.gaussians_offsets.detach(),
        "anchor_feature": anchor_cloud.anchor_features.detach(),
        "anchor_log_scales": anchor_cloud.anchors_log_scales.detach(),
        "anchor_rotation": anchor_cloud.anchors_rotations.detach(),
        "semantic_labels": anchor_cloud.semantic_labels,
        "spatial_lr_scale": spatial_lr_scale,
        "optimizer_state": optimizer.state_dict(),
        "opacity_network": opacity_network.state_dict(),
        "covariance_network": covariance_network.state_dict(),
        "color_network": color_network.state_dict(),
    }


def visualize(step_output, iteration, camera_view, output_dir):
    """
    Visualizes the real image against the rendered image from the rasterizer side by side.
    """
    rendered_image = step_output["rendered_image"]
    real_image = camera_view.original_image.to(
        rendered_image.device
    )  # two images have to be on the same device or torch will crash

    grid = torchvision.utils.make_grid([real_image, rendered_image], nrow=2)
    vis_dir = os.path.join(output_dir, "visualization")
    os.makedirs(vis_dir, exist_ok=True)
    torchvision.utils.save_image(grid, os.path.join(vis_dir, f"iteration_{iteration}.png"))


def save_checkpoint(
    step_output,
    iteration,
    camera_view,
    output_dir,
    checkpoint_manager,
    logger,
    anchor_cloud,
    spatial_lr_scale,
    optimizer,
    opacity_network,
    covariance_network,
    color_network,
    gaussian_type,
    render_mode,
    tile_size_2dgs,
):
    """
    Saves a checkpoint for a given iteration.
    """
    logger.info(f"Saving checkpoint at iteration {iteration}")
    # Per iteration directory
    iter_dir = os.path.join(output_dir, "checkpoints", f"iter_{iteration}")
    os.makedirs(iter_dir, exist_ok=True)
    checkpoint_manager.save_anchor_cloud(
        path=os.path.join(iter_dir, "anchor_cloud.ply")
    )
    checkpoint_manager.save_decoder(
        path=iter_dir,
        gaussian_type=gaussian_type,
        render_mode=render_mode,
        tile_size_2dgs=tile_size_2dgs,
    )
    torch.save(
        state_snapshot(anchor_cloud, spatial_lr_scale, optimizer, opacity_network, covariance_network, color_network),
        os.path.join(iter_dir, "state.pth"),
    )


def get_progress_bar(num_iterations: int) -> tqdm:
    """Creates and returns a tqdm progress bar for training."""
    return tqdm(
        range(1, num_iterations + 1),
        desc="Training progress",
        dynamic_ncols=True,
    )


def update_progress_bar(progress_bar: tqdm, step_output: dict, num_anchors: int) -> None:
    """Updates the postfix metrics on the tqdm progress bar."""
    total_loss = step_output["total_loss"]
    psnr = step_output["psnr"]
    progress_bar.set_postfix({
        "Loss": f"{total_loss:.4f}",
        "PSNR": f"{psnr:.2f}",
        "Anchors": num_anchors
    })
