"""Training orchestration for the VRoom core."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field as dc_field
from random import randint
from typing import List, Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm

from ..models.facade import GaussianModel
from .loss_engine import compute_losses
from ..utils.runtime import exponential_lr_schedule


@dataclass
class TrainingConfig:
    iterations: int = 30_000
    lambda_dssim: float = 0.2
    lambda_dreg: float = 0.01
    lambda_object_loss: float = 0.1
    lambda_zero_penalty: float = 0.01
    lambda_sky_opa: float = 0.0
    lambda_opacity_entropy: float = 0.0
    lambda_normal: float = 0.0
    lambda_dist: float = 0.0
    start_depth: int = 15_000
    depth_l1_weight_init: float = 1.0
    depth_l1_weight_final: float = 0.1
    start_stat: int = 500
    update_from: int = 1_500
    update_until: int = 25_000
    update_interval: int = 100
    densify_grad_threshold: float = 0.0002
    min_opacity: float = 0.005
    success_threshold: float = 0.8
    densification: bool = True
    overlap: bool = False
    growing_type: str = "mean"
    pruning_type: str = "mean"
    update_depth: int = 3
    update_hierachy_factor: int = 4
    update_init_factor: int = 16
    position_lr_init: float = 1.6e-4
    position_lr_final: float = 1.6e-6
    position_lr_delay_mult: float = 0.01
    position_lr_max_steps: int = 30_000
    offset_lr_init: float = 1.6e-4
    offset_lr_final: float = 1.6e-6
    offset_lr_delay_mult: float = 0.01
    offset_lr_max_steps: int = 30_000
    feature_lr: float = 0.0075
    scaling_lr: float = 0.007
    rotation_lr: float = 0.002
    mlp_opacity_lr_init: float = 0.002
    mlp_opacity_lr_final: float = 0.00002
    mlp_opacity_lr_delay_mult: float = 0.01
    mlp_opacity_lr_max_steps: int = 30_000
    mlp_cov_lr_init: float = 0.004
    mlp_cov_lr_final: float = 0.00004
    mlp_cov_lr_delay_mult: float = 0.01
    mlp_cov_lr_max_steps: int = 30_000
    mlp_color_lr_init: float = 0.008
    mlp_color_lr_final: float = 0.00005
    mlp_color_lr_delay_mult: float = 0.01
    mlp_color_lr_max_steps: int = 30_000
    appearance_lr_init: float = 0.05
    appearance_lr_final: float = 0.0005
    appearance_lr_delay_mult: float = 0.01
    appearance_lr_max_steps: int = 30_000
    normal_start_iter: int = 7_000
    dist_start_iter: int = 3_000
    grad_clip_norm: Optional[float] = None


@dataclass
class PipelineConfig:
    add_prefilter: bool = True
    no_prefilter_step: int = 1000
    vis_step: int = 1000
    shuffle: bool = True
    weed_ratio: float = 0.0
    save_explicit: bool = False
    save_vis: bool = True
    save_iterations: List[int] = dc_field(default_factory=lambda: [7000, 20000, 25000, 30000])


PipeConfig = PipelineConfig


class TrainingOrchestrator:
    def __init__(self, opt: TrainingConfig, pipe: PipelineConfig, gaussians: GaussianModel, scene, output_dir: str, logger=None):
        self.opt = opt
        self.pipe = pipe
        self.gaussians = gaussians
        self.scene = scene
        self.output_dir = output_dir
        self.logger = logger
        self.depth_weight = exponential_lr_schedule(
            lr_init=opt.depth_l1_weight_init,
            lr_final=opt.depth_l1_weight_final,
            max_steps=opt.iterations,
        )

    def run(self, first_iter: int = 0):
        from gstrain.gaussian_renderer.render import prefilter_voxel, render
        self.gaussians.setup_training(self.opt, grad_clip_norm=self.opt.grad_clip_norm)
        camera_stack = None
        smoothed_loss = 0.0
        smoothed_depth = 0.0
        densify_counter = 0

        progress = tqdm(range(first_iter, self.opt.iterations), desc="Training progress", dynamic_ncols=True, smoothing=0)
        first_iter += 1
        for iteration in range(first_iter, self.opt.iterations + 1):
            self.gaussians.step_learning_rate(iteration)

            if not camera_stack:
                camera_stack = self.scene.getTrainCameras().copy()
            viewpoint = camera_stack.pop(randint(0, len(camera_stack) - 1))

            self.gaussians.set_anchor_mask(viewpoint.camera_center, viewpoint.resolution_scale)
            visible_mask = prefilter_voxel(viewpoint, self.gaussians).squeeze() if self.pipe.add_prefilter else self.gaussians._anchor_mask
            render_pkg = render(viewpoint, self.gaussians, self.pipe, self.scene.background, visible_mask)

            losses = compute_losses(render_pkg, viewpoint, self.gaussians, self.opt, iteration, self.depth_weight)
            total_loss = sum(losses.values())
            depth_loss = losses.get("depth_loss", torch.tensor(0.0, device=total_loss.device)).item()
            total_loss.backward()
            self.gaussians.clip_gradients()

            with torch.no_grad():
                smoothed_loss = 0.4 * total_loss.item() + 0.6 * smoothed_loss
                smoothed_depth = 0.4 * depth_loss + 0.6 * smoothed_depth
                if iteration % 10 == 0:
                    prediction = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    target = viewpoint.original_image.to(prediction.device)
                    if viewpoint.alpha_mask is not None:
                        alpha_mask = viewpoint.alpha_mask.to(prediction.device)
                        if alpha_mask.max() > 0.5:  # Only mask if binary opacity, not categorical labels
                            prediction = prediction * alpha_mask
                            target = target * alpha_mask
                    mse = F.mse_loss(prediction, target)
                    psnr = -10.0 * torch.log10(mse + 1e-8).item()
                    progress.set_postfix({
                        "Loss": f"{smoothed_loss:.7f}",
                        "Depth Loss": f"{smoothed_depth:.7f}",
                        "psnr": f"{psnr:.3f}",
                        "GS_num": f"{len(self.gaussians.get_anchor)}",
                        "prefilter": f"{self.pipe.add_prefilter}",
                    })
                    progress.update(10)
                if iteration == self.opt.iterations:
                    progress.close()

            # --- Periodic visualization save ---
            if self.pipe.save_vis and (iteration == 1 or (self.pipe.vis_step > 0 and iteration % self.pipe.vis_step == 0)):
                self._save_training_vis(iteration, render_pkg, viewpoint)

            if self.opt.start_stat < iteration < self.opt.update_until:
                self.gaussians.training_statis(self.opt, render_pkg, render_pkg["render"].shape[2], render_pkg["render"].shape[1])
                densify_counter += 1
                if self.opt.densification and iteration > self.opt.update_from and densify_counter % self.opt.update_interval == 0:
                    self.gaussians.run_densify(self.opt, iteration)
            elif iteration == self.opt.update_until:
                self.gaussians.clean()

            if iteration >= self.opt.iterations - self.pipe.no_prefilter_step:
                self.pipe.add_prefilter = False

            if iteration < self.opt.iterations:
                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)

            # --- Checkpoint saves ---
            save_iters = set(self.pipe.save_iterations)
            save_iters.add(self.opt.iterations)  # always save at final iteration
            if iteration in save_iters:
                self._save_checkpoint(iteration)

    train = run

    def _save_training_vis(self, iteration: int, render_pkg: dict, viewpoint):
        """Save a side-by-side rendered vs ground-truth visualization."""
        try:
            import torchvision
            vis_dir = os.path.join(self.output_dir, "vis")
            os.makedirs(vis_dir, exist_ok=True)
            rendered = torch.clamp(render_pkg["render"], 0.0, 1.0)
            gt = viewpoint.original_image.to(rendered.device)
            if viewpoint.alpha_mask is not None:
                alpha = viewpoint.alpha_mask.to(rendered.device)
                if alpha.max() > 0.5:  # Only mask if binary opacity, not categorical labels
                    rendered = rendered * alpha
                    gt = gt * alpha
            grid = torchvision.utils.make_grid([rendered, gt], nrow=2, padding=4)
            torchvision.utils.save_image(grid, os.path.join(vis_dir, f"iter_{iteration:06d}.png"))
        except Exception:
            pass  # non-critical, skip if torchvision unavailable

    def _save_checkpoint(self, iteration: int):
        checkpoint_dir = os.path.join(self.output_dir, f"point_cloud/iteration_{iteration}")
        if self.logger:
            self.logger.info(f"\n[ITER {iteration}] Saving checkpoint")
        else:
            print(f"\n[ITER {iteration}] Saving checkpoint")
        self.gaussians.save_ply(os.path.join(checkpoint_dir, "point_cloud.ply"))
        self.gaussians.save_mlp_checkpoints(checkpoint_dir)
        if self.pipe.save_explicit:
            self.gaussians.save_explicit(os.path.join(checkpoint_dir, "point_cloud_explicit.ply"))
        # Save full training state for resumption
        state_path = os.path.join(checkpoint_dir, "training_state.pth")
        try:
            torch.save(self.gaussians.capture(), state_path)
        except Exception:
            pass  # non-critical
