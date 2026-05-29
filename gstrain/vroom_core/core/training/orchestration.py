import os
import torch
import torchvision
from tqdm import tqdm
from gstrain.vroom_core.core.model.density import DensifcationController
from gstrain.vroom_core.utilities.gaussian_renderer.render import prefilter_voxel, render
from gstrain.vroom_core.utilities.training.optimizer import Optimizer
from gstrain.vroom_core.core.training.loss_engine import LossEngine
from gstrain.vroom_core.utilities.utils.runtime import exponential_lr_schedule
from gstrain.vroom_core.utilities.utils.checkpoints import CheckpointManager


def prepare_gaussian_space_props(
    anchor_cloud,
    visible_anchors_mask,
    negative_opacity_filter,
    rotations_pred,
):
    """Calculate the positions of the gaussians and normalize their rotations"""
    gaussian_positions = anchor_cloud.instantiate_gaussian_positions(
        visible_mask=visible_anchors_mask,
        negative_opacity_filter=negative_opacity_filter
    )
    normalized_rotations = torch.nn.functional.normalize(rotations_pred, dim=-1)
    return gaussian_positions, normalized_rotations


class TrainingOrchestrator:
    def __init__(self, configs, scene, logger):
        self.optmizer_configs = configs["optimization"]
        self.pipeline_configs = configs["pipeline"]
        self.densifier_configs = configs["densifier"]
        self.rendering_configs = configs.get("rendering", {})
        self.output_dir = self.pipeline_configs.get("output_dir", ".")

        self.anchor_cloud = self.optmizer_configs["anchor_cloud"]
        self.decoder = self.optmizer_configs["decoder"]
        self.gaussian_type = self.rendering_configs.get("gaussian_type", "3D")
        self.render_mode = self.rendering_configs.get("render_mode", "RGB+ED")
        self.tile_size_2dgs = self.rendering_configs.get("tile_size_2dgs", 8)
        self.bg_color = self.pipeline_configs.get("bg_color")
        self.dataloader = scene.getTrainCameras()
        self.visualization_interval = self.pipeline_configs.get("visualization_interval", 500)
        self.logger = logger
        self.spatial_lr_scale = scene.cameras_extent

        self.opacity_network = self.decoder.opacity_network
        self.covariance_network = self.decoder.covariance_network
        self.color_network = self.decoder.color_network
        self.CheckPointManager = CheckpointManager(self.anchor_cloud, self.decoder)

  
        self.densifier = DensifcationController(
            voxel_size=self.anchor_cloud.voxel_size,
            anchor_cloud=self.anchor_cloud,
            optimizer=None,
            num_gaussians_per_anchor=self.decoder.number_gaussians_per_anchor,
        )
        self.optimizer = Optimizer(self.optmizer_configs, self.densifier)
        self.densifier.optimizer = self.optimizer

    def _clip_gradients(self):  # to avoid exploding gradients
        params = list(self.anchor_cloud.parameters()) + list(self.decoder.parameters())
        torch.nn.utils.clip_grad_norm_(
            params, self.optmizer_configs["args"].max_grad_norm
        )

    def train_step(self, camera_view, iteration, width, height):
        self.optimizer.zero_grad()
        self.optimizer.step_learning_rate(iteration)

        visible_anchors_mask = prefilter_voxel(camera_view, self.anchor_cloud, self.gaussian_type)
        decoded_output = self.decoder.forward_pass(
            anchor_cloud=self.anchor_cloud,
            visible_anchors_mask=visible_anchors_mask,
            camera=camera_view,
        )

        gaussian_positions, normalized_rotations = prepare_gaussian_space_props(
            anchor_cloud=self.anchor_cloud,
            visible_anchors_mask=visible_anchors_mask,
            negative_opacity_filter=decoded_output["negative_opacity_filter"],
            rotations_pred=decoded_output["rotations"],
        )

        semantics_pred = None
        if self.anchor_cloud.semantic_labels is not None and self.anchor_cloud.semantic_manager is not None:
            semantics_pred = self.anchor_cloud.semantic_manager.instantiate_semantics(
                semantic_labels=self.anchor_cloud.semantic_labels,
                visible_anchors_mask=visible_anchors_mask,
                negative_opacity_filter=decoded_output["negative_opacity_filter"],
                gaussians_per_anchor=self.decoder.number_gaussians_per_anchor,
            )

        rasterizer_output = render(
            viewpoint_camera=camera_view,
            decoded_output=decoded_output,
            gaussian_positions=gaussian_positions,
            normalized_rotations=normalized_rotations,
            bg_color=self.bg_color,
            gaussian_type=self.gaussian_type,
            render_mode=self.render_mode,
            tile_size_2dgs=self.tile_size_2dgs,
            semantics=semantics_pred,
        )
        rasterizer_output["visible_anchors_mask"] = visible_anchors_mask
        rasterizer_output["negative_opacity_filter"] = decoded_output["negative_opacity_filter"]
        loss_engine = LossEngine(self.anchor_cloud.semantic_manager)
        losses = loss_engine.compute_total_losses(
            render_pkg=rasterizer_output,
            viewpoint_cam=camera_view,
            anchor_cloud=self.anchor_cloud,
            opt=self.optmizer_configs["args"],
            iteration=iteration,
        )
        total_loss = sum(losses.values())
        losses["total"] = total_loss
        losses["total"].backward()
        self._clip_gradients()
        self.optimizer.step()

        # Read the 2D points gradients after backward 
        # and detach so we don't keep the full graph in memory
        rendered_2d_points = rasterizer_output["rendered_2d_points"]
        if rendered_2d_points is not None and rendered_2d_points.grad is not None:
            points_grad_detached = rendered_2d_points.grad.detach().clone()
        else:
            points_grad_detached = None
        # densification
        if (
            iteration >= self.densifier_configs["desification_start"]
            and iteration <= self.densifier_configs["desification_end"]
        ):
            self.densifier.update_densification_state(
                visible_anchors_mask,
                rasterizer_output["negative_opacity_filter"],
                rasterizer_output["opacity"],
                points_grad_detached,
                width,
                height,
            )
            if iteration % self.densifier_configs["densification_interval"] == 0:
                self.densifier.growing_operation()
                self.densifier.pruning_operation(opacity_threshold=self.densifier_configs.get("min_opacity", 0.005))
                self.densifier.reset_state()
        else:
            self.densifier.reset_state()
            torch.cuda.empty_cache()

        with torch.no_grad():
            psnr = self.compute_psnr(rasterizer_output, camera_view.original_image)

        # Detach rendered image for visualisation
        rendered_image_detached = rasterizer_output["render"].detach()
        total_loss_value = losses["total"].item()
        losses_copy = {k: v.item() for k, v in losses.items()}

        return {
            "losses": losses_copy,
            "total_loss": total_loss_value,
            "psnr": psnr,
            "rendered_image": rendered_image_detached,
        }

    def compute_psnr(self, rasterizer_output, original_image):
        prediction = rasterizer_output["render"]
        target = original_image.to(prediction.device)
        mse = torch.mean((prediction - target) ** 2)
        if mse == 0:
            return float("inf")
        return 20 * torch.log10(1.0 / torch.sqrt(mse))

    def train(self, cameras):
        width, height = (
            cameras[0].image_width,
            cameras[0].image_height,
        )
        self.optimizer.setup()
        self.densifier.reset_state()
        cameras_iterator = iter(self.dataloader)

        num_iterations = self.optmizer_configs["args"].num_iterations
        progress_bar = tqdm(
            range(1, num_iterations + 1),
            desc="Training progress",
            dynamic_ncols=True,
        )

        for iteration in progress_bar:
            try:
                camera_view = next(cameras_iterator)
            except StopIteration:
                cameras_iterator = iter(
                    self.dataloader
                )  # restart the iterator if we ran out of images
                camera_view = next(cameras_iterator)

            step_output = self.train_step(camera_view, iteration, width, height)

            if iteration % 10 == 0:
                total_loss = step_output["total_loss"]
                psnr = step_output["psnr"]
                num_anchors = self.anchor_cloud.num_anchors
                progress_bar.set_postfix({
                    "Loss": f"{total_loss:.4f}",
                    "PSNR": f"{psnr:.2f}",
                    "Anchors": num_anchors
                })

            if iteration % self.visualization_interval == 0:
                self._visualize(step_output, iteration, camera_view)

            if iteration in self.pipeline_configs["save_iterations"]:
                self._save_checkpoint(step_output, iteration, camera_view)

            del step_output
            torch.cuda.empty_cache()


    def _state_snapshot(self):
        state = {
            "anchor_postions": self.anchor_cloud.anchors_positions.detach(),
            "gaussian_offsests": self.anchor_cloud.gaussians_offsets.detach(),
            "anchor_feature": self.anchor_cloud.anchor_features.detach(),
            "anchor_log_scales": self.anchor_cloud.anchors_log_scales.detach(),
            "anchor_rotation": self.anchor_cloud.anchors_rotations.detach(),
            "semantic_labels": self.anchor_cloud.semantic_labels,
            "spatial_lr_scale": self.spatial_lr_scale,
            "optimizer_state": self.optimizer.state_dict(),
            "opacity_network": self.opacity_network.state_dict(),
            "covariance_network": self.covariance_network.state_dict(),
            "color_network": self.color_network.state_dict(),
        }
        return state

    def _visualize(self, step_output, iteration, camera_view):
        """
        Visualizes the real image agianst the rendered image from the rasterizer side by side
        """
        rendered_image = step_output["rendered_image"]
        real_image = camera_view.original_image.to(
            rendered_image.device
        )  # two images have to be on the same device or torch will crash

        # Guard against NaN/Inf produced by exploding gradients or bad scales
        if not torch.isfinite(rendered_image).all():
            self.logger.warning(
                f"[iteration {iteration}] Rendered image contains NaN/Inf - "
                "clamping for visualization. Check volumetric loss, scales, and gradients."
            )
            rendered_image = torch.nan_to_num(rendered_image, nan=0.0, posinf=1.0, neginf=0.0)

        # Clamp both images to valid display range
        rendered_image = rendered_image.clamp(0.0, 1.0)
        real_image = real_image.clamp(0.0, 1.0)

        grid = torchvision.utils.make_grid([real_image, rendered_image], nrow=2)
        vis_dir = os.path.join(self.output_dir, "visualization")
        os.makedirs(vis_dir, exist_ok=True)
        torchvision.utils.save_image(grid, os.path.join(vis_dir, f"iteration_{iteration}.png"))

    def _save_checkpoint(self, step_output, iteration, camera_view):
        """
        Saves a checkpoint for a given iteration.
        Each call creates its own subdirectory (iter_<N>) so that successive
        checkpoints don't overwrite each other's MLP .pt files.
        """
        self.logger.info(f"Saving checkpoint at iteration {iteration}")
        # Per iteration directory keeps
        iter_dir = os.path.join(self.output_dir, "checkpoints", f"iter_{iteration}")
        os.makedirs(iter_dir, exist_ok=True)
        self.CheckPointManager.save_anchor_cloud(
            path=os.path.join(iter_dir, "anchor_cloud.ply")
        )
        self.CheckPointManager.save_decoder(
            path=iter_dir,
            gaussian_type=self.gaussian_type,
            render_mode=self.render_mode,
            tile_size_2dgs=self.tile_size_2dgs,
        )
        torch.save(self._state_snapshot(), os.path.join(iter_dir, "state.pth"))
