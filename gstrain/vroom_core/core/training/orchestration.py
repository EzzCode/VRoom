import torch
from gstrain.vroom_core.core.model.density import DensifcationController
from gstrain.vroom_core.utilities.render import apply_frustum_culling, render
from gstrain.vroom_core.utilities.training import (
    Optimizer,
    visualize,
    save_checkpoint as _save_checkpoint_util,
    get_progress_bar,
    update_progress_bar,
)
from gstrain.vroom_core.core.training.loss_engine import LossEngine
from gstrain.vroom_core.utilities.utils import (
    CheckpointManager,
)


def prepare_gaussian_space_props(
    anchor_cloud,
    visible_anchors_mask,
    negative_opacity_filter,
    rotations_pred,
):
    """Calculate the positions of the gaussians and normalize their rotations"""
    gaussian_positions = anchor_cloud.instantiate_gaussian_positions(
        visible_mask=visible_anchors_mask,
        negative_opacity_filter=negative_opacity_filter,
    )
    normalized_rotations = torch.nn.functional.normalize(rotations_pred, dim=-1)
    return gaussian_positions, normalized_rotations


def instantiate_semantics(
    semantic_manager,
    semantic_labels,
    visible_anchors_mask,
    negative_opacity_filter,
    gaussians_per_anchor,
):
    """
    Map visible anchor labels to one hot encodings for visible Gaussians
    """
    visible_labels = semantic_labels[visible_anchors_mask]
    visible_label_indices = semantic_manager.build_lookup_table(visible_labels)
    visible_one_hot = semantic_manager.one_hot_encode(visible_label_indices).float()
    expanded_one_hot = visible_one_hot.unsqueeze(1).expand(-1, gaussians_per_anchor, -1)
    return expanded_one_hot[negative_opacity_filter]


class TrainingOrchestrator:
    def __init__(self, configs, scene, logger):
        self.optmizer_configs = configs["optimization"]
        self.pipeline_configs = configs["pipeline"]
        self.densifier_configs = configs["densifier"]
        self.rendering_configs = configs.get("rendering", {})
        self.output_dir = self.pipeline_configs.get("output_dir", ".")

        self.anchor_cloud = self.optmizer_configs["anchor_cloud"]
        self.decoder = self.optmizer_configs["decoder"]
        self.gaussian_type = self.rendering_configs.get("gaussian_type", "2D")
        self.tile_Size = self.rendering_configs.get("tile_Size")
        self.background_color = self.pipeline_configs.get("background_color")
        self.dataloader = scene.getTrainCameras()
        self.visualization_interval = self.pipeline_configs.get(
            "visualization_interval", 500
        )
        self.logger = logger
        self.spatial_lr_scale = scene.cameras_extent

        self.opacity_network = self.decoder.opacity_network
        self.covariance_network = self.decoder.covariance_network
        self.color_network = self.decoder.color_network
        self.CheckPointManager = CheckpointManager(self.anchor_cloud, self.decoder)

        self.densifier = DensifcationController(
            quantization_size=self.anchor_cloud.quantization_size,
            anchor_cloud=self.anchor_cloud,
            optimizer=None,
            num_gaussians_per_anchor=self.decoder.number_gaussians_per_anchor,
            gradient_threshold=self.densifier_configs.get("gradient_threshold"),
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
        within_densification_range = (
            iteration >= self.densifier_configs["desification_start"]
            and iteration <= self.densifier_configs["desification_end"]
        )
        visible_anchors_mask = apply_frustum_culling(
            camera_view, self.anchor_cloud, self.gaussian_type
        )
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
        if (
            self.anchor_cloud.semantic_labels is not None
            and self.anchor_cloud.semantic_manager is not None
        ):
            semantics_pred = instantiate_semantics(
                semantic_manager=self.anchor_cloud.semantic_manager,
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
            background_color=self.background_color,
            gaussian_type=self.gaussian_type,
            tile_Size=self.tile_Size,
            semantics=semantics_pred,
        )
        rasterizer_output["visible_anchors_mask"] = visible_anchors_mask
        rasterizer_output["negative_opacity_filter"] = decoded_output[
            "negative_opacity_filter"
        ]
        loss_engine = LossEngine(self.anchor_cloud.semantic_manager)
        losses = loss_engine.compute_total_losses(
            rasterizer_output=rasterizer_output,
            viewpoint_cam=camera_view,
            anchor_cloud=self.anchor_cloud,
            optimizer_configs=self.optmizer_configs["args"],
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
        if within_densification_range:
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
                self.densifier.pruning_operation(
                    opacity_threshold=self.densifier_configs.get("min_opacity")
                )
                self.densifier.reset_state()
        else:
            self.densifier.reset_state()
            torch.cuda.empty_cache()

        # Detach rendered image for visualisation
        rendered_image_detached = rasterizer_output["render"].detach()
        total_loss_value = losses["total"].item()

        return {
            "total_loss": total_loss_value,
            "rendered_image": rendered_image_detached,
        }

    def train(self, cameras):
        width, height = (
            cameras[0].image_width,
            cameras[0].image_height,
        )
        self.optimizer.setup()
        self.densifier.reset_state()
        cameras_iterator = iter(self.dataloader)

        num_iterations = self.optmizer_configs["args"].num_iterations
        progress_bar = get_progress_bar(num_iterations)

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
                update_progress_bar(
                    progress_bar, step_output, self.anchor_cloud.num_anchors
                )

            if iteration % self.visualization_interval == 0:
                visualize(step_output, iteration, camera_view, self.output_dir)

            if iteration in self.pipeline_configs["save_iterations"]:
                _save_checkpoint_util(
                    step_output=step_output,
                    iteration=iteration,
                    camera_view=camera_view,
                    output_dir=self.output_dir,
                    checkpoint_manager=self.CheckPointManager,
                    logger=self.logger,
                    anchor_cloud=self.anchor_cloud,
                    spatial_lr_scale=self.spatial_lr_scale,
                    optimizer=self.optimizer,
                    opacity_network=self.opacity_network,
                    covariance_network=self.covariance_network,
                    color_network=self.color_network,
                    gaussian_type=self.gaussian_type,
                    render_mode="RGB+ED",
                    tile_Size=self.tile_Size,
                )

            del step_output
            torch.cuda.empty_cache()
