"""
gstrain_bridge.py — thin composition layer for object_refiner.

Instead of reimplementing anything, this module composes the actual gstrain
classes (AnchorCloud, GaussianDecoder, CheckpointManager, DensifcationController,
Optimizer, SemanticsManager) into a single ``VRoomModel`` that presents the
attributes object_refiner's trainer, scene_analysis and debug scripts expect.

Nothing is duplicated from gstrain; every operation delegates to the real class.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from gstrain.vroom_core.core.model.anchor_field import AnchorCloud, AnchorCloudData
from gstrain.vroom_core.utilities.gaussian_decoder import GaussianDecoder
from gstrain.vroom_core.utilities.utils import CheckpointManager, SemanticsManager
from gstrain.vroom_core.core.model.density import DensifcationController
from gstrain.vroom_core.utilities.training import Optimizer


# ---------------------------------------------------------------------------
# SemanticCodec — thin alias for SemanticsManager (same class, different name)
# ---------------------------------------------------------------------------

class SemanticCodec(SemanticsManager):
    """Alias kept so existing code using SemanticCodec.from_labels() still works."""

    @classmethod
    def from_labels(cls, labels: torch.Tensor) -> "SemanticCodec":
        return cls(labels)


# ---------------------------------------------------------------------------
# VRoomModel — composites real gstrain objects; no reimplementation
# ---------------------------------------------------------------------------

class VRoomModel(nn.Module):
    """
    Composite model for object_refiner that holds native gstrain objects.

    Parameters match the old GaussianModel API so existing callers need no
    changes except importing VRoomModel instead.
    """

    def __init__(
        self,
        gs_attr: str,
        feature_dim: int,
        view_dim: int,
        appearance_dim: int,
        gaussians_per_anchor: int,
        voxel_size: float,
        render_mode: str,
        tile_size_2dgs: int,
    ):
        super().__init__()
        self.device = "cuda"

        # Config attrs used by renderer / wrapper
        self.gs_attr = gs_attr
        self.feature_dim = feature_dim
        self.view_dim = view_dim
        self.appearance_dim = appearance_dim
        self.gaussians_per_anchor = gaussians_per_anchor
        self.voxel_size = voxel_size
        self.render_mode = render_mode
        self.tile_size_2dgs = tile_size_2dgs
        self.spatial_lr_scale = 1.0

        # Extra attrs referenced by trainer
        self.weed_ratio: float = 0.0
        self.explicit_gs: bool = False
        self.id_encoder: Optional[SemanticsManager] = None

        # Real gstrain objects
        self.anchor_cloud = AnchorCloud(
            gaussians_per_anchor=gaussians_per_anchor,
            feature_dim=feature_dim,
            voxel_size=voxel_size,
            device=self.device,
        )
        self.decoder = GaussianDecoder(
            feature_dim=feature_dim,
            anchor_cloud=self.anchor_cloud,
        ).to(self.device)
        self.checkpoint_manager = CheckpointManager(self.anchor_cloud, self.decoder)

        self.optimizer: Optional[torch.optim.Optimizer] = None
        self._opt_wrapper: Optional[Optimizer] = None
        self.densifier: Optional[DensifcationController] = None
        self._anchor_mask: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Convenience properties (delegate to anchor_cloud)
    # ------------------------------------------------------------------

    @property
    def label_ids(self) -> Optional[torch.Tensor]:
        return self.anchor_cloud.semantic_labels

    @property
    def _scaling(self) -> torch.Tensor:
        return self.anchor_cloud.anchors_log_scales

    @property
    def _anchor(self) -> torch.Tensor:
        return self.anchor_cloud.anchors_positions

    @property
    def get_anchor(self) -> torch.Tensor:
        return self.anchor_cloud.anchors_positions

    @property
    def _offset(self) -> torch.Tensor:
        return self.anchor_cloud.gaussians_offsets

    # ------------------------------------------------------------------
    # Mask helpers
    # ------------------------------------------------------------------

    def set_anchor_mask(self, camera_center: torch.Tensor, resolution_scale: float) -> None:
        self._anchor_mask = torch.ones(
            self.anchor_cloud.anchors_positions.shape[0],
            dtype=torch.bool,
            device=self.device,
        )

    # ------------------------------------------------------------------
    # Checkpoint I/O — delegates to CheckpointManager
    # ------------------------------------------------------------------

    def load_ply(self, path: str) -> None:
        payload = self.checkpoint_manager.load_anchor_field(path)

        if payload["log_scaling"].numel() > 0:
            voxel_size = float(torch.exp(payload["log_scaling"][:, :3]).mean().item())
        else:
            voxel_size = self.voxel_size if self.voxel_size > 0 else 1.0

        seeds = AnchorCloudData(
            anchors_positions=payload["anchor"],
            gaussians_offsets=payload["offset"],
            anchor_features=payload["feature"],
            anchors_log_scales=payload["log_scaling"],
            anchors_rotations=payload["rotation"],
            labels=payload["labels"],
            semantic_manager=None
            if payload["labels"] is None
            else SemanticsManager(torch.unique(payload["labels"].view(-1))),
            voxel_size=voxel_size,
        )
        self.anchor_cloud.set_anchors_cloud(seeds)

    def load_mlp_checkpoints(self, path: str) -> None:
        self.checkpoint_manager.load_decoder(path)

    def save_ply(self, path: str) -> None:
        self.checkpoint_manager.save_anchor_cloud(path)

    def save_mlp_checkpoints(self, path: str) -> None:
        self.checkpoint_manager.save_decoder(
            path,
            gaussian_type=self.gs_attr,
            render_mode=self.render_mode,
            tile_size_2dgs=self.tile_size_2dgs,
        )

    # ------------------------------------------------------------------
    # Eval mode
    # ------------------------------------------------------------------

    def set_eval(self) -> None:
        self.decoder.eval()
        self.anchor_cloud.eval()

    def eval(self) -> "VRoomModel":  # type: ignore[override]
        self.set_eval()
        return self

    # ------------------------------------------------------------------
    # No-op stubs kept for API compat
    # ------------------------------------------------------------------

    def set_appearance(self, num_cameras: int) -> None:
        pass  # appearance embedding not used in this pipeline

    # ------------------------------------------------------------------
    # Anchor initialisation
    # ------------------------------------------------------------------

    def initialize_anchors(self, pcd, spatial_extent: float, logger=None) -> None:
        self.spatial_lr_scale = spatial_extent
        self.anchor_cloud.initialize_anchors(pcd)

    # ------------------------------------------------------------------
    # Optimizer / training setup
    # Maps old optimizer kwarg names (position_lr_init etc.) → new names
    # ------------------------------------------------------------------

    def training_setup(self, opt) -> None:

        configs = {
            "optimization": {
                "args": opt,
                "spatial_lr_scale": self.spatial_lr_scale,
                "anchor_cloud": self.anchor_cloud,
                "decoder": self.decoder,
            }
        }
        self.densifier = DensifcationController(
            voxel_size=self.anchor_cloud.voxel_size,
            anchor_cloud=self.anchor_cloud,
            optimizer=None,
            num_gaussians_per_anchor=self.decoder.number_gaussians_per_anchor,
        )
        self._opt_wrapper = Optimizer(configs["optimization"], self.densifier)
        self._opt_wrapper.setup()
        self.densifier.optimizer = self._opt_wrapper
        self.optimizer = self._opt_wrapper.optimizer

    def update_learning_rate(self, iteration: int) -> None:
        if self._opt_wrapper is not None:
            self._opt_wrapper.step_learning_rate(iteration)

    # ------------------------------------------------------------------
    # Densification helpers (called from trainer loop)
    # ------------------------------------------------------------------

    def training_statis(self, opt, pkg: dict, width: int, height: int) -> None:
        if self.densifier is None:
            return
        rendered_2d_points = pkg.get("rendered_2d_points")
        points_grad_detached = None
        if rendered_2d_points is not None and rendered_2d_points.grad is not None:
            points_grad_detached = rendered_2d_points.grad.detach().clone()

        self.densifier.update_densification_state(
            visibility_mask=pkg["visible_anchors_mask"],
            negative_opacity_filter=pkg["negative_opacity_filter"],
            opacity=pkg["opacity"],
            points_grad=points_grad_detached,
            width=width,
            height=height,
        )

    def run_densify(self, opt, iteration: int) -> None:
        if self.densifier is None:
            return
        self.densifier.growing_operation()
        self.densifier.pruning_operation(
            opacity_threshold=getattr(opt, "min_opacity", 0.005)
        )
        self.densifier.reset_state()


# ---------------------------------------------------------------------------
# Backward-compat alias so old name still resolves
# ---------------------------------------------------------------------------
GaussianModel = VRoomModel
