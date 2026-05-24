"""Tests for the new TrainingOrchestrator + Optimizer pipeline."""

import sys
import os
import math
import logging
import shutil
import torch
import torch.nn as nn
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vroom_core.training.orchestration import TrainingOrchestrator
from vroom_core.training.optimizer import Optimizer
from vroom_core.models.anchor_field import AnchorCloud
from vroom_core.models.decoder import GaussianDecoder
from vroom_core.models.gaussian_model import GaussianModel
from vroom_core.models.semantics import SemanticCodec

import pytest


# ---------------------------------------------------------------------------
# Minimal stubs
# ---------------------------------------------------------------------------

class MockCamera:
    def __init__(self, uid):
        self.uid = uid
        self.resolution_scale = 1.0
        self.image_width = 128
        self.image_height = 128
        self.original_image = torch.rand(3, 128, 128)
        self.alpha_mask = torch.ones(1, 128, 128)
        self.FoVx = 1.0
        self.FoVy = 1.0
        self.world_view_transform = torch.eye(4).cuda()
        self.full_proj_transform = torch.eye(4).cuda()
        self.camera_center = torch.zeros(3).cuda()
        self.object_mask = torch.zeros(128, 128)
        self.fx = self.image_width  / (2.0 * math.tan(self.FoVx / 2.0))
        self.fy = self.image_height / (2.0 * math.tan(self.FoVy / 2.0))
        self.cx = self.image_width  / 2.0
        self.cy = self.image_height / 2.0


class MockScene:
    def __init__(self):
        self.background = torch.zeros(3).cuda()
        self.cameras_extent = 1.0
        self.cams = [MockCamera(i) for i in range(2)]

    def getTrainCameras(self):
        return self.cams

    def getTestCameras(self):
        return self.cams


def _make_opt_args(**overrides):
    """Build a minimal _OptArgs-compatible namespace for Optimizer.setup()."""
    defaults = dict(
        position_lr_init=0.0,
        position_lr_final=0.0,
        position_lr_delay_mult=0.01,
        position_lr_max_steps=30_000,
        offset_lr_init=0.01,
        offset_lr_final=0.0001,
        offset_lr_delay_mult=0.01,
        offset_lr_max_steps=30_000,
        feature_lr=0.0075,
        scaling_lr=0.007,
        rotation_lr=0.002,
        mlp_opacity_lr_init=0.002,
        mlp_opacity_lr_final=0.00002,
        mlp_opacity_lr_delay_mult=0.01,
        mlp_opacity_lr_max_steps=30_000,
        mlp_cov_lr_init=0.004,
        mlp_cov_lr_final=0.004,
        mlp_cov_lr_delay_mult=0.01,
        mlp_cov_lr_max_steps=30_000,
        mlp_color_lr_init=0.008,
        mlp_color_lr_final=0.00005,
        mlp_color_lr_delay_mult=0.01,
        mlp_color_lr_max_steps=30_000,
        appearance_lr_init=0.05,
        appearance_lr_final=0.0005,
        appearance_lr_delay_mult=0.01,
        appearance_lr_max_steps=30_000,
    )
    defaults.update(overrides)
    ns = type("OptArgs", (), defaults)()
    return ns


def _make_configs(anchor_cloud, decoder, num_iterations=2):
    """Build minimal configs dict for TrainingOrchestrator."""
    return {
        "optimization": {
            "args": _make_opt_args(),
            "spatial_lr_scale": 1.0,
            "anchor_cloud": anchor_cloud,
            "decoder": decoder,
            "num_iterations": num_iterations,
            "max_grad_norm": 1.0,
        },
        "pipeline": {
            "save_iterations": [],
            "save_vis": False,
            "output_dir": "/tmp/test_orchestrator",
        },
        "densifier": {
            "desification_start": 10,
            "desification_end": 100,
            "densification_interval": 50,
            "min_opacity": 0.005,
        },
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_optimizer_setup_creates_density_controller():
    """Optimizer.setup() must instantiate self.density (DensifcationController)."""
    anchor_cloud = AnchorCloud(voxel_size=0.1)
    decoder = GaussianDecoder(n_offsets=2, feat_dim=8)

    # Seed a tiny anchor cloud so DensifcationController has something to work with
    anchor_cloud.anchors_positions = nn.Parameter(
        torch.rand(4, 3, device=anchor_cloud.device)
    )
    anchor_cloud.gaussians_offsets = nn.Parameter(
        torch.zeros(4, 2, 3, device=anchor_cloud.device)
    )
    anchor_cloud.anchor_features = nn.Parameter(
        torch.zeros(4, 8, device=anchor_cloud.device)
    )
    anchor_cloud.anchors_log_scales = nn.Parameter(
        torch.zeros(4, 6, device=anchor_cloud.device)
    )
    anchor_cloud.anchors_rotations = nn.Parameter(
        torch.zeros(4, 4, device=anchor_cloud.device),
        requires_grad=False,
    )

    configs = _make_configs(anchor_cloud, decoder)
    optimizer = Optimizer(configs["optimization"])
    optimizer.setup()

    assert optimizer.density is not None, "Optimizer.setup() must set self.density"
    assert optimizer.optimizer is not None, "Optimizer.setup() must create torch Adam"


def test_orchestrator_constructs_without_crash():
    """TrainingOrchestrator.__init__ must not raise with a valid configs dict."""
    anchor_cloud = AnchorCloud(voxel_size=0.1)
    decoder = GaussianDecoder(n_offsets=2, feat_dim=8)

    anchor_cloud.anchors_positions = nn.Parameter(
        torch.rand(4, 3, device=anchor_cloud.device)
    )
    anchor_cloud.gaussians_offsets = nn.Parameter(
        torch.zeros(4, 2, 3, device=anchor_cloud.device)
    )
    anchor_cloud.anchor_features = nn.Parameter(
        torch.zeros(4, 8, device=anchor_cloud.device)
    )
    anchor_cloud.anchors_log_scales = nn.Parameter(
        torch.zeros(4, 6, device=anchor_cloud.device)
    )
    anchor_cloud.anchors_rotations = nn.Parameter(
        torch.zeros(4, 4, device=anchor_cloud.device),
        requires_grad=False,
    )

    configs = _make_configs(anchor_cloud, decoder)
    gaussian_model = GaussianModel.__new__(GaussianModel)
    gaussian_model.field = anchor_cloud
    gaussian_model.decoder = decoder
    gaussian_model.device = anchor_cloud.device

    # Should not raise
    orchestrator = TrainingOrchestrator(configs, gaussian_model, optimizer=None)
    assert orchestrator.optimizer is not None


if __name__ == "__main__":
    test_optimizer_setup_creates_density_controller()
    print("test_optimizer_setup_creates_density_controller PASSED")
    test_orchestrator_constructs_without_crash()
    print("test_orchestrator_constructs_without_crash PASSED")
    print("ALL TESTS PASSED")
