from types import SimpleNamespace
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trainer import TrainingConfig
from vroom_core.utilities.models.gaussian_model import GaussianModel
from vroom_core.core.models.semantics import SemanticCodec


class MockCamera:
    def __init__(self, center):
        self.camera_center = center
        self.uid = 0


def _set_constant_decoder(model: GaussianModel):
    for module in (model.mlp_opacity, model.mlp_cov, model.mlp_color):
        for layer in module:
            if isinstance(layer, nn.Linear):
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)

    model.mlp_opacity[-1].bias.data.fill_(0.5)
    model.mlp_color[-1].bias.data.copy_(torch.tensor([0.2, 0.4, 0.6] * model.n_offsets, device="cuda"))

    quat = []
    for _ in range(model.n_offsets):
        quat.extend([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    model.mlp_cov[-1].bias.data.copy_(torch.tensor(quat, device="cuda"))


def _build_model(n_anchors=3, n_offsets=2, feat_dim=4):
    model = GaussianModel(n_offsets=n_offsets, feat_dim=feat_dim, appearance_dim=0)
    model._anchor = nn.Parameter(torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], device="cuda"))
    model._offset = nn.Parameter(torch.zeros((n_anchors, n_offsets, 3), device="cuda"))
    model._anchor_feat = nn.Parameter(torch.zeros((n_anchors, feat_dim), device="cuda"))
    model._scaling = nn.Parameter(torch.zeros((n_anchors, 6), device="cuda"))
    model._rotation = nn.Parameter(torch.tensor([[1.0, 0.0, 0.0, 0.0]] * n_anchors, device="cuda"))
    model.label_ids = torch.tensor([[0], [1], [2]], dtype=torch.long, device="cuda")
    model.id_encoder = SemanticCodec.from_labels(model.label_ids.view(-1))
    model._anchor_mask = torch.ones(n_anchors, dtype=torch.bool, device="cuda")
    _set_constant_decoder(model)
    return model


def test_generate_neural_gaussians_returns_vroom_contract_shapes():
    model = _build_model()
    camera = MockCamera(torch.zeros(3, device="cuda"))

    xyz, offsets, color, opacity, scaling, rotation, active_mask, semantics = model.generate_neural_gaussians(camera, training=False)

    assert xyz.shape == (6, 3)
    assert offsets.shape == (6, 3)
    assert color.shape == (6, 3)
    assert opacity.shape == (6, 1)
    assert scaling.shape == (6, 3)
    assert rotation.shape == (6, 4)
    assert active_mask.shape == (6,)
    assert semantics.shape == (6, 3)
    assert torch.allclose(xyz, model._anchor.repeat_interleave(model.n_offsets, dim=0))
    assert torch.allclose(scaling, torch.full((6, 3), 0.5, device="cuda"))
    assert torch.allclose(rotation.norm(dim=-1), torch.ones(6, device="cuda"))


def test_training_statistics_update_accumulators():
    model = _build_model(n_anchors=2, n_offsets=2, feat_dim=4)
    model.setup_training(TrainingConfig())

    viewspace_points = torch.zeros((1, 2), device="cuda", requires_grad=True)
    viewspace_points.grad = torch.tensor([[0.25, 0.5]], device="cuda")
    render_pkg = {
        "selection_mask": torch.tensor([True, False], device="cuda"),
        "visible_mask": torch.tensor([True, False], device="cuda"),
        "opacity": torch.tensor([[0.7]], device="cuda"),
        "viewspace_points": viewspace_points,
        "visibility_filter": torch.tensor([True], device="cuda"),
        "radii": torch.tensor([0.25], device="cuda"),
    }

    opt = SimpleNamespace(pruning_type="mean", growing_type="mean")
    model.training_statis(opt, render_pkg, width=640, height=480)

    assert torch.isclose(model.anchor_opacity_accum[0, 0], torch.tensor(0.35, device="cuda"))
    assert torch.isclose(model.anchor_demon[0, 0], torch.tensor(1.0, device="cuda"))
    assert torch.isclose(model.offset_denom[0, 0], torch.tensor(1.0, device="cuda"))
    assert model.offset_gradient_accum[0, 0] > 0


def test_prune_anchor_removes_state_consistently():
    model = _build_model(n_anchors=3, n_offsets=2, feat_dim=4)
    model.setup_training(TrainingConfig())

    removed = model.prune_anchor(torch.tensor([False, True, False], device="cuda"))

    assert removed.tolist() == [False, True, False]
    assert model._anchor.shape[0] == 2
    assert model._offset.shape[0] == 2
    assert model.label_ids.shape[0] == 2
