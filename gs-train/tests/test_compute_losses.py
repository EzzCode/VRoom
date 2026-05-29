from types import SimpleNamespace
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vroom_core.core.training.loss_engine import compute_losses


class IdentityLabelEncoder:
    def label_to_index(self, labels):
        return labels


class MockGaussians:
    def __init__(self):
        self.id_encoder = IdentityLabelEncoder()


class MockCamera:
    def __init__(self):
        self.original_image = torch.full((3, 4, 4), 0.5)
        self.alpha_mask = torch.ones((1, 4, 4))
        self.object_mask = torch.tensor(
            [[0, 1, 1, 0], [0, 2, 2, 0], [0, 2, 2, 0], [0, 0, 0, 0]],
            dtype=torch.long,
        )
        self.invdepthmap = torch.full((1, 4, 4), 0.5)
        self.depth_mask = torch.ones((1, 4, 4))


def _opt(**overrides):
    defaults = dict(
        lambda_dssim=0.2,
        lambda_dreg=0.1,
        lambda_object_loss=0.3,
        lambda_zero_penalty=0.05,
        lambda_sky_opa=0.2,
        lambda_opacity_entropy=0.1,
        lambda_normal=0.2,
        lambda_dist=0.15,
        normal_start_iter=1,
        dist_start_iter=1,
        start_depth=1,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _render_pkg():
    return {
        "render": torch.full((3, 4, 4), 0.4, device="cuda"),
        "render_alphas": torch.full((1, 4, 4), 0.6, device="cuda"),
        "render_semantics": torch.randn((1, 4, 4, 3), device="cuda"),
        "scaling": torch.full((5, 3), 0.5, device="cuda"),
        "render_normals": torch.ones((1, 4, 4, 3), device="cuda"),
        "render_normals_from_depth": torch.ones((1, 4, 4, 3), device="cuda"),
        "render_distort": torch.full((1, 4, 4, 1), 0.25, device="cuda"),
        "render_depth": torch.full((1, 4, 4), 2.0, device="cuda"),
    }


def test_compute_losses_emits_all_enabled_terms():
    losses = compute_losses(
        _render_pkg(),
        MockCamera(),
        MockGaussians(),
        _opt(),
        iteration=10,
        depth_w_fn=lambda _: 0.4,
    )

    assert {
        "image_loss",
        "scaling_loss",
        "object_loss",
        "zero_penalty",
        "sky_opa_loss",
        "opacity_entropy_loss",
        "normal_loss",
        "distort_loss",
        "depth_loss",
    }.issubset(losses.keys())
    assert all(value.ndim == 0 for value in losses.values())


def test_compute_losses_skips_disabled_optional_terms():
    losses = compute_losses(
        _render_pkg(),
        MockCamera(),
        MockGaussians(),
        _opt(
            lambda_dreg=0.0,
            lambda_object_loss=0.0,
            lambda_sky_opa=0.0,
            lambda_opacity_entropy=0.0,
            lambda_normal=0.0,
            lambda_dist=0.0,
        ),
        iteration=0,
        depth_w_fn=lambda _: 0.0,
    )

    assert set(losses.keys()) == {"image_loss"}
