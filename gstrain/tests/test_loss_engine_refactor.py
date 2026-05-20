"""Unit tests for loss_engine refactor - comparing old vs new implementation."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from unittest.mock import MagicMock

import sys
sys.path.insert(0, "/home/hussein_essam/gs-workspace/VRoom/gs-train/vroom_core")

from training.loss_engine import (
    LossComposer,
    image_loss_fn,
    scaling_loss_fn,
    object_loss_fn,
    sky_opa_loss_fn,
    opacity_entropy_loss_fn,
    normal_loss_fn,
    distort_loss_fn,
    depth_loss_fn,
    l1_loss,
    ssim_loss,
)


def create_mock_opt():
    opt = MagicMock()
    opt.lambda_dssim = 0.2
    opt.lambda_dreg = 0.01
    opt.lambda_object_loss = 0.5
    opt.lambda_zero_penalty = 0.1
    opt.lambda_sky_opa = 0.3
    opt.lambda_opacity_entropy = 0.05
    opt.lambda_normal = 0.4
    opt.lambda_dist = 0.02
    opt.normal_start_iter = 100
    opt.dist_start_iter = 500
    opt.start_depth = 1000
    return opt


def create_mock_viewpoint_cam(device="cuda"):
    cam = MagicMock()
    cam.original_image = torch.rand(1, 3, 256, 256, device=device)
    cam.alpha_mask = torch.rand(1, 1, 256, 256, device=device).gt(0.5).float()
    cam.object_mask = torch.randint(0, 10, (256, 256), device=device).long()
    cam.invdepthmap = torch.rand(1, 1, 256, 256, device=device) * 0.01
    cam.depth_mask = torch.ones(1, 1, 256, 256, device=device)
    return cam


def create_mock_gaussians(device="cuda"):
    gaussians = MagicMock()
    id_encoder = MagicMock()
    def label_to_index(x):
        return x
    id_encoder.label_to_index = label_to_index
    gaussians.id_encoder = id_encoder
    return gaussians


def create_mock_render_pkg(device="cuda"):
    render_pkg = {
        "render": torch.rand(1, 3, 256, 256, device=device),
        "render_alphas": torch.rand(1, 256, 256, device=device).clamp(0.01, 0.99),
        "render_semantics": torch.rand(1, 256, 256, 11, device=device),
        "scaling": torch.rand(100, 3, device=device) * 0.1,
    }
    return render_pkg


def old_compose_impl(render_pkg, viewpoint_cam, gaussians, opt, iteration: int, depth_weight_fn):
    """Old implementation - all logic inline in compose method."""
    prediction = render_pkg["render"]
    alpha = render_pkg["render_alphas"]
    semantics = render_pkg["render_semantics"]
    device = prediction.device
    target = viewpoint_cam.original_image.to(device)
    mask = viewpoint_cam.alpha_mask.to(device)
    prediction = prediction * mask
    target = target * mask

    losses = {}
    losses["image_loss"] = (1.0 - opt.lambda_dssim) * l1_loss(prediction, target) + opt.lambda_dssim * ssim_loss(prediction, target)

    scaling = render_pkg["scaling"]
    if opt.lambda_dreg > 0 and scaling.shape[0] > 0:
        losses["scaling_loss"] = opt.lambda_dreg * scaling.prod(dim=1).mean()

    if opt.lambda_object_loss > 0 and gaussians.id_encoder is not None:
        gt_ids = gaussians.id_encoder.label_to_index(viewpoint_cam.object_mask.to(device)).long()
        logits = semantics.permute(0, 3, 1, 2)
        # Fix dimensions for old_compose_impl in test for comparison
        if gt_ids.dim() == 2:
            gt_ids = gt_ids.unsqueeze(0)
        elif gt_ids.dim() == 4 and gt_ids.shape[1] == 1:
            gt_ids = gt_ids.squeeze(1)
        losses["object_loss"] = opt.lambda_object_loss * F.cross_entropy(logits, gt_ids, ignore_index=0, reduction="mean")
        losses["zero_penalty"] = opt.lambda_zero_penalty * semantics[..., 0].mean()

    if opt.lambda_sky_opa > 0:
        clamped = alpha.clamp(1e-6, 1 - 1e-6)
        losses["sky_opa_loss"] = opt.lambda_sky_opa * (-(1 - mask.float()) * torch.log(1 - clamped)).mean()

    if opt.lambda_opacity_entropy > 0:
        clamped = alpha.clamp(1e-6, 1 - 1e-6)
        losses["opacity_entropy_loss"] = opt.lambda_opacity_entropy * -(clamped * torch.log(clamped)).mean()

    if opt.lambda_normal > 0 and iteration > opt.normal_start_iter and "render_normals" in render_pkg:
        normals = render_pkg["render_normals"].squeeze(0).permute(2, 0, 1)
        a = alpha
        if a.dim() == 4:
            a = a.squeeze(1) if a.shape[1] == 1 else a.squeeze(0)
        depth_normals = render_pkg["render_normals_from_depth"] * a.permute(1, 2, 0).detach()
        if depth_normals.dim() == 4:
            depth_normals = depth_normals.squeeze(0)
        depth_normals = depth_normals.permute(2, 0, 1)
        losses["normal_loss"] = opt.lambda_normal * ((1.0 - (normals * depth_normals).sum(dim=0, keepdim=True)) * mask).mean()

    if opt.lambda_dist > 0 and iteration > opt.dist_start_iter and "render_distort" in render_pkg:
        losses["distort_loss"] = opt.lambda_dist * (render_pkg["render_distort"].squeeze(3) * mask).mean()

    depth_weight = depth_weight_fn(iteration)
    if depth_weight > 0 and iteration > opt.start_depth and viewpoint_cam.invdepthmap is not None and render_pkg["render_depth"] is not None:
        rendered_depth = render_pkg["render_depth"]
        inv_depth = torch.where(rendered_depth > 0, 1.0 / rendered_depth, torch.zeros_like(rendered_depth))
        losses["depth_loss"] = depth_weight * (inv_depth - viewpoint_cam.invdepthmap.to(device)).abs().mul(viewpoint_cam.depth_mask.to(device)).mean()

    return losses


class TestLossEngineRefactor:
    def test_image_loss_equivalence(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        opt = create_mock_opt()
        render_pkg = create_mock_render_pkg(device)
        cam = create_mock_viewpoint_cam(device)

        prediction = render_pkg["render"]
        target = cam.original_image.to(device)
        mask = cam.alpha_mask.to(device)
        prediction = prediction * mask
        target = target * mask

        old_loss = (1.0 - opt.lambda_dssim) * l1_loss(prediction, target) + opt.lambda_dssim * ssim_loss(prediction, target)
        new_loss = image_loss_fn(prediction, target, opt)

        assert torch.allclose(old_loss, new_loss, rtol=1e-5, atol=1e-7), f"image_loss mismatch: {old_loss} vs {new_loss}"

    def test_scaling_loss_equivalence(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        opt = create_mock_opt()
        scaling = torch.rand(100, 3, device=device) * 0.1

        old_loss = opt.lambda_dreg * scaling.prod(dim=1).mean()
        new_loss = scaling_loss_fn(scaling, opt)

        assert torch.allclose(old_loss, new_loss, rtol=1e-5, atol=1e-7), f"scaling_loss mismatch: {old_loss} vs {new_loss}"

    def test_object_loss_equivalence(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        opt = create_mock_opt()
        semantics = torch.rand(1, 256, 256, 11, device=device)
        object_mask = torch.randint(0, 10, (256, 256), device=device).long()
        id_encoder = MagicMock()
        id_encoder.label_to_index = lambda x: x

        gt_ids = id_encoder.label_to_index(object_mask.to(device)).long()
        logits = semantics.permute(0, 3, 1, 2)
        old_object_loss = opt.lambda_object_loss * F.cross_entropy(logits, gt_ids.unsqueeze(0), ignore_index=0, reduction="mean")
        old_zero_penalty = opt.lambda_zero_penalty * semantics[..., 0].mean()

        new_object_loss, new_zero_penalty = object_loss_fn(semantics, object_mask, id_encoder, opt, device)

        assert torch.allclose(old_object_loss, new_object_loss, rtol=1e-5, atol=1e-7), f"object_loss mismatch"
        assert torch.allclose(old_zero_penalty, new_zero_penalty, rtol=1e-5, atol=1e-7), f"zero_penalty mismatch"

    def test_sky_opa_loss_equivalence(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        opt = create_mock_opt()
        alpha = torch.rand(1, 1, 256, 256, device=device).clamp(0.01, 0.99)
        mask = torch.rand(1, 1, 256, 256, device=device).gt(0.5).float()

        clamped = alpha.clamp(1e-6, 1 - 1e-6)
        old_loss = opt.lambda_sky_opa * (-(1 - mask.float()) * torch.log(1 - clamped)).mean()
        new_loss = sky_opa_loss_fn(alpha, mask, opt)

        assert torch.allclose(old_loss, new_loss, rtol=1e-5, atol=1e-7), f"sky_opa_loss mismatch"

    def test_opacity_entropy_loss_equivalence(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        opt = create_mock_opt()
        alpha = torch.rand(1, 1, 256, 256, device=device).clamp(0.01, 0.99)

        clamped = alpha.clamp(1e-6, 1 - 1e-6)
        old_loss = opt.lambda_opacity_entropy * -(clamped * torch.log(clamped)).mean()
        new_loss = opacity_entropy_loss_fn(alpha, opt)

        assert torch.allclose(old_loss, new_loss, rtol=1e-5, atol=1e-7), f"opacity_entropy_loss mismatch"

    def test_full_compose_equivalence(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        opt = create_mock_opt()
        iteration = 2000

        render_pkg = create_mock_render_pkg(device)
        render_pkg["render_normals"] = torch.rand(1, 256, 256, 3, device=device)
        render_pkg["render_normals_from_depth"] = torch.rand(256, 256, 3, device=device)
        render_pkg["render_distort"] = torch.rand(1, 256, 256, 1, device=device)
        render_pkg["render_depth"] = torch.rand(1, 1, 256, 256, device=device)

        cam = create_mock_viewpoint_cam(device)
        gaussians = create_mock_gaussians(device)

        depth_weight_fn = lambda it: 0.15

        old_losses = old_compose_impl(render_pkg, cam, gaussians, opt, iteration, depth_weight_fn)
        composer = LossComposer()
        new_losses = composer.compose(render_pkg, cam, gaussians, opt, iteration, depth_weight_fn)

        for key in old_losses:
            assert key in new_losses, f"Key {key} missing in new_losses"
            assert torch.allclose(old_losses[key], new_losses[key], rtol=1e-4, atol=1e-5), f"{key} mismatch: old={old_losses[key]}, new={new_losses[key]}"

        for key in new_losses:
            assert key in old_losses, f"Key {key} missing in old_losses"


if __name__ == "__main__":
    test = TestLossEngineRefactor()
    test.test_image_loss_equivalence()
    test.test_scaling_loss_equivalence()
    test.test_object_loss_equivalence()
    test.test_sky_opa_loss_equivalence()
    test.test_opacity_entropy_loss_equivalence()
    test.test_full_compose_equivalence()
    print("All tests passed!")