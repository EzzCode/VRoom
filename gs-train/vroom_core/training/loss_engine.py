"""Composable VRoom loss engine."""

from __future__ import annotations

import functools
from typing import Dict

import torch
import torch.nn.functional as F


def l1_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.abs(prediction - target).mean()


def ssim_loss(prediction: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    @functools.lru_cache(maxsize=4)
    def kernel(size: int, channels: int, device_str: str, dtype):
        dist = torch.distributions.Normal(loc=size // 2, scale=1.5)
        coords = torch.arange(size, dtype=torch.float32)
        weights = dist.log_prob(coords).exp()
        weights = weights / weights.sum()
        kernel2d = weights[:, None] @ weights[None, :]
        return kernel2d.unsqueeze(0).unsqueeze(0).expand(channels, 1, size, size).contiguous().to(device=device_str, dtype=dtype)

    if prediction.dim() == 3:
        prediction = prediction.unsqueeze(0)
        target = target.unsqueeze(0)
    prediction = prediction.clamp(0.0, 1.0)
    target = target.clamp(0.0, 1.0)
    channels = prediction.shape[1]
    padding = window_size // 2
    window = kernel(window_size, channels, str(prediction.device), prediction.dtype)
    mu_a = F.conv2d(prediction, window, padding=padding, groups=channels)
    mu_b = F.conv2d(target, window, padding=padding, groups=channels)
    sigma_a = F.conv2d(prediction * prediction, window, padding=padding, groups=channels) - mu_a.pow(2)
    sigma_b = F.conv2d(target * target, window, padding=padding, groups=channels) - mu_b.pow(2)
    sigma_ab = F.conv2d(prediction * target, window, padding=padding, groups=channels) - (mu_a * mu_b)
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    numerator = (2.0 * mu_a * mu_b + c1) * (2.0 * sigma_ab + c2)
    denominator = (mu_a.pow(2) + mu_b.pow(2) + c1) * (sigma_a + sigma_b + c2)
    return 1.0 - (numerator / denominator).mean()


def image_loss_fn(prediction, target, opt):
    return (1.0 - opt.lambda_dssim) * l1_loss(prediction, target) + opt.lambda_dssim * ssim_loss(prediction, target)


def scaling_loss_fn(scaling, opt):
    if opt.lambda_dreg > 0 and scaling.shape[0] > 0:
        return opt.lambda_dreg * scaling.prod(dim=1).mean()
    return None


def object_loss_fn(semantics, object_mask, id_encoder, opt, device):
    if opt.lambda_object_loss > 0 and id_encoder is not None:
        gt_ids = id_encoder.label_to_index(object_mask.to(device)).long()
        logits = semantics.permute(0, 3, 1, 2)
        # Ensure gt_ids has the same spatial dimensions as logits
        if gt_ids.dim() == 2:
            gt_ids = gt_ids.unsqueeze(0)
        elif gt_ids.dim() == 4 and gt_ids.shape[1] == 1:
            gt_ids = gt_ids.squeeze(1)
        
        if (gt_ids != 0).sum() == 0:
            object_loss = (semantics * 0.0).sum()
        else:
            object_loss = opt.lambda_object_loss * F.cross_entropy(logits, gt_ids, ignore_index=0, reduction="mean")
        zero_penalty = opt.lambda_zero_penalty * semantics[..., 0].mean()
        return object_loss, zero_penalty
    return None, None


def sky_opa_loss_fn(alpha, mask, opt):
    if opt.lambda_sky_opa > 0:
        clamped = alpha.clamp(1e-6, 1 - 1e-6)
        return opt.lambda_sky_opa * (-(1 - mask.float()) * torch.log(1 - clamped)).mean()
    return None


def opacity_entropy_loss_fn(alpha, opt):
    if opt.lambda_opacity_entropy > 0:
        clamped = alpha.clamp(1e-6, 1 - 1e-6)
        return opt.lambda_opacity_entropy * -(clamped * torch.log(clamped)).mean()
    return None


def normal_loss_fn(render_pkg, alpha, mask, opt, iteration):
    if opt.lambda_normal > 0 and iteration > opt.normal_start_iter and "render_normals" in render_pkg:
        normals = render_pkg["render_normals"]
        if normals.dim() == 4:
            normals = normals.squeeze(0)
        if normals.shape[-1] == 3:
            normals = normals.permute(2, 0, 1)
        
        a = alpha
        if a.dim() == 4:
            a = a.squeeze(1) if a.shape[1] == 1 else a.squeeze(0)
        if a.dim() == 3 and a.shape[0] == 1:
            a = a.permute(1, 2, 0)
        
        depth_normals = render_pkg["render_normals_from_depth"] * a.detach()
        if depth_normals.dim() == 4:
            depth_normals = depth_normals.squeeze(0)
        if depth_normals.shape[-1] == 3:
            depth_normals = depth_normals.permute(2, 0, 1)
        
        return opt.lambda_normal * ((1.0 - (normals * depth_normals).sum(dim=0, keepdim=True)) * mask).mean()
    return None


def distort_loss_fn(render_pkg, mask, opt, iteration):
    if opt.lambda_dist > 0 and iteration > opt.dist_start_iter and "render_distort" in render_pkg:
        return opt.lambda_dist * (render_pkg["render_distort"].squeeze(3) * mask).mean()
    return None


def depth_loss_fn(render_pkg, viewpoint_cam, depth_weight, device):
    if depth_weight > 0 and viewpoint_cam.invdepthmap is not None and render_pkg["render_depth"] is not None:
        rendered_depth = render_pkg["render_depth"]
        inv_depth = torch.where(rendered_depth > 0, 1.0 / rendered_depth, torch.zeros_like(rendered_depth))
        return depth_weight * (inv_depth - viewpoint_cam.invdepthmap.to(device)).abs().mul(viewpoint_cam.depth_mask.to(device)).mean()
    return None


class LossComposer:
    def compose(self, render_pkg, viewpoint_cam, gaussians, opt, iteration: int, depth_weight_fn) -> Dict[str, torch.Tensor]:
        prediction = render_pkg["render"]
        alpha = render_pkg["render_alphas"]
        semantics = render_pkg["render_semantics"]
        device = prediction.device
        target = viewpoint_cam.original_image.to(device)
        mask = viewpoint_cam.alpha_mask.to(device)
        # Defensive: only use alpha_mask for masking if it looks like a binary opacity mask.
        # Categorical label maps (e.g. SAM id maps with values 0-7, scaled to 0.0-0.027)
        # would zero out the entire image and kill training.
        if mask.max() > 0.5:
            prediction = prediction * mask
            target = target * mask

        losses: Dict[str, torch.Tensor] = {}
        
        # Image Loss
        losses["image_loss"] = image_loss_fn(prediction, target, opt)

        # Scaling Loss
        scaling_loss = scaling_loss_fn(render_pkg["scaling"], opt)
        if scaling_loss is not None:
            losses["scaling_loss"] = scaling_loss

        # Object Loss & Zero Penalty
        object_loss, zero_penalty = object_loss_fn(semantics, viewpoint_cam.object_mask, gaussians.id_encoder, opt, device)
        if object_loss is not None:
            losses["object_loss"] = object_loss
            losses["zero_penalty"] = zero_penalty

        # Sky Opacity Loss
        sky_opa_loss = sky_opa_loss_fn(alpha, mask, opt)
        if sky_opa_loss is not None:
            losses["sky_opa_loss"] = sky_opa_loss

        # Opacity Entropy Loss
        entropy_loss = opacity_entropy_loss_fn(alpha, opt)
        if entropy_loss is not None:
            losses["opacity_entropy_loss"] = entropy_loss

        # Normal Loss
        normal_loss = normal_loss_fn(render_pkg, alpha, mask, opt, iteration)
        if normal_loss is not None:
            losses["normal_loss"] = normal_loss

        # Distort Loss
        distort_loss = distort_loss_fn(render_pkg, mask, opt, iteration)
        if distort_loss is not None:
            losses["distort_loss"] = distort_loss

        # Depth Loss
        depth_weight = depth_weight_fn(iteration)
        if iteration > opt.start_depth:
            depth_loss = depth_loss_fn(render_pkg, viewpoint_cam, depth_weight, device)
            if depth_loss is not None:
                losses["depth_loss"] = depth_loss

        return losses


def compute_losses(render_pkg, viewpoint_cam, gaussians, opt, iteration, depth_w_fn) -> Dict[str, torch.Tensor]:
    return LossComposer().compose(render_pkg, viewpoint_cam, gaussians, opt, iteration, depth_w_fn)
