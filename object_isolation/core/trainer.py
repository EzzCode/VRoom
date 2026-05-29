"""Fresh per-object gstrain training (real + hallucinated views).

This module intentionally does **not** use ``target_replenishment`` optimizers,
backside seeding, or anchors from the already-trained scene model. It
starts from the scene COLMAP point cloud, creates a fresh gstrain
``GaussianModel``, and trains that model directly against the joint real +
hallucinated view set produced by the earlier pipeline stages.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np
import torch

from object_isolation.paths import MODEL_DIR, TRAINING_SUMMARY_FILE
import torch.nn.functional as F
from torch import nn
from tqdm.auto import tqdm

from .colmap_init import load_colmap_object_point_cloud
from .gs_renderer import create_camera
from .object_scope import ObjectScope

logger = logging.getLogger(__name__)

from gstrain.gaussian_renderer.render import prefilter_voxel as _prefilter
from gstrain.gaussian_renderer.render import render as _gstrain_render
from gstrain.vroom_core.core.model.facade import GaussianModel
from gstrain.vroom_core.core.training.loss_engine import ssim_loss as _ssim_loss


# ── Tensor helpers ─────────────────────────────────────────────────────────────────────────

def _to_tensor_image(rgb: np.ndarray) -> torch.Tensor:
    """Convert an HWC numpy RGB image to a CHW float CUDA tensor in [0, 1]."""
    rgb_np = np.asarray(rgb)
    if rgb_np.dtype == np.uint8:
        rgb_np = rgb_np.astype(np.float32) / 255.0
    else:
        rgb_np = np.clip(rgb_np.astype(np.float32), 0.0, 1.0)
    return torch.from_numpy(rgb_np).permute(2, 0, 1).float().cuda()


def _to_tensor_mask(mask: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.asarray(mask).astype(np.float32)).unsqueeze(0).cuda()


def _as_depth_tensor(depth) -> Optional[torch.Tensor]:
    if depth is None:
        return None
    if depth.ndim == 2:
        depth = depth.unsqueeze(0)
    elif depth.ndim == 3:
        depth = depth[0:1]
    else:
        return None
    return depth.float()


def _as_alpha_tensor(alpha) -> torch.Tensor:
    if alpha.ndim == 3:
        return alpha[0:1].float()
    if alpha.ndim == 2:
        return alpha.unsqueeze(0).float()
    return alpha.reshape(1, *alpha.shape[-2:]).float()


def _model_kwargs_from_parent(parent_gaussians=None) -> dict:
    keys = (
        "gs_attr", "feat_dim", "view_dim", "appearance_dim",
        "n_offsets", "voxel_size", "render_mode", "tile_size_2dgs",
    )
    defaults = {
        "gs_attr": "2D",
        "feat_dim": 32,
        "view_dim": 3,
        "appearance_dim": 0,
        "n_offsets": 10,
        "voxel_size": 0.001,
        "render_mode": "RGB+ED",
        "tile_size_2dgs": 8,
    }
    if parent_gaussians is None:
        return defaults
    return {key: getattr(parent_gaussians, key, defaults[key]) for key in keys}


def _prepare_direct_render_model(gaussians: GaussianModel) -> None:
    gaussians.explicit_gs = False
    gaussians.weed_ratio = 0.0


def _render_parent_object_depth_target(
    *,
    parent_gaussians,
    pipe_config,
    cam,
    object_id: int,
    real_mask: torch.Tensor,
    alpha_threshold: float,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if parent_gaussians is None or not hasattr(parent_gaussians, "label_ids"):
        return None, None

    with torch.no_grad():
        labels = parent_gaussians.label_ids.squeeze()
        object_mask = labels == int(object_id)
        if not bool(object_mask.any().item()):
            return None, None

        parent_gaussians.set_anchor_mask(cam.camera_center, cam.resolution_scale)
        try:
            visible_mask = _prefilter(cam, parent_gaussians).squeeze()
        except Exception:
            visible_mask = parent_gaussians._anchor_mask

        bg = torch.zeros(3, dtype=torch.float32, device="cuda")
        pkg = _gstrain_render(
            cam,
            parent_gaussians,
            pipe_config,
            bg,
            visible_mask=visible_mask,
            training=False,
            object_mask=object_mask,
        )
        depth = _as_depth_tensor(pkg.get("render_depth"))
        if depth is None:
            return None, None
        alpha = _as_alpha_tensor(pkg["render_alphas"])
        valid = (
            (real_mask > 0.5)
            & (alpha > float(alpha_threshold))
            & torch.isfinite(depth)
            & (depth > 0.0)
        ).float()
        if int(valid.sum().item()) < 64:
            return None, None
        return depth.detach(), valid.detach()


def _training_options(
    iterations: int,
    lr_scale: float,
    *,
    enable_densification: bool,
    densify_grad_threshold: float,
    densify_extra_ratio: float,
) -> SimpleNamespace:
    iterations = int(iterations)
    return SimpleNamespace(
        iterations=iterations,
        position_lr_init=0.0,
        position_lr_final=0.0,
        position_lr_delay_mult=0.01,
        position_lr_max_steps=iterations,
        offset_lr_init=0.0040 * float(lr_scale),
        offset_lr_final=0.00005 * float(lr_scale),
        offset_lr_delay_mult=0.01,
        offset_lr_max_steps=iterations,
        feature_lr=0.0075 * float(lr_scale),
        scaling_lr=0.0015 * float(lr_scale),
        rotation_lr=0.0020 * float(lr_scale),
        mlp_opacity_lr_init=0.0020 * float(lr_scale),
        mlp_opacity_lr_final=0.000020 * float(lr_scale),
        mlp_opacity_lr_delay_mult=0.01,
        mlp_opacity_lr_max_steps=iterations,
        mlp_cov_lr_init=0.0040 * float(lr_scale),
        mlp_cov_lr_final=0.0040 * float(lr_scale),
        mlp_cov_lr_delay_mult=0.01,
        mlp_cov_lr_max_steps=iterations,
        mlp_color_lr_init=0.0080 * float(lr_scale),
        mlp_color_lr_final=0.000050 * float(lr_scale),
        mlp_color_lr_delay_mult=0.01,
        mlp_color_lr_max_steps=iterations,
        appearance_lr_init=0.0,
        appearance_lr_final=0.0,
        appearance_lr_delay_mult=0.01,
        appearance_lr_max_steps=iterations,
        lambda_dssim=0.2,
        lambda_dreg=0.0001,
        start_stat=max(25, min(500, iterations // 8)),
        update_from=max(50, min(1500, iterations // 4)),
        update_interval=max(25, min(100, iterations // 20)),
        update_until=max(1, iterations) if bool(enable_densification) else 0,
        overlap=False,
        densification=bool(enable_densification),
        growing_type="mean",
        pruning_type="mean",
        min_opacity=0.005,
        success_threshold=0.8,
        densify_grad_threshold=float(densify_grad_threshold),
        update_ratio=0.2,
        extra_ratio=float(densify_extra_ratio),
        extra_up=0.05,
    )


def train_object(
    *,
    supervision_views: list,
    scope: ObjectScope,
    object_id: int,
    model_path: str | Path,
    output_dir: str | Path,
    n_iterations: int,
    extraction_index_path: str | Path | None = None,
    parent_gaussians=None,
    pipe_config=None,
    lr_scale: float = 1.0,
    max_init_points: int = 20000,
    colmap_init_target_points: int = 8000,
    rgb_weight: float = 1.0,
    hallucination_rgb_scale: float = 1.0,
    alpha_weight: float = 1.0,
    outside_alpha_weight: float = 5.0,
    depth_weight: float = 0.1,
    depth_start_iter: int = 100,
    depth_front_weight: float = 1.0,
    depth_back_weight: float = 0.15,
    depth_alpha_threshold: float = 0.35,
    enable_densification: bool = False,
    max_anchor_count: int = 20000,
    densify_grad_threshold: float = 0.00005,
    densify_extra_ratio: float = 0.08,
    max_scale_growth: float = 1.35,
    max_offset_abs: float = 0.45,
) -> dict:
    """Train a fresh object-only gstrain model from COLMAP seed points."""
    if not supervision_views:
        raise RuntimeError("Cannot train object model with no supervision views.")

    out_dir = Path(output_dir)
    model_dir = out_dir / MODEL_DIR
    model_dir.mkdir(parents=True, exist_ok=True)

    entries = []
    for view in supervision_views:
        cam_p = view["camera"]
        cam = create_camera(cam_p["R"], cam_p["T"], cam_p["K"], cam_p["width"], cam_p["height"])
        entries.append({
            "camera": cam,
            "gt_image": _to_tensor_image(view["rgb"]),
            "gt_mask": _to_tensor_mask(view["mask"]),
            "weight": float(view.get("weight", 1.0)),
            "source": str(view.get("source", "unknown")),
        })

    n_depth_targets = 0
    if float(depth_weight) > 0.0 and parent_gaussians is not None:
        for entry in entries:
            if entry["source"] != "real":
                entry["depth_target"] = None
                entry["depth_valid"] = None
                continue
            depth_target, depth_valid = _render_parent_object_depth_target(
                parent_gaussians=parent_gaussians,
                pipe_config=pipe_config,
                cam=entry["camera"],
                object_id=int(object_id),
                real_mask=entry["gt_mask"],
                alpha_threshold=float(depth_alpha_threshold),
            )
            entry["depth_target"] = depth_target
            entry["depth_valid"] = depth_valid
            if depth_target is not None:
                n_depth_targets += 1
    else:
        for entry in entries:
            entry["depth_target"] = None
            entry["depth_valid"] = None

    pcd, init_metadata = load_colmap_object_point_cloud(
        model_path=model_path,
        object_id=int(object_id),
        scope=scope,
        extraction_index_path=extraction_index_path,
        max_points=int(max_init_points),
        target_points=int(colmap_init_target_points),
    )

    gaussians = GaussianModel(**_model_kwargs_from_parent(parent_gaussians))
    _prepare_direct_render_model(gaussians)
    gaussians.set_appearance(len(entries))
    spatial_extent = max(float(scope.radius), float(np.linalg.norm(scope.aabb_max_W - scope.aabb_min_W)))
    gaussians.initialize_anchors(pcd, spatial_extent, logger=logger)

    opt = _training_options(
        int(n_iterations),
        float(lr_scale),
        enable_densification=bool(enable_densification),
        densify_grad_threshold=float(densify_grad_threshold),
        densify_extra_ratio=float(densify_extra_ratio),
    )
    gaussians.training_setup(opt)
    initial_scaling = gaussians._scaling.detach().clone()
    initial_anchor_count = int(gaussians._anchor.shape[0])

    pipe = pipe_config or SimpleNamespace(add_prefilter=True)
    if not hasattr(pipe, "add_prefilter"):
        pipe.add_prefilter = True

    # White background: object-only training. Supervision images (both real gstrain renders
    # and denormalized SV3D hallucinations) have white background for non-object pixels.
    # rgb_bg loss then correctly penalizes any alpha > 0 outside the object mask.
    background = torch.ones(3, dtype=torch.float32, device="cuda")
    loss_history: list[float] = []
    depth_loss_history: list[float] = []
    source_counts: dict[str, int] = {}
    for entry in entries:
        source_counts[entry["source"]] = source_counts.get(entry["source"], 0) + 1

    order = list(range(len(entries)))
    rng = np.random.default_rng(0)
    densify_count = 0

    progress = tqdm(range(1, int(n_iterations) + 1), desc=f"obj {int(object_id)}", dynamic_ncols=True)
    for iteration in progress:
        if (iteration - 1) % len(order) == 0:
            rng.shuffle(order)
        entry = entries[order[(iteration - 1) % len(order)]]
        cam = entry["camera"]
        gt = entry["gt_image"]
        mask = entry["gt_mask"]
        weight = float(entry["weight"])
        is_real = entry["source"] == "real"
        rgb_scale = float(hallucination_rgb_scale) if not is_real else 1.0

        gaussians.update_learning_rate(iteration)
        gaussians.set_anchor_mask(cam.camera_center, cam.resolution_scale)
        visible_mask = _prefilter(cam, gaussians).squeeze() if getattr(pipe, "add_prefilter", True) else gaussians._anchor_mask
        render_pkg = _gstrain_render(cam, gaussians, pipe, background, visible_mask=visible_mask, training=True)
        pred = torch.clamp(render_pkg["render"], 0.0, 1.0)
        alpha = render_pkg["render_alphas"]
        if alpha.ndim == 3:
            alpha = alpha[0:1]
        elif alpha.ndim == 2:
            alpha = alpha.unsqueeze(0)

        n_fg = mask.sum().clamp(min=1.0)
        n_bg = (1.0 - mask).sum().clamp(min=1.0)
        rgb_fg = torch.abs((pred - gt) * mask).sum() / (3.0 * n_fg)
        rgb_bg = torch.abs((pred - gt) * (1.0 - mask)).sum() / (3.0 * n_bg)
        ssim_loss = _ssim_loss(pred.unsqueeze(0) * mask.unsqueeze(0), gt.unsqueeze(0) * mask.unsqueeze(0))
        alpha_fg = (mask * (1.0 - alpha)).mean()
        alpha_bg = ((1.0 - mask) * alpha).mean()
        scale_reg = render_pkg["scaling"].prod(dim=1).mean() if render_pkg["scaling"].numel() else torch.tensor(0.0, device="cuda")
        scale_drift = torch.tensor(0.0, device="cuda")
        if gaussians._scaling.shape == initial_scaling.shape:
            scale_drift = (gaussians._scaling - initial_scaling).pow(2).mean()

        total = weight * float(rgb_weight) * rgb_scale * (0.8 * rgb_fg + 0.2 * ssim_loss + 0.2 * rgb_bg)
        total = total + float(alpha_weight) * alpha_fg + float(outside_alpha_weight) * alpha_bg
        total = total + float(opt.lambda_dreg) * scale_reg + 0.01 * scale_drift

        depth_loss = torch.tensor(0.0, device="cuda")
        depth_target = entry.get("depth_target")
        depth_valid = entry.get("depth_valid")
        pred_depth = _as_depth_tensor(render_pkg.get("render_depth"))
        if (
            float(depth_weight) > 0.0
            and iteration >= int(depth_start_iter)
            and depth_target is not None
            and depth_valid is not None
            and pred_depth is not None
        ):
            rel_delta = (pred_depth - depth_target) / depth_target.detach().abs().clamp_min(1e-3)
            closer_than_target = F.relu(-rel_delta)
            farther_than_target = F.relu(rel_delta)
            depth_loss = (
                (float(depth_front_weight) * closer_than_target + float(depth_back_weight) * farther_than_target)
                * depth_valid
            ).sum() / depth_valid.sum().clamp_min(1.0)
            total = total + float(depth_weight) * depth_loss

        gaussians.optimizer.zero_grad(set_to_none=True)
        total.backward()

        with torch.no_grad():
            if iteration < opt.update_until and iteration > opt.start_stat:
                gaussians.training_statis(opt, render_pkg, pred.shape[2], pred.shape[1])
                densify_count += 1
                if (
                    opt.densification
                    and densify_count % opt.update_interval == 0
                    and int(gaussians._anchor.shape[0]) < int(max_anchor_count)
                ):
                    gaussians.run_densify(opt, iteration)

            gaussians.optimizer.step()
            if gaussians._scaling.shape == initial_scaling.shape:
                gaussians._scaling.data.clamp_(max=initial_scaling.max().item())
            gaussians._offset.data.clamp_(min=-float(max_offset_abs), max=float(max_offset_abs))

        loss_value = float(total.detach().item())
        depth_loss_value = float(depth_loss.detach().item())
        loss_history.append(loss_value)
        depth_loss_history.append(depth_loss_value)
        if iteration == 1 or iteration % 10 == 0 or iteration == int(n_iterations):
            progress.set_postfix({
                "loss": f"{loss_value:.4f}",
                "depth": f"{depth_loss_value:.4f}",
                "src": entry["source"],
                "anchors": int(gaussians._anchor.shape[0]),
            })

    gaussians.eval()
    gaussians.save_ply(str(model_dir / "point_cloud.ply"))
    gaussians.save_mlp_checkpoints(str(model_dir))

    tail = loss_history[-min(50, len(loss_history)):] if loss_history else []
    summary = {
        "object_id": int(object_id),
        "mode": "object_training",
        "init_source": init_metadata.get("init_source", "unknown"),
        "init_metadata": init_metadata,
        "n_supervision_views": len(supervision_views),
        "n_depth_target_views": int(n_depth_targets),
        "source_counts": source_counts,
        "n_init_points": int(len(pcd.points)),
        "n_final_anchors": int(gaussians._anchor.shape[0]),
        "initial_anchor_count": int(initial_anchor_count),
        "densification_enabled": bool(enable_densification),
        "max_anchor_count": int(max_anchor_count),
        "densify_grad_threshold": float(densify_grad_threshold),
        "densify_extra_ratio": float(densify_extra_ratio),
        "hallucination_rgb_scale": float(hallucination_rgb_scale),
        "depth_weight": float(depth_weight),
        "depth_start_iter": int(depth_start_iter),
        "depth_front_weight": float(depth_front_weight),
        "depth_back_weight": float(depth_back_weight),
        "depth_alpha_threshold": float(depth_alpha_threshold),
        "final_sample_loss": float(loss_history[-1]) if loss_history else 0.0,
        "final_loss": float(np.mean(tail)) if tail else 0.0,
        "loss_history": loss_history,
        "depth_loss_history": depth_loss_history,
        "model_dir": str(model_dir),
    }
    with open(out_dir / TRAINING_SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return {"gaussians": gaussians, "summary": summary}
