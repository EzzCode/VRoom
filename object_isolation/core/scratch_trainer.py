"""Fresh object-only ObjectGS training for aligned real + hallucinated views.

This module intentionally does not use target_replenishment optimizers,
backside seeding, or anchors from the already-trained ObjectGS model. It
starts from the scene COLMAP point cloud, creates a fresh ObjectGS
GaussianModel, and trains that model directly against the joint real +
hallucinated view set.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm.auto import tqdm

from .colmap_init import load_colmap_object_point_cloud
from .gs_renderer import create_camera

logger = logging.getLogger(__name__)

_VROOM_ROOT = Path(__file__).resolve().parents[2]
_OBJECTGS_DIR = _VROOM_ROOT / "temp_deps" / "ObjectGS"
if str(_OBJECTGS_DIR) not in sys.path:
    sys.path.insert(0, str(_OBJECTGS_DIR))

from gaussian_renderer.render import render as _ogs_render  # noqa: E402
from gaussian_renderer.render import prefilter_voxel as _prefilter  # noqa: E402
from scene.base_model import GaussianModel  # noqa: E402
from utils.loss_utils import ssim  # noqa: E402


def _to_tensor_image(rgb: np.ndarray) -> torch.Tensor:
    rgb_np = np.asarray(rgb)
    if rgb_np.dtype == np.uint8:
        rgb_np = rgb_np.astype(np.float32) / 255.0
    else:
        rgb_np = np.clip(rgb_np.astype(np.float32), 0.0, 1.0)
    return torch.from_numpy(rgb_np).permute(2, 0, 1).float().cuda()


def _to_tensor_mask(mask: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.asarray(mask).astype(np.float32)).unsqueeze(0).cuda()


def _model_kwargs_from_parent(parent_gaussians=None) -> dict:
    keys = (
        "fork", "gs_attr", "color_attr", "feat_dim", "view_dim", "appearance_dim",
        "n_offsets", "voxel_size", "render_mode", "update_depth",
        "update_init_factor", "update_hierachy_factor",
    )
    defaults = {
        "fork": 2,
        "gs_attr": "2D",
        "color_attr": "RGB",
        "feat_dim": 32,
        "view_dim": 3,
        "appearance_dim": 0,
        "n_offsets": 10,
        "voxel_size": 0.001,
        "render_mode": "RGB+ED",
        "update_depth": 3,
        "update_init_factor": 16,
        "update_hierachy_factor": 4,
    }
    if parent_gaussians is None:
        return defaults
    return {key: getattr(parent_gaussians, key, defaults[key]) for key in keys}


def _prepare_direct_render_model(gaussians: GaussianModel) -> None:
    gaussians.explicit_gs = False
    gaussians.weed_ratio = 0.0


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


def train_scratch_object(
    *,
    supervision_views: list,
    scope,
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
    alpha_weight: float = 1.0,
    outside_alpha_weight: float = 2.0,
    enable_densification: bool = False,
    max_anchor_count: int = 20000,
    densify_grad_threshold: float = 0.00005,
    densify_extra_ratio: float = 0.08,
    max_scale_growth: float = 1.35,
    max_offset_abs: float = 0.45,
) -> dict:
    """Train a fresh object-only ObjectGS model from COLMAP seed points."""
    if not supervision_views:
        raise RuntimeError("Cannot train scratch object model with no supervision views.")

    out_dir = Path(output_dir)
    model_dir = out_dir / "model"
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
    gaussians.create_from_pcd(pcd, spatial_extent, "", logger)

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
    max_scale_log = float(np.log(max(float(max_scale_growth), 1.001)))

    pipe = pipe_config or SimpleNamespace(add_prefilter=True)
    if not hasattr(pipe, "add_prefilter"):
        pipe.add_prefilter = True

    background = torch.ones(3, dtype=torch.float32, device="cuda")
    loss_history: list[float] = []
    source_counts: dict[str, int] = {}
    for entry in entries:
        source_counts[entry["source"]] = source_counts.get(entry["source"], 0) + 1

    order = list(range(len(entries)))
    rng = np.random.default_rng(0)
    densify_count = 0

    progress = tqdm(range(1, int(n_iterations) + 1), desc=f"scratch obj {int(object_id)}", dynamic_ncols=True)
    for iteration in progress:
        if (iteration - 1) % len(order) == 0:
            rng.shuffle(order)
        entry = entries[order[(iteration - 1) % len(order)]]
        cam = entry["camera"]
        gt = entry["gt_image"]
        mask = entry["gt_mask"]
        weight = float(entry["weight"])

        gaussians.update_learning_rate(iteration)
        gaussians.set_anchor_mask(cam.camera_center, cam.resolution_scale)
        visible_mask = _prefilter(cam, gaussians).squeeze() if getattr(pipe, "add_prefilter", True) else gaussians._anchor_mask
        render_pkg = _ogs_render(cam, gaussians, pipe, background, visible_mask=visible_mask, training=True)
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
        ssim_loss = 1.0 - ssim(pred.unsqueeze(0) * mask.unsqueeze(0), gt.unsqueeze(0) * mask.unsqueeze(0))
        alpha_fg = (mask * (1.0 - alpha)).mean()
        alpha_bg = ((1.0 - mask) * alpha).mean()
        scale_reg = render_pkg["scaling"].prod(dim=1).mean() if render_pkg["scaling"].numel() else torch.tensor(0.0, device="cuda")
        scale_drift = torch.tensor(0.0, device="cuda")
        if gaussians._scaling.shape == initial_scaling.shape:
            scale_drift = (gaussians._scaling - initial_scaling).pow(2).mean()

        total = weight * float(rgb_weight) * (0.8 * rgb_fg + 0.2 * ssim_loss + 0.2 * rgb_bg)
        total = total + float(alpha_weight) * alpha_fg + float(outside_alpha_weight) * alpha_bg
        total = total + float(opt.lambda_dreg) * scale_reg + 0.01 * scale_drift

        gaussians.optimizer.zero_grad(set_to_none=True)
        total.backward()

        with torch.no_grad():
            if iteration < opt.update_until and iteration > opt.start_stat:
                gaussians.training_statis(opt, render_pkg, pred.shape[2], pred.shape[1])
                densify_count += 1
                if (
                    opt.densification
                    and iteration > opt.update_from
                    and densify_count % opt.update_interval == 0
                    and int(gaussians._anchor.shape[0]) < int(max_anchor_count)
                ):
                    gaussians.run_densify(opt, iteration)

            gaussians.optimizer.step()
            if gaussians._scaling.shape == initial_scaling.shape:
                gaussians._scaling.data.clamp_(min=initial_scaling - 1.25, max=initial_scaling + max_scale_log)
            gaussians._offset.data.clamp_(min=-float(max_offset_abs), max=float(max_offset_abs))

        loss_value = float(total.detach().item())
        loss_history.append(loss_value)
        if iteration == 1 or iteration % 10 == 0 or iteration == int(n_iterations):
            progress.set_postfix({
                "loss": f"{loss_value:.4f}",
                "src": entry["source"],
                "anchors": int(gaussians._anchor.shape[0]),
            })

    gaussians.eval()
    gaussians.save_ply(str(model_dir / "point_cloud.ply"))
    gaussians.save_mlp_checkpoints(str(model_dir))

    tail = loss_history[-min(50, len(loss_history)):] if loss_history else []
    summary = {
        "object_id": int(object_id),
        "mode": "scratch_object_training",
        "scratch_init_source": init_metadata.get("init_source", "unknown"),
        "scratch_init_metadata": init_metadata,
        "n_supervision_views": len(supervision_views),
        "source_counts": source_counts,
        "n_init_points": int(len(pcd.points)),
        "n_final_anchors": int(gaussians._anchor.shape[0]),
        "initial_anchor_count": int(initial_anchor_count),
        "densification_enabled": bool(enable_densification),
        "max_anchor_count": int(max_anchor_count),
        "densify_grad_threshold": float(densify_grad_threshold),
        "densify_extra_ratio": float(densify_extra_ratio),
        "final_sample_loss": float(loss_history[-1]) if loss_history else 0.0,
        "final_loss": float(np.mean(tail)) if tail else 0.0,
        "loss_history": loss_history,
        "model_dir": str(model_dir),
    }
    with open(out_dir / "scratch_training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return {"gaussians": gaussians, "summary": summary}
