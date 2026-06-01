import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from gstrain.gaussian_renderer.render import prefilter_voxel as _prefilter
from gstrain.gaussian_renderer.render import render as _gstrain_render
from gstrain.vroom_core.models.facade import GaussianModel
from gstrain.vroom_core.training.loss_engine import ssim_loss as _ssim_loss

from .utils.gstrain_wrapper import make_camera
from .utils.colmap_init import load_colmap_object_point_cloud

logger = logging.getLogger(__name__)


def _img_tensor(rgb):
    arr = np.asarray(rgb)
    arr = (arr.astype(np.float32) / 255.0) if arr.dtype == np.uint8 else np.clip(arr.astype(np.float32), 0.0, 1.0)
    return torch.from_numpy(arr).permute(2, 0, 1).cuda()


def _mask_tensor(mask):
    return torch.from_numpy(np.asarray(mask).astype(np.float32)).unsqueeze(0).cuda()


def _depth_tensor(depth):
    if depth is None:
        return None
    d = depth if isinstance(depth, torch.Tensor) else torch.as_tensor(depth)
    return d[0:1].float() if d.ndim == 3 else d.unsqueeze(0).float() if d.ndim == 2 else None


def _alpha_tensor(alpha):
    if alpha.ndim == 3:
        return alpha[0:1].float()
    if alpha.ndim == 2:
        return alpha.unsqueeze(0).float()
    return alpha.reshape(1, *alpha.shape[-2:]).float()


def _model_kwargs(parent=None):
    defaults = {
        "gs_attr": "2D", "feat_dim": 32, "view_dim": 3, "appearance_dim": 0,
        "n_offsets": 10, "voxel_size": 0.001, "render_mode": "RGB+ED", "tile_size_2dgs": 8,
    }
    if parent is None:
        return defaults
    return {k: getattr(parent, k, defaults[k]) for k in defaults}


def _render_depth_target(parent_gaussians, pipe_config, cam, object_id, real_mask, alpha_threshold=0.35):
    """Render per-pixel depth from the parent model as a depth supervision target."""
    if parent_gaussians is None or not hasattr(parent_gaussians, "label_ids"):
        return None, None
    with torch.no_grad():
        obj_mask = parent_gaussians.label_ids.squeeze() == int(object_id)
        if not bool(obj_mask.any().item()):
            return None, None
        parent_gaussians.set_anchor_mask(cam.camera_center, cam.resolution_scale)
        try:
            vis = _prefilter(cam, parent_gaussians).squeeze()
        except Exception:
            vis = parent_gaussians._anchor_mask
        bg = torch.zeros(3, dtype=torch.float32, device="cuda")
        pkg = _gstrain_render(cam, parent_gaussians, pipe_config, bg,
                              visible_mask=vis, training=False, object_mask=obj_mask)
        depth = _depth_tensor(pkg.get("render_depth"))
        if depth is None:
            return None, None
        alpha = _alpha_tensor(pkg["render_alphas"])
        valid = (
            (real_mask > 0.5) & (alpha > float(alpha_threshold))
            & torch.isfinite(depth) & (depth > 0.0)
        ).float()
        return (depth.detach(), valid.detach()) if int(valid.sum().item()) >= 64 else (None, None)


def _make_opt(n, s, *, enable_densification, densify_grad_threshold, densify_extra_ratio):
    return SimpleNamespace(
        iterations=n,
        position_lr_init=0.0, position_lr_final=0.0, position_lr_delay_mult=0.01, position_lr_max_steps=n,
        offset_lr_init=0.0040 * s, offset_lr_final=0.00005 * s, offset_lr_delay_mult=0.01, offset_lr_max_steps=n,
        feature_lr=0.0075 * s, scaling_lr=0.0015 * s, rotation_lr=0.0020 * s,
        mlp_opacity_lr_init=0.0020 * s, mlp_opacity_lr_final=0.000020 * s,
        mlp_opacity_lr_delay_mult=0.01, mlp_opacity_lr_max_steps=n,
        mlp_cov_lr_init=0.0040 * s, mlp_cov_lr_final=0.0040 * s,
        mlp_cov_lr_delay_mult=0.01, mlp_cov_lr_max_steps=n,
        mlp_color_lr_init=0.0080 * s, mlp_color_lr_final=0.000050 * s,
        mlp_color_lr_delay_mult=0.01, mlp_color_lr_max_steps=n,
        appearance_lr_init=0.0, appearance_lr_final=0.0, appearance_lr_delay_mult=0.01, appearance_lr_max_steps=n,
        lambda_dssim=0.2, lambda_dreg=0.0001,
        start_stat=max(25, min(500, n // 8)),
        update_from=max(50, min(1500, n // 4)),
        update_interval=max(25, min(100, n // 20)),
        update_until=max(1, n) if bool(enable_densification) else 0,
        overlap=False, densification=bool(enable_densification),
        growing_type="mean", pruning_type="mean",
        min_opacity=0.005, success_threshold=0.8,
        densify_grad_threshold=float(densify_grad_threshold),
        update_ratio=0.2, extra_ratio=float(densify_extra_ratio), extra_up=0.05,
    )


def train_object(
    *,
    supervision_views,
    scope,
    object_id,
    model_path,
    output_dir,
    n_iterations,
    extraction_index_path=None,
    parent_gaussians=None,
    pipe_config=None,
    lr_scale=1.0,
    max_init_points=20000,
    colmap_init_target_points=8000,
    rgb_weight=1.0,
    hallucination_rgb_scale=1.0,
    alpha_weight=1.0,
    outside_alpha_weight=5.0,
    depth_weight=0.1,
    depth_start_iter=100,
    depth_front_weight=1.0,
    depth_back_weight=0.15,
    depth_alpha_threshold=0.35,
    enable_densification=False,
    max_anchor_count=20000,
    densify_grad_threshold=0.00005,
    densify_extra_ratio=0.08,
    max_offset_abs=0.45,
):
    if not supervision_views:
        raise RuntimeError("No supervision views.")

    out_dir = Path(output_dir)
    model_dir = out_dir / "06_model"
    model_dir.mkdir(parents=True, exist_ok=True)

    entries = [{
        "camera": make_camera(v["camera"]["R"], v["camera"]["T"],
                              v["camera"]["K"], v["camera"]["width"], v["camera"]["height"]),
        "gt_image": _img_tensor(v["rgb"]),
        "gt_mask": _mask_tensor(v["mask"]),
        "weight": float(v.get("weight", 1.0)),
        "source": str(v.get("source", "unknown")),
        "depth_target": None,
        "depth_valid": None,
    } for v in supervision_views]

    n_depth_targets = 0
    if float(depth_weight) > 0.0 and parent_gaussians is not None:
        for entry in entries:
            if entry["source"] != "real":
                continue
            dt, dv = _render_depth_target(
                parent_gaussians, pipe_config, entry["camera"],
                int(object_id), entry["gt_mask"], float(depth_alpha_threshold),
            )
            entry["depth_target"], entry["depth_valid"] = dt, dv
            if dt is not None:
                n_depth_targets += 1

    pcd, init_meta = load_colmap_object_point_cloud(
        model_path=model_path, object_id=int(object_id), scope=scope,
        extraction_index_path=extraction_index_path,
        max_points=int(max_init_points), target_points=int(colmap_init_target_points),
    )

    gaussians = GaussianModel(**_model_kwargs(parent_gaussians))
    gaussians.explicit_gs = False
    gaussians.weed_ratio = 0.0
    gaussians.set_appearance(len(entries))
    spatial_extent = max(
        float(scope.radius),
        float(np.linalg.norm(np.asarray(scope.aabb_max, np.float32) - np.asarray(scope.aabb_min, np.float32))),
    )
    gaussians.initialize_anchors(pcd, spatial_extent, logger=logger)

    opt = _make_opt(
        int(n_iterations), float(lr_scale),
        enable_densification=bool(enable_densification),
        densify_grad_threshold=float(densify_grad_threshold),
        densify_extra_ratio=float(densify_extra_ratio),
    )
    gaussians.training_setup(opt)
    initial_scaling = gaussians._scaling.detach().clone()
    initial_anchor_count = int(gaussians._anchor.shape[0])

    if isinstance(pipe_config, dict):
        pipe = SimpleNamespace(**pipe_config)
    else:
        pipe = pipe_config or SimpleNamespace()
    if not hasattr(pipe, "add_prefilter"):
        pipe.add_prefilter = True

    background = torch.ones(3, dtype=torch.float32, device="cuda")
    loss_hist, depth_hist = [], []
    source_counts = {}
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
        cam, gt, mask = entry["camera"], entry["gt_image"], entry["gt_mask"]
        weight = float(entry["weight"])
        rgb_scale = 1.0 if entry["source"] == "real" else float(hallucination_rgb_scale)

        gaussians.update_learning_rate(iteration)
        gaussians.set_anchor_mask(cam.camera_center, cam.resolution_scale)
        vis = (
            _prefilter(cam, gaussians).squeeze()
            if getattr(pipe, "add_prefilter", True)
            else gaussians._anchor_mask
        )
        pkg = _gstrain_render(cam, gaussians, pipe, background, visible_mask=vis, training=True)
        pred = torch.clamp(pkg["render"], 0.0, 1.0)
        alpha = pkg["render_alphas"]
        alpha = alpha[0:1] if alpha.ndim == 3 else alpha.unsqueeze(0) if alpha.ndim == 2 else alpha

        n_fg = mask.sum().clamp(min=1.0)
        n_bg = (1.0 - mask).sum().clamp(min=1.0)
        rgb_fg = torch.abs((pred - gt) * mask).sum() / (3.0 * n_fg)
        rgb_bg = torch.abs((pred - gt) * (1.0 - mask)).sum() / (3.0 * n_bg)
        ssim_l = _ssim_loss(
            pred.unsqueeze(0) * mask.unsqueeze(0),
            gt.unsqueeze(0) * mask.unsqueeze(0),
        )
        scale_reg = (
            pkg["scaling"].prod(dim=1).mean()
            if pkg["scaling"].numel()
            else torch.tensor(0.0, device="cuda")
        )
        scale_drift = (
            (gaussians._scaling - initial_scaling).pow(2).mean()
            if gaussians._scaling.shape == initial_scaling.shape
            else torch.tensor(0.0, device="cuda")
        )

        total = weight * float(rgb_weight) * rgb_scale * (0.8 * rgb_fg + 0.2 * ssim_l + 0.2 * rgb_bg)
        total = total + float(alpha_weight) * (mask * (1.0 - alpha)).mean()
        total = total + float(outside_alpha_weight) * ((1.0 - mask) * alpha).mean()
        total = total + float(opt.lambda_dreg) * scale_reg + 0.01 * scale_drift

        depth_loss = torch.tensor(0.0, device="cuda")
        dt, dv, pd = entry["depth_target"], entry["depth_valid"], _depth_tensor(pkg.get("render_depth"))
        if float(depth_weight) > 0.0 and iteration >= int(depth_start_iter) and dt is not None and dv is not None and pd is not None:
            rel = (pd - dt) / dt.detach().abs().clamp_min(1e-3)
            depth_loss = (
                (float(depth_front_weight) * F.relu(-rel) + float(depth_back_weight) * F.relu(rel)) * dv
            ).sum() / dv.sum().clamp_min(1.0)
            total = total + float(depth_weight) * depth_loss

        gaussians.optimizer.zero_grad(set_to_none=True)
        total.backward()

        with torch.no_grad():
            if opt.update_until > iteration > opt.start_stat:
                gaussians.training_statis(opt, pkg, pred.shape[2], pred.shape[1])
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

        loss_hist.append(float(total.detach().item()))
        depth_hist.append(float(depth_loss.detach().item()))
        if iteration == 1 or iteration % 10 == 0 or iteration == int(n_iterations):
            progress.set_postfix({
                "loss": f"{loss_hist[-1]:.4f}",
                "depth": f"{depth_hist[-1]:.4f}",
                "src": entry["source"],
                "anchors": int(gaussians._anchor.shape[0]),
            })

    gaussians.eval()
    gaussians.save_ply(str(model_dir / "point_cloud.ply"))
    gaussians.save_mlp_checkpoints(str(model_dir))

    tail = loss_hist[-min(50, len(loss_hist)):]
    summary = {
        "object_id": int(object_id),
        "mode": "object_training",
        "init_source": init_meta.get("init_source", "unknown"),
        "n_supervision_views": len(supervision_views),
        "n_depth_target_views": int(n_depth_targets),
        "source_counts": source_counts,
        "n_init_points": int(len(pcd.points)),
        "n_final_anchors": int(gaussians._anchor.shape[0]),
        "initial_anchor_count": int(initial_anchor_count),
        "densification_enabled": bool(enable_densification),
        "hallucination_rgb_scale": float(hallucination_rgb_scale),
        "depth_weight": float(depth_weight),
        "final_loss": float(np.mean(tail)) if tail else 0.0,
        "model_dir": str(model_dir),
    }
    return {"gaussians": gaussians, "summary": summary}
