import logging
from pathlib import Path
from typing import Any, cast
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from gstrain.gaussian_renderer.render import prefilter_voxel as _prefilter
from gstrain.gaussian_renderer.render import render as _gstrain_render
from gstrain.vroom_core.models.facade import GaussianModel
from gstrain.vroom_core.training.loss_engine import ssim_loss as _ssim_loss
from gstrain.vroom_core.training.orchestration import TrainingConfig, PipelineConfig

from .utils.gstrain_wrapper import make_camera
from .utils.colmap_init import load_colmap_object_point_cloud
from .constants import GAUSSIAN_MODEL_DEFAULTS
from .config import ObjectTrainingConfig

logger = logging.getLogger(__name__)


def train_object(
    *,
    built_views,
    pipeline_config,
    scope,
    object_id,
    model_path,
    output_dir,
    extraction_index_path=None,
    parent_gaussians=None,
    config = ObjectTrainingConfig(),
):
    if not built_views:
        raise RuntimeError("No views provided.")

    n_iterations = config.iterations
    lr_scale = config.lr_scale
    max_init_points = config.max_init_points
    colmap_init_target_points = config.colmap_init_target_points
    rgb_weight = config.rgb_weight
    generated_rgb_scale = config.generated_rgb_scale
    alpha_weight = config.alpha_weight
    outside_alpha_weight = config.outside_alpha_weight
    depth_weight = config.depth_weight
    depth_start_iter = config.depth_start_iter
    depth_front_weight = config.depth_front_weight
    depth_back_weight = config.depth_back_weight
    depth_alpha_threshold = config.depth_alpha_threshold
    enable_densification = config.enable_densification
    max_anchor_count = config.max_anchor_count
    densify_grad_threshold = config.densify_grad_threshold
    max_offset_abs = config.max_offset_abs

    model_dir = Path(output_dir) / "06_model"
    model_dir.mkdir(parents=True, exist_ok=True)

    #convert inputs to cuda tensors
    entries: list[dict[str, Any]] = []
    for view in built_views:
        rgb = np.asarray(view["rgb"], dtype=np.float32) / 255.0

        mask = np.asarray(view["mask"], dtype=np.float32)
        camera = make_camera(view["camera"]["R"],
            view["camera"]["T"],
            view["camera"]["K"],
            view["camera"]["width"],
            view["camera"]["height"]
        )
        target_image = torch.from_numpy(rgb).permute(2, 0, 1).cuda()
        target_mask = torch.from_numpy(mask).unsqueeze(0).cuda()
        weight = float(view["weight"])
        source = view["source"]

        entries.append({
            "camera": camera,
            "target_image": target_image,
            "target_mask": target_mask,
            "weight": weight,
            "source": source,
            "depth_target": None,
            "depth_valid": None,
        })
    if isinstance(pipeline_config, dict):
        pipeline = PipelineConfig(**pipeline_config)
    else:
        raise RuntimeError("Invalid pipeline configuration")

    # Render parent depth targets
    if depth_weight > 0.0 and parent_gaussians is not None:
        with torch.no_grad():
            obj_mask = parent_gaussians.label_ids.squeeze() == object_id
            if obj_mask.any():
                for entry in entries:
                    if entry["source"] != "real":
                        continue

                    parent_gaussians.set_anchor_mask(entry["camera"].camera_center, entry["camera"].resolution_scale)
                    try:
                        vis = _prefilter(entry["camera"], parent_gaussians).squeeze()
                    except Exception:
                        vis = parent_gaussians._anchor_mask

                    pkg = _gstrain_render(
                        entry["camera"], parent_gaussians, pipeline,
                        torch.zeros(3, device="cuda"), visible_mask=vis,
                        training=False, object_mask=obj_mask
                    )

                    depth = pkg.get("render_depth")
                    if depth is None:
                        continue
                    depth = cast(Any, depth)
                    depth = depth[0:1] if depth.ndim == 3 else depth.unsqueeze(0) if depth.ndim == 2 else None
                    if depth is None:
                        continue

                    alpha = pkg.get("render_alphas")
                    if alpha is None:
                        continue
                    alpha = cast(Any, alpha)
                    alpha = alpha[0:1] if alpha.ndim == 3 else alpha.unsqueeze(0) if alpha.ndim == 2 else alpha

                    valid = ((entry["target_mask"] > 0.5) & (alpha > depth_alpha_threshold) & torch.isfinite(depth) & (depth > 0.0)).float()
                    if int(valid.sum().item()) >= 64:
                        entry["depth_target"] = depth.detach()
                        entry["depth_valid"] = valid.detach()

    pcd, _ = load_colmap_object_point_cloud(
        model_path=model_path, object_id=object_id, scope=scope,
        extraction_index_path=extraction_index_path,
        max_points=max_init_points, target_points=colmap_init_target_points,
    )

    # Load model hyper-parameters from parent gaussians or use defaults
    kwargs = {
        k: getattr(parent_gaussians, k, GAUSSIAN_MODEL_DEFAULTS[k])
        for k in GAUSSIAN_MODEL_DEFAULTS
    } if parent_gaussians is not None else GAUSSIAN_MODEL_DEFAULTS.copy()

    gaussians = GaussianModel(
        gs_attr=str(kwargs["gs_attr"]),
        feat_dim=int(kwargs["feat_dim"]),
        view_dim=int(kwargs["view_dim"]),
        appearance_dim=int(kwargs["appearance_dim"]),
        n_offsets=int(kwargs["n_offsets"]),
        voxel_size=float(kwargs["voxel_size"]),
        render_mode=str(kwargs["render_mode"]),
        tile_size_2dgs=int(kwargs["tile_size_2dgs"]),
    )
    object.__setattr__(gaussians, "explicit_gs", False)
    gaussians.weed_ratio = 0.0
    gaussians.set_appearance(len(entries))

    # Initialize anchors
    spatial_extent = max(scope.radius, float(np.linalg.norm(scope.aabb_max - scope.aabb_min)))
    gaussians.initialize_anchors(pcd, spatial_extent, logger=logger)

    # load training configs
    opt = TrainingConfig()
    for k, v in getattr(scope, "optim_params", {}).items():
        if hasattr(opt, k):
            setattr(opt, k, v)

    #freeze anchor
    opt.iterations = n_iterations
    opt.position_lr_init = opt.position_lr_final = opt.appearance_lr_init = opt.appearance_lr_final = 0.0
    opt.position_lr_max_steps = opt.offset_lr_max_steps = opt.mlp_opacity_lr_max_steps = opt.mlp_cov_lr_max_steps = opt.mlp_color_lr_max_steps = opt.appearance_lr_max_steps = n_iterations

    # learning rates scaled by lr_scale
    opt.offset_lr_init = config.offset_lr_init * lr_scale
    opt.offset_lr_final = config.offset_lr_final * lr_scale
    opt.feature_lr = config.feature_lr * lr_scale
    opt.scaling_lr = config.scaling_lr * lr_scale
    opt.rotation_lr = config.rotation_lr * lr_scale

    opt.mlp_opacity_lr_init = config.mlp_opacity_lr_init * lr_scale
    opt.mlp_opacity_lr_final = config.mlp_opacity_lr_final * lr_scale
    opt.mlp_cov_lr_init = opt.mlp_cov_lr_final = config.mlp_cov_lr * lr_scale
    opt.mlp_color_lr_init = config.mlp_color_lr_init * lr_scale
    opt.mlp_color_lr_final = config.mlp_color_lr_final * lr_scale

    opt.densification = enable_densification
    if enable_densification:
        opt.update_until = n_iterations
    else:
        opt.update_until = 0
    opt.densify_grad_threshold = densify_grad_threshold

    opt.start_stat = max(25, min(500, n_iterations // 8))
    opt.update_from = max(50, min(1500, n_iterations // 4))
    opt.update_interval = max(25, min(100, n_iterations // 20))

    gaussians.training_setup(opt)
    optimizer = gaussians.optimizer
    if optimizer is None:
        raise RuntimeError("Optimizer is not initialized")
    initial_scaling = gaussians._scaling.detach().clone()

    background = torch.ones(3, dtype=torch.float32, device="cuda")
    loss_hist, depth_hist = [], []

    order = list(range(len(entries)))
    rng = np.random.default_rng(0)
    densify_count = 0

    progress = tqdm(range(1, n_iterations + 1), desc=f"obj {object_id}", dynamic_ncols=True)
    for iteration in progress:
        if (iteration - 1) % len(order) == 0:
            rng.shuffle(order)
        entry = entries[order[(iteration - 1) % len(order)]]
        camera, target_image, target_mask = entry["camera"], entry["target_image"], entry["target_mask"]
        weight = entry["weight"]
        rgb_scale = 1.0 if entry["source"] == "real" else generated_rgb_scale

        gaussians.update_learning_rate(iteration)
        gaussians.set_anchor_mask(camera.camera_center, camera.resolution_scale)
        vis = _prefilter(camera, gaussians).squeeze() if getattr(pipeline, "add_prefilter", True) else gaussians._anchor_mask

        pkg = _gstrain_render(camera, gaussians, pipeline, background, visible_mask=vis, training=True)
        pred = torch.clamp(pkg["render"], 0.0, 1.0)
        
        # Inline conversion of alpha tensor
        alpha = pkg["render_alphas"]
        alpha = alpha[0:1] if alpha.ndim == 3 else alpha.unsqueeze(0) if alpha.ndim == 2 else alpha

        # Compute reconstruction and regularization losses
        diff = torch.abs(pred - target_image)
        n_fg, n_bg = target_mask.sum().clamp(min=1.0), (1.0 - target_mask).sum().clamp(min=1.0)
        
        rgb_fg = (diff * target_mask).sum() / (3.0 * n_fg)
        rgb_bg = (diff * (1.0 - target_mask)).sum() / (3.0 * n_bg)
        
        ssim_l = _ssim_loss((pred * target_mask).unsqueeze(0), (target_image * target_mask).unsqueeze(0))
        
        scale_reg = pkg["scaling"].prod(dim=1).mean() if pkg["scaling"].numel() else torch.tensor(0.0, device="cuda")
        scale_drift = (
            (gaussians._scaling - initial_scaling).pow(2).mean()
            if gaussians._scaling.shape == initial_scaling.shape
            else torch.tensor(0.0, device="cuda")
        )

        total = weight * rgb_weight * rgb_scale * (0.8 * rgb_fg + 0.2 * ssim_l + 0.2 * rgb_bg)
        total += alpha_weight * (target_mask * (1.0 - alpha)).mean()
        total += outside_alpha_weight * ((1.0 - target_mask) * alpha).mean()
        total += opt.lambda_dreg * scale_reg + 0.01 * scale_drift

        # Inline conversion of depth
        depth_loss = torch.tensor(0.0, device="cuda")
        dt, dv = entry["depth_target"], entry["depth_valid"]
        pd = pkg.get("render_depth")
        if pd is not None:
            pd = cast(Any, pd)
            pd = pd[0:1] if pd.ndim == 3 else pd.unsqueeze(0) if pd.ndim == 2 else pd

        if depth_weight > 0.0 and iteration >= depth_start_iter and dt is not None and dv is not None and pd is not None:
            rel = (pd - dt) / dt.detach().abs().clamp_min(1e-3)
            depth_loss = (
                (depth_front_weight * F.relu(-rel) + depth_back_weight * F.relu(rel)) * dv
            ).sum() / dv.sum().clamp_min(1.0)
            total += depth_weight * depth_loss

        optimizer.zero_grad(set_to_none=True)
        total.backward()

        with torch.no_grad():
            if opt.update_until > iteration > opt.start_stat:
                gaussians.training_statis(opt, pkg, pred.shape[2], pred.shape[1])
                densify_count += 1
                if (
                    opt.densification
                    and densify_count % opt.update_interval == 0
                    and gaussians._anchor.shape[0] < max_anchor_count
                ):
                    gaussians.run_densify(opt, iteration)
            optimizer.step()
            if gaussians._scaling.shape == initial_scaling.shape:
                gaussians._scaling.data.clamp_(max=initial_scaling.max().item())
            gaussians._offset.data.clamp_(min=-max_offset_abs, max=max_offset_abs)

        loss_hist.append(float(total.detach().item()))
        depth_hist.append(float(depth_loss.detach().item()))
        if iteration == 1 or iteration % 10 == 0 or iteration == n_iterations:
            progress.set_postfix({
                "loss": f"{loss_hist[-1]:.4f}",
                "depth": f"{depth_hist[-1]:.4f}",
                "src": entry["source"],
                "anchors": gaussians._anchor.shape[0],
            })

    gaussians.eval()
    gaussians.save_ply(str(model_dir / "point_cloud.ply"))
    gaussians.save_mlp_checkpoints(str(model_dir))

    tail = loss_hist[-min(50, len(loss_hist)):]
    summary = {
        "n_final_anchors": gaussians._anchor.shape[0],
        "final_loss": float(np.mean(tail)) if tail else 0.0,
    }
    return {"gaussians": gaussians, "summary": summary}
