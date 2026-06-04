import logging
import dataclasses
from pathlib import Path
from typing import Any, cast
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from object_refiner.utils.gstrain_wrapper import prefilter_anchors as _prefilter
from object_refiner.utils.gstrain_wrapper import render_rgba as _gstrain_render
from object_refiner.utils.gstrain_wrapper import build_vroom_gaussians, save_vroom_checkpoint, ssim_loss
from gstrain.vroom_core.core.model.density import DensifcationController
from gstrain.vroom_core.utilities.utils.training import Optimizer as GstrainOptimizer
from .utils.gstrain_wrapper import make_camera
from .utils.colmap_init import load_colmap_object_point_cloud
from .constants import GAUSSIAN_MODEL_DEFAULTS
from .config import ObjectTrainingConfig

logger = logging.getLogger(__name__)


def train_object(
    *,
    built_views,
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

    # Render parent depth targets
    if config.depth_weight > 0.0 and parent_gaussians is not None:
        with torch.no_grad():
            if parent_gaussians.anchor_cloud.semantic_labels is not None:
                obj_mask = parent_gaussians.anchor_cloud.semantic_labels.squeeze() == object_id
                if obj_mask.any():
                    for entry in entries:
                        if entry["source"] != "real":
                            continue

                        visible = _prefilter(parent_gaussians, entry["camera"])

                        render = _gstrain_render(
                            parent_gaussians,
                            entry["camera"],
                            bg_white=False,
                            object_label_id=object_id,
                            training=False,
                            visible_mask=visible,
                        )

                        depth = render.get("render_depth")
                        if depth is None:
                            continue
                        depth = cast(Any, depth)
                        depth = depth[0:1] if depth.ndim == 3 else depth.unsqueeze(0) if depth.ndim == 2 else None
                        if depth is None:
                            continue

                        alpha = render.get("render_alphas")
                        if alpha is None:
                            continue
                        alpha = cast(Any, alpha)
                        alpha = alpha[0:1] if alpha.ndim == 3 else alpha.unsqueeze(0) if alpha.ndim == 2 else alpha

                        valid = ((entry["target_mask"] > 0.5) & (alpha > config.depth_alpha_threshold) & torch.isfinite(depth) & (depth > 0.0)).float()
                        if int(valid.sum().item()) >= 64:
                            entry["depth_target"] = depth.detach()
                            entry["depth_valid"] = valid.detach()

    point_cloud, _ = load_colmap_object_point_cloud(
        model_path=model_path, object_id=object_id, scope=scope,
        extraction_index_path=extraction_index_path,
        max_points=config.max_init_points, target_points=config.colmap_init_target_points,
    )

    # Load model hyper-parameters from parent gaussians or use defaults
    kwargs = {
        k: getattr(parent_gaussians, k, GAUSSIAN_MODEL_DEFAULTS[k])
        for k in GAUSSIAN_MODEL_DEFAULTS
    } if parent_gaussians is not None else GAUSSIAN_MODEL_DEFAULTS.copy()

    gaussians = build_vroom_gaussians(kwargs)

    # Initialize anchors
    spatial_extent = max(scope.radius, float(np.linalg.norm(scope.aabb_max - scope.aabb_min)))
    gaussians.anchor_cloud.initialize_anchors(point_cloud)

    opt = config.get_optim_args()

    configs = {
        "optimization": {
            "args": opt,
            "spatial_lr_scale": spatial_extent,
            "anchor_cloud": gaussians.anchor_cloud,
            "decoder": gaussians.decoder,
        }
    }
    densifier = DensifcationController(
        quantization_size=gaussians.anchor_cloud.quantization_size,
        anchor_cloud=gaussians.anchor_cloud,
        optimizer=None,
        num_gaussians_per_anchor=gaussians.decoder.number_gaussians_per_anchor,
        gradient_threshold=opt.densify_grad_threshold,
    )
    opt_wrapper = GstrainOptimizer(configs["optimization"], densifier)
    opt_wrapper.setup()
    densifier.optimizer = opt_wrapper
    optimizer = opt_wrapper.optimizer

    if optimizer is None:
        raise RuntimeError("Optimizer is not initialized")
    initial_scaling = gaussians.anchor_cloud.anchors_log_scales.detach().clone()

    background = torch.ones(3, dtype=torch.float32, device="cuda")
    loss_hist, depth_hist = [], []

    order = list(range(len(entries)))
    rng = np.random.default_rng(0)
    densify_count = 0

    progress = tqdm(range(1, config.iterations + 1), desc=f"obj {object_id}", dynamic_ncols=True)
    for iteration in progress:
        if (iteration - 1) % len(order) == 0:
            rng.shuffle(order)
        entry = entries[order[(iteration - 1) % len(order)]]
        camera, target_image, target_mask = entry["camera"], entry["target_image"], entry["target_mask"]
        weight = entry["weight"]
        rgb_scale = 1.0 if entry["source"] == "real" else config.generated_rgb_scale

        opt_wrapper.step_learning_rate(iteration)
        vis = _prefilter(gaussians, camera)

        pkg = _gstrain_render(
            gaussians,
            camera,
            bg_white=True,
            training=True,
            visible_mask=vis,
        )
        pred = torch.clamp(pkg["render"], 0.0, 1.0)
        
        # Inline conversion of alpha tensor
        alpha = pkg["render_alphas"]
        alpha = alpha[0:1] if alpha.ndim == 3 else alpha.unsqueeze(0) if alpha.ndim == 2 else alpha

        # Compute reconstruction and regularization losses
        diff = torch.abs(pred - target_image)
        n_fg, n_bg = target_mask.sum().clamp(min=1.0), (1.0 - target_mask).sum().clamp(min=1.0)
        
        rgb_fg = (diff * target_mask).sum() / (3.0 * n_fg)
        rgb_bg = (diff * (1.0 - target_mask)).sum() / (3.0 * n_bg)
        
        ssim_l = ssim_loss((pred * target_mask).unsqueeze(0), (target_image * target_mask).unsqueeze(0))
        
        scale_reg = pkg["scaling"].prod(dim=1).mean() if pkg["scaling"].numel() else torch.tensor(0.0, device="cuda")
        scale_drift = (
            (gaussians.anchor_cloud.anchors_log_scales - initial_scaling).pow(2).mean()
            if gaussians.anchor_cloud.anchors_log_scales.shape == initial_scaling.shape
            else torch.tensor(0.0, device="cuda")
        )

        total = weight * config.rgb_weight * rgb_scale * (0.8 * rgb_fg + 0.2 * ssim_l + 0.2 * rgb_bg)
        total += config.alpha_weight * (target_mask * (1.0 - alpha)).mean()
        total += config.outside_alpha_weight * ((1.0 - target_mask) * alpha).mean()
        total += opt.lambda_dreg * scale_reg + 0.01 * scale_drift

        # Inline conversion of depth
        depth_loss = torch.tensor(0.0, device="cuda")
        dt, dv = entry["depth_target"], entry["depth_valid"]
        pd = pkg.get("render_depth")
        if pd is not None:
            pd = cast(Any, pd)
            pd = pd[0:1] if pd.ndim == 3 else pd.unsqueeze(0) if pd.ndim == 2 else pd

        if config.depth_weight > 0.0 and iteration >= config.depth_start_iter and dt is not None and dv is not None and pd is not None:
            rel = (pd - dt) / dt.detach().abs().clamp_min(1e-3)
            depth_loss = (
                (config.depth_front_weight * F.relu(-rel) + config.depth_back_weight * F.relu(rel)) * dv
            ).sum() / dv.sum().clamp_min(1.0)
            total += config.depth_weight * depth_loss

        optimizer.zero_grad(set_to_none=True)
        total.backward()

        with torch.no_grad():
            if opt.update_until > iteration > opt.start_stat:
                rendered_2d_points = pkg.get("rendered_2d_points")
                points_grad_detached = None
                if rendered_2d_points is not None and rendered_2d_points.grad is not None:
                    points_grad_detached = rendered_2d_points.grad.detach().clone()

                densifier.update_densification_state(
                    visibility_mask=pkg["visible_anchors_mask"],
                    negative_opacity_filter=pkg["negative_opacity_filter"],
                    opacity=pkg["opacity"],
                    points_grad=points_grad_detached,
                    width=pred.shape[2],
                    height=pred.shape[1],
                )

                densify_count += 1
                if (
                    opt.densification
                    and densify_count % opt.update_interval == 0
                    and gaussians.anchor_cloud.anchors_positions.shape[0] < config.max_anchor_count
                ):
                    densifier.growing_operation()
                    densifier.pruning_operation(
                        opacity_threshold=getattr(opt, "min_opacity", 0.005)
                    )
                    densifier.reset_state()
            optimizer.step()
            if gaussians.anchor_cloud.anchors_log_scales.shape == initial_scaling.shape:
                gaussians.anchor_cloud.anchors_log_scales.data.clamp_(max=initial_scaling.max().item())
            gaussians.anchor_cloud.gaussians_offsets.data.clamp_(min=-config.max_offset_abs, max=config.max_offset_abs)

        loss_hist.append(float(total.detach().item()))
        depth_hist.append(float(depth_loss.detach().item()))
        if iteration == 1 or iteration % 10 == 0 or iteration == config.iterations:
            progress.set_postfix({
                "loss": f"{loss_hist[-1]:.4f}",
                "depth": f"{depth_hist[-1]:.4f}",
                "src": entry["source"],
                "anchors": gaussians.anchor_cloud.anchors_positions.shape[0],
            })

    gaussians.decoder.eval()
    gaussians.anchor_cloud.eval()
    save_vroom_checkpoint(gaussians, str(model_dir / "point_cloud.ply"), str(model_dir))

    tail = loss_hist[-min(50, len(loss_hist)):]
    summary = {
        "n_final_anchors": gaussians.anchor_cloud.anchors_positions.shape[0],
        "final_loss": float(np.mean(tail)) if tail else 0.0,
    }
    return {"gaussians": gaussians, "summary": summary}
