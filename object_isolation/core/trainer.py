"""Object-only ObjectGS training for aligned real + hallucinated views.

This module starts from the scene COLMAP point cloud, creates a fresh
ObjectGS GaussianModel, and trains that model directly against the joint
real + hallucinated view set.

Key design choices vs. ObjectGS scene-level training:
    * Asymmetric depth supervision: real views use COLMAP-derived inverse-depth
        targets by default, matching ObjectGS' inverse-depth loss. Hallucinated
        (SV3D) views have NO depth target.
    * Camera pose optimization is enabled only for hallucinated SV3D cameras.
        Real COLMAP cameras never receive trainable pose deltas.
  * SSIM is enabled only on real views (SV3D textures are unreliable).
  * Hallucination loss weight decays in the late stage so noisy backside
    textures stop dragging the appearance MLPs after the geometry locks.
    * 2DGS normal consistency and distortion losses are enabled to make the
        generated object behave like a smooth surface instead of round clumps.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import cv2
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


def _skew(v: torch.Tensor) -> torch.Tensor:
    z = torch.zeros((), dtype=v.dtype, device=v.device)
    return torch.stack([
        torch.stack([z, -v[2], v[1]]),
        torch.stack([v[2], z, -v[0]]),
        torch.stack([-v[1], v[0], z]),
    ])


def _axis_angle_to_matrix(rotvec: torch.Tensor) -> torch.Tensor:
    theta = torch.linalg.norm(rotvec).clamp_min(1e-9)
    axis = rotvec / theta
    k = _skew(axis)
    eye = torch.eye(3, dtype=rotvec.dtype, device=rotvec.device)
    return eye + torch.sin(theta) * k + (1.0 - torch.cos(theta)) * (k @ k)


def _apply_pose_delta(cam, pose_delta: torch.Tensor):
    """Apply a small differentiable W2C delta to an SV3D virtual camera."""
    base_r = torch.as_tensor(cam.R, dtype=torch.float32, device="cuda")
    base_t = torch.as_tensor(cam.T, dtype=torch.float32, device="cuda")
    delta_r = _axis_angle_to_matrix(pose_delta[:3])
    delta_t = pose_delta[3:6]
    r = delta_r @ base_r
    t = delta_r @ base_t + delta_t

    rt = torch.eye(4, dtype=torch.float32, device="cuda")
    rt[:3, :3] = r
    rt[:3, 3] = t
    cam.world_view_transform = rt.transpose(0, 1)
    cam.full_proj_transform = (
        cam.world_view_transform.unsqueeze(0)
        .bmm(cam.projection_matrix.unsqueeze(0))
        .squeeze(0)
    )
    cam.camera_center = torch.linalg.inv(cam.world_view_transform)[3, :3]
    return cam


def _precompute_colmap_invdepths(
    *,
    entries: list,
    colmap_points: np.ndarray,
    dilate: int = 3,
    min_pixels: int = 16,
) -> int:
    """Project COLMAP seed points into real cameras as sparse inverse-depth maps.

    This mirrors ObjectGS' inverse-depth supervision path: compare 1/rendered
    depth against a camera-side invdepth map under a trusted mask. Hallucinated
    SV3D views are intentionally skipped.
    """
    points = np.asarray(colmap_points, dtype=np.float32)
    if points.size == 0:
        return 0

    kernel = None
    if int(dilate) > 1:
        k = int(dilate)
        if k % 2 == 0:
            k += 1
        kernel = np.ones((k, k), dtype=np.uint8)

    n_filled = 0
    for entry in entries:
        if entry.get("source") != "real":
            continue
        cam = entry["camera"]
        height, width = int(cam.image_height), int(cam.image_width)
        r = np.asarray(cam.R, dtype=np.float32)
        t = np.asarray(cam.T, dtype=np.float32).reshape(1, 3)
        cam_pts = points @ r.T + t
        z = cam_pts[:, 2]
        valid_z = z > 1e-4
        u = cam.fx * cam_pts[:, 0] / np.maximum(z, 1e-8) + cam.cx
        v = cam.fy * cam_pts[:, 1] / np.maximum(z, 1e-8) + cam.cy
        ui = np.rint(u).astype(np.int64)
        vi = np.rint(v).astype(np.int64)
        valid = valid_z & (ui >= 0) & (ui < width) & (vi >= 0) & (vi < height)
        if not valid.any():
            continue

        depth = np.full((height, width), np.inf, dtype=np.float32)
        np.minimum.at(depth, (vi[valid], ui[valid]), z[valid].astype(np.float32))
        finite = np.isfinite(depth)
        inv_depth = np.zeros((height, width), dtype=np.float32)
        inv_depth[finite] = 1.0 / np.maximum(depth[finite], 1e-6)
        depth_mask = finite.astype(np.uint8)
        if kernel is not None:
            inv_depth = cv2.dilate(inv_depth, kernel, iterations=1)
            depth_mask = cv2.dilate(depth_mask, kernel, iterations=1)

        fg_mask = entry["gt_mask"].detach().cpu().numpy()[0] > 0.5
        depth_mask = (depth_mask > 0) & fg_mask
        if int(depth_mask.sum()) < int(min_pixels):
            continue

        entry["gt_inv_depth"] = torch.from_numpy(inv_depth[None]).float().cuda()
        entry["depth_mask"] = torch.from_numpy(depth_mask.astype(np.float32)[None]).float().cuda()
        n_filled += 1
    return n_filled


def _prune_aabb_outliers(
    gaussians: GaussianModel,
    aabb_min: np.ndarray,
    aabb_max: np.ndarray,
    margin: float,
) -> int:
    """Remove anchors whose world-space position falls outside the object AABB
    (expanded by *margin* in each direction).

    These are floater anchors that drifted outside the object volume — the
    alpha-bg loss often can't catch them because they sit outside the 2D mask
    projection.  Safe to call inside ``torch.no_grad()`` at any time.
    """
    with torch.no_grad():
        lo = torch.tensor(aabb_min - margin, dtype=torch.float32, device="cuda")
        hi = torch.tensor(aabb_max + margin, dtype=torch.float32, device="cuda")
        pos = gaussians.get_anchor  # [N, 3]
        outside = ((pos < lo) | (pos > hi)).any(dim=1)  # [N] bool
        if not outside.any():
            return 0

        n_offsets = int(gaussians.n_offsets)
        keep = ~outside
        n_pruned = int(outside.sum().item())

        gaussians.prune_anchor(outside)

        for attr in ("offset_denom", "offset_gradient_accum", "offset_opacity_accum"):
            buf = getattr(gaussians, attr, None)
            if buf is not None:
                buf = buf.view(-1, n_offsets, 1)[keep].view(-1, 1)
                setattr(gaussians, attr, buf)

        for attr in ("anchor_opacity_accum", "anchor_demon"):
            buf = getattr(gaussians, attr, None)
            if buf is not None:
                setattr(gaussians, attr, buf[keep])

        buf = getattr(gaussians, "max_radii2D", None)
        if buf is not None:
            gaussians.max_radii2D = buf.view(-1, n_offsets)[keep].view(-1)

        return n_pruned


def _prune_oversized_anchors(
    gaussians: GaussianModel, max_world_scale: float) -> int:
    """Remove anchors whose max base scale exceeds *max_world_scale* world units.

    Calls ``prune_anchor`` then manually synchronises the stats buffers
    (``offset_denom``, ``offset_gradient_accum``, ``offset_opacity_accum``,
    ``anchor_opacity_accum``, ``anchor_demon``, ``max_radii2D``) so they remain
    consistent with the new anchor count.  Safe to call immediately after
    ``run_densify``.
    """
    with torch.no_grad():
        # get_scaling returns exp(_scaling), shape [N, 6]
        # Columns 0-2 are the per-axis anchor neighbourhood radius.
        anchor_scales = gaussians.get_scaling
        max_scale = anchor_scales[:, :3].max(dim=1).values  # [N]
        prune_mask = max_scale > float(max_world_scale)
        if not prune_mask.any():
            return 0

        n_offsets = int(gaussians.n_offsets)
        keep = ~prune_mask  # [N]
        n_pruned = int(prune_mask.sum().item())

        # Prune core tensors + optimizer state.
        gaussians.prune_anchor(prune_mask)

        # Sync [N*k, 1] offset stats buffers.
        for attr in ("offset_denom", "offset_gradient_accum", "offset_opacity_accum"):
            buf = getattr(gaussians, attr, None)
            if buf is not None:
                buf = buf.view(-1, n_offsets, 1)[keep].view(-1, 1)
                setattr(gaussians, attr, buf)

        # Sync [N, 1] anchor-level stats buffers.
        for attr in ("anchor_opacity_accum", "anchor_demon"):
            buf = getattr(gaussians, attr, None)
            if buf is not None:
                setattr(gaussians, attr, buf[keep])

        # Sync [N*k] max_radii2D.
        buf = getattr(gaussians, "max_radii2D", None)
        if buf is not None:
            gaussians.max_radii2D = buf.view(-1, n_offsets)[keep].view(-1)

        return n_pruned


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


def _precompute_real_view_depths(
    *,
    parent_gaussians,
    parent_pipe,
    entries: list,
    object_label_id: int,
    depth_alpha_threshold: float = 0.5,
) -> int:
    """Render parent ObjectGS depth at each real view, restricted to the object.

    Stores ``gt_depth`` (1,H,W) and ``depth_mask`` (1,H,W) on each real-view
    entry. Hallucinated views are skipped. Returns number of real views with
    depth populated.
    """
    if parent_gaussians is None:
        return 0
    labels = parent_gaussians.label_ids.squeeze(-1)
    object_anchor_mask = (labels == int(object_label_id))
    if int(object_anchor_mask.sum()) == 0:
        logger.warning("Parent has 0 anchors with label %d; skipping real-view depth.", object_label_id)
        return 0

    pipe = parent_pipe or SimpleNamespace(add_prefilter=True)
    if not hasattr(pipe, "add_prefilter"):
        pipe.add_prefilter = True
    bg = torch.zeros(3, dtype=torch.float32, device="cuda")
    n_filled = 0
    with torch.no_grad():
        for entry in entries:
            if entry.get("source") != "real":
                continue
            cam = entry["camera"]
            parent_gaussians.set_anchor_mask(cam.camera_center, cam.resolution_scale)
            visible_mask = parent_gaussians._anchor_mask & object_anchor_mask
            if int(visible_mask.sum()) == 0:
                continue
            pkg = _ogs_render(cam, parent_gaussians, pipe, bg,
                              visible_mask=visible_mask, training=False)
            depth = pkg.get("render_depth")
            alpha = pkg.get("render_alphas")
            if depth is None or alpha is None:
                continue
            if alpha.ndim == 3 and alpha.shape[0] != 1:
                alpha = alpha[0:1]
            elif alpha.ndim == 2:
                alpha = alpha.unsqueeze(0)
            if depth.ndim == 3 and depth.shape[0] != 1:
                depth = depth[0:1]
            elif depth.ndim == 2:
                depth = depth.unsqueeze(0)
            depth_mask = ((alpha > float(depth_alpha_threshold)).float()
                          * entry["gt_mask"]).clamp(0.0, 1.0)
            if float(depth_mask.sum()) < 16.0:
                continue
            entry["gt_depth"] = depth.detach().clone()
            entry["depth_mask"] = depth_mask.detach().clone()
            n_filled += 1
    return n_filled


def train_object(
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
    alpha_weight: float = 2.0,
    outside_alpha_weight: float = 2.0,
    enable_densification: bool = True,
    max_anchor_count: int = 20000,
    densify_grad_threshold: float = 0.00005,
    densify_extra_ratio: float = 0.08,
    densify_real_views_only: bool = True,
    max_splat_world_size: float = 0.15,
    aabb_prune_margin: float = 0.3,
    aabb_prune_interval: int = 500,
    max_scale_growth: float = 1.35,
    max_offset_abs: float = 0.45,
    enable_depth_supervision: bool = True,
    use_colmap_depth: bool = True,
    colmap_depth_dilate: int = 3,
    depth_weight: float = 0.5,
    depth_alpha_threshold: float = 0.5,
    depth_start_iter_frac: float = 0.0,
    halluc_decay_start_frac: float = 0.6,
    halluc_weight_floor: float = 0.5,
    scale_drift_weight: float = 0.05,
    normal_consistency_weight: float = 0.1,
    normal_start_iter_frac: float = 0.1,
    distortion_weight: float = 0.01,
    distortion_start_iter_frac: float = 0.1,
    enable_cam_pose_opt: bool = True,
    cam_pose_lr: float = 0.0001,
    cam_pose_rot_limit_deg: float = 8.0,
    cam_pose_trans_limit: float = 0.05,
) -> dict:
    """Train an object-only ObjectGS model from COLMAP seed points."""
    if not supervision_views:
        raise RuntimeError("Cannot train object model with no supervision views.")

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
            "gt_depth": None,
            "gt_inv_depth": None,
            "depth_mask": None,
            "pose_delta": None,
        })

    pose_params: list[nn.Parameter] = []
    if bool(enable_cam_pose_opt):
        for entry in entries:
            if entry["source"] == "hallucinated":
                delta = nn.Parameter(torch.zeros(6, dtype=torch.float32, device="cuda"))
                entry["pose_delta"] = delta
                pose_params.append(delta)
    pose_optimizer = (
        torch.optim.Adam(pose_params, lr=float(cam_pose_lr))
        if pose_params and float(cam_pose_lr) > 0.0
        else None
    )

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

    # One-shot prune of COLMAP seed points already outside the object AABB.
    # These are occasional noise points from nearby surfaces that sneak through
    # the extraction filter and become persistent floater anchors.
    _aabb_margin_world = float(aabb_prune_margin) * float(scope.radius)
    _n_init_pruned = _prune_aabb_outliers(
        gaussians,
        np.asarray(scope.aabb_min_W, dtype=np.float32),
        np.asarray(scope.aabb_max_W, dtype=np.float32),
        margin=_aabb_margin_world,
    )
    if _n_init_pruned:
        logger.info("AABB init-prune removed %d out-of-bounds seed anchors.", _n_init_pruned)

    opt = _training_options(
        int(n_iterations),
        float(lr_scale),
        enable_densification=bool(enable_densification),
        densify_grad_threshold=float(densify_grad_threshold),
        densify_extra_ratio=float(densify_extra_ratio),
    )
    gaussians.training_setup(opt)
    if getattr(gaussians, "active_sh_degree", None) is not None:
        logger.warning(
            "Object model color_attr=%s active_sh_degree=%s max_sh_degree=%s; "
            "consider using color_attr='RGB' to avoid SH shadow baking.",
            getattr(gaussians, "color_attr", "unknown"),
            getattr(gaussians, "active_sh_degree", None),
            getattr(gaussians, "max_sh_degree", None),
        )
    initial_scaling = gaussians._scaling.detach().clone()
    initial_anchor_count = int(gaussians._anchor.shape[0])
    max_scale_log = float(np.log(max(float(max_scale_growth), 1.001)))

    pipe = pipe_config or SimpleNamespace(add_prefilter=True)
    if not hasattr(pipe, "add_prefilter"):
        pipe.add_prefilter = True

    background = torch.ones(3, dtype=torch.float32, device="cuda")
    loss_history: list[float] = []
    depth_loss_history: list[float] = []
    normal_loss_history: list[float] = []
    distortion_loss_history: list[float] = []
    source_counts: dict[str, int] = {}
    for entry in entries:
        source_counts[entry["source"]] = source_counts.get(entry["source"], 0) + 1

    n_real_with_depth = 0
    depth_source = "none"
    if bool(enable_depth_supervision):
        if bool(use_colmap_depth):
            n_real_with_depth = _precompute_colmap_invdepths(
                entries=entries,
                colmap_points=np.asarray(pcd.points, dtype=np.float32),
                dilate=int(colmap_depth_dilate),
            )
            depth_source = "colmap_inverse_depth"
            logger.info(
                "COLMAP inverse-depth supervision: %d / %d real views populated.",
                n_real_with_depth, source_counts.get("real", 0),
            )
        else:
            n_real_with_depth = _precompute_real_view_depths(
                parent_gaussians=parent_gaussians,
                parent_pipe=pipe_config,
                entries=entries,
                object_label_id=int(object_id),
                depth_alpha_threshold=float(depth_alpha_threshold),
            )
            depth_source = "parent_render_depth"
            logger.info(
                "Parent-rendered depth supervision: %d / %d real views populated.",
                n_real_with_depth, source_counts.get("real", 0),
            )

    order = list(range(len(entries)))
    rng = np.random.default_rng(0)
    densify_count = 0
    aabb_prune_countdown = int(aabb_prune_interval)
    depth_start_iter = int(max(0, float(depth_start_iter_frac)) * float(n_iterations))
    normal_start_iter = int(max(0, float(normal_start_iter_frac)) * float(n_iterations))
    distortion_start_iter = int(max(0, float(distortion_start_iter_frac)) * float(n_iterations))

    progress = tqdm(range(1, int(n_iterations) + 1), desc=f"obj {int(object_id)}", dynamic_ncols=True)
    for iteration in progress:
        if (iteration - 1) % len(order) == 0:
            rng.shuffle(order)
        entry = entries[order[(iteration - 1) % len(order)]]
        gt = entry["gt_image"]
        mask = entry["gt_mask"]
        is_real = entry["source"] == "real"
        cam = entry["camera"]
        if not is_real and entry.get("pose_delta") is not None:
            cam = _apply_pose_delta(cam, entry["pose_delta"])

        # Hallucinated weight schedule: full early, decay after halluc_decay_start_frac.
        iter_frac = float(iteration) / float(max(1, n_iterations))
        if not is_real and iter_frac > float(halluc_decay_start_frac):
            decay_t = (iter_frac - float(halluc_decay_start_frac)) / max(
                1e-6, 1.0 - float(halluc_decay_start_frac))
            halluc_scale = 1.0 - float(decay_t) * (1.0 - float(halluc_weight_floor))
        else:
            halluc_scale = 1.0
        weight = float(entry["weight"]) * float(halluc_scale)

        gaussians.update_learning_rate(iteration)
        gaussians.set_anchor_mask(cam.camera_center, cam.resolution_scale)
        visible_mask = _prefilter(cam, gaussians).squeeze() if getattr(pipe, "add_prefilter", True) else gaussians._anchor_mask
        render_pkg = _ogs_render(cam, gaussians, pipe, background, visible_mask=visible_mask, training=True)
        pred = torch.clamp(render_pkg["render"], 0.0, 1.0)
        pred_depth = render_pkg.get("render_depth")
        alpha = render_pkg["render_alphas"]
        if alpha.ndim == 3:
            alpha = alpha[0:1]
        elif alpha.ndim == 2:
            alpha = alpha.unsqueeze(0)

        n_fg = mask.sum().clamp(min=1.0)
        n_bg = (1.0 - mask).sum().clamp(min=1.0)
        rgb_fg = torch.abs((pred - gt) * mask).sum() / (3.0 * n_fg)
        rgb_bg = torch.abs((pred - gt) * (1.0 - mask)).sum() / (3.0 * n_bg)
        # SSIM only on real views — SV3D textures are unreliable.
        if is_real:
            ssim_loss = 1.0 - ssim(pred.unsqueeze(0) * mask.unsqueeze(0),
                                   gt.unsqueeze(0) * mask.unsqueeze(0))
        else:
            ssim_loss = torch.tensor(0.0, device="cuda")
        alpha_fg = (mask * (1.0 - alpha)).mean()
        alpha_bg = ((1.0 - mask) * alpha).mean()
        scale_reg = render_pkg["scaling"].prod(dim=1).mean() if render_pkg["scaling"].numel() else torch.tensor(0.0, device="cuda")
        scale_drift = torch.tensor(0.0, device="cuda")
        if gaussians._scaling.shape == initial_scaling.shape:
            scale_drift = (gaussians._scaling - initial_scaling).pow(2).mean()

        # Asymmetric depth loss: only real views, only after warmup.
        # Default path follows ObjectGS: compare inverse rendered depth against
        # COLMAP-derived inverse-depth targets.
        gt_depth = entry.get("gt_depth")
        gt_inv_depth = entry.get("gt_inv_depth")
        depth_mask_t = entry.get("depth_mask")
        if (
            is_real
            and pred_depth is not None
            and depth_mask_t is not None
            and iteration >= depth_start_iter
        ):
            pd = pred_depth
            if pd.ndim == 3 and pd.shape[0] != 1:
                pd = pd[0:1]
            elif pd.ndim == 2:
                pd = pd.unsqueeze(0)
            if gt_inv_depth is not None:
                inv_pd = torch.where(pd > 0.0, 1.0 / pd.clamp(min=1e-6), torch.zeros_like(pd))
                depth_loss = (torch.abs(inv_pd - gt_inv_depth) * depth_mask_t).sum() / depth_mask_t.sum().clamp(min=1.0)
            elif gt_depth is not None:
                depth_loss = (torch.abs(pd - gt_depth) * depth_mask_t).sum() / depth_mask_t.sum().clamp(min=1.0)
            else:
                depth_loss = torch.tensor(0.0, device="cuda")
        else:
            depth_loss = torch.tensor(0.0, device="cuda")

        normal_loss = torch.tensor(0.0, device="cuda")
        if (
            float(normal_consistency_weight) > 0.0
            and iteration >= normal_start_iter
            and render_pkg.get("render_normals") is not None
            and render_pkg.get("render_normals_from_depth") is not None
        ):
            normals = render_pkg["render_normals"]
            normals_from_depth = render_pkg["render_normals_from_depth"]
            if normals.ndim == 4:
                normals = normals.squeeze(0)
            if normals_from_depth.ndim == 4:
                normals_from_depth = normals_from_depth.squeeze(0)
            normals = normals.permute(2, 0, 1)
            normals_from_depth = normals_from_depth * alpha.permute(1, 2, 0).detach()
            normals_from_depth = normals_from_depth.permute(2, 0, 1)
            normal_error = (1.0 - (normals * normals_from_depth).sum(dim=0, keepdim=True)).clamp(min=0.0)
            normal_loss = (normal_error * mask).sum() / mask.sum().clamp(min=1.0)

        distortion_loss = torch.tensor(0.0, device="cuda")
        if (
            float(distortion_weight) > 0.0
            and iteration >= distortion_start_iter
            and render_pkg.get("render_distort") is not None
        ):
            render_distort = render_pkg["render_distort"]
            if render_distort.ndim == 4:
                render_distort = render_distort.squeeze(0)
            if render_distort.ndim == 3 and render_distort.shape[-1] == 1:
                render_distort = render_distort.squeeze(-1)
            if render_distort.ndim == 2:
                render_distort = render_distort.unsqueeze(0)
            distortion_loss = (render_distort * mask).sum() / mask.sum().clamp(min=1.0)

        total = weight * float(rgb_weight) * (0.8 * rgb_fg + 0.2 * ssim_loss + 0.2 * rgb_bg)
        total = total + float(alpha_weight) * alpha_fg + float(outside_alpha_weight) * alpha_bg
        total = total + float(opt.lambda_dreg) * scale_reg + float(scale_drift_weight) * scale_drift
        total = total + float(depth_weight) * depth_loss
        total = total + float(normal_consistency_weight) * normal_loss
        total = total + float(distortion_weight) * distortion_loss

        gaussians.optimizer.zero_grad(set_to_none=True)
        if pose_optimizer is not None:
            pose_optimizer.zero_grad(set_to_none=True)
        total.backward()

        with torch.no_grad():
            if iteration < opt.update_until and iteration > opt.start_stat:
                # Only accumulate gradient stats on real views when requested.
                # Hallucinated back-views would otherwise split anchors in
                # unseen geometry, producing a bloated noisy back shell.
                if not bool(densify_real_views_only) or is_real:
                    gaussians.training_statis(opt, render_pkg, pred.shape[2], pred.shape[1])
                densify_count += 1
                if (
                    opt.densification
                    and iteration > opt.update_from
                    and densify_count % opt.update_interval == 0
                    and int(gaussians._anchor.shape[0]) < int(max_anchor_count)
                ):
                    gaussians.run_densify(opt, iteration)
                    # Kill anchors that grew unreasonably large (covers the
                    # case where a few splats blow up to cover the viewport).
                    if float(max_splat_world_size) > 0.0:
                        n_over = _prune_oversized_anchors(
                            gaussians,
                            float(max_splat_world_size) * float(spatial_extent),
                        )
                        if n_over:
                            logger.debug("Size-prune removed %d oversized anchors.", n_over)

            # Periodic AABB pruning: kill anchors that drifted outside the
            # object bounding box.  These are floaters that produce the
            # trailing specks visible in orbit renders.
            aabb_prune_countdown -= 1
            if aabb_prune_countdown <= 0 and float(aabb_prune_margin) > 0.0:
                aabb_prune_countdown = int(aabb_prune_interval)
                n_aabb = _prune_aabb_outliers(
                    gaussians,
                    np.asarray(scope.aabb_min_W, dtype=np.float32),
                    np.asarray(scope.aabb_max_W, dtype=np.float32),
                    margin=_aabb_margin_world,
                )
                if n_aabb:
                    logger.debug("AABB-prune iter %d: removed %d floater anchors.", iteration, n_aabb)
                    # Resize initial_scaling ref if anchor count changed.
                    if gaussians._scaling.shape != initial_scaling.shape:
                        initial_scaling = gaussians._scaling.detach().clone()

            gaussians.optimizer.step()
            if pose_optimizer is not None:
                pose_optimizer.step()
                rot_limit = float(np.deg2rad(float(cam_pose_rot_limit_deg)))
                trans_limit = float(cam_pose_trans_limit) * float(scope.radius)
                for pose_delta in pose_params:
                    pose_delta[:3].clamp_(min=-rot_limit, max=rot_limit)
                    pose_delta[3:6].clamp_(min=-trans_limit, max=trans_limit)
            if gaussians._scaling.shape == initial_scaling.shape:
                gaussians._scaling.data.clamp_(min=initial_scaling - 1.25, max=initial_scaling + max_scale_log)
            gaussians._offset.data.clamp_(min=-float(max_offset_abs), max=float(max_offset_abs))

        loss_value = float(total.detach().item())
        loss_history.append(loss_value)
        depth_loss_history.append(float(depth_loss.detach().item()))
        normal_loss_history.append(float(normal_loss.detach().item()))
        distortion_loss_history.append(float(distortion_loss.detach().item()))
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
        "mode": "object_training",
        "init_source": init_metadata.get("init_source", "unknown"),
        "init_metadata": init_metadata,
        "n_supervision_views": len(supervision_views),
        "source_counts": source_counts,
        "n_init_points": int(len(pcd.points)),
        "n_final_anchors": int(gaussians._anchor.shape[0]),
        "initial_anchor_count": int(initial_anchor_count),
        "densification_enabled": bool(enable_densification),
        "densify_real_views_only": bool(densify_real_views_only),
        "max_splat_world_size": float(max_splat_world_size),
        "max_anchor_count": int(max_anchor_count),
        "densify_grad_threshold": float(densify_grad_threshold),
        "densify_extra_ratio": float(densify_extra_ratio),
        "depth_supervision_enabled": bool(enable_depth_supervision),
        "use_colmap_depth": bool(use_colmap_depth),
        "depth_source": str(depth_source),
        "colmap_depth_dilate": int(colmap_depth_dilate),
        "n_real_views_with_depth": int(n_real_with_depth),
        "depth_weight": float(depth_weight),
        "depth_alpha_threshold": float(depth_alpha_threshold),
        "depth_start_iter": int(depth_start_iter),
        "cam_pose_opt_enabled": bool(enable_cam_pose_opt and pose_optimizer is not None),
        "cam_pose_lr": float(cam_pose_lr),
        "cam_pose_optimized_views": int(len(pose_params)),
        "cam_pose_rot_limit_deg": float(cam_pose_rot_limit_deg),
        "cam_pose_trans_limit": float(cam_pose_trans_limit),
        "normal_consistency_weight": float(normal_consistency_weight),
        "normal_start_iter": int(normal_start_iter),
        "distortion_weight": float(distortion_weight),
        "distortion_start_iter": int(distortion_start_iter),
        "color_attr": str(getattr(gaussians, "color_attr", "unknown")),
        "active_sh_degree": None if getattr(gaussians, "active_sh_degree", None) is None else int(gaussians.active_sh_degree),
        "max_sh_degree": None if getattr(gaussians, "max_sh_degree", None) is None else int(gaussians.max_sh_degree),
        "halluc_decay_start_frac": float(halluc_decay_start_frac),
        "halluc_weight_floor": float(halluc_weight_floor),
        "scale_drift_weight": float(scale_drift_weight),
        "alpha_weight": float(alpha_weight),
        "outside_alpha_weight": float(outside_alpha_weight),
        "final_sample_loss": float(loss_history[-1]) if loss_history else 0.0,
        "final_loss": float(np.mean(tail)) if tail else 0.0,
        "loss_history": loss_history,
        "depth_loss_history": depth_loss_history,
        "normal_loss_history": normal_loss_history,
        "distortion_loss_history": distortion_loss_history,
        "model_dir": str(model_dir),
    }
    with open(out_dir / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return {"gaussians": gaussians, "summary": summary}
