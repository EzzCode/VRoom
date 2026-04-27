"""
Optimizer — Fine-tune Scaffold-GS 2DGS anchors using novel view supervision.

Uses Era3D-generated novel views as supervision targets to improve object
geometry on the unseen hemisphere. MLPs are frozen by default to prevent
catastrophic forgetting; opacity MLP adaptation is available as an explicit
low-LR experiment for seeded backside anchors.

Public API:
    optimize_with_novel_views(gaussians, pipe, views, ...) -> dict
"""

__all__ = ['optimize_with_novel_views']

import sys
import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

_VROOM_ROOT = Path(__file__).resolve().parent.parent.parent
_OBJECTGS_DIR = _VROOM_ROOT / "temp_deps" / "ObjectGS"
if str(_OBJECTGS_DIR) not in sys.path:
    sys.path.insert(0, str(_OBJECTGS_DIR))


def optimize_with_novel_views(
    gaussians,
    pipe_config,
    novel_views: list,
    n_iterations: int = 1200,
    lr_scale: float = 1.0,
    hallucination_weight: float = 0.6,
    silhouette_weight: float = 1.5,
    lambda_dssim: float = 0.2,
    novel_rgb_weight: float = 1.0,
    target_mask_erode_px: int = 0,
    freeze_feat_when_rgb_off: bool = True,
    object_id: int = None,
    object_anchors = None,
    object_radius: float = None,
    object_center = None,
    anchor_update_mask = None,
    seeded_anchor_mask = None,
    originals_anchor_mask = None,
    seeded_scale_reg_weight: float = 0.0,
    outside_alpha_weight: float = 2.0,
    seeded_offset_reg_weight: float = 0.05,
    seeded_max_scale_delta: float = 0.30,
    seeded_max_offset_delta: float = 0.30,
    originals_lr_scale: float = 0.05,
    originals_max_scale_delta: float = 0.05,
    originals_max_offset_delta: float = 0.05,
    originals_reg_weight: float = 0.5,
    feat_lr_scale: float = 0.25,
    feat_reg_weight: float = 0.05,
    seed_opacity_gate_lr_scale: float = 50.0,
    seed_opacity_gate_reg_weight: float = 0.005,
    seed_opacity_lift_lr_scale: float = 10.0,
    seed_opacity_lift_reg_weight: float = 0.02,
    aabb_min = None,
    aabb_max = None,
    cage_padding_frac: float = 0.05,
    scale_ceiling_log: float = None,
    silhouette_iou_thresh: float = 0.2,
    hole_weight_max: float = 2.5,
    seeded_anisotropy_max: float = 3.0,
    train_mlp_opacity: bool = False,
    mlp_opacity_lr_scale: float = 0.001,
    mlp_opacity_reg_weight: float = 1.0,
    train_mlp_cov: bool = False,
    mlp_cov_lr_scale: float = 0.03,
    preservation_cameras: list = None,
    preservation_weight: float = 1.0,
    save_path: str = None,
    reference_model_path: str = None,
) -> dict:
    """Fine-tune 2DGS model using Era3D novel views as supervision.

    Args:
        gaussians: GaussianModel (modified in-place).
        pipe_config: ObjectGS pipeline config.
        novel_views: list of dicts with 'rgb' (np.ndarray), 'camera' (dict with R,T,K,w,h).
        n_iterations: Fine-tuning iterations.
        lr_scale: Learning rate relative to original training LR.
        hallucination_weight: Loss weight for Era3D views (< 1.0 since hallucinated).
        silhouette_weight: Loss weight for projected anchor silhouette supervision.
        lambda_dssim: Weight for SSIM loss component.
        novel_rgb_weight: Extra multiplier on direct hallucinated RGB L1/SSIM.
            Set to 0.0 to keep geometry/mask supervision while disabling direct
            pixel fitting to Zero123++ output.
        target_mask_erode_px: Optional erosion radius for generated foreground
            masks. Positive values remove uncertain edges/drips from supervision.
        freeze_feat_when_rgb_off: When direct RGB supervision is disabled,
            prevent alpha/silhouette losses from changing anchor features.
            ObjectGS uses anchor features for opacity/color MLP inputs, so
            geometry-only losses can otherwise create colored streaks.
        object_id: Object label to optimize (used for object-masked rendering).
        object_anchors: Optional (N, 3) array of target object anchor positions.
        object_radius: Optional object radius used to scale the silhouette footprint.
        object_center: Optional object center used for diagnostics and future constraints.
        anchor_update_mask: Optional boolean tensor/array with shape (N_anchors,)
            indicating which anchors are allowed to update.
        seeded_anchor_mask: Optional boolean tensor/array with shape (N_anchors,)
            indicating which anchors were newly seeded.
        seeded_scale_reg_weight: Weight for seeded-scale regularization.
        outside_alpha_weight: Weight penalizing rendered alpha outside object mask.
        seeded_offset_reg_weight: Weight for seeded offset drift regularization.
        seeded_max_scale_delta: Hard max absolute delta for seeded scaling params.
        seeded_max_offset_delta: Hard max absolute delta for seeded offset params.
        train_mlp_opacity: If True, adapt mlp_opacity with a very conservative LR.
        mlp_opacity_lr_scale: LR multiplier for mlp_opacity when enabled.
        mlp_opacity_reg_weight: MSE-to-initial-weights regularization for mlp_opacity.
        seed_opacity_lift_lr_scale: LR multiplier for seeded pre-mask opacity lifts.
        seed_opacity_lift_reg_weight: Magnitude regularization for seeded opacity lifts.
        train_mlp_cov: If True, train mlp_cov with a conservative LR.
        mlp_cov_lr_scale: LR multiplier for mlp_cov when enabled.
        preservation_cameras: Optional list of real training camera dicts for
            anti-regression preservation supervision.
        preservation_weight: Weight for preservation loss term.
        save_path: If provided, save updated model here.
        reference_model_path: Path to original ObjectGS model dir containing
            config.yaml / cameras.json for save compatibility.

    Returns:
        dict with 'loss_history', 'final_loss'.
    """
    import torch
    import numpy as np
    import cv2
    from utils.loss_utils import l1_loss, ssim
    from gaussian_renderer.render import render as objectgs_render
    from gaussian_renderer.render import prefilter_voxel
    from target_replenishment.core.objectgs_bridge import create_virtual_camera, project_anchor_silhouette
    from target_replenishment.core.image_alignment import align_image_to_render_bbox

    if not novel_views:
        logger.warning("No novel views — nothing to optimize.")
        return {'loss_history': [], 'final_loss': 0.0}

    # ── Prepare supervision tensors ──
    supervision = []
    supervision_diagnostics = []
    n_views_dropped = 0
    n_aligned = 0
    align_bg = torch.ones(3, dtype=torch.float32, device="cuda")
    aabb_corners_np = None
    if (aabb_min is not None) and (aabb_max is not None):
        amin = np.asarray(aabb_min, dtype=np.float32).reshape(3)
        amax = np.asarray(aabb_max, dtype=np.float32).reshape(3)
        aabb_corners_np = np.array([
            [amin[0], amin[1], amin[2]],
            [amax[0], amin[1], amin[2]],
            [amin[0], amax[1], amin[2]],
            [amax[0], amax[1], amin[2]],
            [amin[0], amin[1], amax[2]],
            [amax[0], amin[1], amax[2]],
            [amin[0], amax[1], amax[2]],
            [amax[0], amax[1], amax[2]],
        ], dtype=np.float32)

    for view in novel_views:
        cam_p = view['camera']
        cam = create_virtual_camera(cam_p['R'], cam_p['T'], cam_p['K'],
                                    cam_p['width'], cam_p['height'])

        rgb_np = view['rgb']
        if rgb_np.dtype == np.uint8:
            rgb_uint8 = rgb_np
        else:
            rgb_uint8 = np.clip(rgb_np * 255.0, 0, 255).astype(np.uint8)

        # ── 2D bbox alignment: warp Zero123++ RGB to match the model's framing ──
        # Without this, the diffusion target is centred/scaled by Zero123++'s
        # convention while the supervision render uses our virtual camera's
        # framing. L1 against differently-framed pixels teaches the optimizer
        # to translate / scale gaussians to chase the diffusion's framing
        # (visible as the object drifting + growing axial spikes).
        try:
            with torch.no_grad():
                gaussians.set_anchor_mask(cam.camera_center, cam.resolution_scale)
                if hasattr(pipe_config, 'add_prefilter') and pipe_config.add_prefilter:
                    a_visible_mask = prefilter_voxel(cam, gaussians).squeeze()
                else:
                    a_visible_mask = gaussians._anchor_mask
                a_pkg = objectgs_render(
                    cam, gaussians, pipe_config, align_bg,
                    visible_mask=a_visible_mask, training=False,
                )
            render_rgb = a_pkg['render'].detach().clamp(0.0, 1.0)
            render_uint8 = (render_rgb.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
            aligned_uint8, dx, dy, ascale = align_image_to_render_bbox(
                rgb_uint8, render_uint8,
                bg_color=(255, 255, 255), return_diag=True,
            )
            # Only accept the warp when it's a sane correction. If the model
            # render at the supervision pose has no meaningful foreground (e.g.
            # backside is completely empty pre-seeding), the renderer-bbox is
            # garbage — skip alignment for that view rather than warp blindly.
            if not (0.25 <= ascale <= 4.0):
                aligned_uint8 = rgb_uint8
                dx = dy = 0.0
                ascale = 1.0
            else:
                n_aligned += 1
            logger.info(
                "Align view az=%s el=%s: dx=%+.1fpx dy=%+.1fpx scale=%.3f",
                cam_p.get('azimuth_offset_deg', '?'),
                cam_p.get('elevation_offset_deg', '?'),
                dx, dy, ascale,
            )
            rgb_uint8 = aligned_uint8
        except Exception as e:
            logger.warning("Bbox alignment failed (%s); using raw Zero123++ RGB.", e)

        rgb_np = rgb_uint8.astype(np.float32) / 255.0

        gt_image = torch.from_numpy(rgb_np).permute(2, 0, 1).float().cuda()
        # Generated views use white background. Use non-white pixels as object supervision mask.
        gt_mask_np = (rgb_np.mean(axis=2) < 0.98)
        gt_mask_np = _largest_component_mask(gt_mask_np, min_pixels=64)
        erode_px = int(max(0, target_mask_erode_px))
        if erode_px > 0 and gt_mask_np.any():
            kernel_size = 2 * erode_px + 1
            kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
            gt_mask_eroded = cv2.erode(gt_mask_np.astype(np.uint8), kernel, iterations=1) > 0
            # If erosion deletes a very thin object view, fall back to the
            # original mask rather than producing an all-background target.
            if gt_mask_eroded.sum() >= 64:
                gt_mask_np = gt_mask_eroded
        gt_object_mask = torch.from_numpy(gt_mask_np.astype(np.float32)).unsqueeze(0).cuda()

        silhouette_target = None
        if object_anchors is not None and len(object_anchors) > 0:
            silhouette_np = project_anchor_silhouette(
                cam,
                object_anchors,
                object_radius=object_radius,
                height=cam.image_height,
                width=cam.image_width,
            )
            silhouette_target = torch.from_numpy(silhouette_np).unsqueeze(0).float().cuda()

        # Per-view confidence: IoU between novel-view foreground mask and the
        # projected AABB silhouette. Drops geometrically implausible Zero123++
        # outputs whose silhouette doesn't agree with the object's AABB.
        confidence = 1.0
        silhouette_iou = None
        if aabb_corners_np is not None:
            aabb_mask_np = _project_aabb_silhouette(cam, aabb_corners_np,
                                                    cam.image_height, cam.image_width)
            inter = float((gt_mask_np & aabb_mask_np).sum())
            union = float((gt_mask_np | aabb_mask_np).sum())
            iou = inter / max(union, 1.0)
            silhouette_iou = float(iou)
            if iou < float(silhouette_iou_thresh):
                logger.info(
                    "Dropping novel view (az_offset=%s): silhouette IoU %.3f < %.3f",
                    view.get('camera', {}).get('azimuth_offset_deg', '?'),
                    iou, silhouette_iou_thresh,
                )
                supervision_diagnostics.append({
                    'supervision_index': None,
                    'azimuth_offset_deg': cam_p.get('azimuth_offset_deg'),
                    'elevation_offset_deg': cam_p.get('elevation_offset_deg'),
                    'bbox_dx': float(dx),
                    'bbox_dy': float(dy),
                    'bbox_scale': float(ascale),
                    'silhouette_iou': silhouette_iou,
                    'confidence': 0.0,
                    'dropped': True,
                })
                n_views_dropped += 1
                continue
            # Soft confidence weighting: bad-but-acceptable views weigh less.
            confidence = float(np.clip(iou, 0.25, 1.0))

        supervision_diagnostics.append({
            'supervision_index': int(len(supervision)),
            'azimuth_offset_deg': cam_p.get('azimuth_offset_deg'),
            'elevation_offset_deg': cam_p.get('elevation_offset_deg'),
            'bbox_dx': float(dx),
            'bbox_dy': float(dy),
            'bbox_scale': float(ascale),
            'silhouette_iou': silhouette_iou,
            'confidence': float(confidence),
            'dropped': False,
        })

        supervision.append({
            'camera': cam,
            'gt_image': gt_image,
            'gt_object_mask': gt_object_mask,
            'gt_silhouette_mask': silhouette_target,
            'weight': float(view.get('weight', hallucination_weight)) * confidence,
        })

    if not supervision:
        logger.warning("All novel views dropped by silhouette IoU filter.")
        return {'loss_history': [], 'final_loss': 0.0,
                'view_usage_counts': [], 'param_delta_norms': {},
            'n_views_dropped': n_views_dropped,
            'supervision_diagnostics': supervision_diagnostics}

    logger.info("Bbox-aligned %d/%d supervision views to model framing.",
                n_aligned, len(supervision))

    # Optional real-view preservation targets to avoid degrading already-good frontsides.
    preservation = []
    n_preserve_real = 0
    if preservation_cameras:
        bg_preserve = torch.ones(3, dtype=torch.float32, device="cuda")
        for cam_p in preservation_cameras:
            cam = create_virtual_camera(cam_p['R'], cam_p['T'], cam_p['K'],
                                        cam_p['width'], cam_p['height'])

            ref_image_tensor = None
            img_path = cam_p.get('image_path')
            if img_path:
                try:
                    from PIL import Image as _PIL
                    img = _PIL.open(str(img_path)).convert('RGB')
                    if (img.size[0], img.size[1]) != (int(cam_p['width']), int(cam_p['height'])):
                        img = img.resize((int(cam_p['width']), int(cam_p['height'])), _PIL.LANCZOS)
                    arr = np.asarray(img).astype(np.float32) / 255.0  # (H,W,3)
                    ref_image_tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous().float().cuda()
                    n_preserve_real += 1
                except Exception as e:
                    logger.warning("Preservation: failed to load real image %s (%s); "
                                   "falling back to self-rendered snapshot.", img_path, e)
                    ref_image_tensor = None

            if ref_image_tensor is None:
                gaussians.set_anchor_mask(cam.camera_center, cam.resolution_scale)
                if hasattr(pipe_config, 'add_prefilter') and pipe_config.add_prefilter:
                    visible_mask = prefilter_voxel(cam, gaussians).squeeze()
                else:
                    visible_mask = gaussians._anchor_mask
                with torch.no_grad():
                    ref_pkg = objectgs_render(
                        cam,
                        gaussians,
                        pipe_config,
                        bg_preserve,
                        visible_mask=visible_mask,
                        training=False,
                    )
                ref_image_tensor = ref_pkg['render'].detach()

            preservation.append({
                'camera': cam,
                'ref_image': ref_image_tensor,
            })
        logger.info("Preservation: %d cameras (%d using real training images, %d self-rendered)",
                    len(preservation), n_preserve_real, len(preservation) - n_preserve_real)

    # ── Setup ──
    gaussians.train()

    # FREEZE most MLPs to prevent catastrophic forgetting. Opacity adaptation
    # is explicit and opt-in: seeded backside anchors are OOD pairs for the
    # pre-trained opacity MLP, so freezing it can make the response field flat.
    logger.info(
        "MLP policy: %s opacity, freeze color, %s cov",
        "train" if train_mlp_opacity else "freeze",
        "train" if train_mlp_cov else "freeze",
    )
    if hasattr(gaussians, 'mlp_opacity'):
        for param in gaussians.mlp_opacity.parameters():
            param.requires_grad = bool(train_mlp_opacity)
    if hasattr(gaussians, 'mlp_color'):
        for param in gaussians.mlp_color.parameters():
            param.requires_grad = False
    if hasattr(gaussians, 'mlp_cov'):
        for param in gaussians.mlp_cov.parameters():
            param.requires_grad = bool(train_mlp_cov)

    _setup_finetune_optimizer(
        gaussians,
        lr_scale,
        train_mlp_opacity=train_mlp_opacity,
        mlp_opacity_lr_scale=mlp_opacity_lr_scale,
        train_mlp_cov=train_mlp_cov,
        mlp_cov_lr_scale=mlp_cov_lr_scale,
        feat_lr_scale=feat_lr_scale,
        seed_opacity_gate_lr_scale=seed_opacity_gate_lr_scale,
        seed_opacity_lift_lr_scale=seed_opacity_lift_lr_scale,
    )

    tracked_param_names = ["_offset", "_anchor_feat", "_scaling"]
    initial_params = {}
    for name in tracked_param_names:
        if hasattr(gaussians, name):
            initial_params[name] = getattr(gaussians, name).detach().clone()

    initial_mlp_opacity = []
    if train_mlp_opacity and hasattr(gaussians, 'mlp_opacity'):
        initial_mlp_opacity = [p.detach().clone() for p in gaussians.mlp_opacity.parameters()]

    bg_color = torch.ones(3, dtype=torch.float32, device="cuda")  # white bg to match Era3D
    loss_history = []
    view_usage_counts = [0 for _ in supervision]

    logger.info(
        f"Starting fine-tuning: {n_iterations} iters, "
        f"{len(supervision)} views, lr_scale={lr_scale}, "
        f"hallucination_weight={hallucination_weight}"
    )

    obj_anchor_mask = None
    if object_id is not None:
        obj_anchor_mask = (gaussians.label_ids.squeeze() == int(object_id))

    grad_update_mask = None
    if anchor_update_mask is not None:
        import torch
        if isinstance(anchor_update_mask, torch.Tensor):
            grad_update_mask = anchor_update_mask.detach().to(device="cuda", dtype=torch.bool)
        else:
            grad_update_mask = torch.tensor(anchor_update_mask, device="cuda", dtype=torch.bool)

    seeded_mask = None
    if seeded_anchor_mask is not None:
        if isinstance(seeded_anchor_mask, torch.Tensor):
            seeded_mask = seeded_anchor_mask.detach().to(device="cuda", dtype=torch.bool)
        else:
            seeded_mask = torch.tensor(seeded_anchor_mask, device="cuda", dtype=torch.bool)

    originals_mask = None
    if originals_anchor_mask is not None:
        if isinstance(originals_anchor_mask, torch.Tensor):
            originals_mask = originals_anchor_mask.detach().to(device="cuda", dtype=torch.bool)
        else:
            originals_mask = torch.tensor(originals_anchor_mask, device="cuda", dtype=torch.bool)

    # Prepare AABB cage tensors and absolute scale ceiling.
    cage_min_t = None
    cage_max_t = None
    cage_padding_t = None
    extent_vec_t = None
    if (aabb_min is not None) and (aabb_max is not None):
        cage_min_t = torch.as_tensor(np.asarray(aabb_min, dtype=np.float32),
                                     device="cuda", dtype=torch.float32).reshape(3)
        cage_max_t = torch.as_tensor(np.asarray(aabb_max, dtype=np.float32),
                                     device="cuda", dtype=torch.float32).reshape(3)
        extent_vec_t = (cage_max_t - cage_min_t).abs()
        cage_padding_t = extent_vec_t * float(cage_padding_frac)

    scale_ceiling_val = None
    if scale_ceiling_log is not None:
        scale_ceiling_val = float(scale_ceiling_log)

    # Per-axis offset cage. Vector of shape (3,) so thin axes get tight caps and
    # long axes stay reasonable. Broadcasted over (N, n_offsets, 3) at clamp time.
    if cage_padding_t is not None:
        offset_abs_cap_vec = cage_padding_t.detach().reshape(1, 1, 3)  # broadcast
        offset_abs_cap = float(cage_padding_t.max().item())  # scalar fallback / telemetry
        extent_max_val = float(extent_vec_t.max().item())
        extent_med_val = float(extent_vec_t.median().item())
        extent_min_val = float(extent_vec_t.min().item())
        # Per-axis delta-from-init caps for the seeded and originals masks.
        seeded_offset_delta_vec = (extent_vec_t * float(seeded_max_offset_delta)).detach().reshape(1, 1, 3)
        originals_offset_delta_vec = (extent_vec_t * float(originals_max_offset_delta)).detach().reshape(1, 1, 3)
    else:
        offset_abs_cap_vec = None
        offset_abs_cap = None
        extent_max_val = None
        extent_med_val = None
        extent_min_val = None
        seeded_offset_delta_vec = None
        originals_offset_delta_vec = None

    last_loss_seeded_scale_reg = torch.tensor(0.0, device="cuda")
    last_loss_seeded_offset_reg = torch.tensor(0.0, device="cuda")
    last_loss_outside_alpha = torch.tensor(0.0, device="cuda")
    last_loss_seed_gate_reg = torch.tensor(0.0, device="cuda")

    # Telemetry counters.
    aabb_escape_total = 0
    n_anchors_caged_last = 0
    n_scale_clipped_last = 0
    n_aniso_capped_last = 0

    for iteration in range(1, n_iterations + 1):
        # Deterministic round-robin ensures every generated view contributes.
        sv_idx = (iteration - 1) % len(supervision)
        sv = supervision[sv_idx]
        view_usage_counts[sv_idx] += 1
        cam = sv['camera']

        # Render
        gaussians.set_anchor_mask(cam.camera_center, cam.resolution_scale)
        if hasattr(pipe_config, 'add_prefilter') and pipe_config.add_prefilter:
            visible_mask = prefilter_voxel(cam, gaussians).squeeze()
        else:
            visible_mask = gaussians._anchor_mask

        render_pkg = objectgs_render(
            cam, gaussians, pipe_config, bg_color,
            visible_mask=visible_mask, training=True, object_mask=obj_anchor_mask,
        )
        rendered = render_pkg['render']

        gt_image = sv['gt_image']
        gt_object_mask = sv['gt_object_mask']
        gt_silhouette_mask = sv.get('gt_silhouette_mask')

        # Hole-focused weighting: emphasize pixels where target says object exists
        # but rendered alpha is weak. Cap multiplier so alpha-spill outside the
        # AABB can't dominate the gradient.
        render_alpha = render_pkg.get('render_alphas')
        if render_alpha is None:
            hole_weight = gt_object_mask
        else:
            alpha = torch.clamp(render_alpha, 0.0, 1.0)
            boost = 2.0 * torch.clamp(gt_object_mask - alpha, min=0.0)
            boost = torch.clamp(boost, max=float(hole_weight_max) - 1.0)
            hole_weight = gt_object_mask * (1.0 + boost)

        hole_weight_3 = hole_weight.expand_as(rendered)
        denom = torch.clamp(hole_weight_3.sum(), min=1.0)
        Ll1_hole = torch.abs(rendered - gt_image).mul(hole_weight_3).sum() / denom

        # Keep SSIM over object region by masking both tensors.
        rendered_masked = rendered * gt_object_mask
        gt_masked = gt_image * gt_object_mask
        Lssim_obj = 1.0 - ssim(rendered_masked, gt_masked)

        loss_novel = (
            float(novel_rgb_weight)
            * sv['weight']
            * ((1.0 - lambda_dssim) * Ll1_hole + lambda_dssim * Lssim_obj)
        )

        loss_silhouette = torch.tensor(0.0, device="cuda")
        if gt_silhouette_mask is not None and render_alpha is not None:
            alpha = torch.clamp(render_alpha, 0.0, 1.0)
            if alpha.ndim == 2:
                alpha = alpha.unsqueeze(0)
            loss_silhouette = l1_loss(alpha, gt_silhouette_mask)

        loss_outside_alpha = torch.tensor(0.0, device="cuda")
        if render_alpha is not None:
            alpha = torch.clamp(render_alpha, 0.0, 1.0)
            if alpha.ndim == 2:
                alpha = alpha.unsqueeze(0)
            loss_outside_alpha = (alpha * (1.0 - gt_object_mask)).mean()

        loss_preserve = torch.tensor(0.0, device="cuda")
        if preservation:
            pv = preservation[(iteration - 1) % len(preservation)]
            pcam = pv['camera']
            gaussians.set_anchor_mask(pcam.camera_center, pcam.resolution_scale)
            if hasattr(pipe_config, 'add_prefilter') and pipe_config.add_prefilter:
                p_visible_mask = prefilter_voxel(pcam, gaussians).squeeze()
            else:
                p_visible_mask = gaussians._anchor_mask
            p_pkg = objectgs_render(
                pcam,
                gaussians,
                pipe_config,
                bg_color,
                visible_mask=p_visible_mask,
                training=True,
            )
            loss_preserve = l1_loss(p_pkg['render'], pv['ref_image'])

        loss_seeded_scale_reg = torch.tensor(0.0, device="cuda")
        if (
            seeded_mask is not None
            and seeded_scale_reg_weight > 0.0
            and seeded_mask.any()
            and '_scaling' in initial_params
            and hasattr(gaussians, '_scaling')
        ):
            cur_scaling = gaussians._scaling[seeded_mask]
            init_scaling = initial_params['_scaling'][seeded_mask]
            loss_seeded_scale_reg = torch.mean((cur_scaling - init_scaling) ** 2)

        loss_seeded_offset_reg = torch.tensor(0.0, device="cuda")
        if (
            seeded_mask is not None
            and seeded_offset_reg_weight > 0.0
            and seeded_mask.any()
            and '_offset' in initial_params
            and hasattr(gaussians, '_offset')
        ):
            cur_offset = gaussians._offset[seeded_mask]
            init_offset = initial_params['_offset'][seeded_mask]
            loss_seeded_offset_reg = torch.mean((cur_offset - init_offset) ** 2)

        # Feature regularization (seeded only): prevents anchor_feat lock-in to
        # hallucinated colors.
        loss_feat_reg = torch.tensor(0.0, device="cuda")
        if (
            seeded_mask is not None
            and feat_reg_weight > 0.0
            and seeded_mask.any()
            and '_anchor_feat' in initial_params
            and hasattr(gaussians, '_anchor_feat')
        ):
            cur_feat = gaussians._anchor_feat[seeded_mask]
            init_feat = initial_params['_anchor_feat'][seeded_mask]
            loss_feat_reg = torch.mean((cur_feat - init_feat) ** 2)

        # Originals regularization: keep originals close to their pre-finetune
        # state when they are also part of the trainable set.
        loss_originals_reg = torch.tensor(0.0, device="cuda")
        if (
            originals_mask is not None
            and originals_reg_weight > 0.0
            and originals_mask.any()
        ):
            terms = []
            for name in ('_offset', '_scaling', '_anchor_feat'):
                if name in initial_params and hasattr(gaussians, name):
                    cur = getattr(gaussians, name)[originals_mask]
                    init_p = initial_params[name][originals_mask]
                    terms.append(torch.mean((cur - init_p) ** 2))
            if terms:
                loss_originals_reg = torch.stack(terms).mean()

        loss_seed_gate_reg = torch.tensor(0.0, device="cuda")
        if (
            seed_opacity_gate_reg_weight > 0.0
            and hasattr(gaussians, 'replenishment_seed_opacity_logit')
        ):
            seed_gates = torch.sigmoid(gaussians.replenishment_seed_opacity_logit)
            if seed_gates.numel() > 0:
                loss_seed_gate_reg = seed_gates.mean()

        loss_seed_lift_reg = torch.tensor(0.0, device="cuda")
        if (
            seed_opacity_lift_reg_weight > 0.0
            and hasattr(gaussians, 'replenishment_seed_opacity_lift')
        ):
            seed_lift = gaussians.replenishment_seed_opacity_lift
            if seed_lift is not None and seed_lift.numel() > 0:
                loss_seed_lift_reg = torch.mean(seed_lift ** 2)

        loss_mlp_opacity_reg = torch.tensor(0.0, device="cuda")
        if (
            train_mlp_opacity
            and mlp_opacity_reg_weight > 0.0
            and initial_mlp_opacity
            and hasattr(gaussians, 'mlp_opacity')
        ):
            reg_terms = []
            for param, start in zip(gaussians.mlp_opacity.parameters(), initial_mlp_opacity):
                reg_terms.append(torch.mean((param - start) ** 2))
            if reg_terms:
                loss_mlp_opacity_reg = torch.stack(reg_terms).mean()

        # Two-pass backward to isolate novel-view (Zero123++) gradients to
        # seeded anchors only. Reasoning: Ll1_hole normalizes by
        # hole_weight.sum() (small denom on back views) so per-pixel grads
        # from the hallucinated target are huge — and they leak into any
        # original anchor still visible at the supervision cam (e.g. 30°/330°
        # azimuth views), translating/warping the real geometry.
        # Pass 1 (novel-only): backprop, then zero originals' per-anchor grads
        # so Zero123++ only teaches seeded anchors. Originals still get
        # corrected by preservation (real images) and silhouette/regs in
        # pass 2.
        loss_novel_total = (
            loss_novel
            + silhouette_weight * loss_silhouette
            + outside_alpha_weight * loss_outside_alpha
        )
        loss_novel_total.backward(retain_graph=True)

        if originals_mask is not None and originals_mask.any():
            # One-shot diag at iter 1: confirm the mask is actually zeroing
            # originals' grads (and that originals have non-zero grads to
            # zero — i.e. that novel-view supervision is reaching them).
            if iteration == 1:
                _diag_lines = []
                for _name in ('_offset', '_scaling', '_anchor_feat'):
                    _p = getattr(gaussians, _name, None)
                    if _p is None or _p.grad is None:
                        _diag_lines.append(f"{_name}=<no_grad>")
                        continue
                    if _p.shape[0] != originals_mask.shape[0]:
                        _diag_lines.append(f"{_name}=<shape_mismatch {_p.shape[0]} vs {originals_mask.shape[0]}>")
                        continue
                    _g_orig = _p.grad[originals_mask]
                    _nnz_orig = int(((_g_orig != 0).reshape(_g_orig.shape[0], -1).any(dim=1)).sum().item()) if _g_orig.numel() > 0 else 0
                    _g_seed = _p.grad[seeded_mask] if (seeded_mask is not None and seeded_mask.any()) else None
                    _nnz_seed = int(((_g_seed != 0).reshape(_g_seed.shape[0], -1).any(dim=1)).sum().item()) if _g_seed is not None and _g_seed.numel() > 0 else 0
                    _diag_lines.append(f"{_name}: orig_nnz_grad={_nnz_orig}/{int(originals_mask.sum().item())}, seed_nnz_grad={_nnz_seed}/{int(seeded_mask.sum().item()) if seeded_mask is not None else 0}")
                logger.info("Two-pass diag (pre-mask, novel-only backward): " + "; ".join(_diag_lines))

            for _name in ('_offset', '_scaling', '_anchor_feat'):
                _p = getattr(gaussians, _name, None)
                if _p is not None and _p.grad is not None and _p.shape[0] == originals_mask.shape[0]:
                    _p.grad[originals_mask] = 0.0

            if iteration == 1:
                _diag_lines2 = []
                for _name in ('_offset', '_scaling', '_anchor_feat'):
                    _p = getattr(gaussians, _name, None)
                    if _p is None or _p.grad is None or _p.shape[0] != originals_mask.shape[0]:
                        continue
                    _g_orig = _p.grad[originals_mask]
                    _nnz_orig_after = int(((_g_orig != 0).reshape(_g_orig.shape[0], -1).any(dim=1)).sum().item()) if _g_orig.numel() > 0 else 0
                    _diag_lines2.append(f"{_name}: orig_nnz_grad_after={_nnz_orig_after}")
                logger.info("Two-pass diag (post-mask): " + "; ".join(_diag_lines2))

        # Pass 2: preservation + regularizers — safe for everyone, accumulate.
        loss_other = (
            preservation_weight * loss_preserve
            + seeded_scale_reg_weight * loss_seeded_scale_reg
            + seeded_offset_reg_weight * loss_seeded_offset_reg
            + feat_reg_weight * loss_feat_reg
            + originals_reg_weight * loss_originals_reg
            + seed_opacity_gate_reg_weight * loss_seed_gate_reg
            + seed_opacity_lift_reg_weight * loss_seed_lift_reg
            + mlp_opacity_reg_weight * loss_mlp_opacity_reg
        )
        loss_other.backward()

        # Reconstructed scalar for telemetry / loss_history.
        loss = loss_novel_total.detach() + loss_other.detach()

        # Keep updates stable while allowing stronger learning rates.
        trainable = [
            p
            for group in gaussians.optimizer.param_groups
            for p in group['params']
            if p.grad is not None
        ]
        if trainable:
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)

        # Restrict updates to selected anchors (typically unseen-side anchors).
        if grad_update_mask is not None:
            if hasattr(gaussians, '_offset') and gaussians._offset.grad is not None:
                gaussians._offset.grad[~grad_update_mask] = 0
            if hasattr(gaussians, '_anchor_feat') and gaussians._anchor_feat.grad is not None:
                gaussians._anchor_feat.grad[~grad_update_mask] = 0
            if hasattr(gaussians, '_scaling') and gaussians._scaling.grad is not None:
                gaussians._scaling.grad[~grad_update_mask] = 0

        if (
            freeze_feat_when_rgb_off
            and float(novel_rgb_weight) <= 0.0
            and hasattr(gaussians, '_anchor_feat')
            and gaussians._anchor_feat.grad is not None
        ):
            if seeded_mask is not None and seeded_mask.any():
                gaussians._anchor_feat.grad[seeded_mask] = 0
            elif grad_update_mask is not None and grad_update_mask.any():
                gaussians._anchor_feat.grad[grad_update_mask] = 0

        with torch.no_grad():
            gaussians.optimizer.step()

            # ── Seeded delta-from-init clamp ──
            if hasattr(gaussians, 'replenishment_seed_opacity_lift'):
                lift_param = gaussians.replenishment_seed_opacity_lift
                if isinstance(lift_param, torch.Tensor) and lift_param.numel() > 0:
                    lift_param.clamp_(min=0.0, max=2.0)

            # NOTE: _scaling is (N, 6). [:, 0:3] = offset-position scales
            # (controls how the K=10 child offsets spread in space),
            # [:, 3:6] = the gaussian ellipsoid log-scales fed into the cov
            # MLP. Both halves get clamped, but with different budgets:
            #   - [:, 3:6]: tight clamp to keep gaussian visual extent near
            #     init (also has scale_ceiling + anisotropy cap).
            #   - [:, 0:3]: looser clamp (2x). When unconstrained, single-view
            #     supervision under depth ambiguity stretches one axis hugely,
            #     and at render time the K=10 child gaussians spread along
            #     that axis → "horizontal streak" splat artifacts. Clamping
            #     keeps offsets near their init log-uniform spread.
            if seeded_mask is not None and seeded_mask.any():
                if (
                    hasattr(gaussians, '_scaling')
                    and '_scaling' in initial_params
                    and gaussians._scaling.shape[-1] >= 6
                ):
                    full_cur = gaussians._scaling[seeded_mask].clone()
                    full_ref = initial_params['_scaling'][seeded_mask]
                    # Gaussian-shape dims [3:6]: tight clamp.
                    full_cur[:, 3:6] = full_ref[:, 3:6] + torch.clamp(
                        full_cur[:, 3:6] - full_ref[:, 3:6],
                        min=-float(seeded_max_scale_delta),
                        max=float(seeded_max_scale_delta),
                    )
                    # Offset-position dims [0:3]: looser clamp (2x).
                    pos_budget = 2.0 * float(seeded_max_scale_delta)
                    full_cur[:, 0:3] = full_ref[:, 0:3] + torch.clamp(
                        full_cur[:, 0:3] - full_ref[:, 0:3],
                        min=-pos_budget,
                        max=pos_budget,
                    )
                    gaussians._scaling[seeded_mask] = full_cur
                if hasattr(gaussians, '_offset') and '_offset' in initial_params:
                    offset_ref = initial_params['_offset'][seeded_mask]
                    offset_cur = gaussians._offset[seeded_mask]
                    delta = offset_cur - offset_ref
                    if seeded_offset_delta_vec is not None:
                        # Per-axis cap (extent-relative).
                        cap = seeded_offset_delta_vec
                        delta = torch.maximum(torch.minimum(delta, cap), -cap)
                    else:
                        delta = torch.clamp(
                            delta,
                            min=-float(seeded_max_offset_delta),
                            max=float(seeded_max_offset_delta),
                        )
                    gaussians._offset[seeded_mask] = offset_ref + delta

            # ── Originals delta-from-init clamp (tighter budget) ──
            # Same restriction as seeded: only the gaussian shape dims [3:6].
            # With novel-view grads now masked from originals, originals only
            # learn from preservation + reg; clamping their offset-position
            # scales [0:3] would crush their pre-trained child-offset spread.
            if originals_mask is not None and originals_mask.any():
                if (
                    hasattr(gaussians, '_scaling')
                    and '_scaling' in initial_params
                    and gaussians._scaling.shape[-1] >= 6
                ):
                    full_cur = gaussians._scaling[originals_mask].clone()
                    scale_ref = initial_params['_scaling'][originals_mask][:, 3:6]
                    scale_cur = full_cur[:, 3:6]
                    full_cur[:, 3:6] = scale_ref + torch.clamp(
                        scale_cur - scale_ref,
                        min=-float(originals_max_scale_delta),
                        max=float(originals_max_scale_delta),
                    )
                    gaussians._scaling[originals_mask] = full_cur
                if hasattr(gaussians, '_offset') and '_offset' in initial_params:
                    offset_ref = initial_params['_offset'][originals_mask]
                    offset_cur = gaussians._offset[originals_mask]
                    delta = offset_cur - offset_ref
                    if originals_offset_delta_vec is not None:
                        cap = originals_offset_delta_vec
                        delta = torch.maximum(torch.minimum(delta, cap), -cap)
                    else:
                        delta = torch.clamp(
                            delta,
                            min=-float(originals_max_offset_delta),
                            max=float(originals_max_offset_delta),
                        )
                    gaussians._offset[originals_mask] = offset_ref + delta

            # ── Absolute scale ceiling (extent-relative, gaussian dims only) ──
            # Ceiling is derived as log(extent_med * frac) — that's only a
            # meaningful cap for the gaussian visual extent dims [3:6]. Do
            # NOT apply to [0:3] (offset-position scale): originals' trained
            # offset radii are structurally larger and would be instantly
            # crushed at iter 1 of finetune, collapsing 10 child gaussians
            # per anchor toward the anchor center → fragmented holes.
            if (
                scale_ceiling_val is not None
                and hasattr(gaussians, '_scaling')
                and gaussians._scaling.shape[-1] >= 6
            ):
                target_mask = None
                if seeded_mask is not None and seeded_mask.any():
                    target_mask = seeded_mask.clone()
                if originals_mask is not None and originals_mask.any():
                    target_mask = (target_mask | originals_mask) if target_mask is not None else originals_mask.clone()
                if target_mask is not None:
                    full_cur = gaussians._scaling[target_mask].clone()
                    cur_gauss = full_cur[:, 3:6]
                    over = (cur_gauss > scale_ceiling_val)
                    n_scale_clipped_last = int(over.any(dim=-1).sum().item()) if over.any() else 0
                    full_cur[:, 3:6] = torch.clamp(cur_gauss, max=scale_ceiling_val)
                    gaussians._scaling[target_mask] = full_cur

            # ── Anisotropy cap on seeded gaussian scales (kills radial spikes) ──
            # _scaling is (N, 6). Dims [3:6] are the 3D Gaussian per-axis log-
            # scales used by the cov MLP. We cap the *largest* axis relative
            # to the *median* (not the min) axis. Surface-fitted gaussians are
            # often oblate (one axis naturally small); capping vs min would
            # then strangle the in-plane axes. Capping vs median kills the
            # single elongated "spike" axis from depth ambiguity while leaving
            # legitimate oblate shapes intact.
            n_aniso_capped_last = 0
            if (
                seeded_anisotropy_max is not None
                and float(seeded_anisotropy_max) > 1.0
                and seeded_mask is not None
                and seeded_mask.any()
                and hasattr(gaussians, '_scaling')
                and gaussians._scaling.shape[-1] >= 6
            ):
                aniso_log = float(np.log(float(seeded_anisotropy_max)))
                gauss_log = gaussians._scaling[seeded_mask, 3:6]  # (M, 3)
                if gauss_log.numel() > 0:
                    median_per_row = gauss_log.median(dim=1, keepdim=True).values
                    max_allowed = median_per_row + aniso_log
                    over_aniso = (gauss_log > max_allowed)
                    n_aniso_capped_last = int(over_aniso.any(dim=1).sum().item())
                    if n_aniso_capped_last > 0:
                        gaussians._scaling[seeded_mask, 3:6] = torch.minimum(
                            gauss_log, max_allowed
                        )

            # The anisotropy cap can lower the larger axes after the first
            # delta clamp. Re-assert the seeded delta budget here so it cannot
            # turn into broad shrinkage/collapse of [3:6].
            if (
                seeded_mask is not None
                and seeded_mask.any()
                and hasattr(gaussians, '_scaling')
                and '_scaling' in initial_params
                and gaussians._scaling.shape[-1] >= 6
            ):
                full_cur = gaussians._scaling[seeded_mask].clone()
                full_ref = initial_params['_scaling'][seeded_mask]
                full_cur[:, 3:6] = full_ref[:, 3:6] + torch.clamp(
                    full_cur[:, 3:6] - full_ref[:, 3:6],
                    min=-float(seeded_max_scale_delta),
                    max=float(seeded_max_scale_delta),
                )
                pos_budget = 2.0 * float(seeded_max_scale_delta)
                full_cur[:, 0:3] = full_ref[:, 0:3] + torch.clamp(
                    full_cur[:, 0:3] - full_ref[:, 0:3],
                    min=-pos_budget,
                    max=pos_budget,
                )
                if scale_ceiling_val is not None:
                    full_cur[:, 3:6] = torch.clamp(full_cur[:, 3:6], max=scale_ceiling_val)
                gaussians._scaling[seeded_mask] = full_cur

            # ── AABB cage on offsets (per-dimension absolute cap) ──
            # SEEDED ONLY. Seeded offsets are initialized to magnitude
            # ~grid_spacing*frac (≈0.1-0.3 world units) and the cage is sized
            # similarly. Originals come from 30k iters of base scaffold-GS
            # training where _offset lives in *scaffold* coords (typical
            # magnitudes ~1-2, scaled down at render time by _scaling[:, 0:3]).
            # Applying a world-unit cage to scaffold-coord offsets chops them
            # to ~10% of their trained values → permanent corruption.
            n_anchors_caged_last = 0
            if (offset_abs_cap_vec is not None) and hasattr(gaussians, '_offset'):
                cage_target = None
                if seeded_mask is not None and seeded_mask.any():
                    cage_target = seeded_mask.clone()
                if cage_target is not None:
                    off_cur = gaussians._offset[cage_target]
                    if off_cur.numel() > 0:
                        cap = offset_abs_cap_vec  # (1, 1, 3)
                        # torch.any in this environment accepts a single dim; flatten trailing dims per anchor.
                        over_mag = (off_cur.abs() > cap).reshape(off_cur.shape[0], -1).any(dim=1)
                        n_anchors_caged_last = int(over_mag.sum().item())
                        if n_anchors_caged_last > 0:
                            gaussians._offset[cage_target] = torch.maximum(
                                torch.minimum(off_cur, cap), -cap
                            )
                aabb_escape_total += n_anchors_caged_last

            gaussians.optimizer.zero_grad(set_to_none=True)

        last_loss_seeded_scale_reg = loss_seeded_scale_reg.detach()
        last_loss_seeded_offset_reg = loss_seeded_offset_reg.detach()
        last_loss_outside_alpha = loss_outside_alpha.detach()
        last_loss_seed_gate_reg = loss_seed_gate_reg.detach()
        last_loss_seed_lift_reg = loss_seed_lift_reg.detach()
        last_loss_mlp_opacity_reg = loss_mlp_opacity_reg.detach()

        loss_val = loss.item()
        loss_history.append(loss_val)

        if iteration % 50 == 0 or iteration == 1:
            logger.info(
                f"  iter {iteration}/{n_iterations}: loss={loss_val:.5f}, "
                f"novel={loss_novel.item():.5f}, silhouette={loss_silhouette.item():.5f}, "
                f"preserve={loss_preserve.item():.5f}, outside={loss_outside_alpha.item():.5f}, "
                f"scale_reg={loss_seeded_scale_reg.item():.5f}, "
                f"offset_reg={loss_seeded_offset_reg.item():.5f}, "
                f"gate_reg={loss_seed_gate_reg.item():.5f}, "
                f"lift_reg={loss_seed_lift_reg.item():.5f}, "
                f"opacity_mlp_reg={loss_mlp_opacity_reg.item():.5f}, view_idx={sv_idx}"
            )

    gaussians.eval()

    if save_path:
        _save_model(gaussians, save_path, reference_model_path=reference_model_path)

    final_loss = loss_history[-1] if loss_history else 0.0
    param_delta_norms = {}
    seeded_delta_norms = {}
    seeded_scaling_stats = {}
    seeded_offset_stats = {}
    seed_opacity_gate_stats = {}
    seed_opacity_lift_stats = {}
    mlp_opacity_delta_norm = None
    with torch.no_grad():
        for name, start in initial_params.items():
            if hasattr(gaussians, name):
                current = getattr(gaussians, name).detach()
                param_delta_norms[name] = float(torch.norm(current - start).item())

        def _axis_stats(tensor):
            if tensor is None or tensor.numel() == 0:
                return []
            stats = []
            for axis in range(tensor.shape[-1]):
                vals = tensor[:, axis].reshape(-1)
                stats.append({
                    'axis': int(axis),
                    'min': float(vals.min().item()),
                    'median': float(vals.median().item()),
                    'mean': float(vals.mean().item()),
                    'max': float(vals.max().item()),
                })
            return stats

        if seeded_mask is not None and seeded_mask.any():
            if hasattr(gaussians, '_scaling') and '_scaling' in initial_params and gaussians._scaling.shape[-1] >= 6:
                scale_cur = gaussians._scaling[seeded_mask].detach()
                scale_start = initial_params['_scaling'][seeded_mask]
                scale_delta = scale_cur - scale_start
                seeded_delta_norms['scaling_total'] = float(torch.norm(scale_delta).item())
                seeded_delta_norms['scaling_pos_0_3'] = float(torch.norm(scale_delta[:, 0:3]).item())
                seeded_delta_norms['scaling_gauss_3_6'] = float(torch.norm(scale_delta[:, 3:6]).item())
                seeded_scaling_stats = {
                    'pos_0_3': _axis_stats(scale_cur[:, 0:3]),
                    'gauss_3_6': _axis_stats(scale_cur[:, 3:6]),
                    'delta_pos_0_3': _axis_stats(scale_delta[:, 0:3]),
                    'delta_gauss_3_6': _axis_stats(scale_delta[:, 3:6]),
                }
            if hasattr(gaussians, '_offset') and '_offset' in initial_params:
                offset_cur = gaussians._offset[seeded_mask].detach()
                offset_start = initial_params['_offset'][seeded_mask]
                offset_delta = offset_cur - offset_start
                seeded_delta_norms['offset_total'] = float(torch.norm(offset_delta).item())
                seeded_offset_stats = {
                    'value': _axis_stats(offset_cur.reshape(-1, offset_cur.shape[-1])),
                    'delta': _axis_stats(offset_delta.reshape(-1, offset_delta.shape[-1])),
                }
            if hasattr(gaussians, 'replenishment_seed_opacity_logit'):
                gates = torch.sigmoid(gaussians.replenishment_seed_opacity_logit.detach()).reshape(-1)
            elif hasattr(gaussians, 'replenishment_seed_opacity_gate'):
                gates = gaussians.replenishment_seed_opacity_gate.detach().reshape(-1)
            else:
                gates = None
            if gates is not None and gates.numel() > 0:
                seed_opacity_gate_stats = {
                    'min': float(gates.min().item()),
                    'median': float(gates.median().item()),
                    'mean': float(gates.mean().item()),
                    'max': float(gates.max().item()),
                }
            if hasattr(gaussians, 'replenishment_seed_opacity_lift'):
                lifts = gaussians.replenishment_seed_opacity_lift.detach().reshape(-1)
                if lifts.numel() > 0:
                    seed_opacity_lift_stats = {
                        'min': float(lifts.min().item()),
                        'median': float(lifts.median().item()),
                        'mean': float(lifts.mean().item()),
                        'max': float(lifts.max().item()),
                    }

        if train_mlp_opacity and initial_mlp_opacity and hasattr(gaussians, 'mlp_opacity'):
            sq_norm = torch.tensor(0.0, device="cuda")
            for param, start in zip(gaussians.mlp_opacity.parameters(), initial_mlp_opacity):
                sq_norm = sq_norm + torch.sum((param.detach() - start) ** 2)
            mlp_opacity_delta_norm = float(torch.sqrt(sq_norm).item())

    # Final scale telemetry.
    max_scale_log = None
    mean_scale_log = None
    if hasattr(gaussians, '_scaling'):
        if seeded_mask is not None and seeded_mask.any():
            sc = gaussians._scaling[seeded_mask].detach()
        else:
            sc = gaussians._scaling.detach()
        if sc.numel() > 0:
            max_scale_log = float(sc.max().item())
            mean_scale_log = float(sc.mean().item())

    logger.info(f"Fine-tuning complete. Final loss: {final_loss:.5f}")
    logger.info(
        "Parameter delta norms: "
        + ", ".join(f"{k}={v:.6e}" for k, v in param_delta_norms.items())
    )
    logger.info(
        "Cage telemetry: aabb_escape_total=%d, last_n_caged=%d, "
        "last_n_scale_clipped=%d, last_n_aniso_capped=%d, "
        "max_scale_log=%s, mean_scale_log=%s, n_views_dropped=%d",
        aabb_escape_total, n_anchors_caged_last, n_scale_clipped_last,
        n_aniso_capped_last,
        f"{max_scale_log:.4f}" if max_scale_log is not None else "n/a",
        f"{mean_scale_log:.4f}" if mean_scale_log is not None else "n/a",
        n_views_dropped,
    )
    if seeded_delta_norms:
        logger.info(
            "Seeded split deltas: scaling_pos_0_3=%.6e, scaling_gauss_3_6=%.6e, "
            "offset_total=%.6e",
            seeded_delta_norms.get('scaling_pos_0_3', 0.0),
            seeded_delta_norms.get('scaling_gauss_3_6', 0.0),
            seeded_delta_norms.get('offset_total', 0.0),
        )

    kept_idx = 0
    for diag in supervision_diagnostics:
        if diag.get('dropped'):
            diag['usage_count'] = 0
        else:
            diag['usage_count'] = int(view_usage_counts[kept_idx]) if kept_idx < len(view_usage_counts) else 0
            kept_idx += 1

    return {
        'loss_history': loss_history,
        'final_loss': final_loss,
        'view_usage_counts': view_usage_counts,
        'param_delta_norms': param_delta_norms,
        'seeded_delta_norms': seeded_delta_norms,
        'seeded_scaling_stats': seeded_scaling_stats,
        'seeded_offset_stats': seeded_offset_stats,
        'seed_opacity_gate_stats': seed_opacity_gate_stats,
        'seed_opacity_lift_stats': seed_opacity_lift_stats,
        'supervision_diagnostics': supervision_diagnostics,
        'seeded_scale_reg_last': float(last_loss_seeded_scale_reg.item()) if loss_history else 0.0,
        'seeded_offset_reg_last': float(last_loss_seeded_offset_reg.item()) if loss_history else 0.0,
        'outside_alpha_loss_last': float(last_loss_outside_alpha.item()) if loss_history else 0.0,
        'seed_opacity_gate_reg_last': float(last_loss_seed_gate_reg.item()) if loss_history else 0.0,
        'seed_opacity_lift_reg_last': float(last_loss_seed_lift_reg.item()) if loss_history else 0.0,
        'mlp_opacity_reg_last': float(last_loss_mlp_opacity_reg.item()) if loss_history else 0.0,
        'mlp_opacity_delta_norm': mlp_opacity_delta_norm,
        'train_mlp_opacity': bool(train_mlp_opacity),
        'mlp_opacity_lr_scale': float(mlp_opacity_lr_scale),
        'mlp_opacity_reg_weight': float(mlp_opacity_reg_weight),
        'aabb_escape_total': int(aabb_escape_total),
        'n_anchors_caged_last': int(n_anchors_caged_last),
        'n_scale_clipped_last': int(n_scale_clipped_last),
        'n_aniso_capped_last': int(n_aniso_capped_last),
        'max_scale_log': max_scale_log,
        'mean_scale_log': mean_scale_log,
        'n_views_dropped': int(n_views_dropped),
        'scale_ceiling_log': scale_ceiling_val,
        'offset_abs_cap': offset_abs_cap,
        'extent_max': extent_max_val,
        'extent_med': extent_med_val,
        'extent_min': extent_min_val,
        'novel_rgb_weight': float(novel_rgb_weight),
        'target_mask_erode_px': int(max(0, target_mask_erode_px)),
        'freeze_feat_when_rgb_off': bool(freeze_feat_when_rgb_off),
    }


def _largest_component_mask(mask: 'np.ndarray', min_pixels: int = 16):
    """Keep largest connected component in binary mask."""
    import cv2
    import numpy as np

    mask_u8 = (mask.astype(np.uint8) > 0).astype(np.uint8)
    if mask_u8.sum() < min_pixels:
        return mask_u8.astype(bool)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num_labels <= 1:
        return mask_u8.astype(bool)

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = int(np.argmax(areas) + 1)
    keep = labels == largest_label
    if keep.sum() < min_pixels:
        return mask_u8.astype(bool)
    return keep


def _setup_finetune_optimizer(
    gaussians,
    lr_scale: float,
    train_mlp_opacity: bool = False,
    mlp_opacity_lr_scale: float = 0.001,
    train_mlp_cov: bool = False,
    mlp_cov_lr_scale: float = 0.03,
    feat_lr_scale: float = 0.25,
    seed_opacity_gate_lr_scale: float = 50.0,
    seed_opacity_lift_lr_scale: float = 10.0,
):
    """Configure optimizer for fine-tuning with reduced learning rates.

    Uses Scaffold-GS's explicit parameter tensors instead of nn.Module iteration.
    """
    import torch

    base_lr = 1e-4 * lr_scale
    params = []

    # Anchor-level parameters (geometry).
    # Reduce _anchor_feat LR to prevent feature lock-in to hallucinated colors.
    feat_factor = float(feat_lr_scale)
    anchor_params = [
        ('_offset', 2e-4 * lr_scale),
        ('_anchor_feat', 2e-3 * lr_scale * feat_factor),
        ('_scaling', 5e-5 * lr_scale),
    ]

    for attr_name, lr in anchor_params:
        if hasattr(gaussians, attr_name):
            param = getattr(gaussians, attr_name)
            if isinstance(param, torch.nn.Parameter) or (isinstance(param, torch.Tensor) and param.requires_grad):
                params.append({'params': [param], 'lr': lr, 'name': attr_name})

    if train_mlp_opacity and hasattr(gaussians, 'mlp_opacity'):
        opacity_params = [p for p in gaussians.mlp_opacity.parameters() if p.requires_grad]
        if opacity_params:
            params.append({
                'params': opacity_params,
                'lr': base_lr * float(mlp_opacity_lr_scale),
                'name': 'mlp_opacity',
            })

    if train_mlp_cov and hasattr(gaussians, 'mlp_cov'):
        cov_params = [p for p in gaussians.mlp_cov.parameters() if p.requires_grad]
        if cov_params:
            params.append({
                'params': cov_params,
                'lr': base_lr * float(mlp_cov_lr_scale),
                'name': 'mlp_cov',
            })

    if hasattr(gaussians, 'replenishment_seed_opacity_logit'):
        gate_param = gaussians.replenishment_seed_opacity_logit
        if isinstance(gate_param, torch.nn.Parameter) or (
            isinstance(gate_param, torch.Tensor) and gate_param.requires_grad
        ):
            params.append({
                'params': [gate_param],
                'lr': base_lr * float(seed_opacity_gate_lr_scale),
                'name': 'seed_opacity_gate',
            })

    if hasattr(gaussians, 'replenishment_seed_opacity_lift'):
        lift_param = gaussians.replenishment_seed_opacity_lift
        if isinstance(lift_param, torch.nn.Parameter) or (
            isinstance(lift_param, torch.Tensor) and lift_param.requires_grad
        ):
            params.append({
                'params': [lift_param],
                'lr': base_lr * float(seed_opacity_lift_lr_scale),
                'name': 'seed_opacity_lift',
            })

    if params:
        gaussians.optimizer = torch.optim.Adam(params, lr=base_lr, eps=1e-15)
        logger.info(
            "Finetune optimizer: %d param groups, lr_scale=%.4f, feat_factor=%.3f, "
            "groups=[%s]",
            len(params), lr_scale, feat_factor,
            ', '.join(f"{p['name']}@{p['lr']:.2e}" for p in params),
        )
    else:
        logger.warning("No trainable parameters found!")


def _project_aabb_silhouette(camera, aabb_corners_np, height: int, width: int):
    """Project the 8 AABB corners and return a filled-rectangle silhouette mask."""
    import numpy as np
    pts = aabb_corners_np
    pts_h = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float32)], axis=1)
    view = camera.world_view_transform.detach().cpu().numpy().astype(np.float32)
    proj = camera.projection_matrix.detach().cpu().numpy().astype(np.float32)
    clip = pts_h @ view @ proj
    w_c = clip[:, 3]
    valid = np.isfinite(w_c) & (w_c > 1e-6)
    if not np.any(valid):
        return np.zeros((height, width), dtype=bool)
    clip = clip[valid]
    ndc = clip[:, :3] / np.clip(clip[:, 3:4], 1e-6, None)
    px = (ndc[:, 0] * 0.5 + 0.5) * (width - 1)
    py = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * (height - 1)
    x0 = int(np.clip(np.floor(px.min()), 0, width - 1))
    x1 = int(np.clip(np.ceil(px.max()), 0, width - 1))
    y0 = int(np.clip(np.floor(py.min()), 0, height - 1))
    y1 = int(np.clip(np.ceil(py.max()), 0, height - 1))
    mask = np.zeros((height, width), dtype=bool)
    if x1 >= x0 and y1 >= y0:
        mask[y0:y1 + 1, x0:x1 + 1] = True
    return mask


def _save_model(gaussians, save_path: str, reference_model_path: str = None):
    """Save the fine-tuned model in both simple and ObjectGS-loadable formats."""
    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    try:
        # Legacy/simple dump.
        gaussians.save_ply(str(save_dir / "point_cloud.ply"))
        gaussians.save_mlp_checkpoints(str(save_dir))

        # ObjectGS-compatible layout for load_gaussians().
        iter_dir = save_dir / "point_cloud" / "iteration_1"
        iter_dir.mkdir(parents=True, exist_ok=True)
        gaussians.save_ply(str(iter_dir / "point_cloud.ply"))
        gaussians.save_mlp_checkpoints(str(iter_dir))

        if reference_model_path:
            ref = Path(reference_model_path)
            for name in ("config.yaml", "cameras.json"):
                src = ref / name
                dst = save_dir / name
                if src.exists() and not dst.exists():
                    shutil.copy2(src, dst)

        logger.info(f"Saved model to {save_dir}")
    except Exception as e:
        logger.error(f"Failed to save model: {e}")
