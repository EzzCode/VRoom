"""
Optimizer — Fine-tune ObjectGS anchors + MLPs with inpainted image supervision.

PAInpainter §3.5: Uses accepted inpainted views as ground-truth supervision to
update the 3DGS model. Computes L1+SSIM loss in the repair region and a
preservation loss in the non-masked region to prevent drift.

Public API:
    optimize_with_inpainted_views(gaussians, pipe, views, ...) -> dict
"""

__all__ = ['optimize_with_inpainted_views']

import sys
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_VROOM_ROOT = Path(__file__).resolve().parent.parent.parent
_OBJECTGS_DIR = _VROOM_ROOT / "temp_deps" / "ObjectGS"
if str(_OBJECTGS_DIR) not in sys.path:
    sys.path.insert(0, str(_OBJECTGS_DIR))


def optimize_with_inpainted_views(
    gaussians,
    pipe_config,
    accepted_views: list,
    n_iterations: int = 500,
    lr_scale: float = 0.1,
    lambda_dssim: float = 0.2,
    lambda_preserve: float = 0.5,
    save_path: str = None,
    training_cameras: list = None,
) -> dict:
    """Fine-tune ObjectGS model using inpainted images as supervision.

    For each iteration:
        1. Pick a random accepted view.
        2. Render via ObjectGS with training=True (gradient flow through MLPs).
        3. Compute inpainting loss (L1 + SSIM) in the repair region.
        4. Compute preservation loss (L1) in the non-masked region.
        5. Backprop and step optimizer.

    No densification or pruning — we're fine-tuning, not training from scratch.

    Args:
        gaussians: GaussianModel (modified in-place).
        pipe_config: ObjectGS pipeline config.
        accepted_views: list of dicts with 'rgb_inpainted', 'mask', 'camera_params'.
        n_iterations: Number of fine-tuning iterations.
        lr_scale: Learning rate relative to original training LR.
        lambda_dssim: Weight for SSIM loss component.
        lambda_preserve: Weight for preservation loss (non-masked region).
        save_path: If provided, save updated PLY + MLPs here.

    Returns:
        dict with 'loss_history', 'final_loss'.
    """
    import torch
    import numpy as np
    from random import choice
    import random

    from utils.loss_utils import l1_loss, ssim
    from gaussian_renderer.render import render as objectgs_render
    from gaussian_renderer.render import prefilter_voxel
    from target_replenishment.core.objectgs_bridge import create_virtual_camera

    if not accepted_views:
        logger.warning("No accepted views — nothing to optimize.")
        return {'loss_history': [], 'final_loss': 0.0}

    bg_color = torch.zeros(3, dtype=torch.float32, device="cuda")
    original_renders = []
    if training_cameras:
        logger.info("Pre-computing original renders for multi-view regularization...")
        cams = random.sample(training_cameras, min(len(training_cameras), 8))
        for cam_p in cams:
            cam = create_virtual_camera(cam_p['R'], cam_p['T'], cam_p['K'], cam_p['width'], cam_p['height'])
            with torch.no_grad():
                render_pkg = objectgs_render(cam, gaussians, pipe_config, bg_color, training=False)
                original_renders.append({'camera': cam, 'render': render_pkg['render'].clone().detach()})

    # Prepare supervision data (convert to tensors once)
    supervision = []
    for view in accepted_views:
        cam_p = view['camera_params']
        cam = create_virtual_camera(cam_p['R'], cam_p['T'], cam_p['K'],
                                    cam_p['width'], cam_p['height'])

        rgb_np = view['rgb_inpainted']
        if rgb_np.dtype == np.uint8:
            rgb_np = rgb_np.astype(np.float32) / 255.0

        gt_image = torch.from_numpy(rgb_np).permute(2, 0, 1).float().cuda()

        mask_np = view['mask']
        if mask_np.ndim == 2:
            mask_tensor = torch.from_numpy(mask_np).float().cuda().unsqueeze(0)
        else:
            mask_tensor = torch.from_numpy(mask_np).float().cuda()

        supervision.append({
            'camera': cam,
            'gt_image': gt_image,
            'mask': mask_tensor,     # (1, H, W) — repair region
            'inv_mask': 1.0 - mask_tensor,  # non-masked region
        })

    # Set up optimizer with reduced LR
    gaussians.train()

    # FREEZE MLPs to prevent catastrophic forgetting. Only fine-tune anchors.
    logger.info("Freezing MLPs to prevent multi-view degradation...")
    if hasattr(gaussians, 'mlp_opacity'):
        for param in gaussians.mlp_opacity.parameters(): param.requires_grad = False
    if hasattr(gaussians, 'mlp_cov'):
        for param in gaussians.mlp_cov.parameters(): param.requires_grad = False
    if hasattr(gaussians, 'mlp_color'):
        for param in gaussians.mlp_color.parameters(): param.requires_grad = False

    _setup_finetune_optimizer(gaussians, lr_scale)

    bg_color = torch.zeros(3, dtype=torch.float32, device="cuda")
    loss_history = []

    logger.info(f"Starting fine-tuning: {n_iterations} iters, "
                f"{len(supervision)} views, lr_scale={lr_scale}")

    for iteration in range(1, n_iterations + 1):
        sv = choice(supervision)
        cam = sv['camera']

        # Render with training=True for gradient flow through MLPs
        gaussians.set_anchor_mask(cam.camera_center, cam.resolution_scale)
        if hasattr(pipe_config, 'add_prefilter') and pipe_config.add_prefilter:
            visible_mask = prefilter_voxel(cam, gaussians).squeeze()
        else:
            visible_mask = gaussians._anchor_mask

        render_pkg = objectgs_render(
            cam, gaussians, pipe_config, bg_color,
            visible_mask=visible_mask, training=True,
        )
        rendered = render_pkg['render']

        # Inpainting loss: L1 over masked region specifically to prevent catastrophic gradient squashing
        mask_bool = sv['mask'] > 0.5
        mask_3ch = sv['mask'].expand_as(rendered)
        
        masked_render = rendered * mask_3ch
        masked_gt = sv['gt_image'] * mask_3ch
        
        if mask_bool.sum() > 0:
            mask_3ch_bool = mask_bool.expand_as(rendered)
            Ll1_inpaint = l1_loss(rendered[mask_3ch_bool], sv['gt_image'][mask_3ch_bool])
        else:
            Ll1_inpaint = torch.tensor(0.0, device="cuda")
            
        # SSIM averages over entire image, standard isolated masked SSIM without rescaling
        Lssim_inpaint = 1.0 - ssim(masked_render, masked_gt)
        loss_inpaint = (1.0 - lambda_dssim) * Ll1_inpaint + lambda_dssim * Lssim_inpaint

        # Preservation loss: L1 in non-masked region (prevent drift on target view)
        inv_mask_3ch = sv['inv_mask'].expand_as(rendered)
        if lambda_preserve > 0:
            loss_preserve = lambda_preserve * l1_loss(
                rendered * inv_mask_3ch, sv['gt_image'] * inv_mask_3ch
            )
        else:
            loss_preserve = torch.tensor(0.0, device="cuda")

        # Multi-view Consistency Regularization
        loss_mv = torch.tensor(0.0, device="cuda")
        if original_renders:
            reg_view = choice(original_renders)
            reg_cam = reg_view['camera']
            gaussians.set_anchor_mask(reg_cam.camera_center, reg_cam.resolution_scale)
            if hasattr(pipe_config, 'add_prefilter') and pipe_config.add_prefilter:
                v_mask = prefilter_voxel(reg_cam, gaussians).squeeze()
            else:
                v_mask = gaussians._anchor_mask
            reg_pkg = objectgs_render(reg_cam, gaussians, pipe_config, bg_color, visible_mask=v_mask, training=True)
            loss_mv = l1_loss(reg_pkg['render'], reg_view['render'])

        total_loss = loss_inpaint + loss_preserve + 0.5 * loss_mv
        total_loss.backward()

        with torch.no_grad():
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)

        loss_val = total_loss.item()
        loss_history.append(loss_val)

        if iteration % 50 == 0 or iteration == 1:
            logger.info(
                f"  iter {iteration}/{n_iterations}: "
                f"loss={loss_val:.5f} (inpaint={loss_inpaint.item():.5f}, "
                f"preserve={loss_preserve.item():.5f})"
            )

    gaussians.eval()

    if save_path:
        _save_model(gaussians, save_path)

    final_loss = loss_history[-1] if loss_history else 0.0
    logger.info(f"Fine-tuning complete. Final loss: {final_loss:.5f}")

    return {
        'loss_history': loss_history,
        'final_loss': final_loss,
    }


def _setup_finetune_optimizer(gaussians, lr_scale: float):
    """Create a fine-tuning optimizer with scaled-down learning rates."""
    import torch
    
    # Ensure optimizer was successfully restored in load_gaussians
    if gaussians.optimizer is None:
        raise RuntimeError("ObjectGS optimizer must be explicitly initialized using training_setup(op) "
                           "prior to replenishment. Falling back to static values risks explosive gradients.")

    params = []
    for group in gaussians.optimizer.param_groups:
        for p in group['params']:
            if p.requires_grad:
                # The spatial_lr_scale is already incorporated into group['lr'] via training_setup
                params.append({
                    'params': [p],
                    'lr': group['lr'] * lr_scale,
                    'name': group.get('name', 'unknown'),
                })

    gaussians.optimizer = torch.optim.Adam(params, lr=0.0, eps=1e-15)
    logger.info(f"Fine-tune optimizer: {len(params)} param groups, "
                f"lr_scale={lr_scale}")


def _save_model(gaussians, save_path: str):
    """Save the fine-tuned model (PLY + MLP checkpoints)."""
    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    ply_path = save_dir / "point_cloud.ply"
    gaussians.save_ply(str(ply_path))
    gaussians.save_mlp_checkpoints(str(save_dir))

    logger.info(f"Saved fine-tuned model to {save_dir}")
