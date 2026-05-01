"""Joint optimizer — train Scaffold-GS anchors with real + hallucinated views.

Replaces ``target_replenishment.core.optimizer`` with a clean implementation
that uses a single aligned supervision set. Real Phase-3 frames and SV3D
Phase-5 frames are both represented by RGB, mask and camera. No bbox alignment
is applied; alignment is guaranteed by using the matching camera intrinsics for
the exact image resolution being optimized.

Public API
----------
optimize_sv3d(gaussians, pipe_config, supervision_views, n_iterations, ...)
    -> dict with 'loss_history', 'final_loss', 'n_iters_done'
"""

from __future__ import annotations

__all__ = ["optimize_sv3d"]

import logging
import random
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from .gs_renderer import create_camera, render_rgba

logger = logging.getLogger(__name__)


# ── Small helpers ─────────────────────────────────────────────────────────────

def _largest_cc_mask(mask_np: np.ndarray, min_pixels: int = 64) -> np.ndarray:
    """Keep only the largest connected component in a binary mask."""
    import cv2
    m = mask_np.astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n_labels <= 1:
        return mask_np
    areas = stats[1:, cv2.CC_STAT_AREA]
    if areas.max() < min_pixels:
        return mask_np
    best = int(areas.argmax()) + 1
    return (labels == best)


def _build_supervision(
    views: list,
    *,
    hallucination_weight: float,
    target_mask_erode_px: int,
) -> list:
    """Convert raw supervision_views dicts into pre-processed tensors."""
    import cv2  # local import — cv2 not available on all build targets

    entries = []
    for view in views:
        cam_p = view["camera"]
        cam = create_camera(cam_p["R"], cam_p["T"], cam_p["K"],
                            cam_p["width"], cam_p["height"])

        rgb_np = np.asarray(view["rgb"])
        if rgb_np.dtype == np.uint8:
            rgb_f = rgb_np.astype(np.float32) / 255.0
        else:
            rgb_f = np.clip(rgb_np.astype(np.float32), 0.0, 1.0)

        if "mask" in view and view["mask"] is not None:
            mask_np = np.asarray(view["mask"]).astype(bool)
            if mask_np.shape != rgb_f.shape[:2]:
                mask_np = cv2.resize(
                    mask_np.astype(np.uint8),
                    (rgb_f.shape[1], rgb_f.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ) > 0
        else:
            mask_np = (rgb_f.mean(axis=2) < 0.98)
        mask_np = _largest_cc_mask(mask_np, min_pixels=64)
        erode_px = max(0, int(target_mask_erode_px))
        if erode_px > 0 and mask_np.any():
            k = 2 * erode_px + 1
            kernel = np.ones((k, k), dtype=np.uint8)
            eroded = cv2.erode(mask_np.astype(np.uint8), kernel) > 0
            if eroded.sum() >= 64:
                mask_np = eroded

        gt_image = torch.from_numpy(rgb_f).permute(2, 0, 1).float().cuda()   # (3,H,W)
        gt_mask = torch.from_numpy(mask_np.astype(np.float32)).unsqueeze(0).cuda()  # (1,H,W)

        weight = float(view.get("weight", hallucination_weight))
        entries.append({
            "camera": cam,
            "gt_image": gt_image,
            "gt_mask": gt_mask,
            "weight": weight,
            "source": view.get("source", "hallucinated"),
            "az": cam_p.get("azimuth_offset_deg", 0.0),
            "el": cam_p.get("elevation_offset_deg", 0.0),
        })

    return entries


def _setup_optimizer(
    gaussians,
    *,
    base_lr: float,
    feat_lr_scale: float,
    gate_lr_scale: float,
) -> torch.optim.Optimizer:
    """Create an Adam optimizer over anchor + gate parameters."""
    param_groups = [
        {"params": [gaussians._anchor_feat], "lr": base_lr * feat_lr_scale, "name": "anchor_feat"},
        {"params": [gaussians._offset],      "lr": base_lr,                  "name": "offset"},
        {"params": [gaussians._scaling],     "lr": base_lr * 0.1,            "name": "scaling"},
        {"params": [gaussians._anchor],      "lr": base_lr * 0.01,           "name": "anchor"},
    ]

    if hasattr(gaussians, "replenishment_seed_opacity_logit"):
        gate = gaussians.replenishment_seed_opacity_logit
        if isinstance(gate, nn.Parameter):
            param_groups.append(
                {"params": [gate], "lr": base_lr * gate_lr_scale, "name": "opacity_gate"}
            )

    if hasattr(gaussians, "replenishment_seed_opacity_lift"):
        lift = gaussians.replenishment_seed_opacity_lift
        if isinstance(lift, nn.Parameter):
            param_groups.append(
                {"params": [lift], "lr": base_lr * gate_lr_scale * 0.2, "name": "opacity_lift"}
            )

    return torch.optim.Adam(param_groups, eps=1e-15)


def _l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.abs(pred - target).mean()


def _ssim_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """1 - SSIM, computed with a simple sliding-window approximation."""
    try:
        import sys
        from pathlib import Path
        _ogs_dir = Path(__file__).resolve().parents[3] / "temp_deps" / "ObjectGS"
        if str(_ogs_dir) not in sys.path:
            sys.path.insert(0, str(_ogs_dir))
        from utils.loss_utils import ssim
        return 1.0 - ssim(pred.unsqueeze(0), target.unsqueeze(0))
    except Exception:
        return torch.tensor(0.0, device=pred.device)


# ── Main optimizer ────────────────────────────────────────────────────────────

def optimize_sv3d(
    gaussians,
    pipe_config,
    supervision_views: list,
    n_iterations: int,
    *,
    object_id: int,
    n_original_anchors: int,
    # Loss weights
    rgb_weight: float = 1.0,
    lambda_dssim: float = 0.2,
    silhouette_weight: float = 2.5,
    outside_alpha_weight: float = 8.0,
    scale_reg_weight: float = 0.20,
    offset_reg_weight: float = 0.20,
    feat_reg_weight: float = 0.05,
    gate_reg_weight: float = 0.005,
    # Learning rates
    base_lr: float = 5e-4,
    feat_lr_scale: float = 0.25,
    gate_lr_scale: float = 50.0,
    # Constraints
    aabb_min: Optional[np.ndarray] = None,
    aabb_max: Optional[np.ndarray] = None,
    cage_padding_frac: float = 0.02,
    scale_ceiling_log: Optional[float] = None,
    seeded_max_scale_delta: float = 0.20,
    seeded_max_offset_delta: float = 0.20,
    # Mask erosion
    target_mask_erode_px: int = 0,
    # Logging
    log_every: int = 100,
) -> dict:
    """Fine-tune seeded anchors using aligned real + hallucinated views.

    No bbox alignment is applied: every RGB/mask pair is optimized with the
    camera intrinsics/extrinsics that generated or extracted that exact image.

    Parameters
    ----------
    gaussians:
        Parent ``GaussianModel`` already seeded (Phase 7a done).
    pipe_config:
        Rendering pipeline config.
    supervision_views:
        Output of ``dataset_builder.build_supervision_views``.
    n_iterations:
        Number of gradient steps.
    object_id:
        Label of the object being replenished (used for masked render).
    n_original_anchors:
        Number of anchors before seeding; gradients for indices < this are
        zeroed each step so originals are frozen.

    Returns
    -------
    dict with 'loss_history', 'final_loss', 'n_iters_done'.
    """
    if not supervision_views:
        logger.warning("No supervision views — skipping fine-tuning.")
        return {"loss_history": [], "final_loss": 0.0, "n_iters_done": 0}

    # ── Pre-process supervision ────────────────────────────────────────────
    entries = _build_supervision(
        supervision_views,
        hallucination_weight=1.0,   # weight is already stored per-view
        target_mask_erode_px=target_mask_erode_px,
    )
    if not entries:
        logger.warning("All supervision views invalid after pre-processing.")
        return {"loss_history": [], "final_loss": 0.0, "n_iters_done": 0}

    source_counts: dict[str, int] = {}
    for entry in entries:
        source = str(entry.get("source", "unknown"))
        source_counts[source] = source_counts.get(source, 0) + 1

    logger.info("optimize_sv3d: %d iters, %d views, obj_id=%d, n_orig=%d, sources=%s",
                n_iterations, len(entries), object_id, n_original_anchors, source_counts)

    # ── Initialise opacity gate / lift if not already set ─────────────────
    n_total = int(gaussians._anchor.shape[0])
    n_seeded = n_total - n_original_anchors

    if n_seeded > 0:
        if not hasattr(gaussians, "replenishment_seed_opacity_logit") or \
                gaussians.replenishment_seed_opacity_logit is None:
            gaussians.replenishment_seed_opacity_logit = nn.Parameter(
                torch.zeros((n_seeded, 1), dtype=torch.float32, device="cuda")
            )
        if not hasattr(gaussians, "replenishment_seed_opacity_lift") or \
                gaussians.replenishment_seed_opacity_lift is None:
            gaussians.replenishment_seed_opacity_lift = nn.Parameter(
                torch.full((n_seeded, gaussians.n_offsets), 0.3,
                           dtype=torch.float32, device="cuda")
            )
        gaussians.n_original_anchors = int(n_original_anchors)

    # ── Freeze MLPs ────────────────────────────────────────────────────────
    gaussians.train()
    for mlp_attr in ("mlp_opacity", "mlp_cov", "mlp_color"):
        if hasattr(gaussians, mlp_attr):
            for p in getattr(gaussians, mlp_attr).parameters():
                p.requires_grad = False

    # ── Create optimizer ───────────────────────────────────────────────────
    optimizer = _setup_optimizer(
        gaussians,
        base_lr=base_lr,
        feat_lr_scale=feat_lr_scale,
        gate_lr_scale=gate_lr_scale,
    )

    # Snapshot originals for regularisation deltas.
    snap: dict[str, torch.Tensor] = {}
    for attr in ("_offset", "_scaling", "_anchor_feat"):
        if hasattr(gaussians, attr):
            snap[attr] = getattr(gaussians, attr).detach().clone()

    # ── AABB cage ─────────────────────────────────────────────────────────
    cage_min = cage_max = cage_pad = None
    if aabb_min is not None and aabb_max is not None:
        cage_min = torch.as_tensor(np.asarray(aabb_min, np.float32), device="cuda").reshape(3)
        cage_max = torch.as_tensor(np.asarray(aabb_max, np.float32), device="cuda").reshape(3)
        ext = (cage_max - cage_min).abs()
        cage_pad = ext * float(cage_padding_frac)

    bg = torch.ones(3, dtype=torch.float32, device="cuda")  # white bg = white background

    loss_history: list[float] = []
    running = {
        "loss": 0.0,
        "rgb": 0.0,
        "sil": 0.0,
        "out": 0.0,
    }
    order = list(range(len(entries)))
    random.shuffle(order)

    progress = tqdm(
        range(int(n_iterations)),
        desc=f"phase7 obj {int(object_id)}",
        dynamic_ncols=True,
        leave=True,
    )
    for it in progress:
        if it % len(order) == 0:
            random.shuffle(order)
        entry = entries[order[it % len(order)]]
        cam = entry["camera"]
        gt_image: torch.Tensor = entry["gt_image"]       # (3,H,W)
        gt_mask: torch.Tensor = entry["gt_mask"]         # (1,H,W)
        view_weight = entry["weight"]
        source = str(entry.get("source", "unknown"))

        # ── Forward render ─────────────────────────────────────────────────
        pkg = render_rgba(
            gaussians, cam, pipe_config,
            bg_white=True,
            object_label_id=int(object_id),
            training=True,
        )
        pred_rgb: torch.Tensor = pkg["rgb"]       # (3,H,W)
        pred_alpha: torch.Tensor = pkg["alpha"]   # (H,W)

        # ── RGB loss (masked to object region) ────────────────────────────
        mask = gt_mask                             # (1,H,W)
        n_mask_px = mask.sum().clamp(min=1.0)

        if float(rgb_weight) > 0.0:
            pred_rgb_masked = pred_rgb * mask
            gt_rgb_masked = gt_image * mask
            l1 = torch.abs(pred_rgb_masked - gt_rgb_masked).sum() / n_mask_px
            ssim_l = _ssim_loss(pred_rgb * mask, gt_image * mask)
            rgb_loss = (1.0 - lambda_dssim) * l1 + lambda_dssim * ssim_l
        else:
            rgb_loss = torch.tensor(0.0, device="cuda")

        # ── Silhouette loss (foreground coverage) ─────────────────────────
        pred_alpha_1hw = pred_alpha.unsqueeze(0)   # (1,H,W)
        silh_loss = torch.tensor(0.0, device="cuda")
        if float(silhouette_weight) > 0.0:
            # Where GT says foreground, penalise missing alpha.
            fg_coverage = (gt_mask * (1.0 - pred_alpha_1hw)).mean()
            silh_loss = fg_coverage

        # ── Outside-alpha penalty (no alpha where GT is bg) ───────────────
        outside_loss = torch.tensor(0.0, device="cuda")
        if float(outside_alpha_weight) > 0.0:
            bg_mask = 1.0 - gt_mask
            outside_loss = (bg_mask * pred_alpha_1hw).mean()

        # ── Scale regularisation (seeded anchors only) ────────────────────
        scale_loss = torch.tensor(0.0, device="cuda")
        if float(scale_reg_weight) > 0.0 and n_seeded > 0 and "_scaling" in snap:
            s_now = gaussians._scaling[n_original_anchors: n_original_anchors + n_seeded]
            s_orig = snap["_scaling"][n_original_anchors: n_original_anchors + n_seeded]
            delta = s_now - s_orig.detach()
            # Hard clamp + soft MSE penalty for large drifts.
            scale_loss = (delta.pow(2)).mean()

        # ── Offset regularisation ─────────────────────────────────────────
        offset_loss = torch.tensor(0.0, device="cuda")
        if float(offset_reg_weight) > 0.0 and n_seeded > 0 and "_offset" in snap:
            o_now = gaussians._offset[n_original_anchors: n_original_anchors + n_seeded]
            o_orig = snap["_offset"][n_original_anchors: n_original_anchors + n_seeded]
            offset_loss = ((o_now - o_orig.detach()).pow(2)).mean()

        # ── Feature regularisation ─────────────────────────────────────────
        feat_loss = torch.tensor(0.0, device="cuda")
        if float(feat_reg_weight) > 0.0 and n_seeded > 0 and "_anchor_feat" in snap:
            f_now = gaussians._anchor_feat[n_original_anchors: n_original_anchors + n_seeded]
            f_orig = snap["_anchor_feat"][n_original_anchors: n_original_anchors + n_seeded]
            feat_loss = ((f_now - f_orig.detach()).pow(2)).mean()

        # ── Gate regularisation (L2 on logits; no directional bias) ───────
        gate_loss = torch.tensor(0.0, device="cuda")
        if float(gate_reg_weight) > 0.0 and hasattr(gaussians, "replenishment_seed_opacity_logit"):
            gate = gaussians.replenishment_seed_opacity_logit
            if isinstance(gate, nn.Parameter):
                gate_loss = (gate ** 2).mean()

        # ── Total loss ────────────────────────────────────────────────────
        total = (
            view_weight * rgb_weight * rgb_loss
            + float(silhouette_weight) * silh_loss
            + float(outside_alpha_weight) * outside_loss
            + float(scale_reg_weight) * scale_loss
            + float(offset_reg_weight) * offset_loss
            + float(feat_reg_weight) * feat_loss
            + float(gate_reg_weight) * gate_loss
        )

        optimizer.zero_grad(set_to_none=True)
        total.backward()

        # ── Zero gradients for original anchors (freeze them) ────────────
        with torch.no_grad():
            for attr in ("_anchor", "_anchor_feat", "_offset", "_scaling"):
                param = getattr(gaussians, attr, None)
                if param is not None and isinstance(param, nn.Parameter) and param.grad is not None:
                    param.grad[:n_original_anchors].zero_()

        # ── Hard clamp seeded scale / offset deltas ───────────────────────
        if float(seeded_max_scale_delta) > 0.0 and n_seeded > 0 and "_scaling" in snap:
            with torch.no_grad():
                s_init = snap["_scaling"][n_original_anchors: n_original_anchors + n_seeded]
                lo = s_init - float(seeded_max_scale_delta)
                hi = s_init + float(seeded_max_scale_delta)
                gaussians._scaling.data[n_original_anchors: n_original_anchors + n_seeded] = \
                    torch.clamp(gaussians._scaling.data[n_original_anchors: n_original_anchors + n_seeded],
                                min=lo, max=hi)

        if float(seeded_max_offset_delta) > 0.0 and n_seeded > 0 and "_offset" in snap:
            with torch.no_grad():
                o_init = snap["_offset"][n_original_anchors: n_original_anchors + n_seeded]
                lo = o_init - float(seeded_max_offset_delta)
                hi = o_init + float(seeded_max_offset_delta)
                gaussians._offset.data[n_original_anchors: n_original_anchors + n_seeded] = \
                    torch.clamp(gaussians._offset.data[n_original_anchors: n_original_anchors + n_seeded],
                                min=lo, max=hi)

        # ── AABB cage for seeded anchor positions ─────────────────────────
        if cage_min is not None and n_seeded > 0:
            with torch.no_grad():
                lo = cage_min - cage_pad
                hi = cage_max + cage_pad
                gaussians._anchor.data[n_original_anchors: n_original_anchors + n_seeded] = \
                    torch.clamp(
                        gaussians._anchor.data[n_original_anchors: n_original_anchors + n_seeded],
                        min=lo, max=hi,
                    )

        # ── Optional scale ceiling ─────────────────────────────────────────
        if scale_ceiling_log is not None and n_seeded > 0:
            with torch.no_grad():
                gaussians._scaling.data[n_original_anchors: n_original_anchors + n_seeded] = \
                    torch.clamp(
                        gaussians._scaling.data[n_original_anchors: n_original_anchors + n_seeded],
                        max=float(scale_ceiling_log),
                    )

        optimizer.step()

        loss_val = float(total.item())
        loss_history.append(loss_val)

        running["loss"] = 0.95 * running["loss"] + 0.05 * loss_val if it else loss_val
        running["rgb"] = 0.95 * running["rgb"] + 0.05 * float(rgb_loss.item()) if it else float(rgb_loss.item())
        running["sil"] = 0.95 * running["sil"] + 0.05 * float(silh_loss.item()) if it else float(silh_loss.item())
        running["out"] = 0.95 * running["out"] + 0.05 * float(outside_loss.item()) if it else float(outside_loss.item())
        progress.set_postfix({
            "loss": f"{running['loss']:.3f}",
            "rgb": f"{running['rgb']:.3f}",
            "sil": f"{running['sil']:.3f}",
            "out": f"{running['out']:.3f}",
        })



    # ── Persist opacity gates as plain tensors (not nn.Parameter) ─────────
    gaussians.eval()
    with torch.no_grad():
        if hasattr(gaussians, "replenishment_seed_opacity_logit") and \
                isinstance(gaussians.replenishment_seed_opacity_logit, nn.Parameter):
            gaussians.replenishment_seed_opacity_gate = \
                torch.sigmoid(gaussians.replenishment_seed_opacity_logit).detach()

    final_loss = float(loss_history[-1]) if loss_history else 0.0
    logger.info("optimize_sv3d done: %d iters, final_loss=%.5f", n_iterations, final_loss)
    return {
        "loss_history": loss_history,
        "final_loss": final_loss,
        "n_iters_done": int(n_iterations),
        "supervision_diagnostics": [
            {"source": source, "n_views": count}
            for source, count in sorted(source_counts.items())
        ],
    }
