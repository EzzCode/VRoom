"""Anchor Seeding — Grid-Based Backside Filling.

Identical algorithm to the original target_replenishment anchor_seeding.py,
rewritten here with NO dependency on target_replenishment.

Public API
----------
seed_backside(gaussians, object_center, view_direction, object_id, ...) -> int
    Appends new grid anchors to the back-hemisphere of the object in-place.
    Returns the number of new anchors added.
"""

from __future__ import annotations

__all__ = ["seed_backside"]

import logging

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def seed_backside(
    gaussians,
    object_center: np.ndarray,
    view_direction: np.ndarray,
    object_id: int,
    *,
    grid_resolution: int = 25,
    k_neighbors: int = 5,
    max_new_anchors: int = 2000,
    bounds_quantile_low: float = 0.01,
    bounds_quantile_high: float = 0.99,
    hemisphere_margin: float = 0.02,
    bounds_expand_frac: float = 0.05,
    offset_scale_frac: float = 0.5,
    scale_max_frac: float = 0.10,
    conservative_seed_render: bool = True,
    visual_hull_constraints: list | None = None,
    visual_hull_min_views: int = 2,
    surface_shell_filter: bool = True,
    surface_shell_min_norm: float = 0.65,
) -> int:
    """Seed a dense voxel grid on the unseen backside of an object.

    Strategy:
    1. Identify frontside anchors (facing camera) for feature borrowing.
    2. Compute a robust 3-D AABB for the object.
    3. Generate a uniform 3-D grid inside the AABB.
    4. Filter to the unseen hemisphere (back of the camera plane).
    5. KNN-initialise features from frontside anchors (stays in-distribution).
    6. Append to ``gaussians`` in-place.

    Returns
    -------
    Number of new anchors added (0 if nothing was seeded).
    """
    labels = gaussians.label_ids.squeeze(-1)
    obj_mask = labels == int(object_id)
    n_obj = int(obj_mask.sum().item())

    if n_obj < 5:
        logger.warning("Object %d has only %d anchors — too few to bound.", object_id, n_obj)
        return 0

    obj_indices = torch.where(obj_mask)[0]
    obj_positions = gaussians._anchor.detach()[obj_indices]   # (N_obj, 3)
    obj_feats = gaussians._anchor_feat.detach()[obj_indices]
    obj_rotation = gaussians._rotation.detach()[obj_indices]
    obj_scaling = gaussians._scaling.detach()[obj_indices]    # (N_obj, 6)

    center = torch.tensor(object_center, dtype=torch.float32, device="cuda")
    view_dir = torch.tensor(view_direction, dtype=torch.float32, device="cuda")
    vnorm = float(view_dir.norm().item())
    if vnorm < 1e-6:
        logger.warning("Object %d: near-zero view_direction norm (%e); using +Z.", object_id, vnorm)
        view_dir = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device="cuda")
    else:
        view_dir = view_dir / view_dir.norm()

    # ── Frontside / backside classification ───────────────────────────────
    relative = obj_positions - center.unsqueeze(0)
    front_scores = (relative * view_dir.unsqueeze(0)).sum(dim=1)

    front_mask = front_scores > float(hemisphere_margin)
    n_front = int(front_mask.sum().item())
    if n_front < 3:
        logger.warning("Object %d: only %d frontside anchors — too few to borrow features.", object_id, n_front)
        return 0

    front_positions = obj_positions[front_mask]
    front_feats = obj_feats[front_mask]
    front_rotation = obj_rotation[front_mask]

    # ── Robust AABB ───────────────────────────────────────────────────────
    q_low = float(np.clip(bounds_quantile_low, 0.0, 0.49))
    q_high = float(np.clip(bounds_quantile_high, 0.51, 1.0))
    k_lo = max(1, min(n_obj, int(np.floor(q_low * n_obj))))
    k_hi = max(1, min(n_obj, int(np.ceil(q_high * n_obj))))

    if k_hi <= k_lo:
        bounds_min = obj_positions.min(dim=0)[0]
        bounds_max = obj_positions.max(dim=0)[0]
    else:
        bounds_min, _ = torch.kthvalue(obj_positions, k_lo, dim=0)
        bounds_max, _ = torch.kthvalue(obj_positions, k_hi, dim=0)

    # Density-trim: keep densest 50% to drop floaters.
    if n_obj >= 32:
        try:
            k_density = int(min(8, max(3, n_obj // 16)))
            dist_mat = torch.cdist(obj_positions, obj_positions)
            knn_d, _ = torch.topk(dist_mat, k=k_density + 1, dim=1, largest=False)
            knn_d = knn_d[:, k_density]
            density_thr = float(torch.median(knn_d).item())
            core_keep = knn_d <= density_thr
            n_core = int(core_keep.sum().item())
            if n_core >= 16:
                core_pos = obj_positions[core_keep]
                k_c_lo = max(1, int(0.05 * n_core))
                k_c_hi = max(1, min(n_core, int(np.ceil(0.95 * n_core))))
                core_min, _ = torch.kthvalue(core_pos, k_c_lo, dim=0)
                core_max, _ = torch.kthvalue(core_pos, k_c_hi, dim=0)
                bounds_min = torch.maximum(bounds_min, core_min)
                bounds_max = torch.minimum(bounds_max, core_max)
        except Exception as exc:
            logger.warning("Object %d: density-trim failed (%s); using quantile bounds.", object_id, exc)

    extent = bounds_max - bounds_min
    bounds_min -= extent * float(bounds_expand_frac)
    bounds_max += extent * float(bounds_expand_frac)
    extent = bounds_max - bounds_min
    extent_max = float(extent.max().item())

    # ── Generate 3-D grid ─────────────────────────────────────────────────
    x = torch.linspace(float(bounds_min[0]), float(bounds_max[0]), grid_resolution, device="cuda")
    y = torch.linspace(float(bounds_min[1]), float(bounds_max[1]), grid_resolution, device="cuda")
    z = torch.linspace(float(bounds_min[2]), float(bounds_max[2]), grid_resolution, device="cuda")
    gx, gy, gz = torch.meshgrid(x, y, z, indexing="ij")
    grid_pts = torch.stack([gx.flatten(), gy.flatten(), gz.flatten()], dim=-1)

    # ── Filter to backside hemisphere ─────────────────────────────────────
    grid_rel = grid_pts - center.unsqueeze(0)
    grid_scores = (grid_rel * view_dir.unsqueeze(0)).sum(dim=1)
    back_mask = grid_scores < -float(hemisphere_margin)
    new_positions = grid_pts[back_mask]

    if new_positions.shape[0] == 0:
        logger.warning("Object %d: no backside grid points after hemisphere filter.", object_id)
        return 0

    # Remove points too close to existing anchors (inside the surface hull).
    dists_to_existing = torch.cdist(new_positions, obj_positions)
    min_dists, _ = dists_to_existing.min(dim=1)
    avg_spacing = extent.norm() / max(int(grid_resolution), 1)
    new_positions = new_positions[min_dists > (avg_spacing * 0.5)]

    if new_positions.shape[0] == 0:
        logger.warning("Object %d: all backside grid points too close to existing anchors.", object_id)
        return 0

    # ── Optional visual-hull filter ───────────────────────────────────────
    if visual_hull_constraints:
        support = torch.zeros(new_positions.shape[0], dtype=torch.int32, device="cuda")
        for constraint in visual_hull_constraints:
            try:
                cam_c = constraint["camera"]
                mask_np = np.asarray(constraint["mask"]).astype(bool)
                if mask_np.ndim != 2 or mask_np.sum() == 0:
                    continue
                H, W = mask_np.shape
                mask_t = torch.from_numpy(mask_np).to(device="cuda", dtype=torch.bool)
                R = torch.as_tensor(cam_c["R"], dtype=torch.float32, device="cuda")
                T = torch.as_tensor(cam_c["T"], dtype=torch.float32, device="cuda").reshape(1, 3)
                K = torch.as_tensor(cam_c["K"], dtype=torch.float32, device="cuda")
                cam_pts = (new_positions @ R.T) + T
                z_v = cam_pts[:, 2]
                valid = z_v > 1e-4
                u = K[0, 0] * cam_pts[:, 0] / (z_v + 1e-8) + K[0, 2]
                v = K[1, 1] * cam_pts[:, 1] / (z_v + 1e-8) + K[1, 2]
                ui = torch.round(u).long()
                vi = torch.round(v).long()
                valid = valid & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
                inside = torch.zeros_like(valid, dtype=torch.bool)
                if valid.any():
                    inside[valid] = mask_t[vi[valid], ui[valid]]
                support += inside.to(torch.int32)
            except Exception as exc:
                logger.warning("Object %d: visual-hull constraint failed (%s).", object_id, exc)

        min_v = max(1, int(visual_hull_min_views))
        keep_hull = support >= min_v
        if not bool(keep_hull.any().item()):
            max_sup = int(support.max().item()) if support.numel() else 0
            keep_hull = (support >= max_sup) if max_sup > 0 else torch.ones_like(support, dtype=torch.bool)
            logger.warning("Object %d: visual-hull min_views=%d had no hits; keeping max_support=%d.",
                           object_id, min_v, max_sup)
        new_positions = new_positions[keep_hull]

    if new_positions.shape[0] == 0:
        logger.warning("Object %d: visual-hull filter removed all candidates.", object_id)
        return 0

    # ── Optional surface-shell filter (keep outer shell only) ────────────
    if surface_shell_filter:
        half_ext = torch.clamp((bounds_max - bounds_min) * 0.5, min=1e-6)
        box_ctr = (bounds_min + bounds_max) * 0.5
        normalized = ((new_positions - box_ctr.unsqueeze(0)).abs() / half_ext.unsqueeze(0)).clamp(0, 1)
        shell_score = normalized.max(dim=1).values
        thresh = float(np.clip(surface_shell_min_norm, 0.0, 0.99))
        keep_shell = shell_score >= thresh
        if keep_shell.any():
            new_positions = new_positions[keep_shell]
        else:
            q90 = torch.quantile(shell_score, 0.90)
            new_positions = new_positions[shell_score >= q90]
            logger.warning("Object %d: shell thresh %.2f kept nothing; using top 10%%.", object_id, thresh)

    if new_positions.shape[0] == 0:
        logger.warning("Object %d: surface-shell filter removed all candidates.", object_id)
        return 0

    # ── Hard cap ─────────────────────────────────────────────────────────
    n_new = new_positions.shape[0]
    if n_new > int(max_new_anchors):
        perm = torch.randperm(n_new, device="cuda")[: int(max_new_anchors)]
        new_positions = new_positions[perm]
        n_new = int(max_new_anchors)

    # ── KNN feature init from frontside anchors ───────────────────────────
    k = min(k_neighbors, n_front)
    dists = torch.cdist(new_positions, front_positions)
    _, knn_idx = dists.topk(k, dim=1, largest=False)

    knn_feats = front_feats[knn_idx]       # (n_new, k, feat_dim)
    knn_rots = front_rotation[knn_idx]     # (n_new, k, 4)
    knn_d = torch.gather(dists, 1, knn_idx)
    weights = 1.0 / (knn_d + 1e-4)
    weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)

    new_feats = (knn_feats * weights.unsqueeze(-1)).sum(dim=1)
    new_rotation = (knn_rots * weights.unsqueeze(-1)).sum(dim=1)
    new_rotation = new_rotation / (new_rotation.norm(dim=-1, keepdim=True) + 1e-8)

    # ── Scale initialisation ──────────────────────────────────────────────
    grid_spacing = extent.norm() / (max(int(grid_resolution), 1) * 1.732)
    grid_spacing = torch.clamp(grid_spacing, min=1e-4, max=1e2)
    scale_floor_log = float(np.clip(torch.log(grid_spacing).item(), -10.0, 2.0))

    if obj_scaling.shape[0] >= 4:
        full_dists = torch.cdist(new_positions, obj_positions)
        k_obj = min(int(k_neighbors), int(obj_scaling.shape[0]))
        obj_knn_d, obj_knn_idx = full_dists.topk(k_obj, dim=1, largest=False)
        obj_w = 1.0 / (obj_knn_d + 1e-4)
        obj_w = obj_w / (obj_w.sum(dim=1, keepdim=True) + 1e-8)
        knn_scaling = obj_scaling[obj_knn_idx]
        new_scaling = (knn_scaling * obj_w.unsqueeze(-1)).sum(dim=1)
        extent_med = float(extent.median().item())
        scale_ceil = float(np.log(max(extent_med * float(scale_max_frac), 1e-6)))
        new_scaling = torch.clamp(new_scaling, min=scale_floor_log - 1.0, max=scale_ceil)
        new_scaling = torch.maximum(new_scaling, torch.full_like(new_scaling, scale_floor_log - 1.0))
    else:
        new_scaling = torch.ones(n_new, 6, device="cuda") * scale_floor_log

    if conservative_seed_render:
        seed_log = float(np.clip(np.log(max(float(grid_spacing.item()) * 0.35, 1e-5)), -10.0, 2.0))
        new_scaling[:, 3:6] = seed_log
        new_scaling[:, 0:3] = torch.minimum(new_scaling[:, 0:3],
                                             torch.full_like(new_scaling[:, 0:3], seed_log))

    # ── Offset initialisation ─────────────────────────────────────────────
    n_offsets = gaussians.n_offsets
    if conservative_seed_render:
        new_offsets = torch.zeros(n_new, n_offsets, 3, device="cuda")
        seed_offset_mag = 0.0
    else:
        rand_unit = torch.randn(n_new, n_offsets, 3, device="cuda")
        rand_unit = rand_unit / (rand_unit.norm(dim=-1, keepdim=True) + 1e-8)
        extent_med2 = float(extent.median().item())
        seed_offset_mag = min(float(offset_scale_frac) * float(grid_spacing.item()),
                              float(scale_max_frac) * extent_med2)
        new_offsets = rand_unit * seed_offset_mag

    new_labels = torch.full((n_new, 1), int(object_id),
                            dtype=gaussians.label_ids.dtype, device="cuda")

    # ── Append to model tensors in-place ─────────────────────────────────
    with torch.no_grad():
        gaussians._anchor = nn.Parameter(
            torch.cat([gaussians._anchor, new_positions], dim=0).requires_grad_(True)
        )
        gaussians._anchor_feat = nn.Parameter(
            torch.cat([gaussians._anchor_feat, new_feats], dim=0).requires_grad_(True)
        )
        gaussians._offset = nn.Parameter(
            torch.cat([gaussians._offset, new_offsets], dim=0).requires_grad_(True)
        )
        gaussians._scaling = nn.Parameter(
            torch.cat([gaussians._scaling, new_scaling], dim=0).requires_grad_(True)
        )
        gaussians._rotation = nn.Parameter(
            torch.cat([gaussians._rotation, new_rotation], dim=0).requires_grad_(False)
        )
        gaussians.label_ids = torch.cat([gaussians.label_ids, new_labels], dim=0)

        # Auxiliary statistics tensors (all models have some of these).
        # Shape is inferred from the existing tensor so we never mismatch ndim.
        # offset-level tensors are stored flat: (N*k, 1) or (N*k,).
        def _cat_zeros(attr: str, n_rows: int):
            if hasattr(gaussians, attr):
                old = getattr(gaussians, attr)
                tail = old.shape[1:]  # trailing dims after the leading count dim
                new = torch.zeros((n_rows,) + tail, dtype=old.dtype, device=old.device)
                setattr(gaussians, attr, torch.cat([old, new], dim=0))

        n_new_offsets = n_new * n_offsets
        _cat_zeros("anchor_demon", n_new)
        _cat_zeros("anchor_opacity_accum", n_new)
        _cat_zeros("offset_gradient_accum", n_new_offsets)
        _cat_zeros("offset_denom", n_new_offsets)
        _cat_zeros("offset_opacity_accum", n_new_offsets)
        _cat_zeros("max_radii2D", n_new_offsets)

        if hasattr(gaussians, "_anchor_mask"):
            gaussians._anchor_mask = torch.ones(
                gaussians._anchor.shape[0], dtype=torch.bool, device="cuda"
            )

    # Store AABB so the optimizer can cage seeds.
    if not hasattr(gaussians, "_replenishment_aabb"):
        gaussians._replenishment_aabb = {}
    gaussians._replenishment_aabb[int(object_id)] = {
        "min": bounds_min.detach().clone(),
        "max": bounds_max.detach().clone(),
        "extent": extent.detach().clone(),
        "extent_max": float(extent_max),
        "extent_med": float(extent.median().item()),
        "grid_spacing": float(grid_spacing.item()),
    }

    logger.info(
        "Seeded %d backside anchors for object %d "
        "(grid_res=%d, conservative=%s, offset_mag=%.5f, total now %d)",
        n_new, object_id, grid_resolution, conservative_seed_render,
        seed_offset_mag, int(gaussians._anchor.shape[0]),
    )
    return n_new
