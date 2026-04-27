"""
Anchor Seeding — Grid-Based Backside Filling

Replaces the mirroring strategy with a dense voxel grid bounded by the object's
physical dimensions.

Public API:
    seed_backside_anchors(...) -> int
"""

__all__ = ['seed_backside_anchors']

import sys
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

_VROOM_ROOT = Path(__file__).resolve().parent.parent.parent
_OBJECTGS_DIR = _VROOM_ROOT / "temp_deps" / "ObjectGS"
if str(_OBJECTGS_DIR) not in sys.path:
    sys.path.insert(0, str(_OBJECTGS_DIR))


def seed_backside_anchors(
    gaussians,
    object_center: np.ndarray,
    view_direction: np.ndarray,
    object_id: int,
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
    visual_hull_constraints: list = None,
    visual_hull_min_views: int = 2,
    surface_shell_filter: bool = True,
    surface_shell_min_norm: float = 0.65,
) -> int:
    """Seed a dense voxel grid on the unseen backside of an object.

    Strategy:
    1. Identify frontside anchors (facing camera) for feature borrowing.
    2. Compute the 3D bounding box (bounds_min, bounds_max) of the object.
    3. Generate a uniform 3D grid within these bounds.
    4. Filter the grid to ONLY keep points in the unseen hemisphere.
    5. Initialize features via K-nearest-neighbor weighted average from
       frontside anchors (stays in-distribution).
    6. Initialize all 6 aux tensors for training_statis().

    Args:
        gaussians: The GaussianModel to modify in-place.
        object_center: (3,) centroid of the target object.
        view_direction: (3,) unit vector FROM object TO best camera.
        object_id: Label ID of the target object.
        grid_resolution: Number of grid points per dimension.
        k_neighbors: Number of nearest frontside neighbors for feature interpolation.
        max_new_anchors: Hard cap for appended anchors to avoid memory spikes.
        bounds_quantile_low: Lower quantile for robust object bounds.
        bounds_quantile_high: Upper quantile for robust object bounds.
        hemisphere_margin: Margin around the separating plane to reduce unstable labels.

    Returns:
        Number of new grid anchors added.
    """
    labels = gaussians.label_ids.squeeze(-1)
    obj_mask = (labels == object_id)
    n_obj = int(obj_mask.sum().item())

    if n_obj < 5:
        logger.warning(f"Object {object_id} has only {n_obj} anchors, too few to bound.")
        return 0

    # Current object anchors
    obj_indices = torch.where(obj_mask)[0]
    obj_positions = gaussians._anchor.detach()[obj_indices]  # (N_obj, 3)
    obj_feats = gaussians._anchor_feat.detach()[obj_indices]
    obj_rotation = gaussians._rotation.detach()[obj_indices]
    obj_scaling = gaussians._scaling.detach()[obj_indices]  # (N_obj, 6) per-anchor log-scales

    center = torch.tensor(object_center, dtype=torch.float32, device='cuda')
    view_dir = torch.tensor(view_direction, dtype=torch.float32, device='cuda')
    view_norm = float(view_dir.norm().item())
    if view_norm < 1e-6:
        logger.warning(
            f"Object {object_id}: near-zero view direction norm ({view_norm:.3e}); "
            "falling back to +Z for seeding."
        )
        view_dir = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device='cuda')
    else:
        view_dir = view_dir / view_dir.norm()

    # Classify: frontside (facing camera) vs backside
    relative = obj_positions - center.unsqueeze(0)
    front_scores = (relative * view_dir.unsqueeze(0)).sum(dim=1)
    
    front_mask = front_scores > float(hemisphere_margin)
    n_front = int(front_mask.sum().item())

    if n_front < 3:
        logger.warning(f"Object {object_id}: only {n_front} frontside anchors, too few to borrow features.")
        return 0

    front_positions = obj_positions[front_mask]
    front_feats = obj_feats[front_mask]
    front_rotation = obj_rotation[front_mask]

    # ── 1. Create Bounded Grid ──
    # Calculate robust object bounding box using clamped quantile indices.
    q_low = float(np.clip(bounds_quantile_low, 0.0, 0.49))
    q_high = float(np.clip(bounds_quantile_high, 0.51, 1.0))
    k_low = max(1, min(n_obj, int(np.floor(q_low * n_obj))))
    k_high = max(1, min(n_obj, int(np.ceil(q_high * n_obj))))

    if k_high <= k_low:
        bounds_min = obj_positions.min(dim=0)[0]
        bounds_max = obj_positions.max(dim=0)[0]
    else:
        bounds_min, _ = torch.kthvalue(obj_positions, k_low, dim=0)
        bounds_max, _ = torch.kthvalue(obj_positions, k_high, dim=0)

    # ── 1a. Trim AABB to the dense core (kNN-density-based) ──
    # Pre-existing floaters surround the object cluster (often symmetrically),
    # so JOINT distance-from-median filtering doesn't help: the median sits
    # between core and floaters, and MAD captures both. Density does work:
    # core anchors have small kNN distance, floaters have large kNN distance.
    # We compute each anchor's distance to its k-th nearest neighbor and keep
    # the densest 50%. Their AABB is the "core" AABB used to cage seeds.
    if n_obj >= 32:
        try:
            k_density = int(min(8, max(3, n_obj // 16)))
            # pairwise dist (cdist is fine for n_obj up to a few thousand).
            dist = torch.cdist(obj_positions, obj_positions)  # (N, N)
            # k-th NN distance (k+1 because self is at distance 0).
            knn_d, _ = torch.topk(dist, k=k_density + 1, dim=1, largest=False)
            knn_d = knn_d[:, k_density]  # (N,)
            # Keep densest 50% (median-thresholded). Exclude lonely points.
            density_thr = float(torch.median(knn_d).item())
            core_keep = knn_d <= density_thr
            n_core = int(core_keep.sum().item())
            if n_core >= 16:
                core_pos = obj_positions[core_keep]
                # Tighter quantile on the core-only set rejects any residual
                # mid-density tendrils (e.g. spider-leg streaks).
                k_lo = max(1, int(0.05 * n_core))
                k_hi = max(1, min(n_core, int(np.ceil(0.95 * n_core))))
                core_min, _ = torch.kthvalue(core_pos, k_lo, dim=0)
                core_max, _ = torch.kthvalue(core_pos, k_hi, dim=0)
                # Only contract — never expand past the quantile bounds.
                bounds_min = torch.maximum(bounds_min, core_min)
                bounds_max = torch.minimum(bounds_max, core_max)
                ext_before = (obj_positions.max(0).values - obj_positions.min(0).values)
                ext_after = (bounds_max - bounds_min)
                logger.info(
                    "Object %d: density-trim core %d/%d (k=%d, knn_d_med=%.4f); "
                    "extent before=[%.3f,%.3f,%.3f] after=[%.3f,%.3f,%.3f]",
                    object_id, n_core, n_obj, k_density, density_thr,
                    float(ext_before[0]), float(ext_before[1]), float(ext_before[2]),
                    float(ext_after[0]), float(ext_after[1]), float(ext_after[2]),
                )
            else:
                logger.warning("Object %d: density-trim left only %d anchors, skipping.",
                               object_id, n_core)
        except Exception as e:
            logger.warning("Object %d: density-trim failed (%s); using quantile bounds.",
                           object_id, e)

    # Expand bounds slightly to ensure we don't clip the outer shell.
    extent = bounds_max - bounds_min
    expand = float(bounds_expand_frac)
    bounds_min -= extent * expand
    bounds_max += extent * expand
    extent = bounds_max - bounds_min  # refresh after expansion
    extent_max = float(extent.max().item())

    # Generate 3D grid
    x = torch.linspace(bounds_min[0], bounds_max[0], grid_resolution, device='cuda')
    y = torch.linspace(bounds_min[1], bounds_max[1], grid_resolution, device='cuda')
    z = torch.linspace(bounds_min[2], bounds_max[2], grid_resolution, device='cuda')
    grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing='ij')
    grid_positions = torch.stack([grid_x.flatten(), grid_y.flatten(), grid_z.flatten()], dim=-1)

    # ── 2. Filter Grid to Unseen Hemisphere ──
    # Keep only grid points that are "behind" the center plane relative to the camera
    grid_rel = grid_positions - center.unsqueeze(0)
    grid_front_scores = (grid_rel * view_dir.unsqueeze(0)).sum(dim=1)
    
    # Negative score = backside (away from camera)
    backside_mask = grid_front_scores < -float(hemisphere_margin)
    new_positions = grid_positions[backside_mask]
    if new_positions.shape[0] == 0:
        logger.warning("No backside grid points after hemisphere filtering.")
        return 0
    
    # Optional: also remove grid points that are *inside* the existing object hull
    # (By checking distance to nearest existing anchor. If too close, discard)
    dists_to_existing = torch.cdist(new_positions, obj_positions)
    min_dists, _ = dists_to_existing.min(dim=1)
    avg_spacing = extent.norm() / max(int(grid_resolution), 1)
    outside_mask = min_dists > (avg_spacing * 0.5)
    new_positions = new_positions[outside_mask]

    visual_hull_stats = None
    if visual_hull_constraints:
        support = torch.zeros(new_positions.shape[0], dtype=torch.int32, device='cuda')
        for constraint in visual_hull_constraints:
            try:
                cam = constraint['camera']
                mask_np = np.asarray(constraint['mask']).astype(bool)
                if mask_np.ndim != 2 or mask_np.sum() == 0:
                    continue
                height, width = mask_np.shape
                mask_t = torch.from_numpy(mask_np).to(device='cuda', dtype=torch.bool)
                R = torch.as_tensor(cam['R'], dtype=torch.float32, device='cuda')
                T = torch.as_tensor(cam['T'], dtype=torch.float32, device='cuda').reshape(1, 3)
                K = torch.as_tensor(cam['K'], dtype=torch.float32, device='cuda')

                cam_pts = (new_positions @ R.T) + T
                z = cam_pts[:, 2]
                valid = z > 1e-4
                u = K[0, 0] * cam_pts[:, 0] / (z + 1e-8) + K[0, 2]
                v = K[1, 1] * cam_pts[:, 1] / (z + 1e-8) + K[1, 2]
                ui = torch.round(u).long()
                vi = torch.round(v).long()
                valid = valid & (ui >= 0) & (ui < width) & (vi >= 0) & (vi < height)

                inside = torch.zeros_like(valid, dtype=torch.bool)
                if valid.any():
                    inside[valid] = mask_t[vi[valid], ui[valid]]
                support += inside.to(torch.int32)
            except Exception as e:
                logger.warning("Object %d: visual-hull constraint failed (%s); skipping one view.",
                               object_id, e)

        before_hull = int(new_positions.shape[0])
        min_views = max(1, int(visual_hull_min_views))
        keep_hull = support >= min_views
        if before_hull > 0 and not bool(keep_hull.any().item()):
            max_support = int(support.max().item()) if support.numel() else 0
            if max_support > 0:
                keep_hull = support >= max_support
                logger.warning(
                    "Object %d: no seeds met visual-hull min_views=%d; keeping max-support=%d candidates.",
                    object_id, min_views, max_support,
                )
            else:
                logger.warning(
                    "Object %d: visual hull had zero support for all candidates; leaving seed grid unfiltered.",
                    object_id,
                )
                keep_hull = torch.ones_like(support, dtype=torch.bool)

        new_positions = new_positions[keep_hull]
        support_kept = support[keep_hull]
        visual_hull_stats = {
            'enabled': True,
            'n_constraints': int(len(visual_hull_constraints)),
            'min_views': int(min_views),
            'before': int(before_hull),
            'after': int(new_positions.shape[0]),
            'support_min': int(support_kept.min().item()) if support_kept.numel() else 0,
            'support_mean': float(support_kept.float().mean().item()) if support_kept.numel() else 0.0,
            'support_max': int(support_kept.max().item()) if support_kept.numel() else 0,
        }
        gaussians._replenishment_seed_filter_stats = visual_hull_stats
        logger.info(
            "Object %d: visual-hull seed filter kept %d/%d candidates "
            "(views=%d, min_views=%d, support mean=%.2f max=%d).",
            object_id,
            visual_hull_stats['after'],
            visual_hull_stats['before'],
            visual_hull_stats['n_constraints'],
            visual_hull_stats['min_views'],
            visual_hull_stats['support_mean'],
            visual_hull_stats['support_max'],
        )

    shell_stats = None
    if surface_shell_filter and new_positions.shape[0] > 0:
        half_extent = torch.clamp((bounds_max - bounds_min) * 0.5, min=1e-6)
        box_center = (bounds_min + bounds_max) * 0.5
        normalized = torch.abs((new_positions - box_center.unsqueeze(0)) / half_extent.unsqueeze(0))
        shell_score = torch.clamp(normalized, 0.0, 1.0).max(dim=1).values
        shell_thresh = float(np.clip(surface_shell_min_norm, 0.0, 0.99))
        keep_shell = shell_score >= shell_thresh
        before_shell = int(new_positions.shape[0])
        if keep_shell.any():
            new_positions = new_positions[keep_shell]
            shell_score_kept = shell_score[keep_shell]
        else:
            # Keep the outermost decile rather than falling back to volume seeds.
            q = torch.quantile(shell_score, 0.90)
            keep_shell = shell_score >= q
            new_positions = new_positions[keep_shell]
            shell_score_kept = shell_score[keep_shell]
            logger.warning(
                "Object %d: surface-shell threshold %.2f kept nothing; keeping outermost %.0f%% instead.",
                object_id, shell_thresh, 100.0 * float(keep_shell.float().mean().item()),
            )
        shell_stats = {
            'enabled': True,
            'min_norm': float(shell_thresh),
            'before': int(before_shell),
            'after': int(new_positions.shape[0]),
            'score_min': float(shell_score_kept.min().item()) if shell_score_kept.numel() else 0.0,
            'score_mean': float(shell_score_kept.mean().item()) if shell_score_kept.numel() else 0.0,
            'score_max': float(shell_score_kept.max().item()) if shell_score_kept.numel() else 0.0,
        }
        if not hasattr(gaussians, '_replenishment_seed_filter_stats'):
            gaussians._replenishment_seed_filter_stats = {}
        gaussians._replenishment_seed_filter_stats['surface_shell'] = shell_stats
        logger.info(
            "Object %d: surface-shell seed filter kept %d/%d candidates "
            "(min_norm=%.2f, score mean=%.2f).",
            object_id,
            shell_stats['after'],
            shell_stats['before'],
            shell_stats['min_norm'],
            shell_stats['score_mean'],
        )

    n_new = new_positions.shape[0]
    if n_new == 0:
        logger.warning("No grid points fell into the valid backside region.")
        return 0

    if n_new > int(max_new_anchors):
        keep = int(max_new_anchors)
        perm = torch.randperm(n_new, device='cuda')[:keep]
        new_positions = new_positions[perm]
        logger.info(
            f"Object {object_id}: capping backside seeds from {n_new} to {keep}."
        )
        n_new = keep

    # ── 3. K-NN Weighted Feature Initialization ──
    k = min(k_neighbors, n_front)
    dists = torch.cdist(new_positions, front_positions)  # (n_new, n_front)
    _, knn_indices = dists.topk(k, dim=1, largest=False)

    knn_feats = front_feats[knn_indices]     # (n_new, k, feat_dim)
    knn_rotation = front_rotation[knn_indices]  # (n_new, k, 4)

    knn_dists = torch.gather(dists, 1, knn_indices)  # (n_new, k)
    weights = 1.0 / (knn_dists + 1e-4)
    weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)

    new_feats = (knn_feats * weights.unsqueeze(-1)).sum(dim=1)
    new_rotation = (knn_rotation * weights.unsqueeze(-1)).sum(dim=1)
    new_rotation = new_rotation / (new_rotation.norm(dim=-1, keepdim=True) + 1e-8)

    # ── Per-anchor scale init ──
    grid_spacing = extent.norm() / (max(int(grid_resolution), 1) * 1.732)
    grid_spacing = torch.clamp(grid_spacing, min=1e-4, max=1e2)
    scale_floor_log = float(torch.log(grid_spacing).item())
    scale_floor_log = float(np.clip(scale_floor_log, -10.0, 2.0))

    # KNN-weighted distance-based blend of original anchors' log-scales.
    # Use full object anchors (not just front) to get a representative
    # surface-density prior; fall back to grid_spacing if too sparse.
    if obj_scaling.shape[0] >= 4:
        full_dists = torch.cdist(new_positions, obj_positions)
        k_obj = min(int(k_neighbors), int(obj_scaling.shape[0]))
        obj_knn_dists, obj_knn_idx = full_dists.topk(k_obj, dim=1, largest=False)
        obj_knn_w = 1.0 / (obj_knn_dists + 1e-4)
        obj_knn_w = obj_knn_w / (obj_knn_w.sum(dim=1, keepdim=True) + 1e-8)
        knn_scaling = obj_scaling[obj_knn_idx]  # (n_new, k_obj, 6)
        new_scaling = (knn_scaling * obj_knn_w.unsqueeze(-1)).sum(dim=1)
        # Floor: at least the grid spacing so seeds are visible.
        # Ceiling: scale_max_frac * extent_med (median axis) so a Gaussian
        # never grows larger than the object's characteristic dimension.
        extent_med_local = float(extent.median().item())
        scale_ceiling_log_seed = float(np.log(max(extent_med_local * float(scale_max_frac), 1e-6)))
        new_scaling = torch.clamp(
            new_scaling,
            min=scale_floor_log - 1.0,
            max=scale_ceiling_log_seed,
        )
        # Enforce a hard floor so degenerate KNN can't shrink below visibility.
        new_scaling = torch.maximum(
            new_scaling,
            torch.full_like(new_scaling, scale_floor_log - 1.0),
        )
    else:
        new_scaling = torch.ones(n_new, 6, device='cuda') * scale_floor_log

    if conservative_seed_render:
        # A seeded ObjectGS anchor immediately expands into n_offsets child
        # Gaussians through the frozen opacity/color/cov MLPs. Borrowed KNN
        # scales from visible front anchors can make those children render as
        # a comb of textured strokes before any optimization happens. Start
        # seeds as small, isotropic, local blobs instead; optimization can
        # still move/shape them through the normal update path.
        seed_log_scale = float(np.log(max(float(grid_spacing.item()) * 0.35, 1e-5)))
        seed_log_scale = float(np.clip(seed_log_scale, -10.0, 2.0))
        new_scaling[:, 3:6] = seed_log_scale
        # Keep the child-offset scale local as well. Dims [0:3] multiply
        # _offset at render time, so using the same local scale prevents
        # randomly initialized children from fanning into visible streaks.
        new_scaling[:, 0:3] = torch.minimum(
            new_scaling[:, 0:3],
            torch.full_like(new_scaling[:, 0:3], seed_log_scale),
        )

    # ── Per-anchor offsets: object-relative magnitude ──
    n_offsets = gaussians.n_offsets
    if conservative_seed_render:
        seed_offset_mag = 0.0
        new_offsets = torch.zeros(n_new, n_offsets, 3, device='cuda')
    else:
        rand_vecs = torch.randn(n_new, n_offsets, 3, device='cuda')
        rand_unit = rand_vecs / (rand_vecs.norm(dim=-1, keepdim=True) + 1e-8)
        # Couple offset spread to local grid spacing so small objects get
        # small offsets. Default offset_scale_frac=0.5 of grid spacing.
        seed_offset_mag = float(offset_scale_frac) * float(grid_spacing.item())
        # Hard ceiling: never seed offsets larger than scale_max_frac * extent_med.
        extent_med_local = float(extent.median().item())
        seed_offset_mag = min(seed_offset_mag, float(scale_max_frac) * extent_med_local)
        new_offsets = rand_unit * seed_offset_mag

    new_labels = torch.full((n_new, 1), object_id, dtype=gaussians.label_ids.dtype, device='cuda')
    n_offsets = gaussians.n_offsets

    # ── 4. Concatenate into model tensors ──
    with torch.no_grad():
        gaussians._anchor = nn.Parameter(
            torch.cat([gaussians._anchor, new_positions], dim=0).requires_grad_(True))
        gaussians._anchor_feat = nn.Parameter(
            torch.cat([gaussians._anchor_feat, new_feats], dim=0).requires_grad_(True))
        gaussians._offset = nn.Parameter(
            torch.cat([gaussians._offset, new_offsets], dim=0).requires_grad_(True))
        gaussians._scaling = nn.Parameter(
            torch.cat([gaussians._scaling, new_scaling], dim=0).requires_grad_(True))
        gaussians._rotation = nn.Parameter(
            torch.cat([gaussians._rotation, new_rotation], dim=0).requires_grad_(False))

        gaussians.label_ids = torch.cat([gaussians.label_ids, new_labels], dim=0)

        # ── 5. Initialize ALL SIX auxiliary tensors ──
        if hasattr(gaussians, 'anchor_demon'):
            gaussians.anchor_demon = torch.cat([
                gaussians.anchor_demon, torch.zeros(n_new, 1, device='cuda')], dim=0)
        if hasattr(gaussians, 'anchor_opacity_accum'):
            gaussians.anchor_opacity_accum = torch.cat([
                gaussians.anchor_opacity_accum, torch.zeros(n_new, 1, device='cuda')], dim=0)
        if hasattr(gaussians, 'offset_gradient_accum'):
            gaussians.offset_gradient_accum = torch.cat([
                gaussians.offset_gradient_accum, torch.zeros(n_new * n_offsets, 1, device='cuda')], dim=0)
        if hasattr(gaussians, 'offset_denom'):
            gaussians.offset_denom = torch.cat([
                gaussians.offset_denom, torch.zeros(n_new * n_offsets, 1, dtype=torch.int32, device='cuda')], dim=0)
        if hasattr(gaussians, 'offset_opacity_accum'):
            gaussians.offset_opacity_accum = torch.cat([
                gaussians.offset_opacity_accum, torch.zeros(n_new * n_offsets, 1, dtype=torch.int32, device='cuda')], dim=0)
        if hasattr(gaussians, 'max_radii2D'):
            gaussians.max_radii2D = torch.cat([
                gaussians.max_radii2D, torch.zeros(n_new * n_offsets, device='cuda')], dim=0)
        if hasattr(gaussians, '_anchor_mask'):
            gaussians._anchor_mask = torch.ones(
                gaussians._anchor.shape[0], dtype=torch.bool, device='cuda')

    # Persist this object's AABB so the optimizer can cage seeded anchors.
    extent_min_val = float(extent.min().item())
    extent_med_val = float(extent.median().item())
    if not hasattr(gaussians, '_replenishment_aabb'):
        gaussians._replenishment_aabb = {}
    gaussians._replenishment_aabb[int(object_id)] = {
        'min': bounds_min.detach().clone(),
        'max': bounds_max.detach().clone(),
        'extent': extent.detach().clone(),
        'extent_max': float(extent_max),
        'extent_min': float(extent_min_val),
        'extent_med': float(extent_med_val),
        'grid_spacing': float(grid_spacing.item()),
    }

    logger.info(
        f"Seeded {n_new} backside grid anchors for object {object_id} "
        f"(q=({q_low:.2f},{q_high:.2f}), margin={hemisphere_margin:.3f}, "
        f"bounds_min={bounds_min.cpu().numpy().round(4)}, bounds_max={bounds_max.cpu().numpy().round(4)}, "
        f"extent_max={extent_max:.4f}, grid_spacing={float(grid_spacing.item()):.5f}, "
        f"seed_offset_mag={seed_offset_mag:.5f}, "
        f"conservative_seed_render={bool(conservative_seed_render)}, "
        f"total anchors now: {gaussians._anchor.shape[0]})"
    )
    return n_new
