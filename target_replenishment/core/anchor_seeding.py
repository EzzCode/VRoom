"""
Anchor Seeding — Grid-Based Backside Filling

Replaces the mirroring strategy with a dense voxel grid bounded by the object's
physical dimensions.

Public API:
    seed_backside_anchors(...) -> int
"""

__all__ = ['seed_backside_anchors', 'prune_object_floaters_dense_surface']

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


def _chunked_kth_knn_distance(points: torch.Tensor, k: int, batch_size: int = 1024) -> torch.Tensor:
    """Compute per-point k-th nearest-neighbor distance with chunked cdist."""
    n = int(points.shape[0])
    if n <= 1:
        return torch.zeros((n,), dtype=torch.float32, device=points.device)

    k_eff = int(max(1, min(int(k), n - 1)))
    out = torch.empty((n,), dtype=torch.float32, device=points.device)

    for start in range(0, n, int(batch_size)):
        end = min(n, start + int(batch_size))
        chunk = points[start:end]
        dist = torch.cdist(chunk, points)
        knn_d, _ = torch.topk(dist, k=k_eff + 1, dim=1, largest=False)
        out[start:end] = knn_d[:, k_eff]
    return out


def _chunked_knn_indices(points: torch.Tensor, k: int, batch_size: int = 1024) -> torch.Tensor:
    """Compute per-point k-NN neighbor indices with chunked cdist."""
    n = int(points.shape[0])
    if n <= 1:
        return torch.zeros((n, 0), dtype=torch.long, device=points.device)

    k_eff = int(max(1, min(int(k), n - 1)))
    out = torch.empty((n, k_eff), dtype=torch.long, device=points.device)

    for start in range(0, n, int(batch_size)):
        end = min(n, start + int(batch_size))
        chunk = points[start:end]
        dist = torch.cdist(chunk, points)
        _, knn_idx = torch.topk(dist, k=k_eff + 1, dim=1, largest=False)
        out[start:end] = knn_idx[:, 1:k_eff + 1]
    return out


def _largest_component_from_knn(knn_idx: torch.Tensor) -> torch.Tensor:
    """Return a boolean mask for the largest connected component in a k-NN graph."""
    import numpy as _np

    n = int(knn_idx.shape[0])
    if n <= 1:
        return torch.ones((n,), dtype=torch.bool, device=knn_idx.device)

    edges = knn_idx.detach().cpu().numpy().astype(_np.int64)
    parent = _np.arange(n, dtype=_np.int64)
    size = _np.ones(n, dtype=_np.int64)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return int(x)

    def union(a: int, b: int):
        ra = find(a)
        rb = find(b)
        if ra == rb:
            return
        if size[ra] < size[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        size[ra] += size[rb]

    for i in range(n):
        row = edges[i]
        for j in row:
            j_int = int(j)
            if j_int < 0 or j_int >= n or j_int == i:
                continue
            union(i, j_int)

    roots = _np.empty(n, dtype=_np.int64)
    for i in range(n):
        roots[i] = find(i)

    uniq, counts = _np.unique(roots, return_counts=True)
    largest_root = int(uniq[_np.argmax(counts)])
    keep_np = roots == largest_root
    return torch.from_numpy(keep_np).to(device=knn_idx.device, dtype=torch.bool)


def _select_dense_surface_mask(
    obj_positions: torch.Tensor,
    density_quantile: float = 0.65,
    min_keep_ratio: float = 0.35,
    knn_k: int = 8,
    connectivity_knn: int = 6,
    batch_size: int = 1024,
):
    """Select a balanced dense-surface mask, ignoring sparse floater anchors."""
    n_obj = int(obj_positions.shape[0])
    keep_all = torch.ones((n_obj,), dtype=torch.bool, device=obj_positions.device)
    if n_obj < 32:
        return keep_all, {
            'enabled': True,
            'n_before': n_obj,
            'n_after': n_obj,
            'n_pruned': 0,
            'reason': 'too_few_anchors',
        }

    k_density = int(max(3, min(int(knn_k), n_obj - 1)))
    q_density_base = float(np.clip(density_quantile, 0.30, 0.95))
    q_density = q_density_base
    min_keep = int(max(16, min(n_obj, round(float(min_keep_ratio) * n_obj))))

    kth_dist = _chunked_kth_knn_distance(
        obj_positions,
        k=k_density,
        batch_size=int(max(128, batch_size)),
    )

    q50 = float(torch.quantile(kth_dist, 0.50).item())
    q90 = float(torch.quantile(kth_dist, 0.90).item())
    spread_ratio = float(q90 / max(q50, 1e-8))

    # Adaptive quantile: noisy long-tail objects should tighten (remove more);
    # compact objects should relax (remove less).
    if spread_ratio >= 2.4:
        q_density = max(0.30, q_density_base - 0.18)
    elif spread_ratio >= 2.0:
        q_density = max(0.30, q_density_base - 0.12)
    elif spread_ratio >= 1.7:
        q_density = max(0.30, q_density_base - 0.06)
    elif spread_ratio <= 1.20:
        q_density = min(0.95, q_density_base + 0.10)
    elif spread_ratio <= 1.35:
        q_density = min(0.95, q_density_base + 0.05)

    def _dense_keep_for_quantile(q_val: float):
        thr = float(torch.quantile(kth_dist, float(np.clip(q_val, 0.30, 0.95))).item())
        keep = kth_dist <= thr
        n_keep_local = int(keep.sum().item())
        if n_keep_local < min_keep:
            _, top_idx = torch.topk(kth_dist, k=min_keep, largest=False)
            keep = torch.zeros_like(keep)
            keep[top_idx] = True
            n_keep_local = int(keep.sum().item())
        return keep, thr, n_keep_local

    dense_keep, density_thr, n_dense = _dense_keep_for_quantile(q_density)

    # Secondary adaptivity from geometric shrink amount to avoid over/under-prune.
    extent_before = torch.clamp(
        obj_positions.max(dim=0).values - obj_positions.min(dim=0).values,
        min=1e-6,
    )
    extent_after_dense = torch.clamp(
        obj_positions[dense_keep].max(dim=0).values - obj_positions[dense_keep].min(dim=0).values,
        min=1e-6,
    )
    extent_ratio_max = float((extent_after_dense / extent_before).max().item())

    # If shrink is too extreme, we likely removed true surface anchors; relax.
    if extent_ratio_max < 0.30:
        q_relax = min(0.95, q_density + 0.10)
        dense_keep, density_thr, n_dense = _dense_keep_for_quantile(q_relax)
        q_density = q_relax
        extent_after_dense = torch.clamp(
            obj_positions[dense_keep].max(dim=0).values - obj_positions[dense_keep].min(dim=0).values,
            min=1e-6,
        )
        extent_ratio_max = float((extent_after_dense / extent_before).max().item())
    # If shrink is too weak, tighten to suppress residual floaters.
    elif extent_ratio_max > 0.78:
        q_tight = max(0.30, q_density - 0.08)
        dense_keep, density_thr, n_dense = _dense_keep_for_quantile(q_tight)
        q_density = q_tight
        extent_after_dense = torch.clamp(
            obj_positions[dense_keep].max(dim=0).values - obj_positions[dense_keep].min(dim=0).values,
            min=1e-6,
        )
        extent_ratio_max = float((extent_after_dense / extent_before).max().item())

    n_dense = int(dense_keep.sum().item())

    final_keep = dense_keep.clone()
    n_component = n_dense
    comp_ratio = 1.0
    k_conn_used = 0

    if n_dense >= 32 and int(connectivity_knn) > 0:
        dense_idx = torch.nonzero(dense_keep, as_tuple=False).squeeze(1)
        dense_pos = obj_positions[dense_keep]
        k_conn = int(max(2, min(int(connectivity_knn), int(dense_pos.shape[0]) - 1)))
        k_conn_used = k_conn
        knn_dense = _chunked_knn_indices(
            dense_pos,
            k=k_conn,
            batch_size=int(max(128, batch_size)),
        )
        largest_local = _largest_component_from_knn(knn_dense)
        n_component = int(largest_local.sum().item())
        comp_ratio = float(n_component / max(n_dense, 1))

        # Balanced mode: prefer the largest coherent component only when it
        # captures a substantial part of the dense set; otherwise keep dense set.
        if n_component >= max(16, int(0.40 * n_dense), int(0.20 * n_obj)):
            final_keep = torch.zeros_like(dense_keep)
            final_keep[dense_idx[largest_local]] = True

    n_after = int(final_keep.sum().item())
    if n_after < min_keep:
        _, top_idx = torch.topk(kth_dist, k=min_keep, largest=False)
        final_keep = torch.zeros_like(final_keep)
        final_keep[top_idx] = True
        n_after = int(final_keep.sum().item())

    stats = {
        'enabled': True,
        'n_before': int(n_obj),
        'n_after': int(n_after),
        'n_pruned': int(n_obj - n_after),
        'density_quantile_base': float(q_density_base),
        'density_quantile': float(q_density),
        'density_threshold': float(density_thr),
        'density_spread_ratio_q90_q50': float(spread_ratio),
        'knn_k': int(k_density),
        'n_dense': int(n_dense),
        'extent_ratio_max_dense': float(extent_ratio_max),
        'connectivity_knn': int(k_conn_used),
        'n_largest_component': int(n_component),
        'largest_component_ratio': float(comp_ratio),
        'min_keep_ratio': float(min_keep_ratio),
    }
    return final_keep, stats


def _slice_like(param, mask: torch.Tensor):
    """Slice Tensor/Parameter by a boolean mask on dim 0 while preserving type."""
    sliced = param[mask].clone()
    if isinstance(param, torch.nn.Parameter):
        return torch.nn.Parameter(sliced, requires_grad=param.requires_grad)
    return sliced


def _apply_anchor_prune_mask(gaussians, keep_mask: torch.Tensor):
    """Apply a global keep-mask to anchor and child-stat tensors."""
    n_total = int(keep_mask.shape[0])
    n_offsets = int(getattr(gaussians, 'n_offsets', 1))
    child_keep = keep_mask.unsqueeze(1).expand(-1, n_offsets).reshape(-1)

    per_anchor_attrs = [
        '_anchor', '_anchor_feat', '_offset', '_scaling', '_rotation',
        'label_ids', 'anchor_demon', 'anchor_opacity_accum', '_anchor_mask',
    ]
    per_child_attrs = [
        'offset_gradient_accum', 'offset_denom', 'offset_opacity_accum', 'max_radii2D',
    ]

    for name in per_anchor_attrs:
        if not hasattr(gaussians, name):
            continue
        tensor = getattr(gaussians, name)
        if tensor is None or int(tensor.shape[0]) != n_total:
            continue
        setattr(gaussians, name, _slice_like(tensor, keep_mask))

    for name in per_child_attrs:
        if not hasattr(gaussians, name):
            continue
        tensor = getattr(gaussians, name)
        if tensor is None or int(tensor.shape[0]) != int(child_keep.shape[0]):
            continue
        setattr(gaussians, name, _slice_like(tensor, child_keep))


def prune_object_floaters_dense_surface(
    gaussians,
    object_id: int,
    density_quantile: float = 0.65,
    min_keep_ratio: float = 0.35,
    knn_k: int = 8,
    connectivity_knn: int = 6,
):
    """Permanently remove sparse floater anchors for one object label.

    Returns a diagnostics dictionary with before/after counts and extents.
    """
    labels = gaussians.label_ids.squeeze(-1)
    obj_mask = labels == int(object_id)
    obj_idx = torch.nonzero(obj_mask, as_tuple=False).squeeze(1)
    n_obj = int(obj_idx.numel())
    if n_obj < 32:
        return {
            'enabled': True,
            'object_id': int(object_id),
            'n_before': int(n_obj),
            'n_after': int(n_obj),
            'n_pruned': 0,
            'reason': 'too_few_anchors',
        }

    obj_positions = gaussians._anchor.detach()[obj_idx]
    keep_local, stats = _select_dense_surface_mask(
        obj_positions=obj_positions,
        density_quantile=float(density_quantile),
        min_keep_ratio=float(min_keep_ratio),
        knn_k=int(knn_k),
        connectivity_knn=int(connectivity_knn),
    )

    n_keep = int(keep_local.sum().item())
    n_pruned = int(n_obj - n_keep)
    if n_pruned <= 0:
        stats.update({
            'enabled': True,
            'object_id': int(object_id),
            'n_before': int(n_obj),
            'n_after': int(n_obj),
            'n_pruned': 0,
            'reason': 'no_sparse_floaters_detected',
        })
        return stats

    keep_global = torch.ones(
        (int(gaussians._anchor.shape[0]),),
        dtype=torch.bool,
        device=gaussians._anchor.device,
    )
    remove_local = ~keep_local
    keep_global[obj_idx[remove_local]] = False

    extent_before = obj_positions.max(dim=0).values - obj_positions.min(dim=0).values
    kept_positions = obj_positions[keep_local]
    extent_after = kept_positions.max(dim=0).values - kept_positions.min(dim=0).values

    _apply_anchor_prune_mask(gaussians, keep_global)

    if hasattr(gaussians, 'n_original_anchors') and gaussians.n_original_anchors is not None:
        gaussians.n_original_anchors = int(min(int(gaussians.n_original_anchors), int(gaussians._anchor.shape[0])))

    stats.update({
        'enabled': True,
        'object_id': int(object_id),
        'n_before': int(n_obj),
        'n_after': int(n_keep),
        'n_pruned': int(n_pruned),
        'extent_before': [float(v) for v in extent_before.detach().cpu().tolist()],
        'extent_after': [float(v) for v in extent_after.detach().cpu().tolist()],
    })
    logger.info(
        "Object %d: pre-seed floater prune removed %d/%d anchors; extent before=[%.3f,%.3f,%.3f] after=[%.3f,%.3f,%.3f]",
        int(object_id),
        int(n_pruned),
        int(n_obj),
        float(extent_before[0]), float(extent_before[1]), float(extent_before[2]),
        float(extent_after[0]), float(extent_after[1]), float(extent_after[2]),
    )
    return stats


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