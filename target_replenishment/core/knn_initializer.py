"""
Stage C — KNN Direct-Drive Initializer.

For each seed_xyz position, finds the k nearest *survivor* anchors of the
same object and copies their features (KNN-blended). This is the
"copy the couch fabric" trick that worked in earlier experiments.

Side-effects on `gaussians`:
  - Replaces _anchor, _offset, _scaling, _rotation, _anchor_feat with new
    nn.Parameter tensors that are concat(originals, seeds).
  - Extends label_ids with `object_id`.
  - Sets `gaussians._anchor_mask` to all-True.
  - Stamps `gaussians._replenishment_aabb[object_id] = (bounds_min, bounds_max)`.
  - Stamps `gaussians._replenishment_seeded_mask` to a (N_total,) bool tensor.
  - Stamps `gaussians._replenishment_originals_snapshot` with detached copies
    of the original tensors, indexed by attribute name.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch import nn


@dataclass
class PruneResult:
    n_before: int
    n_after: int
    n_target_before: int
    n_target_after: int
    n_pruned: int
    survivor_indices_mapped: np.ndarray

    def to_dict(self) -> dict:
        return {
            "n_before": int(self.n_before),
            "n_after": int(self.n_after),
            "n_target_before": int(self.n_target_before),
            "n_target_after": int(self.n_target_after),
            "n_pruned": int(self.n_pruned),
        }


@dataclass
class KNNInitResult:
    n_originals: int
    n_seeded: int
    n_total: int
    knn_k: int
    scale_clip: tuple
    seed_indices_global: np.ndarray  # (n_seeded,) global anchor indices
    seed_opacity_lift: float = 0.0
    seed_opacity_gate: float = 1.0
    seed_fixed_opacity: float = 0.0
    seed_scaling_boost: float = 1.0

    def to_dict(self) -> dict:
        return {
            "n_originals": int(self.n_originals),
            "n_seeded": int(self.n_seeded),
            "n_total": int(self.n_total),
            "knn_k": int(self.knn_k),
            "scale_clip_log": [float(self.scale_clip[0]), float(self.scale_clip[1])],
            "seed_opacity_lift": float(self.seed_opacity_lift),
            "seed_opacity_gate": float(self.seed_opacity_gate),
            "seed_fixed_opacity": float(self.seed_fixed_opacity),
            "seed_scaling_boost": float(self.seed_scaling_boost),
        }


def prune_target_object_floaters(
    gaussians,
    object_id: int,
    survivor_global_indices: np.ndarray,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    padding: float = 0.0,
) -> PruneResult:
    """Delete target-object anchors lying outside the survivor AABB envelope.

    Survivors are always kept; non-target anchors are always preserved.
    """
    labels_np = gaussians.label_ids.detach().squeeze(-1).cpu().numpy()
    anchor_np = gaussians._anchor.detach().cpu().numpy()
    n_before = int(labels_np.shape[0])
    target_mask_np = labels_np == int(object_id)
    keep_mask_np = ~target_mask_np
    survivor_global_indices = np.asarray(survivor_global_indices, dtype=np.int64)

    lo = np.asarray(bounds_min, dtype=np.float32) - float(padding)
    hi = np.asarray(bounds_max, dtype=np.float32) + float(padding)
    in_box = np.all((anchor_np >= lo.reshape(1, 3)) & (anchor_np <= hi.reshape(1, 3)), axis=1)
    keep_mask_np[target_mask_np & in_box] = True
    keep_mask_np[survivor_global_indices] = True

    if keep_mask_np.all():
        old_to_new = np.arange(n_before, dtype=np.int64)
        mapped = old_to_new[survivor_global_indices]
        return PruneResult(
            n_before=n_before,
            n_after=n_before,
            n_target_before=int(target_mask_np.sum()),
            n_target_after=int((target_mask_np & keep_mask_np).sum()),
            n_pruned=0,
            survivor_indices_mapped=mapped,
        )

    device = gaussians._anchor.device
    keep_mask = torch.from_numpy(keep_mask_np).to(device=device, dtype=torch.bool)
    keep_idx = torch.nonzero(keep_mask, as_tuple=False).squeeze(1)

    old_to_new = np.full(n_before, -1, dtype=np.int64)
    old_to_new[keep_mask_np] = np.arange(int(keep_mask_np.sum()), dtype=np.int64)
    mapped_survivors = old_to_new[survivor_global_indices]
    if (mapped_survivors < 0).any():
        raise RuntimeError("Survivor remap failed after target-object pruning.")

    def _filter_param(name: str, requires_grad: bool | None = None):
        old = getattr(gaussians, name)
        filtered = old.detach()[keep_idx].clone()
        if requires_grad is None:
            requires_grad = bool(getattr(old, "requires_grad", False))
        setattr(gaussians, name, nn.Parameter(filtered.requires_grad_(requires_grad)))

    _filter_param("_anchor")
    _filter_param("_anchor_feat")
    _filter_param("_scaling")
    _filter_param("_rotation")
    _filter_param("_offset")
    gaussians.label_ids = gaussians.label_ids.detach()[keep_idx].clone()

    n_after = int(gaussians._anchor.shape[0])
    n_offsets = int(gaussians._offset.shape[1])
    gaussians._anchor_mask = torch.ones(n_after, dtype=torch.bool, device=device)
    gaussians.anchor_opacity_accum = torch.zeros((n_after, 1), device=device)
    gaussians.anchor_demon = torch.zeros((n_after, 1), device=device)
    gaussians.offset_opacity_accum = torch.zeros((n_after * n_offsets, 1), device=device)
    gaussians.offset_gradient_accum = torch.zeros((n_after * n_offsets, 1), device=device)
    gaussians.offset_denom = torch.zeros((n_after * n_offsets, 1), device=device)
    gaussians.max_radii2D = torch.zeros(n_after * n_offsets, dtype=torch.float, device=device)

    return PruneResult(
        n_before=n_before,
        n_after=n_after,
        n_target_before=int(target_mask_np.sum()),
        n_target_after=int((target_mask_np & keep_mask_np).sum()),
        n_pruned=int(n_before - n_after),
        survivor_indices_mapped=mapped_survivors.astype(np.int64),
    )


def prune_anchor_indices(gaussians, prune_global_indices: np.ndarray) -> dict:
    """Remove explicit global anchor indices from all anchor-sized tensors."""
    prune_global_indices = np.asarray(prune_global_indices, dtype=np.int64)
    n_before = int(gaussians._anchor.shape[0])
    if prune_global_indices.size == 0:
        return {"n_before": n_before, "n_after": n_before, "n_pruned": 0}

    valid = prune_global_indices[(prune_global_indices >= 0) & (prune_global_indices < n_before)]
    if valid.size == 0:
        return {"n_before": n_before, "n_after": n_before, "n_pruned": 0}

    keep_mask_np = np.ones(n_before, dtype=bool)
    keep_mask_np[np.unique(valid)] = False
    device = gaussians._anchor.device
    keep_idx = torch.from_numpy(np.where(keep_mask_np)[0]).to(device=device, dtype=torch.long)

    def _filter_param(name: str, requires_grad: bool | None = None):
        old = getattr(gaussians, name)
        filtered = old.detach()[keep_idx].clone()
        if requires_grad is None:
            requires_grad = bool(getattr(old, "requires_grad", False))
        setattr(gaussians, name, nn.Parameter(filtered.requires_grad_(requires_grad)))

    _filter_param("_anchor")
    _filter_param("_anchor_feat")
    _filter_param("_scaling")
    _filter_param("_rotation")
    _filter_param("_offset")
    gaussians.label_ids = gaussians.label_ids.detach()[keep_idx].clone()

    # Keep seed-override arrays (renderer reads these by
    # seed_indices = global_idx - n_original_anchors) aligned with survivors.
    n_orig_before = int(getattr(gaussians, "n_original_anchors", n_before) or n_before)
    n_orig_before = max(0, min(n_orig_before, n_before))
    keep_mask_t = torch.from_numpy(keep_mask_np).to(device=device)
    seed_keep_mask = keep_mask_t[n_orig_before:]
    n_orig_after = int(keep_mask_np[:n_orig_before].sum()) if n_orig_before > 0 else 0
    if hasattr(gaussians, "n_original_anchors"):
        gaussians.n_original_anchors = n_orig_after
    for attr in (
        "replenishment_seed_opacity_lift",
        "replenishment_seed_opacity_gate",
        "replenishment_seed_fixed_opacity",
        "replenishment_seed_color_rgb",
    ):
        val = getattr(gaussians, attr, None)
        if val is None:
            continue
        if isinstance(val, torch.Tensor) and val.shape[0] == (n_before - n_orig_before):
            setattr(gaussians, attr, val[seed_keep_mask].clone())

    n_after = int(gaussians._anchor.shape[0])
    n_offsets = int(gaussians._offset.shape[1])
    gaussians._anchor_mask = torch.ones(n_after, dtype=torch.bool, device=device)
    gaussians.anchor_opacity_accum = torch.zeros((n_after, 1), device=device)
    gaussians.anchor_demon = torch.zeros((n_after, 1), device=device)
    gaussians.offset_opacity_accum = torch.zeros((n_after * n_offsets, 1), device=device)
    gaussians.offset_gradient_accum = torch.zeros((n_after * n_offsets, 1), device=device)
    gaussians.offset_denom = torch.zeros((n_after * n_offsets, 1), device=device)
    gaussians.max_radii2D = torch.zeros(n_after * n_offsets, dtype=torch.float, device=device)

    return {
        "n_before": n_before,
        "n_after": n_after,
        "n_pruned": int(n_before - n_after),
    }


def _estimate_survivor_normals(survivor_xyz: np.ndarray, k: int = 12) -> np.ndarray:
    """Per-survivor outward unit normal via local PCA, oriented away from centroid."""
    n = survivor_xyz.shape[0]
    if n < 4:
        return np.zeros((n, 3), dtype=np.float32)
    k = int(min(max(k, 4), n))
    tree = cKDTree(survivor_xyz)
    _, nn_idx = tree.query(survivor_xyz, k=k)
    centroid = survivor_xyz.mean(axis=0)
    normals = np.zeros((n, 3), dtype=np.float32)
    for i in range(n):
        pts = survivor_xyz[nn_idx[i]]
        c = pts.mean(axis=0)
        cov = np.cov((pts - c).T)
        try:
            evals, evecs = np.linalg.eigh(cov)
        except np.linalg.LinAlgError:
            continue
        normal = evecs[:, 0]  # smallest eigenvalue → surface normal
        # Orient outward from object centroid
        if np.dot(normal, survivor_xyz[i] - centroid) < 0:
            normal = -normal
        normals[i] = normal
    return normals.astype(np.float32)


def knn_direct_drive_init(
    gaussians,
    object_id: int,
    seed_xyz: np.ndarray,
    survivor_global_indices: np.ndarray,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    grid_spacing: float,
    knn_k: int = 4,
    scale_log_floor_offset: float = -1.0,
    scale_log_ceil_offset: float = 0.5,
    seed_opacity_lift: float = 1.0,
    seed_opacity_gate: float = 1.0,
    seed_fixed_opacity: float = 0.0,
    seed_scaling_boost: float = 1.0,
    seed_sheet_tangent_u: np.ndarray | None = None,
    seed_sheet_tangent_v: np.ndarray | None = None,
    seed_sheet_radius_factor: float = 0.45,
    seed_normal: np.ndarray | None = None,
    normal_align_min_cos: float = 0.0,
    normal_donor_pool_k: int = 32,
) -> KNNInitResult:
    """Append normal-aware seed anchors to ``gaussians`` for the given object_id.

    Donor selection is normal-aware: for each seed, donors are restricted to
    survivors whose locally-estimated outward normal aligns with the seed's
    wall normal (cosine >= ``normal_align_min_cos``). This is what aligns the
    new shrinkwrap shell with the original 2DGS surface frame so the frozen
    cov-MLP produces flat splats coplanar with the donor surface.

    Offsets are placed on a deterministic planar mini-patch in the wall's
    tangent basis. Anchor features are taken as the medoid of the eligible
    donor pool to avoid latent averaging color drift.
    """
    seed_xyz = np.asarray(seed_xyz, dtype=np.float32)
    n_seed = int(seed_xyz.shape[0])
    n_orig = int(gaussians._anchor.shape[0])

    if n_seed == 0:
        _stamp_meta(
            gaussians, object_id, bounds_min, bounds_max,
            seeded_mask=torch.zeros(n_orig, dtype=torch.bool, device="cuda"),
        )
        return KNNInitResult(
            n_originals=n_orig, n_seeded=0, n_total=n_orig, knn_k=knn_k,
            scale_clip=(0.0, 0.0), seed_indices_global=np.zeros(0, dtype=np.int64),
            seed_opacity_lift=seed_opacity_lift,
            seed_opacity_gate=seed_opacity_gate,
            seed_fixed_opacity=seed_fixed_opacity,
            seed_scaling_boost=seed_scaling_boost,
        )

    survivor_global_indices = np.asarray(survivor_global_indices, dtype=np.int64)
    if survivor_global_indices.size == 0:
        raise ValueError("knn_direct_drive_init requires at least 1 survivor donor.")

    device = gaussians._anchor.device
    dtype = gaussians._anchor.dtype

    survivor_xyz = gaussians._anchor.detach().cpu().numpy()[survivor_global_indices]

    # Per-survivor outward normals (Task 2: 2DGS-aligned donor selection).
    survivor_normals = _estimate_survivor_normals(survivor_xyz, k=12)

    # Build per-seed donor selection. We pull a wider candidate pool, then
    # filter to those whose normal aligns with the seed wall normal.
    pool_k = int(min(max(normal_donor_pool_k, knn_k), survivor_global_indices.size))
    tree = cKDTree(survivor_xyz)
    pool_dists, pool_idx = tree.query(seed_xyz, k=pool_k)
    if pool_k == 1:
        pool_dists = pool_dists[:, None]
        pool_idx = pool_idx[:, None]

    eps = 1e-8
    k = int(min(knn_k, pool_k))

    if seed_normal is not None and survivor_normals.size > 0:
        seed_normal_arr = np.asarray(seed_normal, dtype=np.float32)
        if seed_normal_arr.ndim == 1:
            seed_normal_arr = np.repeat(seed_normal_arr.reshape(1, 3), n_seed, axis=0)
        # cos similarity between each seed's wall normal and pool donor normals.
        pool_normals = survivor_normals[pool_idx]                      # (M, pool_k, 3)
        cos = (pool_normals * seed_normal_arr[:, None, :]).sum(axis=-1)  # (M, pool_k)
        eligible = cos >= float(normal_align_min_cos)
        # Guarantee each seed has at least k donors.
        for m in range(n_seed):
            if int(eligible[m].sum()) < k:
                # take top-k by cosine alignment, regardless of threshold
                top = np.argsort(-cos[m])[:k]
                eligible[m] = False
                eligible[m, top] = True
        # For each seed pick the best-k aligned donors closest in space.
        dists = np.where(eligible, pool_dists, np.inf)
        order = np.argsort(dists, axis=1)[:, :k]
        knn_idx_local = np.take_along_axis(pool_idx, order, axis=1)
        knn_dists = np.take_along_axis(pool_dists, order, axis=1)
    else:
        knn_idx_local = pool_idx[:, :k]
        knn_dists = pool_dists[:, :k]

    weights = 1.0 / np.maximum(knn_dists, eps)
    weights = weights / weights.sum(axis=1, keepdims=True)        # (M, k)
    donor_global = survivor_global_indices[knn_idx_local]          # (M, k)

    # Pull donor tensors
    src_anchor_feat = gaussians._anchor_feat.detach()                       # (N, feat)
    src_scaling = gaussians._scaling.detach()                               # (N, 6)
    src_rotation = gaussians._rotation.detach()                             # (N, 4)
    src_offset = gaussians._offset.detach()                                 # (N, n_offsets, 3)
    n_offsets = src_offset.shape[1]

    weights_t = torch.from_numpy(weights).to(device=device, dtype=dtype)            # (M, k)
    donor_global_t = torch.from_numpy(donor_global).to(device=device, dtype=torch.long)  # (M, k)
    nearest_idx = weights_t.argmax(dim=1)                                           # (M,)

    # Anchor-feat: medoid of the eligible donor pool. The frozen color MLP has
    # seen each donor feature during training, so a single real donor avoids
    # the freckled artifacts of latent averaging.
    donor_pool = torch.unique(donor_global_t.reshape(-1))
    pool_feat = src_anchor_feat[donor_pool]
    mean_feat = pool_feat.mean(dim=0, keepdim=True)
    medoid_idx = torch.argmin(((pool_feat - mean_feat) ** 2).sum(dim=1))
    seed_anchor_feat = pool_feat[medoid_idx:medoid_idx + 1].expand(n_seed, -1).clone()

    # Scaling (log-space, knn-blended), then clamp.
    scale_donors = src_scaling[donor_global_t]                                      # (M, k, 6)
    seed_scaling = (scale_donors * weights_t.unsqueeze(-1)).sum(dim=1)              # (M, 6)
    log_grid = float(np.log(max(grid_spacing, 1e-6)))
    log_lo = log_grid + scale_log_floor_offset
    log_hi = log_grid + scale_log_ceil_offset
    seed_scaling = seed_scaling.clamp(min=log_lo, max=log_hi)

    # Rotation: nearest aligned-donor's quaternion. Renderer uses cov_mlp
    # output, so this is mostly bookkeeping, but downstream tools (mesh export)
    # and any unfrozen cov MLP read this.
    rot_donors = src_rotation[donor_global_t]                                       # (M, k, 4)
    seed_rotation = rot_donors[torch.arange(rot_donors.size(0), device=device), nearest_idx]
    seed_rotation = seed_rotation / seed_rotation.norm(dim=1, keepdim=True).clamp(min=eps)

    # Sheet offsets: deterministic planar mini-patch in per-seed wall tangents.
    if seed_sheet_tangent_u is None or seed_sheet_tangent_v is None:
        tangent_u_np = np.repeat(np.array([[1.0, 0.0, 0.0]], dtype=np.float32), n_seed, axis=0)
        tangent_v_np = np.repeat(np.array([[0.0, 1.0, 0.0]], dtype=np.float32), n_seed, axis=0)
    else:
        tangent_u_np = np.asarray(seed_sheet_tangent_u, dtype=np.float32)
        tangent_v_np = np.asarray(seed_sheet_tangent_v, dtype=np.float32)
        if tangent_u_np.ndim == 1:
            tangent_u_np = np.repeat(tangent_u_np.reshape(1, 3), n_seed, axis=0)
        if tangent_v_np.ndim == 1:
            tangent_v_np = np.repeat(tangent_v_np.reshape(1, 3), n_seed, axis=0)
    if tangent_u_np.shape != (n_seed, 3) or tangent_v_np.shape != (n_seed, 3):
        raise ValueError("seed_sheet_tangent_u/v must be shape (3,) or (n_seed, 3).")
    tangent_u_norm = np.linalg.norm(tangent_u_np, axis=1, keepdims=True)
    tangent_u_np = tangent_u_np / np.maximum(tangent_u_norm, 1e-8)
    tangent_v_np = tangent_v_np - (tangent_v_np * tangent_u_np).sum(axis=1, keepdims=True) * tangent_u_np
    tangent_v_norm = np.linalg.norm(tangent_v_np, axis=1, keepdims=True)
    tangent_v_np = tangent_v_np / np.maximum(tangent_v_norm, 1e-8)

    pattern = [
        (0.0, 0.0), (-1.0, 0.0), (1.0, 0.0), (0.0, -1.0), (0.0, 1.0),
        (-0.7, -0.7), (-0.7, 0.7), (0.7, -0.7), (0.7, 0.7), (0.0, 0.0),
    ]
    if n_offsets != len(pattern):
        angles = np.linspace(0.0, 2.0 * np.pi, max(n_offsets - 1, 1), endpoint=False)
        pattern = [(0.0, 0.0)] + [(float(np.cos(a)), float(np.sin(a))) for a in angles]
        pattern = pattern[:n_offsets]
    coeffs = torch.tensor(pattern, device=device, dtype=dtype)
    tangent_u = torch.from_numpy(tangent_u_np).to(device=device, dtype=dtype)
    tangent_v = torch.from_numpy(tangent_v_np).to(device=device, dtype=dtype)
    world_offsets = (
        coeffs[:, 0].reshape(1, -1, 1) * tangent_u.unsqueeze(1)
        + coeffs[:, 1].reshape(1, -1, 1) * tangent_v.unsqueeze(1)
    ) * float(seed_sheet_radius_factor * grid_spacing)
    scale_xyz = torch.exp(seed_scaling[:, :3]).clamp(min=1e-6).unsqueeze(1)
    seed_offset = world_offsets / scale_xyz

    seed_anchor = torch.from_numpy(seed_xyz).to(device=device, dtype=dtype)         # (M, 3)
    seed_label_ids = torch.full(
        (n_seed, 1), int(object_id), device=device, dtype=gaussians.label_ids.dtype
    )

    originals_snapshot = {
        "_anchor": gaussians._anchor.detach().clone(),
        "_anchor_feat": gaussians._anchor_feat.detach().clone(),
        "_scaling": gaussians._scaling.detach().clone(),
        "_rotation": gaussians._rotation.detach().clone(),
        "_offset": gaussians._offset.detach().clone(),
    }

    new_anchor = torch.cat([gaussians._anchor.detach(), seed_anchor], dim=0)
    new_feat = torch.cat([gaussians._anchor_feat.detach(), seed_anchor_feat], dim=0)
    new_scaling = torch.cat([gaussians._scaling.detach(), seed_scaling], dim=0)
    new_rotation = torch.cat([gaussians._rotation.detach(), seed_rotation], dim=0)
    new_offset = torch.cat([gaussians._offset.detach(), seed_offset], dim=0)
    new_label = torch.cat([gaussians.label_ids, seed_label_ids], dim=0)

    gaussians._anchor = nn.Parameter(new_anchor.requires_grad_(True))
    gaussians._anchor_feat = nn.Parameter(new_feat.requires_grad_(True))
    gaussians._scaling = nn.Parameter(new_scaling.requires_grad_(True))
    gaussians._rotation = nn.Parameter(new_rotation.requires_grad_(False))
    gaussians._offset = nn.Parameter(new_offset.requires_grad_(True))
    gaussians.label_ids = new_label
    gaussians._anchor_mask = torch.ones(new_anchor.shape[0], dtype=torch.bool, device=device)

    n_total = new_anchor.shape[0]
    gaussians.anchor_opacity_accum = torch.zeros((n_total, 1), device=device)
    gaussians.anchor_demon = torch.zeros((n_total, 1), device=device)
    gaussians.offset_opacity_accum = torch.zeros((n_total * n_offsets, 1), device=device)
    gaussians.offset_gradient_accum = torch.zeros((n_total * n_offsets, 1), device=device)
    gaussians.offset_denom = torch.zeros((n_total * n_offsets, 1), device=device)
    gaussians.max_radii2D = torch.zeros(n_total * n_offsets, dtype=torch.float, device=device)

    seeded_mask = torch.zeros(n_total, dtype=torch.bool, device=device)
    seeded_mask[n_orig:] = True

    if seed_opacity_lift != 0.0:
        gaussians.replenishment_seed_opacity_lift = torch.full(
            (n_seed, 1), float(seed_opacity_lift), device=device, dtype=dtype,
        )
    elif hasattr(gaussians, "replenishment_seed_opacity_lift"):
        gaussians.replenishment_seed_opacity_lift = None

    if seed_opacity_gate != 1.0:
        gaussians.replenishment_seed_opacity_gate = torch.full(
            (n_seed, 1), float(seed_opacity_gate), device=device, dtype=dtype,
        )
    elif hasattr(gaussians, "replenishment_seed_opacity_gate"):
        gaussians.replenishment_seed_opacity_gate = None

    if seed_fixed_opacity > 0.0:
        gaussians.replenishment_seed_fixed_opacity = torch.full(
            (n_seed, 1), float(seed_fixed_opacity), device=device, dtype=dtype,
        )
    elif hasattr(gaussians, "replenishment_seed_fixed_opacity"):
        gaussians.replenishment_seed_fixed_opacity = None

    if seed_scaling_boost != 1.0:
        gaussians.replenishment_seed_scaling_boost = float(seed_scaling_boost)
    elif hasattr(gaussians, "replenishment_seed_scaling_boost"):
        gaussians.replenishment_seed_scaling_boost = None

    seed_indices_global = np.arange(n_orig, n_total, dtype=np.int64)
    _stamp_meta(
        gaussians, object_id, bounds_min, bounds_max,
        seeded_mask=seeded_mask, originals_snapshot=originals_snapshot,
    )

    return KNNInitResult(
        n_originals=n_orig,
        n_seeded=n_seed,
        n_total=n_total,
        knn_k=k,
        scale_clip=(log_lo, log_hi),
        seed_indices_global=seed_indices_global,
        seed_opacity_lift=seed_opacity_lift,
        seed_opacity_gate=seed_opacity_gate,
        seed_fixed_opacity=seed_fixed_opacity,
        seed_scaling_boost=seed_scaling_boost,
    )


def find_3d_floater_anchors(
    gaussians,
    object_id: int,
    edge_factor: float = 2.5,
    min_component_size: int = 12,
    knn_k: int = 8,
) -> np.ndarray:
    """Return target-anchor global indices that form small disconnected 3D
    components after seeding. Used as a final Stage E floater pass.
    """
    labels_np = gaussians.label_ids.detach().squeeze(-1).cpu().numpy()
    target_global = np.where(labels_np == int(object_id))[0].astype(np.int64)
    if target_global.size < max(min_component_size + 1, 8):
        return np.zeros(0, dtype=np.int64)
    xyz = gaussians._anchor.detach().cpu().numpy()[target_global]

    k = int(min(knn_k, xyz.shape[0]))
    tree = cKDTree(xyz)
    dists, _ = tree.query(xyz, k=k)
    r_med = float(np.median(dists[:, 1:].mean(axis=1))) if k > 1 else float(np.median(dists))
    if not np.isfinite(r_med) or r_med <= 0.0:
        return np.zeros(0, dtype=np.int64)
    r_link = float(edge_factor * r_med)

    pairs = tree.query_pairs(r=r_link, output_type="ndarray")
    n = xyz.shape[0]
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    if pairs.size == 0:
        return target_global  # no edges → all floaters; let caller decide
    rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
    cols = np.concatenate([pairs[:, 1], pairs[:, 0]])
    data = np.ones(rows.size, dtype=np.uint8)
    graph = csr_matrix((data, (rows, cols)), shape=(n, n))
    n_comp, comp_labels = connected_components(graph, directed=False)
    if n_comp <= 1:
        return np.zeros(0, dtype=np.int64)

    sizes = np.bincount(comp_labels, minlength=n_comp)
    largest = int(np.argmax(sizes))
    drop_local = np.where(
        (comp_labels != largest) & (sizes[comp_labels] < int(min_component_size))
    )[0]
    return target_global[drop_local].astype(np.int64)



def _stamp_meta(
    gaussians,
    object_id: int,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    seeded_mask: torch.Tensor,
    originals_snapshot: dict | None = None,
):
    if not hasattr(gaussians, "_replenishment_aabb") or gaussians._replenishment_aabb is None:
        gaussians._replenishment_aabb = {}
    gaussians._replenishment_aabb[int(object_id)] = (
        torch.tensor(bounds_min, dtype=torch.float32, device="cuda"),
        torch.tensor(bounds_max, dtype=torch.float32, device="cuda"),
    )
    gaussians._replenishment_seeded_mask = seeded_mask
    if originals_snapshot is not None:
        gaussians._replenishment_originals_snapshot = originals_snapshot
    gaussians.n_original_anchors = int(seeded_mask.numel() - int(seeded_mask.sum().item()))
