"""
Stage B (alternative) — Dynamic 6-Sided Directional Shrinkwrap.

Object-agnostic missing-side completion. Replaces the AABB-band volumetric
shrinkwrap with a per-side projected height-field scanner:

    1. Build an oriented frame on the survivor anchors via PCA (with world-axis
       fallback when eigenvalues are ill-conditioned).
     2. For each of the 6 signed frame axes, project survivors onto the
         perpendicular tangent plane and rasterize a 2D wall mask + outer-depth
         map.
    3. Score each side as "missing" using: (a) low camera support from that
       side, (b) sufficient projected support area, (c) low existing coverage
       near the candidate cap.
     4. For sides selected as missing, generate a smooth virtual wall: close
         small cracks in the projected silhouette, preserve large interior voids,
         then push the wall forward to the side's outer survivor depth.
    5. Subtract seeds already covered by existing target anchors.

This is intentionally not a volumetric AABB fill: broad projected holes remain
empty, which prevents seeding deep voids inside concave objects (e.g., the
empty space above couch seats).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.ndimage import binary_closing, binary_fill_holes, binary_opening, gaussian_filter, label
from scipy.spatial import cKDTree


# ───────────────────────────────────────────────────────────────────────────
@dataclass
class SideScanResult:
    side_id: int                       # 0..5
    axis_index: int                    # 0,1,2 (PCA axis)
    sign: int                          # +1 / -1
    normal: np.ndarray                 # (3,) world-space unit vector u_i
    tangent_u: np.ndarray              # (3,) world-space tangent basis 1
    tangent_v: np.ndarray              # (3,) world-space tangent basis 2
    cell_size: float                   # tangent-plane cell edge in world units
    grid_origin_uv: np.ndarray         # (2,) (u_min, v_min)
    grid_shape: tuple                  # (nu, nv)
    occupancy: np.ndarray              # (nu, nv) bool — survivors projected here
    occupancy_filled: np.ndarray       # (nu, nv) bool — after morphological close+fill
    outer_depth: np.ndarray            # (nu, nv) float, depth percentile along +u_i
    camera_support: float              # in [0,1]; 1 = many cameras look from this side
    projected_area_cells: int
    projected_area_frac: float         # cells / max_cells across 6 sides
    existing_coverage_frac: float
    uncovered_frac: float
    n_candidate_cells: int
    selected: bool
    selection_score: float
    n_seeds: int

    def to_dict(self) -> dict:
        return {
            "side_id": int(self.side_id),
            "axis_index": int(self.axis_index),
            "sign": int(self.sign),
            "normal": self.normal.tolist(),
            "cell_size": float(self.cell_size),
            "grid_shape": [int(x) for x in self.grid_shape],
            "n_occupied": int(self.occupancy.sum()),
            "n_occupied_filled": int(self.occupancy_filled.sum()),
            "camera_support": float(self.camera_support),
            "projected_area_cells": int(self.projected_area_cells),
            "projected_area_frac": float(self.projected_area_frac),
            "existing_coverage_frac": float(self.existing_coverage_frac),
            "uncovered_frac": float(self.uncovered_frac),
            "n_candidate_cells": int(self.n_candidate_cells),
            "selected": bool(self.selected),
            "selection_score": float(self.selection_score),
            "n_seeds": int(self.n_seeds),
        }


@dataclass
class DirectionalShrinkwrapResult:
    seed_xyz: np.ndarray
    seed_tangent_u: np.ndarray
    seed_tangent_v: np.ndarray
    object_center: np.ndarray
    pca_axes: np.ndarray               # (3,3) rows are unit axes
    pca_eigvals: np.ndarray            # (3,)
    pca_used_world_fallback: bool
    cell_size: float
    sides: list                        # list[SideScanResult]
    selected_side_ids: list            # list[int]
    n_seeds_total: int

    def to_dict(self) -> dict:
        return {
            "object_center": self.object_center.tolist(),
            "pca_axes": self.pca_axes.tolist(),
            "pca_eigvals": self.pca_eigvals.tolist(),
            "pca_used_world_fallback": bool(self.pca_used_world_fallback),
            "cell_size": float(self.cell_size),
            "selected_side_ids": [int(i) for i in self.selected_side_ids],
            "n_seeds_total": int(self.n_seeds_total),
            "sides": [s.to_dict() for s in self.sides],
        }


# ───────────────────────────────────────────────────────────────────────────
def _pca_frame(
    points: np.ndarray,
    eigval_ratio_threshold: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Return (center, axes(3,3 row-major unit), eigvals, used_world_fallback).

    Axes are sorted descending by variance. Falls back to world axes when the
    smallest eigenvalue is far smaller than the largest (degenerate / planar).
    """
    center = points.mean(axis=0).astype(np.float32)
    centered = points - center
    cov = np.cov(centered.T) + np.eye(3, dtype=np.float64) * 1e-12
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order].astype(np.float32)
    eigvecs = eigvecs[:, order]
    # axes[i] = i-th principal axis as a row vector
    axes = eigvecs.T.astype(np.float32)
    used_fallback = False
    if eigvals.max() <= 0.0 or (eigvals.min() / max(eigvals.max(), 1e-12)) < eigval_ratio_threshold:
        # Degenerate distribution → fall back to world axes.
        axes = np.eye(3, dtype=np.float32)
        used_fallback = True
    # Force right-handed
    if np.dot(np.cross(axes[0], axes[1]), axes[2]) < 0:
        axes[2] = -axes[2]
    return center, axes, eigvals, used_fallback


def _fill_small_holes(mask: np.ndarray, max_hole_cells: int) -> np.ndarray:
    filled = binary_fill_holes(mask)
    holes = filled & ~mask
    if not holes.any():
        return mask
    hole_labels, n_labels = label(holes)
    out = mask.copy()
    for comp_id in range(1, n_labels + 1):
        comp = hole_labels == comp_id
        if int(comp.sum()) <= max_hole_cells:
            out[comp] = True
    return out


def _build_wall_mask(
    occupancy: np.ndarray,
    close_iters: int,
    max_hole_area_frac: float = 0.003,
    max_hole_cells: int = 64,
) -> np.ndarray:
    """Close tiny silhouette cracks while preserving broad couch-seat voids."""
    wall = occupancy.copy()
    if close_iters > 0:
        struct = np.ones((3, 3), dtype=bool)
        wall = binary_closing(wall, structure=struct, iterations=close_iters)
    hole_limit = min(max_hole_cells, max(8, int(wall.size * max_hole_area_frac)))
    return _fill_small_holes(wall, hole_limit)


def _rasterize_side(
    survivor_xyz: np.ndarray,
    center: np.ndarray,
    normal: np.ndarray,
    tangent_u: np.ndarray,
    tangent_v: np.ndarray,
    cell_size: float,
    depth_percentile: float = 95.0,
    close_iters: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple]:
    """Return (occupancy, occupancy_filled, outer_depth, origin_uv, shape).

    outer_depth at cell (i,j) is the requested depth percentile of survivors
    falling in that cell, measured along +normal. Cells without survivors get
    ``-inf`` depth (and are not seeded).
    """
    rel = survivor_xyz - center.reshape(1, 3)
    u = rel @ tangent_u
    v = rel @ tangent_v
    d = rel @ normal

    u_min, u_max = float(u.min()), float(u.max())
    v_min, v_max = float(v.min()), float(v.max())
    pad = 0.5 * cell_size
    u_min -= pad; v_min -= pad; u_max += pad; v_max += pad
    nu = max(1, int(np.ceil((u_max - u_min) / cell_size)))
    nv = max(1, int(np.ceil((v_max - v_min) / cell_size)))

    iu = np.clip(((u - u_min) / cell_size).astype(np.int64), 0, nu - 1)
    iv = np.clip(((v - v_min) / cell_size).astype(np.int64), 0, nv - 1)

    occupancy = np.zeros((nu, nv), dtype=bool)
    occupancy[iu, iv] = True

    outer_depth = np.full((nu, nv), -np.inf, dtype=np.float32)
    # For each cell collect survivors and take percentile of d.
    flat_idx = iu * nv + iv
    order = np.argsort(flat_idx, kind="stable")
    flat_sorted = flat_idx[order]
    d_sorted = d[order]
    # Find run boundaries.
    if flat_sorted.size:
        boundaries = np.concatenate(([0], np.where(np.diff(flat_sorted) != 0)[0] + 1, [flat_sorted.size]))
        for k in range(len(boundaries) - 1):
            s, e = boundaries[k], boundaries[k + 1]
            cell = flat_sorted[s]
            ci, cj = divmod(cell, nv)
            outer_depth[ci, cj] = float(np.percentile(d_sorted[s:e], depth_percentile))

    occupancy_filled = _build_wall_mask(occupancy, close_iters=close_iters)

    return occupancy, occupancy_filled, outer_depth, np.array([u_min, v_min], dtype=np.float32), (nu, nv)


def _camera_support(
    cam_centers: np.ndarray | None,
    object_center: np.ndarray,
    normal: np.ndarray,
) -> float:
    """Fraction of cameras viewing the object from the +normal half-space."""
    if cam_centers is None or len(cam_centers) == 0:
        return 0.5
    dirs = np.asarray(cam_centers, dtype=np.float32) - object_center.reshape(1, 3)
    norms = np.linalg.norm(dirs, axis=1)
    mask = norms > 1e-6
    if not mask.any():
        return 0.5
    dirs = dirs[mask] / norms[mask].reshape(-1, 1)
    return float((dirs @ normal > 0.0).mean())


def _filled_depth_map(
    side: SideScanResult,
    smooth_iters: int,
    max_fill_cells: float = 1.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Local-fill mode: occupied cells + cells within ``max_fill_cells`` of
    an occupied cell. Prevents bridging wide concave gaps.
    """
    depth = side.outer_depth.copy()
    depth_filled = depth.copy()
    occ_idx = np.argwhere(side.occupancy)
    if occ_idx.size == 0:
        empty_mask = np.zeros_like(side.occupancy, dtype=bool)
        return depth_filled, empty_mask

    fill_mask = side.occupancy_filled & ~side.occupancy
    if fill_mask.any():
        tree = cKDTree(occ_idx.astype(np.float32))
        fill_idx = np.argwhere(fill_mask)
        dists, nn = tree.query(fill_idx.astype(np.float32), k=1)
        within = dists <= float(max_fill_cells)
        if within.any():
            src = occ_idx[nn[within]]
            tgt = fill_idx[within]
            depth_filled[tgt[:, 0], tgt[:, 1]] = depth[src[:, 0], src[:, 1]]
        # Cells beyond the local-fill radius are dropped so the mask stops
        # at small cracks and never bridges large concave gaps.
        if (~within).any():
            far = fill_idx[~within]
            close_mask = side.occupancy_filled.copy()
            close_mask[far[:, 0], far[:, 1]] = False
            mask = close_mask & np.isfinite(depth_filled)
        else:
            mask = side.occupancy_filled & np.isfinite(depth_filled)
    else:
        mask = side.occupancy & np.isfinite(depth_filled)

    if smooth_iters > 0 and mask.any():
        sigma = max(0.4, float(smooth_iters) * 0.75)
        weights = mask.astype(np.float32)
        values = np.where(mask, depth_filled, 0.0).astype(np.float32)
        smooth_values = gaussian_filter(values, sigma=sigma, mode="nearest")
        smooth_weights = gaussian_filter(weights, sigma=sigma, mode="nearest")
        smooth = smooth_values / np.maximum(smooth_weights, 1e-6)
        depth_filled[mask] = smooth[mask]

    return depth_filled, mask


def _smooth_depth_on_mask(
    depth_filled: np.ndarray,
    mask: np.ndarray,
    smooth_iters: int,
) -> np.ndarray:
    if smooth_iters <= 0 or not mask.any():
        return depth_filled
    sigma = max(0.4, float(smooth_iters) * 0.75)
    weights = mask.astype(np.float32)
    values = np.where(mask, depth_filled, 0.0).astype(np.float32)
    smooth_values = gaussian_filter(values, sigma=sigma, mode="nearest")
    smooth_weights = gaussian_filter(weights, sigma=sigma, mode="nearest")
    smooth = smooth_values / np.maximum(smooth_weights, 1e-6)
    out = depth_filled.copy()
    out[mask] = smooth[mask]
    return out


def _silhouette_mask(occupancy: np.ndarray) -> np.ndarray:
    """Row+column convex closure of a 2D occupancy grid.

    For each row, mark every cell between the leftmost and rightmost True;
    same per column. Their AND is a non-strict convex hull of the silhouette
    that fills missing-back-side gaps without bridging spurious holes that
    are open on at least one axis.
    """
    if not occupancy.any():
        return occupancy.copy()
    nu, nv = occupancy.shape
    row_fill = np.zeros_like(occupancy)
    col_fill = np.zeros_like(occupancy)
    rows_any = occupancy.any(axis=1)
    cols_any = occupancy.any(axis=0)
    if rows_any.any():
        for i in np.where(rows_any)[0]:
            cols = np.where(occupancy[i])[0]
            row_fill[i, cols.min():cols.max() + 1] = True
    if cols_any.any():
        for j in np.where(cols_any)[0]:
            rows = np.where(occupancy[:, j])[0]
            col_fill[rows.min():rows.max() + 1, j] = True
    return row_fill & col_fill


def _silhouette_depth_map(
    side: SideScanResult,
    smooth_iters: int,
    far_percentile: float = 90.0,
    seam_blend_cells: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Silhouette-fill mode: extend mask to row/col convex closure of the
    side's occupancy. Cells inside the silhouette but without survivors get
    a single far-extent depth (``far_percentile`` of the occupied depths).
    Used to seed missing back/front faces with no survivors of their own.
    """
    depth = side.outer_depth.copy()
    if not side.occupancy.any():
        empty = np.zeros_like(side.occupancy, dtype=bool)
        return depth, empty

    mask_full = _silhouette_mask(side.occupancy)
    far_depth = float(np.percentile(depth[side.occupancy], far_percentile))

    depth_filled = depth.copy()
    fill_only = mask_full & ~side.occupancy
    if fill_only.any():
        fill_idx = np.argwhere(fill_only)
        occ_idx = np.argwhere(side.occupancy)
        if occ_idx.size:
            tree = cKDTree(occ_idx.astype(np.float32))
            d_grid, nn = tree.query(fill_idx.astype(np.float32), k=1)
            src = occ_idx[nn]
            near = d_grid <= float(seam_blend_cells)
            if near.any():
                depth_filled[fill_idx[near, 0], fill_idx[near, 1]] = depth[src[near, 0], src[near, 1]]
            if (~near).any():
                far_idx = fill_idx[~near]
                depth_filled[far_idx[:, 0], far_idx[:, 1]] = far_depth
        else:
            depth_filled[fill_only] = far_depth

    if smooth_iters > 0 and mask_full.any():
        sigma = max(0.4, float(smooth_iters) * 0.75)
        weights = mask_full.astype(np.float32)
        values = np.where(mask_full, depth_filled, 0.0).astype(np.float32)
        sv = gaussian_filter(values, sigma=sigma, mode="nearest")
        sw = gaussian_filter(weights, sigma=sigma, mode="nearest")
        smooth = sv / np.maximum(sw, 1e-6)
        depth_filled[mask_full] = smooth[mask_full]

    return depth_filled, mask_full


def _remove_small_mask_components(mask: np.ndarray, min_cells: int) -> np.ndarray:
    if not mask.any() or int(min_cells) <= 1:
        return mask
    comp, n_comp = label(mask)
    if n_comp <= 1:
        return mask
    sizes = np.bincount(comp.reshape(-1), minlength=n_comp + 1)
    keep_ids = np.where(sizes >= int(min_cells))[0]
    keep_ids = keep_ids[keep_ids != 0]
    if keep_ids.size == 0:
        keep_ids = np.array([int(np.argmax(sizes[1:]) + 1)], dtype=np.int64)
    return np.isin(comp, keep_ids)


def _cleanup_seed_mask_edges(
    mask: np.ndarray,
    min_component_cells: int,
    opening_iters: int,
) -> np.ndarray:
    if not mask.any():
        return mask
    cleaned = mask.copy()
    if opening_iters > 0:
        struct = np.ones((3, 3), dtype=bool)
        cleaned = binary_opening(cleaned, structure=struct, iterations=int(opening_iters))
        if not cleaned.any():
            cleaned = mask.copy()
    cleaned = _remove_small_mask_components(cleaned, int(min_component_cells))
    return cleaned


def _align_vertical_side_to_base_line(
    side: SideScanResult,
    center: np.ndarray,
    depth_filled: np.ndarray,
    seed_mask: np.ndarray,
    scene_up_unit: np.ndarray,
    base_signed: float,
    max_extend_cells: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Extend a vertical side sheet down to a shared object-base line.

    This is deliberately different from floor filling: it only grows the
    front/back/side wall masks along their vertical tangent axis, so it creates
    vertical couch-side surface instead of a horizontal slab under the couch.
    """
    if not seed_mask.any():
        return depth_filled, seed_mask

    n_up = float(np.dot(side.normal, scene_up_unit))
    if abs(n_up) > 0.35:
        return depth_filled, seed_mask

    tu_up = float(np.dot(side.tangent_u, scene_up_unit))
    tv_up = float(np.dot(side.tangent_v, scene_up_unit))
    if max(abs(tu_up), abs(tv_up)) < 0.35:
        return depth_filled, seed_mask

    vertical_axis = 0 if abs(tu_up) >= abs(tv_up) else 1
    vertical_up = tu_up if vertical_axis == 0 else tv_up
    other_up = tv_up if vertical_axis == 0 else tu_up
    if abs(vertical_up) < 1e-6:
        return depth_filled, seed_mask

    nu, nv = seed_mask.shape
    out_mask = seed_mask.copy()
    out_depth = depth_filled.copy()
    center_h = float(np.dot(center, scene_up_unit))
    max_cells = max(0, int(round(float(max_extend_cells))))
    if max_cells == 0:
        return out_depth, out_mask

    if vertical_axis == 0:
        for j in np.where(seed_mask.any(axis=0))[0]:
            rows = np.where(seed_mask[:, j])[0]
            if rows.size == 0:
                continue
            v_coord = side.grid_origin_uv[1] + (float(j) + 0.5) * side.cell_size
            u_base = (float(base_signed) - center_h - v_coord * other_up) / vertical_up
            base_i = int(np.floor((u_base - side.grid_origin_uv[0]) / side.cell_size))
            base_i = int(np.clip(base_i, 0, nu - 1))
            edge_i = int(rows.min() if vertical_up > 0.0 else rows.max())
            if abs(base_i - edge_i) > max_cells:
                base_i = edge_i - max_cells if base_i < edge_i else edge_i + max_cells
                base_i = int(np.clip(base_i, 0, nu - 1))
            lo, hi = sorted((base_i, edge_i))
            if lo == hi:
                continue
            out_mask[lo:hi + 1, j] = True
            out_depth[lo:hi + 1, j] = depth_filled[edge_i, j]
    else:
        for i in np.where(seed_mask.any(axis=1))[0]:
            cols = np.where(seed_mask[i, :])[0]
            if cols.size == 0:
                continue
            u_coord = side.grid_origin_uv[0] + (float(i) + 0.5) * side.cell_size
            v_base = (float(base_signed) - center_h - u_coord * other_up) / vertical_up
            base_j = int(np.floor((v_base - side.grid_origin_uv[1]) / side.cell_size))
            base_j = int(np.clip(base_j, 0, nv - 1))
            edge_j = int(cols.min() if vertical_up > 0.0 else cols.max())
            if abs(base_j - edge_j) > max_cells:
                base_j = edge_j - max_cells if base_j < edge_j else edge_j + max_cells
                base_j = int(np.clip(base_j, 0, nv - 1))
            lo, hi = sorted((base_j, edge_j))
            if lo == hi:
                continue
            out_mask[i, lo:hi + 1] = True
            out_depth[i, lo:hi + 1] = depth_filled[i, edge_j]

    return out_depth, out_mask


def _candidate_seeds_for_side(
    side: SideScanResult,
    center: np.ndarray,
    depth_filled: np.ndarray,
    mask: np.ndarray,
    cap_offset: float,
    samples_per_cell: int,
) -> np.ndarray:
    if not mask.any():
        return np.zeros((0, 3), dtype=np.float32)
    ii, jj = np.where(mask)
    samples = max(1, int(samples_per_cell))
    offsets = (np.arange(samples, dtype=np.float32) + 0.5) / float(samples)
    du, dv = np.meshgrid(offsets, offsets, indexing="ij")
    du = du.reshape(-1)
    dv = dv.reshape(-1)

    all_chunks = []
    d_base = depth_filled[ii, jj] + cap_offset
    for ou, ov in zip(du, dv):
        u_coords = side.grid_origin_uv[0] + (ii.astype(np.float32) + ou) * side.cell_size
        v_coords = side.grid_origin_uv[1] + (jj.astype(np.float32) + ov) * side.cell_size
        seeds_local = (
            center.reshape(1, 3)
            + u_coords.reshape(-1, 1) * side.tangent_u.reshape(1, 3)
            + v_coords.reshape(-1, 1) * side.tangent_v.reshape(1, 3)
            + d_base.reshape(-1, 1) * side.normal.reshape(1, 3)
        ).astype(np.float32)
        all_chunks.append(seeds_local)
    return np.concatenate(all_chunks, axis=0) if all_chunks else np.zeros((0, 3), dtype=np.float32)


# ───────────────────────────────────────────────────────────────────────────
def build_directional_seeds(
    survivor_xyz: np.ndarray,
    existing_object_xyz: np.ndarray,
    r_med: float,
    cam_centers: Optional[np.ndarray] = None,
    cell_size_factor: float = 1.0,
    depth_percentile: float = 95.0,
    cap_offset_factor: float = 0.0,
    morphological_close_iters: int = 1,
    depth_smooth_iters: int = 2,
    samples_per_cell: int = 3,
    min_camera_support: float = 0.15,
    min_projected_area_frac: float = 0.10,
    min_uncovered_frac: float = 0.12,
    existing_coverage_alpha: float = 0.6,
    max_fill_cells: float = 1.5,
    scene_xyz: Optional[np.ndarray] = None,
    scene_up: Optional[np.ndarray] = None,
    floor_clearance_factor: float = 0.0,
    floor_percentile: float = 1.5,
    extend_to_floor: bool = False,
    align_side_bottoms: bool = True,
    side_base_percentile: float = 2.0,
    side_bottom_max_extend_factor: float = 8.0,
    seed_bottom_cap: bool = False,
    mask_min_component_cells: int = 9,
    mask_edge_open_iters: int = 1,
    seam_blend_cells: float = 2.0,
    bottom_alignment_threshold: float = 0.5,
    silhouette_camera_support_max: float = 0.25,
    silhouette_far_percentile: float = 90.0,
    extend_to_walls: bool = True,
    wall_search_factor: float = 8.0,
) -> DirectionalShrinkwrapResult:
    """Build cap seed_xyz on dynamically selected missing sides.

    Args:
        survivor_xyz: (N_s, 3) Stage A surface anchors.
        existing_object_xyz: (N_o, 3) all current target-object anchors.
        r_med: median kNN distance from Stage A.
        cam_centers: world-space camera centers (Cx3) or None.
        cell_size_factor: tangent-plane cell edge = factor * r_med.
        depth_percentile: outer survivor depth percentile along +normal used
            as the cap surface depth (95 ≈ outermost shell, robust to noise).
        cap_offset_factor: cap is placed (factor * r_med) outside outer_depth.
        depth_smooth_iters: smooth the per-side outer-depth sheet before seeding.
        samples_per_cell: deterministic sub-cell samples per tangent cell axis.
        min_camera_support: sides with camera_support BELOW this can be flagged
            as missing.
        min_projected_area_frac: a candidate side's projected support area must
            be at least this fraction of the largest side's area.
        min_uncovered_frac: also seed camera-supported sides when the projected
            shrinkwrap sheet has enough uncovered cells after existing-coverage
            testing. This turns the stage into a real shrinkwrap instead of
            only a hidden-side cap.
        existing_coverage_alpha: drop seeds within (alpha * cell_size) of any
            existing anchor.

    Side selection is automatic and uncapped: every side that satisfies the
    camera-support / uncovered / area thresholds contributes seeds.
    """
    survivor_xyz = np.asarray(survivor_xyz, dtype=np.float32)
    existing_xyz = np.asarray(existing_object_xyz, dtype=np.float32)

    if survivor_xyz.shape[0] < 4:
        return DirectionalShrinkwrapResult(
            seed_xyz=np.zeros((0, 3), dtype=np.float32),
            seed_tangent_u=np.zeros((0, 3), dtype=np.float32),
            seed_tangent_v=np.zeros((0, 3), dtype=np.float32),
            object_center=np.zeros(3, dtype=np.float32),
            pca_axes=np.eye(3, dtype=np.float32),
            pca_eigvals=np.zeros(3, dtype=np.float32),
            pca_used_world_fallback=True,
            cell_size=0.0, sides=[], selected_side_ids=[], n_seeds_total=0,
        )

    cell_size = max(float(cell_size_factor * r_med), 1e-5)
    cap_offset = max(float(cap_offset_factor * r_med), 0.0)

    # ---- Scene/object base references -------------------------------------
    # The room floor is only used as a lower safety clamp or for explicitly
    # requested legged-object floor filling. The default extension target is
    # the object's own lower silhouette line, so front/back/sides land on one
    # clean base without creating a horizontal floor slab.
    floor_signed: Optional[float] = None
    object_base_signed: Optional[float] = None
    scene_up_unit: Optional[np.ndarray] = None
    if scene_xyz is not None and scene_up is not None:
        scene_xyz_arr = np.asarray(scene_xyz, dtype=np.float32)
        up_arr = np.asarray(scene_up, dtype=np.float32).reshape(3)
        up_norm = float(np.linalg.norm(up_arr))
        if scene_xyz_arr.shape[0] >= 16 and up_norm > 1e-6:
            scene_up_unit = up_arr / up_norm
            heights = scene_xyz_arr @ scene_up_unit
            floor_signed = float(np.percentile(heights, float(floor_percentile)))
            object_heights = survivor_xyz @ scene_up_unit
            object_base_signed = float(np.percentile(object_heights, float(side_base_percentile)))
    floor_clearance = float(floor_clearance_factor) * r_med

    center, axes, eigvals, used_fallback = _pca_frame(survivor_xyz)

    sides: list[SideScanResult] = []
    side_id = 0
    for axis_idx in range(3):
        for sign in (+1, -1):
            normal = (sign * axes[axis_idx]).astype(np.float32)
            # Pick two stable in-plane tangents from the other axes.
            other_idx = [i for i in range(3) if i != axis_idx]
            tu = axes[other_idx[0]].astype(np.float32)
            tv = axes[other_idx[1]].astype(np.float32)
            occ, occ_f, depth, origin_uv, shape = _rasterize_side(
                survivor_xyz, center, normal, tu, tv,
                cell_size=cell_size,
                depth_percentile=depth_percentile,
                close_iters=morphological_close_iters,
            )
            cs = _camera_support(cam_centers, center, normal)
            sides.append(SideScanResult(
                side_id=side_id, axis_index=axis_idx, sign=sign,
                normal=normal, tangent_u=tu, tangent_v=tv,
                cell_size=cell_size, grid_origin_uv=origin_uv, grid_shape=shape,
                occupancy=occ, occupancy_filled=occ_f, outer_depth=depth,
                camera_support=cs,
                projected_area_cells=int(occ.sum()),
                projected_area_frac=0.0,
                existing_coverage_frac=0.0,
                uncovered_frac=1.0,
                n_candidate_cells=0,
                selected=False, selection_score=0.0, n_seeds=0,
            ))
            side_id += 1

    # Normalize projected_area against the largest side.
    max_area = max(int(s.projected_area_cells) for s in sides) or 1
    for s in sides:
        s.projected_area_frac = float(s.projected_area_cells) / float(max_area)

    # Identify the bottom-pointing side (normal most aligned with -scene_up).
    # Its outer-depth field is what gets extended down to the floor.
    bottom_side_id: Optional[int] = None
    if scene_up_unit is not None:
        # Larger negative dot = more downward.
        align = [(s.side_id, float(np.dot(s.normal, scene_up_unit))) for s in sides]
        sid, dotv = min(align, key=lambda kv: kv[1])
        if dotv < -float(bottom_alignment_threshold):
            bottom_side_id = sid

    side_candidates: dict[int, np.ndarray] = {}
    for s in sides:
        # Use silhouette mode on hidden faces (no/low camera support): they
        # rarely have survivor coverage and need the row/col closure to
        # synthesize a back/under cap. Use local-fill mode on visible faces
        # to preserve concave grooves on the front/cushions.
        use_silhouette = float(s.camera_support) <= float(silhouette_camera_support_max)
        if use_silhouette:
            depth_filled, seed_mask = _silhouette_depth_map(
                s, smooth_iters=depth_smooth_iters,
                far_percentile=silhouette_far_percentile,
                seam_blend_cells=seam_blend_cells,
            )
        else:
            depth_filled, seed_mask = _filled_depth_map(
                s, smooth_iters=depth_smooth_iters, max_fill_cells=max_fill_cells,
            )

        # Wall extension (room-aware): for hidden sides, find the nearest
        # non-target anchor along +normal from the survivors' outer depth
        # and treat that as a wall stop. Push hull-only depths up to the
        # wall (or, if none found, leave them at far_percentile).
        if (
            extend_to_walls
            and use_silhouette
            and scene_xyz is not None
            and seed_mask.any()
            and s.occupancy.any()
        ):
            scene_arr = np.asarray(scene_xyz, dtype=np.float32)
            if scene_arr.shape[0] > 0:
                rel = scene_arr - center.reshape(1, 3)
                d_along = rel @ s.normal
                far_extent = float(np.max(s.outer_depth[s.occupancy])) if s.occupancy.any() else 0.0
                search_max = far_extent + float(wall_search_factor) * cell_size
                near = (d_along > far_extent + 0.25 * cell_size) & (d_along < search_max)
                if near.any():
                    wall_depth = float(np.percentile(d_along[near], 5.0)) - 0.5 * cell_size
                    fill_only = seed_mask & ~s.occupancy
                    if fill_only.any():
                        depth_filled = depth_filled.copy()
                        depth_filled[fill_only] = np.minimum(
                            depth_filled[fill_only], wall_depth
                        )

        if (
            align_side_bottoms
            and object_base_signed is not None
            and scene_up_unit is not None
            and seed_mask.any()
        ):
            depth_filled, seed_mask = _align_vertical_side_to_base_line(
                s,
                center,
                depth_filled,
                seed_mask,
                scene_up_unit,
                object_base_signed,
                max_extend_cells=float(side_bottom_max_extend_factor) * cell_size / max(cell_size, 1e-8),
            )

        # Floor extension: for the down-pointing side, push every occupied
        # column's cap depth out to the floor plane (along this side's
        # normal) when the survivor outer depth doesn't already reach it.
        # This grounds couch legs/base instead of sealing them mid-air.
        if (
            extend_to_floor
            and bottom_side_id is not None
            and s.side_id == bottom_side_id
            and floor_signed is not None
            and scene_up_unit is not None
        ):
            n_dot_up = float(np.dot(s.normal, scene_up_unit))
            if n_dot_up < -1e-3:
                # depth d along +normal where (center + d*n) @ up == floor_signed
                floor_depth = (floor_signed - float(np.dot(center, scene_up_unit))) / n_dot_up
                # clearance: stop just short of the floor by clearance distance.
                # Moving 'clearance' along +up corresponds to (-clearance/n_dot_up) along +normal.
                if floor_clearance > 0.0:
                    floor_depth -= floor_clearance / abs(n_dot_up)
                if seed_mask.any():
                    extend_mask = seed_mask & (depth_filled < floor_depth)
                    if extend_mask.any():
                        depth_filled = depth_filled.copy()
                        depth_filled[extend_mask] = floor_depth

        if seed_mask.any():
            cleaned_mask = _cleanup_seed_mask_edges(
                seed_mask,
                min_component_cells=int(mask_min_component_cells),
                opening_iters=int(mask_edge_open_iters),
            )
            if not np.array_equal(cleaned_mask, seed_mask):
                seed_mask = cleaned_mask
                depth_filled = np.where(seed_mask, depth_filled, -np.inf).astype(np.float32)
            depth_filled = _smooth_depth_on_mask(
                depth_filled,
                seed_mask,
                smooth_iters=max(1, int(depth_smooth_iters)),
            )
        s.n_candidate_cells = int(seed_mask.sum())
        if s.n_candidate_cells > 0:
            # Coverage is measured in projected sheet cells, not at the final
            # offset cap position. Otherwise every visible surface appears
            # uncovered simply because the candidate sheet is pushed outward.
            s.existing_coverage_frac = float(s.occupancy.sum()) / float(s.n_candidate_cells)
            s.uncovered_frac = float(np.clip(1.0 - s.existing_coverage_frac, 0.0, 1.0))
        else:
            s.existing_coverage_frac = 0.0
            s.uncovered_frac = 1.0

        # Score: higher = more shrinkwrap-worthy. Low camera support finds
        # hidden sides; high uncovered fraction catches visible-but-incomplete
        # front/side surfaces without hardcoding object semantics.
        support_term = float(np.clip(1.0 - s.camera_support, 0.0, 1.0))
        gap_term = float(np.clip(s.uncovered_frac, 0.0, 1.0))
        area_term = float(s.projected_area_frac)
        s.selection_score = max(support_term, gap_term) * area_term
        side_candidates[s.side_id] = _candidate_seeds_for_side(
            s, center, depth_filled, seed_mask,
            cap_offset=cap_offset,
            samples_per_cell=samples_per_cell,
        )

    # Selection: auto + unlimited. A side is eligible when it has either low
    # camera support OR a meaningful uncovered shrinkwrap sheet, and its
    # projected footprint is non-trivial.
    eligible = [
        s for s in sides
        if ((s.camera_support < min_camera_support) or (s.uncovered_frac >= min_uncovered_frac))
        and (s.projected_area_frac >= min_projected_area_frac)
        and (seed_bottom_cap or bottom_side_id is None or s.side_id != bottom_side_id)
    ]
    eligible.sort(key=lambda s: s.selection_score, reverse=True)
    selected = [s.side_id for s in eligible]
    selected_set = set(selected)

    # Generate cap seeds for selected sides.
    all_seeds = []
    all_seed_tangent_u = []
    all_seed_tangent_v = []
    for s in sides:
        if s.side_id not in selected_set:
            continue
        seeds_local = side_candidates.get(s.side_id, np.zeros((0, 3), dtype=np.float32))
        if seeds_local.shape[0] == 0:
            continue

        s.n_seeds = int(seeds_local.shape[0])
        s.selected = True
        all_seeds.append(seeds_local)
        all_seed_tangent_u.append(np.repeat(s.tangent_u.reshape(1, 3), seeds_local.shape[0], axis=0))
        all_seed_tangent_v.append(np.repeat(s.tangent_v.reshape(1, 3), seeds_local.shape[0], axis=0))

    if all_seeds:
        seed_xyz = np.concatenate(all_seeds, axis=0)
        seed_tangent_u = np.concatenate(all_seed_tangent_u, axis=0).astype(np.float32)
        seed_tangent_v = np.concatenate(all_seed_tangent_v, axis=0).astype(np.float32)
    else:
        seed_xyz = np.zeros((0, 3), dtype=np.float32)
        seed_tangent_u = np.zeros((0, 3), dtype=np.float32)
        seed_tangent_v = np.zeros((0, 3), dtype=np.float32)

    # Existing-coverage subtraction. Directional wall seeds are deliberately
    # placed just outside survivor geometry; using the legacy large coverage
    # radius punches holes around armrests/back/base and leaves a dotted rim.
    # Keep only exact near-duplicates out, so the pushed wall can meet the
    # current surface continuously.
    if existing_xyz.shape[0] > 0 and seed_xyz.shape[0] > 0:
        cov_r = float(min(existing_coverage_alpha, 0.18) * cell_size)
        tree = cKDTree(existing_xyz)
        d_exist, _ = tree.query(seed_xyz, k=1, distance_upper_bound=cov_r)
        keep = ~(np.isfinite(d_exist) & (d_exist <= cov_r))
        seed_xyz = seed_xyz[keep]
        seed_tangent_u = seed_tangent_u[keep]
        seed_tangent_v = seed_tangent_v[keep]
        # Recount per side approximately by re-thresholding (best-effort).
        # We won't re-attribute to sides; n_seeds above is pre-subtraction.

    # Floor clamp: drop any seed that would land below the room floor.
    if (
        floor_signed is not None
        and scene_up_unit is not None
        and seed_xyz.shape[0] > 0
    ):
        seed_heights = seed_xyz @ scene_up_unit
        keep_floor = seed_heights >= (floor_signed + floor_clearance)
        if not keep_floor.all():
            seed_xyz = seed_xyz[keep_floor]
            seed_tangent_u = seed_tangent_u[keep_floor]
            seed_tangent_v = seed_tangent_v[keep_floor]

    return DirectionalShrinkwrapResult(
        seed_xyz=seed_xyz.astype(np.float32),
        seed_tangent_u=seed_tangent_u.astype(np.float32),
        seed_tangent_v=seed_tangent_v.astype(np.float32),
        object_center=center.astype(np.float32),
        pca_axes=axes.astype(np.float32),
        pca_eigvals=eigvals.astype(np.float32),
        pca_used_world_fallback=bool(used_fallback),
        cell_size=cell_size,
        sides=sides,
        selected_side_ids=selected,
        n_seeds_total=int(seed_xyz.shape[0]),
    )


# ───────────────────────────────────────────────────────────────────────────
def save_side_scan_pngs(result: DirectionalShrinkwrapResult, out_dir):
    """Write per-side PNGs: occupancy + outer_depth heatmap + selected flag.

    Lightweight, uses cv2 if available; otherwise no-op.
    """
    try:
        import cv2  # noqa: WPS433
    except Exception:
        return
    from pathlib import Path
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for s in result.sides:
        nu, nv = s.grid_shape
        if nu < 2 or nv < 2:
            continue
        occ = s.occupancy.astype(np.uint8) * 255
        occ_f = s.occupancy_filled.astype(np.uint8) * 255
        depth = s.outer_depth.copy()
        finite = np.isfinite(depth)
        if finite.any():
            dmin = float(depth[finite].min()); dmax = float(depth[finite].max())
            d_norm = np.zeros_like(depth, dtype=np.float32)
            if dmax - dmin > 1e-12:
                d_norm[finite] = (depth[finite] - dmin) / (dmax - dmin)
            depth_img = (d_norm * 255).astype(np.uint8)
        else:
            depth_img = np.zeros((nu, nv), dtype=np.uint8)
        depth_color = cv2.applyColorMap(depth_img, cv2.COLORMAP_VIRIDIS)
        # Tile horizontally: occ | occ_filled | depth.
        occ_rgb = cv2.cvtColor(occ, cv2.COLOR_GRAY2BGR)
        occ_f_rgb = cv2.cvtColor(occ_f, cv2.COLOR_GRAY2BGR)
        tile = np.hstack([occ_rgb, occ_f_rgb, depth_color])
        # Upscale for visibility.
        scale = max(1, 256 // max(nu, nv))
        if scale > 1:
            tile = cv2.resize(tile, (tile.shape[1] * scale, tile.shape[0] * scale),
                              interpolation=cv2.INTER_NEAREST)
        label = (
            f"side={s.side_id} axis={s.axis_index} sign={s.sign:+d} "
            f"sup={s.camera_support:.2f} area={s.projected_area_frac:.2f} "
            f"score={s.selection_score:.3f} sel={'Y' if s.selected else '.'}"
        )
        cv2.putText(tile, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(tile, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 1, cv2.LINE_AA)
        out_path = out_dir / f"side_{s.side_id}_axis{s.axis_index}_{'p' if s.sign>0 else 'n'}.png"
        cv2.imwrite(str(out_path), tile)
