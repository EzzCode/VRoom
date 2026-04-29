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
from scipy.ndimage import binary_closing, binary_fill_holes, gaussian_filter, label
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
    max_hole_area_frac: float = 0.01,
    max_hole_cells: int = 256,
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


def _filled_depth_map(side: SideScanResult, smooth_iters: int) -> tuple[np.ndarray, np.ndarray]:
    depth = side.outer_depth.copy()
    depth_filled = depth.copy()
    mask = side.occupancy_filled & np.isfinite(depth_filled)
    if side.occupancy_filled.sum() > side.occupancy.sum():
        occ_idx = np.argwhere(side.occupancy)
        if occ_idx.size:
            tree = cKDTree(occ_idx.astype(np.float32))
            fill_idx = np.argwhere(side.occupancy_filled & ~side.occupancy)
            if fill_idx.size:
                _, nn = tree.query(fill_idx.astype(np.float32), k=1)
                src = occ_idx[nn]
                depth_filled[fill_idx[:, 0], fill_idx[:, 1]] = depth[src[:, 0], src[:, 1]]
        mask = side.occupancy_filled & np.isfinite(depth_filled)

    if smooth_iters > 0 and mask.any():
        sigma = max(0.4, float(smooth_iters) * 0.75)
        weights = mask.astype(np.float32)
        values = np.where(mask, depth_filled, 0.0).astype(np.float32)
        smooth_values = gaussian_filter(values, sigma=sigma, mode="nearest")
        smooth_weights = gaussian_filter(weights, sigma=sigma, mode="nearest")
        smooth = smooth_values / np.maximum(smooth_weights, 1e-6)
        depth_filled[mask] = smooth[mask]

    return depth_filled, mask


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
    cap_offset_factor: float = 0.35,
    morphological_close_iters: int = 1,
    depth_smooth_iters: int = 2,
    samples_per_cell: int = 3,
    min_camera_support: float = 0.15,
    min_projected_area_frac: float = 0.10,
    min_uncovered_frac: float = 0.12,
    existing_coverage_alpha: float = 0.6,
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
    cap_offset = max(float(cap_offset_factor * r_med), 1e-6)

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

    side_candidates: dict[int, np.ndarray] = {}
    for s in sides:
        depth_filled, seed_mask = _filled_depth_map(s, smooth_iters=depth_smooth_iters)
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
