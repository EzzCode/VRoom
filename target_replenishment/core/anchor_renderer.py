"""
VRoom Anchor Renderer — Defect Detection & Anchor View Rendering (Production)

Detects two types of defects in a trained ObjectGS 2DGS model:
  Type A — Holes:    Missing geometry (transparent gaps via spatial coverage).
  Type B — Degraded: Low-quality rendered regions (via anchor ID map + normal map).

Places an optimal "anchor" camera facing the worst defect region and renders
RGB, Alpha, Normal, Depth maps plus a unified repair mask for downstream inpainting.

Architecture decisions:
  - Normal consistency scored on RENDERED normal map, not anchor quaternions.
  - Holes detected spatially by projecting anchor bboxes (no MLP invocation).
  - Degraded regions identified via Anchor ID buffer from patched ObjectGS render.
  - The rasterizer will be reimplemented later — until then, ObjectGS gsplat is used.

Public API:
    compute_quality_scores(gaussians, pipe, anchor_camera)  -> per-anchor scores
    detect_defect_regions(anchor_xyz, quality_scores)        -> ranked defect regions
    compute_anchor_camera(defect_region, anchor_xyz)         -> camera params dict
    render_anchor_views(gaussians, camera, pipe, ...)        -> renders + repair mask
    run_anchor_detection(model_path, ...)                    -> end-to-end pipeline
"""

import logging
import numpy as np
import cv2
from pathlib import Path

logger = logging.getLogger(__name__)


# ── Quality Scoring ──────────────────────────────────────────────────────────

def _normalize_signal(values: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1]. 0 = best, 1 = worst."""
    vmin, vmax = values.min(), values.max()
    if vmax - vmin < 1e-8:
        return np.zeros_like(values)
    return (values - vmin) / (vmax - vmin)


def compute_quality_scores(
    gaussians,
    pipe_config,
    anchor_camera_params: dict,
    k: int = 16,
    weights: dict | None = None,
) -> np.ndarray:
    """Compute per-anchor quality score in [0, 1]. 0 = perfect, 1 = severely degraded.

    Three geometry signals (no MLP invocation):
        1. scale         — Oversized anchor voxels (blur/smear)
        2. density       — Sparse neighborhoods (isolated anchors)
        3. spatial_holes — Low spatial coverage from projected anchor bboxes

    Two render-based signals (single render from anchor camera):
        4. alpha_deficit  — Low alpha at anchor projections in the rendered view
        5. normal_wrinkling — High-frequency normal deviations in rendered normal map

    Signal 4 and 5 require the anchor camera → a single render, not multi-view probes.

    Returns:
        (N,) float32 quality score per anchor.
    """
    import torch
    from scipy.spatial import cKDTree
    from target_replenishment.core.objectgs_bridge import (
        get_anchor_positions, get_anchor_scales,
        create_virtual_camera, render_view, detect_spatial_holes,
        build_anchor_id_map,
    )

    if weights is None:
        weights = {
            'scale': 0.20,
            'density': 0.20,
            'spatial_holes': 0.25,
            'alpha_deficit': 0.20,
            'normal_wrinkling': 0.15,
        }

    xyz = get_anchor_positions(gaussians)
    n = len(xyz)
    signals = {}
    bg_color = torch.zeros(3, dtype=torch.float32, device="cuda")

    # ── Signal 1: Scale — oversized anchors → blur ──
    scales = get_anchor_scales(gaussians)
    max_scale = np.exp(scales).max(axis=1)
    signals['scale'] = _normalize_signal(max_scale)

    # ── Signal 2: Density — sparse neighborhood → isolated ──
    logger.info(f"Building KD-tree for {n} anchors (k={k})...")
    tree = cKDTree(xyz)
    distances, _ = tree.query(xyz, k=k + 1)
    mean_knn_dist = distances[:, 1:].mean(axis=1)
    signals['density'] = _normalize_signal(mean_knn_dist)

    # ── Signal 3: Spatial holes — project anchor bboxes, no MLP ──
    R, T, K = anchor_camera_params['R'], anchor_camera_params['T'], anchor_camera_params['K']
    W, H = anchor_camera_params['width'], anchor_camera_params['height']
    cam = create_virtual_camera(R, T, K, W, H)

    coverage = detect_spatial_holes(gaussians, cam, coverage_threshold=0.1)
    # Map per-pixel coverage back to per-anchor score
    u, v, visible = _project_points(xyz, R, T, K, W, H)
    anchor_coverage = np.zeros(n, dtype=np.float32)
    for i in range(n):
        if visible[i]:
            px, py = int(u[i]), int(v[i])
            anchor_coverage[i] = coverage[py, px]
        else:
            anchor_coverage[i] = 0.0  # not visible = potential hole
    signals['spatial_holes'] = _normalize_signal(1.0 - anchor_coverage)

    # ── Signal 4 & 5: Render-based (single render from anchor camera) ──
    result = render_view(gaussians, cam, pipe_config, bg_color)

    # Alpha deficit: low alpha where anchors project
    alpha_map = result['alpha'].squeeze(0).cpu().numpy()
    anchor_alpha = np.zeros(n, dtype=np.float32)
    for i in range(n):
        if visible[i]:
            px, py = int(u[i]), int(v[i])
            anchor_alpha[i] = alpha_map[py, px]
        # else: remains 0 → high defect score
    signals['alpha_deficit'] = _normalize_signal(1.0 - anchor_alpha)

    # Normal wrinkling: per-anchor normal roughness from rendered normal map
    normal_map = result.get('normal')
    if normal_map is not None:
        normal_np = normal_map.squeeze(0).cpu().numpy()  # (H, W, 3)
        if normal_np.ndim == 4:
            normal_np = normal_np.squeeze(0)
        if normal_np.shape[0] == 3 and normal_np.shape[-1] != 3:
            normal_np = np.transpose(normal_np, (1, 2, 0))  # (3,H,W) → (H,W,3)

        # Compute per-pixel normal gradient magnitude (Laplacian proxy for wrinkling)
        nx = normal_np[..., 0]
        ny = normal_np[..., 1]
        nz = normal_np[..., 2]
        laplacian = (
            np.abs(cv2.Laplacian(nx, cv2.CV_32F)) +
            np.abs(cv2.Laplacian(ny, cv2.CV_32F)) +
            np.abs(cv2.Laplacian(nz, cv2.CV_32F))
        )

        # Map per-pixel wrinkling to per-anchor via anchor ID map
        anchor_id_map = build_anchor_id_map(result, H, W, n)
        anchor_wrinkling = np.zeros(n, dtype=np.float32)
        anchor_wrinkle_count = np.zeros(n, dtype=np.float32)
        for y in range(H):
            for x in range(W):
                aid = anchor_id_map[y, x]
                if aid >= 0:
                    anchor_wrinkling[aid] += laplacian[y, x]
                    anchor_wrinkle_count[aid] += 1.0
        mask = anchor_wrinkle_count > 0
        anchor_wrinkling[mask] /= anchor_wrinkle_count[mask]
        signals['normal_wrinkling'] = _normalize_signal(anchor_wrinkling)
    else:
        signals['normal_wrinkling'] = np.full(n, 0.5, dtype=np.float32)

    # ── Weighted combination ──
    quality_score = np.zeros(n, dtype=np.float32)
    for key, weight in weights.items():
        if key in signals:
            quality_score += weight * signals[key]

    logger.info(
        f"Quality scores: mean={quality_score.mean():.3f}, "
        f"std={quality_score.std():.3f}, "
        f"degraded (>0.5): {(quality_score > 0.5).sum()} / {n}"
    )
    return quality_score


# ── Defect Region Detection ──────────────────────────────────────────────────

def detect_defect_regions(
    anchor_xyz: np.ndarray,
    quality_scores: np.ndarray,
    quality_threshold: float = 0.5,
    voxel_size: float = 0.02,
    min_region_anchors: int = 10,
) -> list:
    """Detect coherent defect regions by clustering degraded anchors + holes.

    Returns list of dicts sorted by severity (worst first):
        'center', 'boundary_indices', 'defect_indices', 'severity'
    """
    from scipy.spatial import cKDTree
    from scipy.ndimage import label as ndimage_label, binary_fill_holes

    n = len(anchor_xyz)
    is_degraded = quality_scores > quality_threshold
    degraded_indices = np.where(is_degraded)[0]
    healthy_indices = np.where(~is_degraded)[0]
    logger.info(f"Degraded anchors: {len(degraded_indices)} / {n}")

    # Voxel occupancy grid
    xyz_min = anchor_xyz.min(axis=0) - voxel_size
    xyz_max = anchor_xyz.max(axis=0) + voxel_size
    grid_dims = np.ceil((xyz_max - xyz_min) / voxel_size).astype(int)

    voxel_coords = ((anchor_xyz - xyz_min) / voxel_size).astype(int)
    voxel_coords = np.clip(voxel_coords, 0, grid_dims - 1)

    occupancy = np.zeros(grid_dims, dtype=np.uint8)
    occupancy[voxel_coords[:, 0], voxel_coords[:, 1], voxel_coords[:, 2]] = 1

    degraded_occ = np.zeros(grid_dims, dtype=np.uint8)
    if len(degraded_indices) > 0:
        deg_vox = voxel_coords[degraded_indices]
        degraded_occ[deg_vox[:, 0], deg_vox[:, 1], deg_vox[:, 2]] = 1

    healthy_occ = occupancy.copy()
    healthy_occ[degraded_occ == 1] = 0
    filled = binary_fill_holes(healthy_occ)
    hole_voxels = filled & (~healthy_occ.astype(bool))
    defect_map = hole_voxels | degraded_occ.astype(bool)

    labeled_array, num_features = ndimage_label(defect_map)
    logger.info(f"Found {num_features} raw defect cluster(s)")
    if num_features == 0:
        return []

    tree_healthy = cKDTree(anchor_xyz[healthy_indices]) if len(healthy_indices) > 0 else None
    regions = []

    for region_id in range(1, num_features + 1):
        region_voxels = np.argwhere(labeled_array == region_id)
        region_world = region_voxels * voxel_size + xyz_min

        if len(degraded_indices) > 0:
            deg_voxels = voxel_coords[degraded_indices]
            labels_at = labeled_array[deg_voxels[:, 0], deg_voxels[:, 1], deg_voxels[:, 2]]
            region_defect_indices = degraded_indices[labels_at == region_id]
        else:
            region_defect_indices = np.array([], dtype=int)

        if len(region_voxels) < min_region_anchors:
            continue

        center = region_world.mean(axis=0).astype(np.float32)

        boundary_indices = np.array([], dtype=int)
        if tree_healthy is not None and len(region_world) > 0:
            nearby_sets = tree_healthy.query_ball_point(region_world, r=voxel_size * 3)
            nearby_flat = set()
            for s in nearby_sets:
                nearby_flat.update(s)
            if nearby_flat:
                boundary_indices = healthy_indices[np.array(sorted(nearby_flat))]

        severity = (
            len(region_voxels) * quality_scores[region_defect_indices].mean()
            if len(region_defect_indices) > 0 else float(len(region_voxels))
        )

        regions.append({
            'center': center,
            'boundary_indices': boundary_indices,
            'defect_indices': region_defect_indices,
            'severity': severity,
        })

    regions.sort(key=lambda r: r['severity'], reverse=True)
    logger.info(f"Detected {len(regions)} defect region(s) after filtering")
    return regions


# ── Anchor Camera Placement ──────────────────────────────────────────────────

def compute_anchor_camera(
    defect_region: dict,
    anchor_xyz: np.ndarray,
    fov_deg: float = 60.0,
    render_size: int = 512,
    standoff_multiplier: float = 2.5,
) -> dict:
    """Place a virtual camera looking at the defect region.

    Returns dict: 'R', 'T', 'K', 'width', 'height', 'position', 'look_at', 'up'
    """
    defect_center = defect_region['center']
    obj_centroid = anchor_xyz.mean(axis=0)

    view_dir = defect_center - obj_centroid
    dist = np.linalg.norm(view_dir)
    if dist < 1e-8:
        view_dir = np.array([0, 0, 1], dtype=np.float32)
    else:
        view_dir = view_dir / dist

    boundary_idx = defect_region['boundary_indices']
    if len(boundary_idx) > 3:
        boundary_pos = anchor_xyz[boundary_idx]
        extent = np.linalg.norm(boundary_pos.max(axis=0) - boundary_pos.min(axis=0))
        centered = boundary_pos - boundary_pos.mean(axis=0)
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        pca_up = Vt[1]
    else:
        extent = dist * 0.5
        pca_up = np.array([0, 1, 0], dtype=np.float32)

    standoff = max(extent * standoff_multiplier, 0.5)
    cam_pos = defect_center + view_dir * standoff

    forward = -view_dir
    right = np.cross(forward, pca_up)
    if np.linalg.norm(right) < 1e-6:
        pca_up = np.array([0, 0, 1], dtype=np.float32)
        right = np.cross(forward, pca_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    up /= np.linalg.norm(up)

    R = np.stack([right, -up, forward], axis=0).astype(np.float32)
    T = (-R @ cam_pos).reshape(3, 1).astype(np.float32)

    fov_rad = np.radians(fov_deg)
    focal = render_size / (2 * np.tan(fov_rad / 2))
    K = np.array([
        [focal, 0, render_size / 2.0],
        [0, focal, render_size / 2.0],
        [0, 0, 1],
    ], dtype=np.float32)

    logger.info(
        f"Anchor camera: pos={cam_pos}, standoff={standoff:.3f}, "
        f"fov={fov_deg}°, res={render_size}x{render_size}"
    )

    return {
        'R': R, 'T': T, 'K': K,
        'width': render_size, 'height': render_size,
        'position': cam_pos.astype(np.float32),
        'look_at': defect_center.astype(np.float32),
        'up': up.astype(np.float32),
    }


# ── Anchor View Rendering + Repair Mask ──────────────────────────────────────

def render_anchor_views(
    gaussians,
    camera_params: dict,
    pipe_config,
    bg_color,
    quality_scores: np.ndarray,
    defect_region: dict,
    quality_threshold: float = 0.5,
    mask_dilation_px: int = 5,
) -> dict:
    """Render anchor view via ObjectGS and build the repair mask.

    Repair mask = union of:
      - Hole pixels (alpha < 0.5)
      - Degraded anchor pixels (from Anchor ID map + quality scores)

    Returns dict: 'rgb', 'alpha', 'normal', 'depth', 'repair_mask',
                  'anchor_id_map', 'camera_params'
    """
    import torch
    from target_replenishment.core.objectgs_bridge import (
        create_virtual_camera, render_view, get_anchor_positions,
        build_anchor_id_map,
    )

    H, W = camera_params['height'], camera_params['width']
    R, T, K = camera_params['R'], camera_params['T'], camera_params['K']

    cam = create_virtual_camera(R, T, K, W, H)
    result = render_view(gaussians, cam, pipe_config, bg_color)

    n_anchors = get_anchor_positions(gaussians).shape[0]

    # Convert to numpy
    rgb_np = (result['rgb'].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    alpha_np = result['alpha'].squeeze(0).cpu().numpy()
    normal_np = (
        result['normal'].squeeze(0).cpu().numpy()
        if result['normal'] is not None
        else np.zeros((H, W, 3), dtype=np.float32)
    )
    depth_np = result['depth'].squeeze(0).cpu().numpy() if result['depth'] is not None else None

    # Build Anchor ID map
    anchor_id_map = build_anchor_id_map(result, H, W, n_anchors)

    # ── Repair mask ──
    # Type A: holes from alpha
    hole_mask = (alpha_np < 0.5).astype(np.uint8)

    # Type B: degraded anchors via Anchor ID map
    # Pixels whose parent anchor has quality > threshold are degraded
    degraded_mask = np.zeros((H, W), dtype=np.uint8)
    for y in range(H):
        for x in range(W):
            aid = anchor_id_map[y, x]
            if aid >= 0 and quality_scores[aid] > quality_threshold:
                degraded_mask[y, x] = 1

    repair_mask = np.clip(hole_mask + degraded_mask, 0, 1).astype(np.uint8)

    if mask_dilation_px > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (mask_dilation_px * 2 + 1, mask_dilation_px * 2 + 1)
        )
        repair_mask = cv2.dilate(repair_mask, kernel, iterations=1)

    logger.info(
        f"Repair mask: {repair_mask.sum()}/{H*W} px "
        f"({100 * repair_mask.sum() / (H*W):.1f}%) — "
        f"holes={hole_mask.sum()}, degraded={degraded_mask.sum()}"
    )

    return {
        'rgb': rgb_np,
        'alpha': alpha_np,
        'normal': normal_np,
        'depth': depth_np,
        'repair_mask': repair_mask,
        'anchor_id_map': anchor_id_map,
        'camera_params': camera_params,
    }


# ── Internal Helpers ─────────────────────────────────────────────────────────

def _project_points(points, R, T, K, W, H):
    """Project Nx3 world points to pixel coords. Returns (u, v, in_bounds)."""
    T_flat = T.flatten()
    cam_pts = (R @ points.T).T + T_flat[np.newaxis, :]
    z = cam_pts[:, 2]
    valid = z > 0

    u = np.full(len(points), -1.0)
    v = np.full(len(points), -1.0)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    u[valid] = fx * cam_pts[valid, 0] / z[valid] + cx
    v[valid] = fy * cam_pts[valid, 1] / z[valid] + cy

    ui = np.round(u).astype(int)
    vi = np.round(v).astype(int)
    in_bounds = valid & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    return ui, vi, in_bounds


# ── Full Pipeline ────────────────────────────────────────────────────────────

def run_anchor_detection(
    model_path: str,
    output_dir: str | None = None,
    iteration: int = -1,
    quality_threshold: float = 0.5,
    fov_deg: float = 60.0,
    render_size: int = 512,
) -> dict:
    """Run the complete anchor detection pipeline.

    Flow: load model → preliminary camera → spatial quality scoring → detect regions
          → final camera at worst defect → render + mask.
    """
    import torch
    from target_replenishment.core.objectgs_bridge import load_gaussians, get_anchor_positions

    gaussians, pipe_config = load_gaussians(model_path, iteration)
    anchor_xyz = get_anchor_positions(gaussians)
    bg_color = torch.zeros(3, dtype=torch.float32, device="cuda")

    # Preliminary camera: aim at object centroid for initial quality assessment
    centroid = anchor_xyz.mean(axis=0)
    extent = np.linalg.norm(anchor_xyz.max(axis=0) - anchor_xyz.min(axis=0))
    prelim_region = {
        'center': centroid.astype(np.float32),
        'boundary_indices': np.arange(min(10, len(anchor_xyz))),
        'defect_indices': np.array([], dtype=int),
        'severity': 0.0,
    }
    prelim_camera = compute_anchor_camera(
        prelim_region, anchor_xyz, fov_deg=fov_deg, render_size=render_size,
        standoff_multiplier=3.0,
    )

    # Quality scoring (uses spatial + single render from prelim camera)
    quality_scores = compute_quality_scores(gaussians, pipe_config, prelim_camera)

    # Defect detection
    defect_regions = detect_defect_regions(anchor_xyz, quality_scores, quality_threshold)

    if not defect_regions:
        logger.warning("No defect regions detected — model appears healthy.")
        return {
            'gaussians': gaussians, 'quality_scores': quality_scores,
            'defect_regions': [], 'camera_params': None, 'renders': None,
        }

    # Final anchor camera at worst defect
    primary = defect_regions[0]
    camera_params = compute_anchor_camera(
        primary, anchor_xyz, fov_deg=fov_deg, render_size=render_size,
    )

    # Render
    renders = render_anchor_views(
        gaussians, camera_params, pipe_config, bg_color,
        quality_scores, primary, quality_threshold=quality_threshold,
    )

    if output_dir:
        _save_outputs(renders, output_dir)

    return {
        'gaussians': gaussians,
        'quality_scores': quality_scores,
        'defect_regions': defect_regions,
        'camera_params': camera_params,
        'renders': renders,
    }


def _save_outputs(renders: dict, output_dir: str):
    """Save anchor view images to disk."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(out / "anchor_rgb.png"), cv2.cvtColor(renders['rgb'], cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out / "anchor_mask.png"), renders['repair_mask'] * 255)

    if renders['normal'] is not None:
        normal_vis = ((renders['normal'] + 1) * 0.5 * 255).astype(np.uint8)
        cv2.imwrite(str(out / "anchor_normal.png"), normal_vis)

    if renders['depth'] is not None:
        depth = renders['depth'].squeeze()
        depth_vis = (depth / (depth.max() + 1e-6) * 255).astype(np.uint8)
        cv2.imwrite(str(out / "anchor_depth.png"), depth_vis)

    # Anchor ID visualization
    aid = renders['anchor_id_map']
    aid_vis = ((aid - aid.min()) / (aid.max() - aid.min() + 1e-6) * 255).astype(np.uint8)
    cv2.imwrite(str(out / "anchor_id_map.png"), cv2.applyColorMap(aid_vis, cv2.COLORMAP_TURBO))

    # Overlay
    overlay = renders['rgb'].copy()
    overlay[renders['repair_mask'] == 1] = [255, 0, 0]
    blended = cv2.addWeighted(overlay, 0.5, renders['rgb'], 0.5, 0)
    cv2.imwrite(str(out / "anchor_overlay.png"), cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))

    logger.info(f"Saved anchor outputs to {out}")


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S',
    )

    parser = argparse.ArgumentParser(description="VRoom Anchor Renderer — Defect Detection")
    parser.add_argument("--model_path", required=True, help="ObjectGS training output directory")
    parser.add_argument("--iteration", type=int, default=-1, help="Training iteration to load (-1 = latest)")
    parser.add_argument("--output_dir", default="anchor_output", help="Directory for output images")
    parser.add_argument("--quality_threshold", type=float, default=0.5, help="Quality score cutoff (0-1)")
    parser.add_argument("--fov", type=float, default=60.0, help="FOV in degrees")
    parser.add_argument("--render_size", type=int, default=512, help="Render resolution (square)")
    args = parser.parse_args()

    result = run_anchor_detection(
        args.model_path,
        output_dir=args.output_dir,
        iteration=args.iteration,
        quality_threshold=args.quality_threshold,
        fov_deg=args.fov,
        render_size=args.render_size,
    )

    if result['defect_regions']:
        print(f"\nDetected {len(result['defect_regions'])} defect region(s).")
        print(f"Primary defect center: {result['defect_regions'][0]['center']}")
        print(f"Anchor camera position: {result['camera_params']['position']}")
        print(f"Repair mask coverage: {result['renders']['repair_mask'].sum()} px")
    else:
        print("\nNo defects detected. Model appears healthy.")
