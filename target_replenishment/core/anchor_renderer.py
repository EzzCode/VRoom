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

__all__ = [
    'compute_quality_scores',
    'detect_defect_regions',
    'compute_anchor_camera',
    'render_anchor_views',
    'run_anchor_detection',
]

import logging
import numpy as np
import cv2
from pathlib import Path

logger = logging.getLogger(__name__)


# ── Quality Scoring ──────────────────────────────────────────────────────────

def _normalize_signal(values: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1]. 0 = best, 1 = worst."""
    if values.size == 0:
        return values
    vmin, vmax = values.min(), values.max()
    if vmax - vmin < 1e-8:
        return np.zeros_like(values)
    return (values - vmin) / (vmax - vmin)


def compute_static_signals(gaussians, n_neighbors: int = 16) -> dict:
    """Compute camera-independent geometry signals for all anchors once."""
    from target_replenishment.core.objectgs_bridge import get_anchor_positions, get_anchor_scales
    from scipy.spatial import cKDTree
    
    xyz = get_anchor_positions(gaussians)
    n = len(xyz)
    signals = {}
    
    # Scale signal (raw values)
    scales = get_anchor_scales(gaussians)
    signals['scale_raw'] = np.exp(scales).max(axis=1)
    
    # Density signal (raw values)
    logger.info(f"Building KD-tree for {n} anchors (n_neighbors={n_neighbors})...")
    tree = cKDTree(xyz)
    distances, _ = tree.query(xyz, k=n_neighbors + 1)
    signals['density_raw'] = distances[:, 1:].mean(axis=1)
    
    return signals


def compute_quality_scores(
    gaussians,
    pipe_config,
    anchor_camera_params: dict,
    n_neighbors: int = 16,
    weights: dict | None = None,
    object_id: int = None,
    static_signals: dict | None = None,
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
    from target_replenishment.core.objectgs_bridge import (
        get_anchor_positions, create_virtual_camera, render_view, 
        detect_spatial_holes, build_anchor_id_map,
    )

    VALID_SIGNAL_KEYS = {'scale', 'density', 'spatial_holes', 'alpha_deficit', 'normal_wrinkling'}
    if weights:
        if not set(weights).issubset(VALID_SIGNAL_KEYS):
            raise ValueError(f"Unknown weight keys: {set(weights) - VALID_SIGNAL_KEYS}")
    else:
        weights = {'scale': 0.20, 'density': 0.20, 'spatial_holes': 0.25, 'alpha_deficit': 0.20, 'normal_wrinkling': 0.15}

    xyz = get_anchor_positions(gaussians)
    n = len(xyz)
    signals = {}
    bg_color = torch.zeros(3, dtype=torch.float32, device="cuda")
    
    # Use provided static signals or compute them
    if static_signals and 'scale_raw' in static_signals:
        scale_raw = static_signals['scale_raw']
        density_raw = static_signals['density_raw']
    else:
        static = compute_static_signals(gaussians, n_neighbors=n_neighbors)
        scale_raw = static['scale_raw']
        density_raw = static['density_raw']

    # Local normalization logic
    if object_id is not None:
        labels = gaussians.label_ids.squeeze(-1).cpu().numpy()
        mask = labels == object_id
        
        signals['scale'] = np.zeros(n, dtype=np.float32)
        signals['density'] = np.zeros(n, dtype=np.float32)
        
        if mask.any():
            signals['scale'][mask] = _normalize_signal(scale_raw[mask])
            signals['density'][mask] = _normalize_signal(density_raw[mask])
    else:
        signals['scale'] = _normalize_signal(scale_raw)
        signals['density'] = _normalize_signal(density_raw)

    # ── Signal 3: Spatial holes ──
    R, T, K = anchor_camera_params['R'], anchor_camera_params['T'], anchor_camera_params['K']
    W, H = anchor_camera_params['width'], anchor_camera_params['height']
    cam = create_virtual_camera(R, T, K, W, H)

    coverage = detect_spatial_holes(gaussians, cam, coverage_threshold=0.1, object_label_id=object_id)
    ui, vi, visible = _project_points(xyz, R, T, K, W, H)
    anchor_coverage = np.zeros(n, dtype=np.float32)
    in_idx = np.where(visible)[0]
    if len(in_idx) > 0:
        anchor_coverage[in_idx] = coverage[vi[in_idx], ui[in_idx]]
    signals['spatial_holes'] = 1.0 - anchor_coverage

    # ── Signal 4 & 5: Render-based (single render from anchor camera) ──
    result = render_view(gaussians, cam, pipe_config, bg_color, object_label_id=object_id)

    # Alpha deficit: low alpha where anchors project
    alpha_map = result['alpha'].squeeze(0).cpu().numpy()
    anchor_alpha = np.zeros(n, dtype=np.float32)
    if len(in_idx) > 0:
        anchor_alpha[in_idx] = alpha_map[vi[in_idx], ui[in_idx]]
    signals['alpha_deficit'] = 1.0 - anchor_alpha

    # Normal wrinkling: per-anchor normal roughness from rendered normal map
    normal_map = result.get('normal')
    if normal_map is not None:
        normal_np = normal_map.squeeze(0).cpu().numpy()  # (H, W, 3)
        if normal_np.ndim == 4:
            normal_np = normal_np.squeeze(0)
        if normal_np.shape[0] == 3:  # channel-first → channel-last
            normal_np = np.transpose(normal_np, (1, 2, 0))

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
        
        # Fast vectorized aggregation
        valid_px = anchor_id_map >= 0
        aids = anchor_id_map[valid_px]
        lap_vals = laplacian[valid_px]
        np.add.at(anchor_wrinkling, aids, lap_vals)
        np.add.at(anchor_wrinkle_count, aids, 1.0)
        
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
    quality_threshold: float = 0.2,
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

    xyz_min = anchor_xyz.min(axis=0)
    xyz_max = anchor_xyz.max(axis=0)
    # Dynamically cap voxel grid to ~100^3 to avoid OOM from stray floaters
    max_extent = np.max(xyz_max - xyz_min)
    effective_voxel_size = max(voxel_size, max_extent / 100.0)
    
    xyz_min -= effective_voxel_size
    xyz_max += effective_voxel_size
    grid_dims = np.ceil((xyz_max - xyz_min) / effective_voxel_size).astype(int)

    voxel_coords = ((anchor_xyz - xyz_min) / effective_voxel_size).astype(int)
    voxel_coords = np.clip(voxel_coords, 0, grid_dims - 1)

    occupancy = np.zeros(grid_dims, dtype=np.uint8)
    occupancy[voxel_coords[:, 0], voxel_coords[:, 1], voxel_coords[:, 2]] = 1

    degraded_occ = np.zeros(grid_dims, dtype=np.uint8)
    if len(degraded_indices) > 0:
        deg_vox = voxel_coords[degraded_indices]
        degraded_occ[deg_vox[:, 0], deg_vox[:, 1], deg_vox[:, 2]] = 1

    healthy_occ = occupancy.copy()
    healthy_occ[degraded_occ == 1] = 0
    
    # Use conservative structural hole filling to avoid convex hulls spanning across concave background gaps (e.g. bananas).
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
        region_world = region_voxels * effective_voxel_size + xyz_min

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
            nearby_sets = tree_healthy.query_ball_point(region_world, r=effective_voxel_size * 3)
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
            'cavity_points': region_world,
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
    force_view_dir=None,
) -> dict:
    """Place a virtual camera looking at the defect region.

    Returns dict: 'R', 'T', 'K', 'width', 'height', 'position', 'look_at', 'up'
    """
    defect_center = defect_region['center']
    obj_centroid = anchor_xyz.mean(axis=0)

    if force_view_dir is not None:
        view_dir = np.array(force_view_dir, dtype=np.float32)
        view_dir = view_dir / np.linalg.norm(view_dir)
        dist = np.linalg.norm(defect_center - obj_centroid)
    else:
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
    object_id: int = None,
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
    
    # Render 1: Isolated object (for pristine hole detection and isolated alpha/normals)
    result_obj = render_view(gaussians, cam, pipe_config, bg_color, object_label_id=object_id)
    
    # Render 2: Full scene context (for the inpainter's background reference)
    result_full = render_view(gaussians, cam, pipe_config, bg_color)

    n_anchors = get_anchor_positions(gaussians).shape[0]

    # Convert to numpy
    rgb_isolated_np = (result_obj['rgb'].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    rgb_full_np = (result_full['rgb'].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    alpha_np = result_obj['alpha'].squeeze(0).cpu().numpy()
    normal_np = (
        result_obj['normal'].squeeze(0).cpu().numpy()
        if result_obj['normal'] is not None
        else np.zeros((H, W, 3), dtype=np.float32)
    )
    depth_np = result_obj['depth'].squeeze(0).cpu().numpy() if result_obj['depth'] is not None else None

    # Build Anchor ID map
    anchor_id_map = build_anchor_id_map(result_obj, H, W, n_anchors)

    # ── Repair mask ──
    # Type A: holes from alpha bounded by 3D cavity
    hole_mask = np.zeros((H, W), dtype=np.uint8)
    if 'cavity_points' in defect_region and len(defect_region['cavity_points']) > 0:
        pts = defect_region['cavity_points']
        
        # Internal helper for projecting points
        ui, vi, in_bounds = _project_points(pts, R, T, K, W, H)
        
        cavity_mask = np.zeros((H, W), dtype=np.uint8)
        if in_bounds.sum() > 0:
            import cv2
            u_vis, v_vis = ui[in_bounds], vi[in_bounds]
            cavity_mask[v_vis, u_vis] = 1
            # Adjust kernel size based on projected cavity extent
            k_size = max(3, int(W * 0.015))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
            cavity_mask = cv2.dilate(cavity_mask, kernel, iterations=2)
            # Erode to snap strictly inside the body
            cavity_mask = cv2.erode(cavity_mask, kernel, iterations=4)
        
        # Hole is where the ISOLATED object alpha is transparent, BUT we are physically inside the 3D cavity
        hole_mask = ((alpha_np < 0.5) & (cavity_mask == 1)).astype(np.uint8)
    else:
        hole_mask = (alpha_np < 0.5).astype(np.uint8)

    # Type B: degraded anchors via Vectorized Indexing
    degraded_mask = np.zeros((H, W), dtype=np.uint8)
    valid_px = anchor_id_map >= 0
    if valid_px.any():
        aids = anchor_id_map[valid_px]
        # Strip out background spillage by enforcing alpha > 0.1
        is_degraded = (quality_scores[aids] > quality_threshold) & (alpha_np[valid_px] > 0.1)
        degraded_mask[valid_px] = is_degraded.astype(np.uint8)

    repair_mask = np.clip(hole_mask + degraded_mask, 0, 1).astype(np.uint8)

    if mask_dilation_px > 0:
        import cv2
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
        'rgb_full': rgb_full_np,
        'rgb_isolated': rgb_isolated_np,
        'rgb': rgb_full_np,  # Expose full scene context as primary RGB for downstream
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
    """Run the complete anchor detection pipeline per-object in the scene.

    Flow: load model → extract unique labels → for each object:
          preliminary camera array → spatial quality scoring → detect regions
          → final camera at worst defect → render + mask.
    """
    import torch
    from target_replenishment.core.objectgs_bridge import load_gaussians, get_anchor_positions

    gaussians, pipe_config = load_gaussians(model_path, iteration)
    anchor_xyz_global = get_anchor_positions(gaussians)
    bg_color = torch.zeros(3, dtype=torch.float32, device="cuda")

    # Compute static signals once for the entire scene
    static_signals = compute_static_signals(gaussians)

    labels = gaussians.label_ids.squeeze(-1).cpu().numpy()
    unique_ids = np.unique(labels)
    
    results = {}

    for obj_id in unique_ids:
        logger.info(f"--- Processing Object ID {obj_id} ---")
        obj_mask = (labels == obj_id)
        anchor_xyz = anchor_xyz_global[obj_mask]
        
        if len(anchor_xyz) < 10:
            logger.warning(f"Object {obj_id} has too few anchors ({len(anchor_xyz)}). Skipping.")
            continue

        # Preliminary multi-view quality assessment
        centroid = anchor_xyz.mean(axis=0)
        prelim_region = {
            'center': centroid.astype(np.float32),
            'boundary_indices': np.array([], dtype=int),
            'defect_indices': np.array([], dtype=int),
            'severity': 0.0,
        }
        
        # Use the raw global static signals
        obj_static = {
            'scale_raw': static_signals['scale_raw'],
            'density_raw': static_signals['density_raw'],
        }
        
        # Maintain scores in global space
        quality_scores_full = np.zeros(len(anchor_xyz_global), dtype=np.float32)
        # 6-axis orthogonal sweep
        view_dirs = [[1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1]] 
        
        for vd in view_dirs:
            prelim_camera = compute_anchor_camera(
                prelim_region, anchor_xyz, fov_deg=fov_deg, render_size=render_size,
                standoff_multiplier=1.2, force_view_dir=vd
            )
            # Normalization happens locally INSIDE compute_quality_scores based on obj_id
            scores = compute_quality_scores(
                gaussians, pipe_config, prelim_camera, object_id=obj_id, 
                static_signals=obj_static
            )
            # Update object's quality scores in the global array
            quality_scores_full[obj_mask] = np.maximum(quality_scores_full[obj_mask], scores[obj_mask])

        # Defect detection on the object's subset
        quality_scores_obj = quality_scores_full[obj_mask]
        defect_regions = detect_defect_regions(anchor_xyz, quality_scores_obj, quality_threshold)

        if not defect_regions:
            logger.warning(f"No defect regions detected for object {obj_id} — model appears healthy.")
            results[obj_id] = {
                'quality_scores': quality_scores_obj,
                'defect_regions': [], 'camera_params': None, 'renders': None,
            }
            continue

        # Final anchor camera at worst defect
        primary = defect_regions[0]
        camera_params = compute_anchor_camera(
            primary, anchor_xyz, fov_deg=fov_deg, render_size=render_size,
        )

        # Render uses full global arrays
        renders = render_anchor_views(
            gaussians, camera_params, pipe_config, bg_color,
            quality_scores_full, primary, quality_threshold=quality_threshold,
            object_id=obj_id,
        )

        if output_dir:
            obj_out_dir = str(Path(output_dir) / f"obj_{obj_id}")
            _save_outputs(renders, obj_out_dir)

        results[obj_id] = {
            'quality_scores': quality_scores_obj,
            'defect_regions': defect_regions,
            'camera_params': camera_params,
            'renders': renders,
        }

    return results


def _save_outputs(renders: dict, output_dir: str):
    """Save anchor view images to disk."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(out / "anchor_rgb.png"), cv2.cvtColor(renders['rgb_full'], cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out / "anchor_rgb_isolated.png"), cv2.cvtColor(renders['rgb_isolated'], cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out / "anchor_mask.png"), renders['repair_mask'] * 255)

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

    # Overlay (using full scene so context is visible)
    overlay = renders['rgb_full'].copy()
    overlay[renders['repair_mask'] == 1] = [255, 0, 0]
    blended = cv2.addWeighted(overlay, 0.5, renders['rgb_full'], 0.5, 0)
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
    parser.add_argument("--quality_threshold", type=float, default=0.2, help="Quality score cutoff (0-1)")
    parser.add_argument("--fov", type=float, default=60.0, help="FOV in degrees")
    parser.add_argument("--render_size", type=int, default=512, help="Render resolution (square)")
    args = parser.parse_args()

    results = run_anchor_detection(
        args.model_path,
        output_dir=args.output_dir,
        iteration=args.iteration,
        quality_threshold=args.quality_threshold,
        fov_deg=args.fov,
        render_size=args.render_size,
    )

    for obj_id, result in results.items():
        if result['defect_regions']:
            print(f"\n--- Object {obj_id} ---")
            print(f"Detected {len(result['defect_regions'])} defect region(s).")
            print(f"Primary defect center: {result['defect_regions'][0]['center']}")
            print(f"Anchor camera position: {result['camera_params']['position']}")
            print(f"Repair mask coverage: {result['renders']['repair_mask'].sum()} px")
        else:
            print(f"\n--- Object {obj_id} ---")
            print("No defects detected. Model appears healthy.")

