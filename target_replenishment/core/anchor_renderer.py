"""
VRoom Anchor Renderer — Defect Detection & Anchor View Rendering (Production)

Designed for the object enhancement pipeline: select an object, separate it from
the scene, find defects, and prepare inpainting data for model re-training.

Detects two types of defects in a trained ObjectGS 2DGS model:
  Type A — Holes:    Missing geometry (low alpha inside object's projected bbox).
  Type B — Degraded: Low-quality rendered regions (via Anchor ID map + normal map).

Selects anchor views from the perspective graph of existing training cameras
and provides neighbor views for downstream content propagation.

Architecture decisions:
  - Multi-view quality scoring from top-K training cameras (max per anchor).
  - Holes detected via alpha inside projected object bbox (not full-scene comparison).
  - Normal wrinkling scored on interior surface pixels, silhouette edges excluded.
  - Defect indices returned in GLOBAL anchor space for consistency.
  - Anchor view selected with frontality weighting (head-on to defect surface).

Public API:
    compute_quality_scores(gaussians, pipe, cam, ...)  -> per-anchor scores
    detect_defect_regions(anchor_xyz, scores, ...)     -> ranked defect regions
    render_anchor_views(gaussians, cam, pipe, ...)     -> renders + repair mask
    run_anchor_detection(model_path, ...)              -> end-to-end pipeline
"""

__all__ = [
    'compute_static_signals',
    'compute_quality_scores',
    'compute_multiview_quality',
    'detect_defect_regions',
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

    scales = get_anchor_scales(gaussians)
    scale_raw = np.exp(scales).max(axis=1)

    logger.info(f"Building KD-tree for {n} anchors (n_neighbors={n_neighbors})...")
    tree = cKDTree(xyz)
    distances, _ = tree.query(xyz, k=n_neighbors + 1)
    density_raw = distances[:, 1:].mean(axis=1)

    return {'scale_raw': scale_raw, 'density_raw': density_raw}


def compute_quality_scores(
    gaussians,
    pipe_config,
    camera_params: dict,
    n_neighbors: int = 16,
    weights: dict | None = None,
    object_id: int | None = None,
    static_signals: dict | None = None,
) -> np.ndarray:
    """Compute per-anchor quality score in [0, 1]. 0 = perfect, 1 = severely degraded.

    Scores anchors from a SINGLE camera view. For full-object coverage,
    call this from multiple views and take max per anchor.

    Returns:
        (N_global,) float32 quality score per anchor.
    """
    import torch
    from target_replenishment.core.objectgs_bridge import (
        get_anchor_positions, create_virtual_camera, render_view,
        build_anchor_id_map,
    )

    VALID_KEYS = {'scale', 'density', 'alpha_deficit', 'normal_wrinkling'}
    if weights:
        if not set(weights).issubset(VALID_KEYS):
            raise ValueError(f"Unknown weight keys: {set(weights) - VALID_KEYS}")
    else:
        weights = {
            'scale': 0.25, 'density': 0.25, 'alpha_deficit': 0.30,
            'normal_wrinkling': 0.20,
        }

    sum_w = sum(weights.values())
    if sum_w != 1.0:
        weights = {k: v / sum_w for k, v in weights.items()}

    xyz = get_anchor_positions(gaussians)
    n = len(xyz)
    signals = {}
    bg_color = torch.zeros(3, dtype=torch.float32, device="cuda")

    # ── Static signals ──
    if static_signals and 'scale_raw' in static_signals:
        scale_raw, density_raw = static_signals['scale_raw'], static_signals['density_raw']
    else:
        static = compute_static_signals(gaussians, n_neighbors=n_neighbors)
        scale_raw, density_raw = static['scale_raw'], static['density_raw']

    if object_id is not None:
        labels = gaussians.label_ids.squeeze(-1).cpu().numpy()
        obj_mask = labels == object_id
        signals['scale'] = np.zeros(n, dtype=np.float32)
        signals['density'] = np.zeros(n, dtype=np.float32)
        if obj_mask.any():
            signals['scale'][obj_mask] = _normalize_signal(scale_raw[obj_mask])
            signals['density'][obj_mask] = _normalize_signal(density_raw[obj_mask])
    else:
        signals['scale'] = _normalize_signal(scale_raw)
        signals['density'] = _normalize_signal(density_raw)

    # ── Rendered view properties ──
    R, T, K = camera_params['R'], camera_params['T'], camera_params['K']
    W, H = camera_params['width'], camera_params['height']
    cam = create_virtual_camera(R, T, K, W, H)

    ui, vi, visible = _project_points(xyz, R, T, K, W, H)
    in_idx = np.where(visible)[0]

    # ── Render-based signals (single render) ──
    result = render_view(gaussians, cam, pipe_config, bg_color, object_label_id=object_id)

    alpha_map = result['alpha'].squeeze(0).cpu().numpy()
    anchor_alpha = np.zeros(n, dtype=np.float32)
    if len(in_idx) > 0:
        anchor_alpha[in_idx] = alpha_map[vi[in_idx], ui[in_idx]]
    signals['alpha_deficit'] = 1.0 - anchor_alpha

    # Normal wrinkling (silhouette edges excluded)
    normal_tensor = result.get('normal')
    if normal_tensor is not None:
        normal_np = normal_tensor.squeeze(0).cpu().numpy()
        if normal_np.ndim == 4:
            normal_np = normal_np.squeeze(0)
        if normal_np.shape[0] == 3:
            normal_np = np.transpose(normal_np, (1, 2, 0))

        # Mathematically principled surface roughness: 1 - ||E[N]||
        # A perfectly smooth surface has locally identical normals, so ||E[N]|| = 1.
        # Wildly varying normals average towards 0, resulting in roughness near 1.
        w_size = max(3, int(min(H, W) * 0.005))
        mean_nx = cv2.boxFilter(normal_np[..., 0], cv2.CV_32F, (w_size, w_size))
        mean_ny = cv2.boxFilter(normal_np[..., 1], cv2.CV_32F, (w_size, w_size))
        mean_nz = cv2.boxFilter(normal_np[..., 2], cv2.CV_32F, (w_size, w_size))
        
        mean_norm = np.clip(np.sqrt(mean_nx**2 + mean_ny**2 + mean_nz**2), 0.0, 1.0)
        laplacian = 1.0 - mean_norm

        alpha_grad = np.abs(cv2.Laplacian(alpha_map, cv2.CV_32F))
        silhouette_mask = alpha_grad > 0.1
        interior_mask = (alpha_map > 0.5) & (~silhouette_mask)
        laplacian[~interior_mask] = 0.0

        anchor_id_map = build_anchor_id_map(result, H, W, n)
        anchor_wrinkling = np.zeros(n, dtype=np.float32)
        anchor_wrinkle_count = np.zeros(n, dtype=np.float32)

        valid_px = (anchor_id_map >= 0) & interior_mask
        if valid_px.any():
            aids = anchor_id_map[valid_px]
            np.add.at(anchor_wrinkling, aids, laplacian[valid_px])
            np.add.at(anchor_wrinkle_count, aids, 1.0)

        nonzero = anchor_wrinkle_count > 0
        anchor_wrinkling[nonzero] /= anchor_wrinkle_count[nonzero]
        signals['normal_wrinkling'] = _normalize_signal(anchor_wrinkling)
    else:
        signals['normal_wrinkling'] = np.full(n, 0.0, dtype=np.float32)

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


def compute_multiview_quality(
    gaussians,
    pipe_config,
    camera_list: list,
    object_id: int = None,
    static_signals: dict | None = None,
    **kwargs,
) -> np.ndarray:
    """Score from multiple training cameras, take MAX per anchor.

    This ensures all sides of the object are checked — defects on the back
    are caught by cameras facing that side.

    Args:
        camera_list: List of camera param dicts (from get_top_k_views_for_object).

    Returns:
        (N_global,) float32 worst-case quality score per anchor.
    """
    from target_replenishment.core.objectgs_bridge import get_anchor_positions
    n = len(get_anchor_positions(gaussians))
    combined = np.zeros(n, dtype=np.float32)

    for i, cam_params in enumerate(camera_list):
        logger.info(f"Scoring from view {i+1}/{len(camera_list)}...")
        scores = compute_quality_scores(
            gaussians, pipe_config, cam_params,
            object_id=object_id, static_signals=static_signals, **kwargs,
        )
        combined = np.maximum(combined, scores)

    logger.info(f"Multi-view scores: mean={combined.mean():.3f}, "
                f"degraded (>0.5): {(combined > 0.5).sum()} / {n}")
    return combined


# ── Defect Region Detection ──────────────────────────────────────────────────

def detect_defect_regions(
    anchor_xyz: np.ndarray,
    quality_scores: np.ndarray,
    quality_threshold: float = 0.2,
    voxel_size: float = 0.02,
    min_region_anchors: int = 10,
    global_index_offset: np.ndarray = None,
) -> list:
    """Detect coherent defect regions by clustering degraded anchors.

    Holes are detected from the rendered alpha map in render_anchor_views(),
    not here. This function only clusters degraded anchors in 3D.

    Args:
        anchor_xyz: (N,3) positions of this object's anchors.
        quality_scores: (N,) scores for this object's anchors.
        global_index_offset: (N,) int array mapping local indices → global anchor IDs.
            If provided, defect_indices in the output will be GLOBAL indices.

    Returns:
        List of dicts sorted by severity (worst first):
            'center', 'boundary_indices', 'defect_indices', 'severity'
    """
    from scipy.spatial import cKDTree
    from scipy.ndimage import label as ndimage_label

    n = len(anchor_xyz)
    # Adaptive Z-score thresholding: target the top ~15% worst anchors mathematically
    mean_score = quality_scores.mean()
    std_score = quality_scores.std()
    effective_threshold = mean_score + 1.0 * std_score

    is_degraded = quality_scores > effective_threshold
    degraded_indices = np.where(is_degraded)[0]
    healthy_indices = np.where(~is_degraded)[0]
    logger.info(f"Degraded anchors: {len(degraded_indices)} / {n}")

    if len(degraded_indices) == 0:
        return []

    # Dynamic voxel grid (capped at ~100^3) robust to floater artifacts
    xyz_min = np.percentile(anchor_xyz, 1, axis=0)
    xyz_max = np.percentile(anchor_xyz, 99, axis=0)
    max_extent = np.max(xyz_max - xyz_min)
    effective_vs = max(voxel_size, max_extent / 100.0)

    xyz_min -= effective_vs
    xyz_max += effective_vs
    grid_dims = np.ceil((xyz_max - xyz_min) / effective_vs).astype(int)

    voxel_coords = ((anchor_xyz - xyz_min) / effective_vs).astype(int)
    voxel_coords = np.clip(voxel_coords, 0, grid_dims - 1)

    # Cluster degraded anchors only
    degraded_occ = np.zeros(grid_dims, dtype=np.uint8)
    deg_vox = voxel_coords[degraded_indices]
    degraded_occ[deg_vox[:, 0], deg_vox[:, 1], deg_vox[:, 2]] = 1

    labeled_array, num_features = ndimage_label(degraded_occ)
    logger.info(f"Found {num_features} degraded cluster(s)")
    if num_features == 0:
        return []

    tree_healthy = cKDTree(anchor_xyz[healthy_indices]) if len(healthy_indices) > 0 else None
    regions = []

    for region_id in range(1, num_features + 1):
        labels_at = labeled_array[deg_vox[:, 0], deg_vox[:, 1], deg_vox[:, 2]]
        local_defect_idx = degraded_indices[labels_at == region_id]

        if len(local_defect_idx) < min_region_anchors:
            continue

        defect_xyz = anchor_xyz[local_defect_idx]
        center = defect_xyz.mean(axis=0).astype(np.float32)

        # Healthy boundary ring
        boundary_local = np.array([], dtype=int)
        if tree_healthy is not None:
            nearby_sets = tree_healthy.query_ball_point(defect_xyz, r=effective_vs * 3)
            nearby_flat = set()
            for s in nearby_sets:
                nearby_flat.update(s)
            if nearby_flat:
                boundary_local = healthy_indices[np.array(sorted(nearby_flat))]

        severity = len(local_defect_idx) * quality_scores[local_defect_idx].mean()

        # Convert to global indices if mapping provided
        if global_index_offset is not None:
            defect_global = global_index_offset[local_defect_idx]
            boundary_global = global_index_offset[boundary_local] if len(boundary_local) > 0 else np.array([], dtype=int)
        else:
            defect_global = local_defect_idx
            boundary_global = boundary_local

        regions.append({
            'center': center,
            'boundary_indices': boundary_global,
            'defect_indices': defect_global,
            'defect_indices_local': local_defect_idx,
            'severity': severity,
        })

    regions.sort(key=lambda r: r['severity'], reverse=True)
    logger.info(f"Detected {len(regions)} defect region(s) after filtering")
    return regions


# ── Anchor View Rendering + Repair Mask ──────────────────────────────────────

def render_anchor_views(
    gaussians,
    camera_params: dict,
    pipe_config,
    bg_color,
    quality_scores: np.ndarray,
    quality_threshold: float = 0.2,
    mask_dilation_px: int = 5,
    object_id: int | None = None,
    object_anchors: np.ndarray = None,
) -> dict:
    """Render anchor view via ObjectGS and build the repair mask.

    Repair mask = union of:
      - Type A: Hole pixels (low alpha inside object's projected bounding box)
      - Type B: Degraded anchor pixels (from Anchor ID map + quality scores)

    Args:
        quality_scores: GLOBAL-space per-anchor quality scores.
        object_anchors: (N_obj, 3) object anchor positions for bbox hole detection.

    Returns dict: 'rgb_full', 'rgb_isolated', 'rgb', 'alpha', 'normal', 'depth',
                  'repair_mask', 'anchor_id_map', 'camera_params'
    """
    import torch
    from target_replenishment.core.objectgs_bridge import (
        create_virtual_camera, render_view, get_anchor_positions,
        build_anchor_id_map,
    )

    H, W = camera_params['height'], camera_params['width']
    R, T, K = camera_params['R'], camera_params['T'], camera_params['K']
    cam = create_virtual_camera(R, T, K, W, H)

    # Render 1: Isolated object
    result_obj = render_view(gaussians, cam, pipe_config, bg_color, object_label_id=object_id)
    # Render 2: Full scene context
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
    if normal_np.ndim > 2 and normal_np.shape[0] == 3:
        normal_np = np.transpose(normal_np, (1, 2, 0))
    depth_np = result_obj['depth'].squeeze(0).cpu().numpy() if result_obj['depth'] is not None else None

    anchor_id_map = build_anchor_id_map(result_obj, H, W, n_anchors)

    # ── Repair mask ──

    # Type A: Holes via object bounding box projection (strict: alpha < 0.1)
    hole_mask = _detect_object_holes(alpha_np, object_anchors, R, T, K, W, H)

    # Type B: Degraded anchors via Anchor ID map (vectorized)
    degraded_mask = np.zeros((H, W), dtype=np.uint8)
    valid_px = anchor_id_map >= 0
    aids = anchor_id_map[valid_px] if valid_px.any() else np.array([], dtype=int)
    if len(aids) > 0:
        local_scores = quality_scores[aids]
        # Dynamically scale defect threshold per-view based purely on visible geometry
        effective_threshold = local_scores.mean() + 1.0 * local_scores.std()
            
        is_degraded = (local_scores > effective_threshold) & (alpha_np[valid_px] > 0.3)
        degraded_mask[valid_px] = is_degraded.astype(np.uint8)

    repair_mask = np.clip(hole_mask + degraded_mask, 0, 1).astype(np.uint8)

    if mask_dilation_px > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (mask_dilation_px * 2 + 1, mask_dilation_px * 2 + 1)
        )
        repair_mask = cv2.dilate(repair_mask, kernel, iterations=1)
        repair_mask = cv2.morphologyEx(repair_mask, cv2.MORPH_CLOSE, kernel)

    # Mask size cap removed to align with PAInpainter's requirement for
    # large solid canvas masks during structural completion.

    logger.info(
        f"Repair mask: {repair_mask.sum()}/{H*W} px "
        f"({100 * repair_mask.sum() / (H*W):.1f}%) — "
        f"holes={hole_mask.sum()}, degraded={degraded_mask.sum()}"
    )

    return {
        'rgb_full': rgb_full_np,
        'rgb_isolated': rgb_isolated_np,
        'rgb': rgb_full_np,
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
    assert T.size == 3, f"T must have size 3, got {T.size}"
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


def _detect_object_holes(
    alpha_np: np.ndarray,
    object_anchors: np.ndarray | None,
    R, T, K, W, H,
) -> np.ndarray:
    """Detect holes by finding empty areas immediately adjacent or inside the object.

    Instead of a Convex Hull (which incorrectly fills large natural empty spaces like 
    the curve of a crescent), we dilate the existing valid alpha mask. Pockets of low
    alpha within this dilated perimeter are flagged as holes.
    """
    hole_mask = np.zeros((H, W), dtype=np.uint8)

    # 1. Create binary mask of existing geometry
    valid_geometry = (alpha_np > 0.3).astype(np.uint8)
    
    if valid_geometry.sum() < 10:
        return hole_mask
        
    # 2. Dilate the geometry to form a tight boundary "halo"
    # The halo bridges small physical cracks but rejects massive empty concavities
    k_size = max(5, int(min(W, H) * 0.02))  # ~2% of image size, e.g. 21px for 1080p
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    
    # Close first to bridge small gaps, then dilate
    closed_geom = cv2.morphologyEx(valid_geometry, cv2.MORPH_CLOSE, kernel)
    dilated_geom = cv2.dilate(closed_geom, kernel, iterations=1)
    
    # 3. Holes = empty space (alpha < 0.1) INSIDE the dilated boundary
    hole_mask = ((alpha_np < 0.1) & (dilated_geom == 1)).astype(np.uint8)

    return hole_mask


# ── Full Pipeline ────────────────────────────────────────────────────────────

def run_anchor_detection(
    model_path: str,
    output_dir: str | None = None,
    iteration: int = -1,
    quality_threshold: float = 0.2,
    render_size: int = 512,
    target_object_ids: list | None = None,
    scoring_views: int = 4,
    mask_dilation_px: int = 35,
) -> dict:
    """Run the complete object enhancement detection pipeline.

    Flow: load model → build perspective graph → for each target object:
          multi-view quality scoring → detect defect regions (global indices)
          → select anchor view with frontality → render + mask.

    Args:
        target_object_ids: Optional list of object IDs to process. None = all objects.
        scoring_views: Number of training cameras to use for multi-view scoring.

    Returns:
        Dict keyed by object_id, each containing:
            'quality_scores'   — (N_obj,) per-anchor scores
            'defect_regions'   — list of defect dicts (GLOBAL indices)
            'camera_params'    — anchor view camera + neighbor list
            'renders'          — rendered views + repair mask
            'obj_mask'         — (N_global,) bool mask of this object's anchors
    """
    import torch
    from target_replenishment.core.objectgs_bridge import load_gaussians, get_anchor_positions
    from target_replenishment.core.perspective_graph import (
        build_perspective_graph, select_anchor_views, get_top_k_views_for_object,
    )

    gaussians, pipe_config = load_gaussians(model_path, iteration)
    anchor_xyz_global = get_anchor_positions(gaussians)
    bg_color = torch.zeros(3, dtype=torch.float32, device="cuda")

    cameras_json = Path(model_path) / "cameras.json"
    graph = build_perspective_graph(str(cameras_json), anchor_xyz_global, overlap_method='visibility')

    static_signals = compute_static_signals(gaussians)

    labels = gaussians.label_ids.squeeze(-1).cpu().numpy()
    process_ids = target_object_ids if target_object_ids is not None else np.unique(labels).tolist()

    results = {}

    for obj_id in process_ids:
        logger.info(f"--- Processing Object ID {obj_id} ---")
        obj_mask = (labels == obj_id)
        anchor_xyz = anchor_xyz_global[obj_mask]
        global_indices = np.where(obj_mask)[0]  # mapping: local → global

        if len(anchor_xyz) < 10:
            logger.warning(f"Object {obj_id} has too few anchors ({len(anchor_xyz)}). Skipping.")
            continue

        # Multi-view quality scoring from top-K training cameras
        top_cams = get_top_k_views_for_object(graph, anchor_xyz, k=scoring_views)
        if not top_cams:
            logger.warning(f"No training cameras see object {obj_id}. Skipping.")
            continue

        cam_param_list = [
            {'R': c['R'], 'T': c['T'], 'K': c['K'].copy(), 'width': c['width'], 'height': c['height']}
            for c in top_cams
        ]

        quality_scores_global = compute_multiview_quality(
            gaussians, pipe_config, cam_param_list,
            object_id=obj_id, static_signals=static_signals,
        )
        quality_scores_obj = quality_scores_global[obj_mask]

        # Defect detection (returns GLOBAL indices via global_indices mapping)
        defect_regions = detect_defect_regions(
            anchor_xyz, quality_scores_obj, quality_threshold,
            global_index_offset=global_indices,
        )

        if not defect_regions:
            logger.warning(f"No degraded clusters for {obj_id}. Using object center to check for structural holes.")
            primary = {
                'center': anchor_xyz.mean(axis=0).astype(np.float32),
                'boundary_indices': np.array([]),
                'defect_indices': np.array([]),
                'severity': 0.0
            }
            defect_regions = [primary]
        else:
            primary = defect_regions[0]
        defect_normal = _estimate_defect_normal(anchor_xyz_global, primary)
        anchor_views = select_anchor_views(graph, primary['center'], k=4, defect_normal=defect_normal)
        anchor_cam = anchor_views[0]

        camera_params = {
            'R': anchor_cam['R'], 'T': anchor_cam['T'], 'K': anchor_cam['K'],
            'width': anchor_cam['width'], 'height': anchor_cam['height'],
            'position': anchor_cam['position'],
            'look_at': primary['center'],
            'neighbors': anchor_views[1:],
        }

        renders = render_anchor_views(
            gaussians, camera_params, pipe_config, bg_color,
            quality_scores_global, quality_threshold=quality_threshold,
            object_id=obj_id, object_anchors=anchor_xyz,
            mask_dilation_px=mask_dilation_px,
        )

        if output_dir:
            _save_outputs(renders, str(Path(output_dir) / f"obj_{obj_id}"))

        results[obj_id] = {
            'quality_scores': quality_scores_obj,
            'defect_regions': defect_regions,
            'camera_params': camera_params,
            'renders': renders,
            'obj_mask': obj_mask,
        }

    return results


def _estimate_defect_normal(anchor_xyz_global, defect_region):
    """Estimate the surface normal at a defect region for frontality scoring.

    Uses PCA on boundary anchors: the smallest principal component
    is approximately the surface normal.
    """
    boundary_idx = defect_region['boundary_indices']
    if len(boundary_idx) < 5:
        return None

    boundary_pts = anchor_xyz_global[boundary_idx]
    centered = boundary_pts - boundary_pts.mean(axis=0)
    try:
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        # Smallest eigenvector ≈ surface normal
        normal = Vt[2]
        # Orient normal to point away from object center (towards outside/camera)
        object_center = anchor_xyz_global.mean(axis=0)
        to_outside = boundary_pts.mean(axis=0) - object_center
        if np.dot(normal, to_outside) < 0:
            normal = -normal
        return normal.astype(np.float32)
    except np.linalg.LinAlgError:
        return None


def _save_outputs(renders: dict, output_dir: str):
    """Save anchor view images to disk."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(out / "anchor_rgb.png"), cv2.cvtColor(renders['rgb_full'], cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out / "anchor_rgb_isolated.png"), cv2.cvtColor(renders['rgb_isolated'], cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out / "anchor_mask.png"), (renders['repair_mask'] * 255).astype(np.uint8))

    normal_np = renders['normal']
    if normal_np is not None and normal_np.size > 0:
        normal_vis = ((normal_np + 1) * 0.5 * 255).clip(0, 255).astype(np.uint8)
        cv2.imwrite(str(out / "anchor_normal.png"), normal_vis)

    if renders['depth'] is not None:
        depth = renders['depth'].squeeze()
        depth_vis = (depth / (depth.max() + 1e-6) * 255).astype(np.uint8)
        cv2.imwrite(str(out / "anchor_depth.png"), depth_vis)

    aid = renders['anchor_id_map']
    if aid.max() > aid.min():
        aid_vis = ((aid - aid.min()) / (aid.max() - aid.min() + 1e-6) * 255).astype(np.uint8)
        cv2.imwrite(str(out / "anchor_id_map.png"), cv2.applyColorMap(aid_vis, cv2.COLORMAP_TURBO))

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

    parser = argparse.ArgumentParser(description="VRoom Anchor Renderer — Object Enhancement")
    parser.add_argument("--model_path", required=True, help="ObjectGS training output directory")
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--output_dir", default="anchor_output")
    parser.add_argument("--quality_threshold", type=float, default=0.2)
    parser.add_argument("--render_size", type=int, default=512)
    parser.add_argument("--object_ids", type=int, nargs='+', default=None, help="Specific object IDs to process (default: all)")
    parser.add_argument("--scoring_views", type=int, default=4, help="Number of training views for multi-view scoring")
    args = parser.parse_args()

    results = run_anchor_detection(
        args.model_path,
        output_dir=args.output_dir,
        iteration=args.iteration,
        quality_threshold=args.quality_threshold,
        render_size=args.render_size,
        target_object_ids=args.object_ids,
        scoring_views=args.scoring_views,
    )

    for obj_id, result in results.items():
        if result['defect_regions']:
            print(f"\n--- Object {obj_id} ---")
            print(f"Detected {len(result['defect_regions'])} defect region(s).")
            print(f"Primary defect: {result['defect_regions'][0]['center']}")
            print(f"Global defect anchors: {len(result['defect_regions'][0]['defect_indices'])}")
            print(f"Repair mask: {result['renders']['repair_mask'].sum()} px")
        else:
            print(f"\n--- Object {obj_id} --- Healthy")
