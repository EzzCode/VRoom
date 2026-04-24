"""
Content Propagation — Warp inpainted content from anchor view to neighbor views.

PAInpainter §3.2: Uses depth from the anchor view to unproject inpainted pixels
to 3D, then projects them into each neighbor camera. This provides visual priors
for the neighbor views' inpainting, ensuring multi-view consistency.

Public API:
    propagate_to_neighbors(inpainted, depth, anchor_cam, neighbors, mask) -> list
"""

__all__ = ['propagate_to_neighbors']

import logging
import numpy as np
import cv2

logger = logging.getLogger(__name__)


def propagate_to_neighbors(
    inpainted_rgb: np.ndarray,
    depth: np.ndarray,
    anchor_cam: dict,
    neighbor_cams: list,
    repair_mask: np.ndarray,
) -> list:
    """Warp inpainted content from anchor view to each neighbor view.

    Args:
        inpainted_rgb: (H, W, 3) uint8 — inpainted anchor image.
        depth: (H, W) float32 — depth map from anchor view rendering.
        anchor_cam: dict with R, T, K, width, height.
        neighbor_cams: list of camera dicts.
        repair_mask: (H, W) uint8 — binary mask of repaired region.

    Returns:
        List of dicts per neighbor:
            'rgb_warped'    — (H_n, W_n, 3) uint8, warped inpainted content
            'mask_warped'   — (H_n, W_n) uint8, validity mask (1 = has content)
            'camera_params' — the neighbor's camera dict
    """
    if depth is None or depth.size == 0:
        logger.warning("No depth available — skipping content propagation.")
        return [{'rgb_warped': np.zeros_like(inpainted_rgb),
                 'mask_warped': np.zeros(inpainted_rgb.shape[:2], dtype=np.uint8),
                 'camera_params': nc} for nc in neighbor_cams]

    # Flatten depth to 2D if needed
    if depth.ndim == 3:
        depth = depth.squeeze()

    H_a, W_a = inpainted_rgb.shape[:2]
    R_a, T_a, K_a = anchor_cam['R'], anchor_cam['T'].flatten(), anchor_cam['K']

    # Build 3D points from anchor view's repaired pixels
    ys, xs = np.where(repair_mask > 0)
    if len(xs) == 0:
        return [{'rgb_warped': np.zeros_like(inpainted_rgb),
                 'mask_warped': np.zeros(inpainted_rgb.shape[:2], dtype=np.uint8),
                 'camera_params': nc} for nc in neighbor_cams]

    zs = depth[ys, xs].astype(np.float64)
    valid_depth = zs > 1e-4
    xs, ys, zs = xs[valid_depth], ys[valid_depth], zs[valid_depth]
    colors = inpainted_rgb[ys, xs]  # (N, 3)

    # Unproject to camera space
    fx_a, fy_a = K_a[0, 0], K_a[1, 1]
    cx_a, cy_a = K_a[0, 2], K_a[1, 2]
    X_cam = (xs.astype(np.float64) - cx_a) * zs / fx_a
    Y_cam = (ys.astype(np.float64) - cy_a) * zs / fy_a
    pts_cam = np.stack([X_cam, Y_cam, zs], axis=1)  # (N, 3) in anchor cam space

    # To world space: p_world = R_a^T @ (p_cam - T_a)
    R_a_64 = R_a.astype(np.float64)
    T_a_64 = T_a.astype(np.float64)
    pts_world = (R_a_64.T @ (pts_cam - T_a_64[np.newaxis, :]).T).T  # (N, 3)

    results = []
    for nc in neighbor_cams:
        result = _project_to_neighbor(pts_world, colors, nc)
        results.append(result)

    logger.info(f"Propagated to {len(results)} neighbor(s), "
                f"{len(xs)} source pixels")
    return results


def _project_to_neighbor(
    pts_world: np.ndarray,
    colors: np.ndarray,
    neighbor_cam: dict,
) -> dict:
    """Project 3D points into a neighbor camera and splat colors."""
    R_n = neighbor_cam['R'].astype(np.float64)
    T_n = neighbor_cam['T'].flatten().astype(np.float64)
    K_n = neighbor_cam['K']
    W_n, H_n = neighbor_cam['width'], neighbor_cam['height']

    # World to neighbor camera space
    pts_ncam = (R_n @ pts_world.T).T + T_n[np.newaxis, :]
    z_n = pts_ncam[:, 2]
    valid = z_n > 1e-4

    fx_n, fy_n = K_n[0, 0], K_n[1, 1]
    cx_n, cy_n = K_n[0, 2], K_n[1, 2]

    u_n = (fx_n * pts_ncam[valid, 0] / z_n[valid] + cx_n).astype(int)
    v_n = (fy_n * pts_ncam[valid, 1] / z_n[valid] + cy_n).astype(int)

    in_bounds = (u_n >= 0) & (u_n < W_n) & (v_n >= 0) & (v_n < H_n)
    u_valid = u_n[in_bounds]
    v_valid = v_n[in_bounds]
    z_valid = z_n[valid][in_bounds]
    c_valid = colors[valid][in_bounds]

    # Z-buffer splatting: closest point wins
    rgb_warped = np.zeros((H_n, W_n, 3), dtype=np.uint8)
    mask_warped = np.zeros((H_n, W_n), dtype=np.uint8)
    zbuf = np.full((H_n, W_n), np.inf, dtype=np.float64)

    for i in range(len(u_valid)):
        if z_valid[i] < zbuf[v_valid[i], u_valid[i]]:
            zbuf[v_valid[i], u_valid[i]] = z_valid[i]
            rgb_warped[v_valid[i], u_valid[i]] = c_valid[i]
            mask_warped[v_valid[i], u_valid[i]] = 1

    # Small dilation to fill single-pixel gaps from forward warping
    if mask_warped.sum() > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_dilated = cv2.dilate(mask_warped, kernel, iterations=1)
        # Inpaint tiny holes in the warped result
        if mask_dilated.sum() > mask_warped.sum():
            gap_mask = (mask_dilated > 0) & (mask_warped == 0)
            if gap_mask.any():
                rgb_warped = cv2.inpaint(
                    rgb_warped, gap_mask.astype(np.uint8), 3, cv2.INPAINT_TELEA
                )
                mask_warped = mask_dilated

    coverage = mask_warped.sum() / (H_n * W_n) * 100
    logger.debug(f"Neighbor warp: {mask_warped.sum()} px ({coverage:.1f}%) covered")

    return {
        'rgb_warped': rgb_warped,
        'mask_warped': mask_warped,
        'camera_params': neighbor_cam,
    }
