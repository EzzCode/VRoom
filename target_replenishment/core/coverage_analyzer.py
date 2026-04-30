"""
Coverage Analyzer — Detect unseen hemisphere gaps in training camera coverage.

Analyzes the spatial distribution of training cameras relative to an object's
centroid to identify azimuth sectors with no/low coverage. These gaps correspond
to the unseen backside of objects (e.g., couch against a wall).

Public API:
    analyze_coverage(object_anchors, training_cameras, ...) -> CoverageResult
"""

__all__ = ['analyze_coverage']

import logging
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CoverageResult:
    """Result of coverage gap analysis for a single object."""
    object_center: np.ndarray          # (3,) centroid
    object_radius: float               # bounding sphere radius
    coverage_map: np.ndarray           # (n_bins,) per-azimuth coverage fraction
    gap_azimuths: list                 # azimuth angles (radians) with low coverage
    best_input_cam: dict               # training camera with most object visibility
    input_azimuth: float               # azimuth of best_input_cam relative to object
    up_vector: np.ndarray              # (3,) estimated world up
    orbit_radius: float                # median camera distance to object


def analyze_coverage(
    object_anchors: np.ndarray,
    training_cameras: list,
    n_bins: int = 36,
    gap_threshold: float = 0.1,
    up_axis: str = 'auto',
) -> CoverageResult:
    """Analyze training camera coverage around an object.

    Args:
        object_anchors: (N, 3) float32 — object anchor positions in world space.
        training_cameras: list of camera dicts with 'R', 'T', 'K', 'position',
                          'width', 'height' keys.
        n_bins: Number of azimuth bins (default 36 = 10° each).
        gap_threshold: Bins below this fraction of max coverage are gaps.
        up_axis: 'x', 'y', 'z', or 'auto' (detect from camera distribution).

    Returns:
        CoverageResult with gap analysis.
    """
    center = object_anchors.mean(axis=0).astype(np.float32)
    dists_to_center = np.linalg.norm(object_anchors - center, axis=1)
    radius = float(np.quantile(dists_to_center, 0.98))

    logger.info(f"Object: center={center}, radius={radius:.3f}, {len(object_anchors)} anchors")

    # ── Estimate up vector ──
    up = _estimate_up_vector(training_cameras, up_axis)
    logger.info(f"Up vector: {up} (method={up_axis})")

    # ── Compute camera positions and azimuths ──
    cam_positions = np.array([c['position'] for c in training_cameras], dtype=np.float32)
    cam_dists = np.linalg.norm(cam_positions - center, axis=1)
    orbit_radius = float(np.median(cam_dists))

    # Project camera-to-object vectors onto horizontal plane
    to_cams = cam_positions - center  # (N_cams, 3)
    horizontal = to_cams - np.outer(np.dot(to_cams, up), up)  # remove vertical component

    # Compute azimuths using two horizontal basis vectors
    basis_h, basis_v = _compute_horizontal_basis(up)
    az_components_x = horizontal @ basis_h
    az_components_y = horizontal @ basis_v
    cam_azimuths = np.arctan2(az_components_y, az_components_x)

    # ── Build coverage histogram ──
    bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    coverage_map = np.zeros(n_bins, dtype=np.float32)

    # Count visible anchors per camera, weight by azimuth bin
    for ci, cam in enumerate(training_cameras):
        visible = _count_visible(cam, object_anchors)
        n_visible = visible.sum()
        if n_visible == 0:
            continue
        bin_idx = np.digitize(cam_azimuths[ci], bin_edges) - 1
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)
        coverage_map[bin_idx] += n_visible

    # Normalize
    if coverage_map.max() > 0:
        coverage_map /= coverage_map.max()

    # ── Identify gaps ──
    gap_mask = coverage_map < gap_threshold
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    gap_azimuths = bin_centers[gap_mask].tolist()

    n_gap_bins = gap_mask.sum()
    gap_degrees = n_gap_bins * (360.0 / n_bins)
    logger.info(
        f"Coverage: {n_bins - n_gap_bins}/{n_bins} bins covered, "
        f"gap={gap_degrees:.0f}° ({n_gap_bins} bins)"
    )

    # ── Select best input camera ──
    camera_scores = []
    camera_metrics = []
    for cam in training_cameras:
        score, metrics = _score_input_camera(cam, object_anchors)
        camera_scores.append(score)
        camera_metrics.append(metrics)

    if np.max(camera_scores) <= 0.0:
        visibility_counts = [int(_count_visible(cam, object_anchors).sum()) for cam in training_cameras]
        best_idx = int(np.argmax(visibility_counts))
        best_metrics = {
            'vis_ratio': float(visibility_counts[best_idx] / max(len(object_anchors), 1)),
            'area_ratio': 0.0,
            'edge_penalty': 1.0,
            'center_penalty': 1.0,
            'score': 0.0,
        }
    else:
        best_idx = int(np.argmax(camera_scores))
        best_metrics = camera_metrics[best_idx]

    best_cam = training_cameras[best_idx]
    input_azimuth = float(cam_azimuths[best_idx])

    logger.info(
        f"Best input camera: idx={best_idx} "
        f"(id={best_cam.get('id', '?')}), "
        f"azimuth={np.degrees(input_azimuth):.1f}°, "
        f"score={best_metrics['score']:.3f}, "
        f"vis={best_metrics['vis_ratio']:.3f}, "
        f"area={best_metrics['area_ratio']:.3f}, "
        f"edge_pen={best_metrics['edge_penalty']:.3f}, "
        f"center_pen={best_metrics['center_penalty']:.3f}"
    )

    return CoverageResult(
        object_center=center,
        object_radius=radius,
        coverage_map=coverage_map,
        gap_azimuths=gap_azimuths,
        best_input_cam=best_cam,
        input_azimuth=input_azimuth,
        up_vector=up,
        orbit_radius=orbit_radius,
    )


def _estimate_up_vector(cameras: list, method: str = 'auto') -> np.ndarray:
    """Estimate the world up vector.

    Methods:
      'x' / 'y' / 'z': force a canonical axis.
      'auto' (default): average each camera's *own* world-up direction. In
          COLMAP convention, R rows are (right, -up, forward), so each
          camera's world-up is -R[1, :]. Averaging across all training
          cameras gives a continuous up vector that:
            - is NOT axis-snapped (works for tilted scenes);
            - is identical across objects (uses the SAME training cameras);
            - matches whatever frame Zero123++'s elevation is measured in
              IF the dataset was captured roughly with cameras upright.
          This replaces the prior position-spread heuristic that snapped
          to {x,y,z} based on per-object camera distribution and produced
          different up axes for different objects in the same scene.
      'spread': legacy axis-snap heuristic (kept for parity / debugging).
    """
    if method == 'x':
        return np.array([1, 0, 0], dtype=np.float32)
    elif method == 'y':
        return np.array([0, 1, 0], dtype=np.float32)
    elif method == 'z':
        return np.array([0, 0, 1], dtype=np.float32)

    if method == 'spread':
        positions = np.array([c['position'] for c in cameras], dtype=np.float32)
        spreads = positions.std(axis=0)
        up_idx = int(np.argmin(spreads))
        up = np.zeros(3, dtype=np.float32)
        up[up_idx] = 1.0
        mean_pos = positions.mean(axis=0)
        if mean_pos[up_idx] < 0:
            up[up_idx] = -1.0
        logger.info(
            f"Up vector (spread heuristic): {up} "
            f"(spreads x={spreads[0]:.2f} y={spreads[1]:.2f} z={spreads[2]:.2f})"
        )
        return up

    # Default 'auto': camera-local-up consensus.
    local_ups = []
    for c in cameras:
        R = np.asarray(c['R'], dtype=np.float32)
        if R.shape != (3, 3):
            continue
        # COLMAP: R rows = (right, -up, forward) → world-up = -R[1]
        local_ups.append(-R[1, :])
    if not local_ups:
        logger.warning("No valid camera R matrices — falling back to +Y up.")
        return np.array([0, 1, 0], dtype=np.float32)

    up = np.mean(np.stack(local_ups, axis=0), axis=0)
    n = float(np.linalg.norm(up))
    if n < 1e-6:
        # Cameras point in conflicting up directions — rare; use principal axis
        # of the camera-local-up cloud (largest eigenvector of its covariance).
        ups_arr = np.stack(local_ups, axis=0)
        cov = ups_arr.T @ ups_arr
        eigvals, eigvecs = np.linalg.eigh(cov)
        up = eigvecs[:, -1]
        # Sign-align with majority of local ups
        if np.dot(up, ups_arr.mean(axis=0)) < 0:
            up = -up
        n = float(np.linalg.norm(up))
        logger.warning(
            "Camera local ups cancel out (mean norm=0). Falling back to "
            "principal axis: %s", up,
        )
    up = (up / max(n, 1e-8)).astype(np.float32)

    # Diagnostic: how axis-aligned is the result, and how concentrated were
    # the local ups? Spread close to 1 means cameras agree (axis-aligned
    # capture); much less means tilted scene.
    ups_arr = np.stack(local_ups, axis=0)
    agreement = float(np.mean(ups_arr @ up))
    logger.info(
        "Up vector (camera-local consensus): [%.3f, %.3f, %.3f] "
        "(agreement=%.3f over %d cams)",
        up[0], up[1], up[2], agreement, len(local_ups),
    )
    return up


def _compute_horizontal_basis(up: np.ndarray):
    """Compute two orthonormal horizontal basis vectors perpendicular to up."""
    # Pick a vector not parallel to up
    if abs(up[0]) < 0.9:
        arbitrary = np.array([1, 0, 0], dtype=np.float32)
    else:
        arbitrary = np.array([0, 1, 0], dtype=np.float32)

    basis_h = np.cross(up, arbitrary)
    basis_h /= np.linalg.norm(basis_h)
    basis_v = np.cross(up, basis_h)
    basis_v /= np.linalg.norm(basis_v)
    return basis_h, basis_v


def _count_visible(cam: dict, points: np.ndarray) -> np.ndarray:
    """Return boolean array of which points are visible in this camera."""
    R, T = cam['R'], cam['T']
    K = cam['K']
    W, H = cam['width'], cam['height']
    cam_pts = (R @ points.T).T + T.flatten()[np.newaxis, :]
    z = cam_pts[:, 2]
    valid = z > 0.01
    u = K[0, 0] * cam_pts[:, 0] / (z + 1e-8) + K[0, 2]
    v = K[1, 1] * cam_pts[:, 1] / (z + 1e-8) + K[1, 2]
    return valid & (u >= 0) & (u < W) & (v >= 0) & (v < H)


def _score_input_camera(cam: dict, points: np.ndarray):
    """Score camera suitability for novel-view input rendering.

    The score favors cameras where the object is visible, reasonably large,
    centered, and not clipped near image borders.
    """
    R, T = cam['R'], cam['T']
    K = cam['K']
    W, H = cam['width'], cam['height']

    cam_pts = (R @ points.T).T + T.flatten()[np.newaxis, :]
    z = cam_pts[:, 2]
    valid = z > 0.01
    if not np.any(valid):
        return 0.0, {
            'score': 0.0,
            'vis_ratio': 0.0,
            'area_ratio': 0.0,
            'edge_penalty': 1.0,
            'center_penalty': 1.0,
        }

    u = K[0, 0] * cam_pts[:, 0] / (z + 1e-8) + K[0, 2]
    v = K[1, 1] * cam_pts[:, 1] / (z + 1e-8) + K[1, 2]
    visible = valid & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    n_visible = int(visible.sum())
    vis_ratio = float(n_visible / max(len(points), 1))

    if n_visible == 0:
        return 0.0, {
            'score': 0.0,
            'vis_ratio': vis_ratio,
            'area_ratio': 0.0,
            'edge_penalty': 1.0,
            'center_penalty': 1.0,
        }

    u_vis = u[visible]
    v_vis = v[visible]
    x0, x1 = float(np.min(u_vis)), float(np.max(u_vis))
    y0, y1 = float(np.min(v_vis)), float(np.max(v_vis))

    bbox_w = max(x1 - x0, 1.0)
    bbox_h = max(y1 - y0, 1.0)
    area_ratio_raw = float((bbox_w * bbox_h) / max(float(W * H), 1.0))

    target_area = 0.10
    area_ratio = float(np.clip(1.0 - abs(area_ratio_raw - target_area) / max(target_area, 1e-6), 0.0, 1.0))

    edge_dist = min(x0, y0, (W - 1.0) - x1, (H - 1.0) - y1)
    margin = 0.12 * min(W, H)
    edge_penalty = float(np.clip((margin - edge_dist) / max(margin, 1e-6), 0.0, 1.0))

    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    dx = cx - (W / 2.0)
    dy = cy - (H / 2.0)
    half_diag = max(np.sqrt((W / 2.0) ** 2 + (H / 2.0) ** 2), 1e-6)
    center_penalty = float(np.clip(np.sqrt(dx * dx + dy * dy) / half_diag, 0.0, 1.0))

    score = (
        0.35 * vis_ratio
        + 0.25 * area_ratio
        + 0.25 * (1.0 - edge_penalty)
        + 0.15 * (1.0 - center_penalty)
    )

    return float(score), {
        'score': float(score),
        'vis_ratio': vis_ratio,
        'area_ratio': area_ratio_raw,
        'edge_penalty': edge_penalty,
        'center_penalty': center_penalty,
    }