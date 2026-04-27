"""
Perspective Graph — Training camera adjacency and view selection for object enhancement.

Builds a perspective graph from ObjectGS training cameras (cameras.json) that models
spatial overlap between viewpoints. Used by run_replenishment to:
  - Select multiple training cameras that see an object (for multi-view scoring)
  - Pick the best anchor view for a defect (with frontality weighting)
  - Provide neighbor views for content propagation

Public API:
    build_perspective_graph(cameras_json_path, anchor_xyz) -> PerspectiveGraph
    select_anchor_views(graph, defect_center, defect_normal, k) -> list of cam dicts
    get_top_k_views_for_object(graph, object_anchors, k) -> list of cam dicts
"""

__all__ = [
    'PerspectiveGraph',
    'build_perspective_graph',
    'select_anchor_views',
    'get_top_k_views_for_object',
]

import json
import logging
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


# ── Perspective Graph ────────────────────────────────────────────────────────

class PerspectiveGraph:
    """Camera adjacency graph built from training views.

    Attributes:
        cameras     : list of camera dicts (R, T, K, position, width, height, ...)
        adjacency   : (N, N) float32 overlap matrix (0 = no overlap, 1 = identical)
        positions   : (N, 3) camera world positions
    """

    def __init__(self, cameras: list, adjacency: np.ndarray):
        self.cameras = cameras
        self.adjacency = adjacency
        self.positions = np.array([c['position'] for c in cameras], dtype=np.float32)
        self._n = len(cameras)

    def get_neighbors(self, cam_idx: int, k: int = 4) -> list:
        """Get k most overlapping cameras for a given camera index."""
        scores = self.adjacency[cam_idx].copy()
        scores[cam_idx] = -1  # exclude self
        top_k = np.argsort(scores)[::-1][:k]
        return [int(i) for i in top_k if scores[i] > 0]

    def get_camera_params(self, cam_idx: int) -> dict:
        """Get camera params dict for a given index."""
        return self.cameras[cam_idx]


def build_perspective_graph(
    cameras_json_path: str,
    anchor_xyz: np.ndarray = None,
    overlap_method: str = 'frustum',
) -> PerspectiveGraph:
    """Build a perspective graph from ObjectGS's cameras.json.

    Args:
        cameras_json_path: Path to cameras.json (written by ObjectGS during training).
        anchor_xyz: Optional (N, 3) anchor positions for visibility-based overlap.
        overlap_method: 'frustum' (angular) or 'visibility' (anchor-based).

    Returns:
        PerspectiveGraph with adjacency matrix.
    """
    path = Path(cameras_json_path)
    if not path.exists():
        raise FileNotFoundError(f"cameras.json not found: {path}")

    with open(path) as f:
        raw_cameras = json.load(f)

    logger.info(f"Loading {len(raw_cameras)} training cameras from {path}")

    cameras = []
    for cam in raw_cameras:
        rot = np.array(cam['rotation'], dtype=np.float32)
        pos = np.array(cam['position'], dtype=np.float32)
        fx = cam['fx']
        fy = cam['fy']
        w = cam['width']
        h = cam['height']

        R = rot.T
        T = -R @ pos

        K = np.array([
            [fx, 0, w / 2.0],
            [0, fy, h / 2.0],
            [0, 0, 1],
        ], dtype=np.float32)

        cameras.append({
            'id': cam['id'],
            'img_name': cam.get('img_name', f'cam_{cam["id"]}'),
            'R': R, 'T': T, 'K': K,
            'position': pos,
            'width': w, 'height': h,
            'fx': fx, 'fy': fy,
        })

    n = len(cameras)

    if overlap_method == 'visibility' and anchor_xyz is not None:
        adjacency = _compute_visibility_overlap(cameras, anchor_xyz)
    else:
        adjacency = _compute_angular_overlap(cameras)

    logger.info(f"Perspective graph: {n} cameras, "
                f"mean adjacency={adjacency.mean():.3f}")

    return PerspectiveGraph(cameras, adjacency)


def _compute_angular_overlap(cameras: list) -> np.ndarray:
    """Pairwise overlap from viewing direction similarity + proximity."""
    n = len(cameras)
    positions = np.array([c['position'] for c in cameras], dtype=np.float32)
    forwards = np.array([c['R'][2, :] for c in cameras], dtype=np.float32)
    forwards /= np.linalg.norm(forwards, axis=1, keepdims=True) + 1e-8

    angular = (forwards @ forwards.T + 1.0) / 2.0
    dists = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=2)
    proximity = 1.0 - dists / (dists.max() + 1e-8)

    adjacency = (0.6 * angular + 0.4 * proximity).astype(np.float32)
    np.fill_diagonal(adjacency, 1.0)
    return adjacency


def _compute_visibility_overlap(cameras: list, anchor_xyz: np.ndarray) -> np.ndarray:
    """Overlap as fraction of shared visible anchors (Jaccard)."""
    n_cams = len(cameras)
    n_anchors = len(anchor_xyz)

    visibility = np.zeros((n_cams, n_anchors), dtype=bool)
    for ci, cam in enumerate(cameras):
        R, T = cam['R'], cam['T']
        K = cam['K']
        W, H = cam['width'], cam['height']
        cam_pts = (R @ anchor_xyz.T).T + T.flatten()[np.newaxis, :]
        z = cam_pts[:, 2]
        valid = z > 0.01
        u = K[0, 0] * cam_pts[:, 0] / (z + 1e-8) + K[0, 2]
        v = K[1, 1] * cam_pts[:, 1] / (z + 1e-8) + K[1, 2]
        visibility[ci] = valid & (u >= 0) & (u < W) & (v >= 0) & (v < H)

    adjacency = np.zeros((n_cams, n_cams), dtype=np.float32)
    for i in range(n_cams):
        for j in range(i, n_cams):
            inter = (visibility[i] & visibility[j]).sum()
            union = (visibility[i] | visibility[j]).sum()
            score = inter / (union + 1e-8)
            adjacency[i, j] = score
            adjacency[j, i] = score
    np.fill_diagonal(adjacency, 1.0)
    return adjacency


# ── View Selection ───────────────────────────────────────────────────────────

def _count_visible_anchors(cam: dict, points: np.ndarray) -> np.ndarray:
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


def get_top_k_views_for_object(
    graph: PerspectiveGraph,
    object_anchors: np.ndarray,
    k: int = 4,
) -> list:
    """Find the top-k training cameras that see the most anchors of a specific object.

    Used for multi-view quality scoring: score from each of these cameras
    and take max per anchor to ensure all sides are checked.

    Returns:
        List of camera param dicts, sorted by visible anchor count (best first).
    """
    counts = []
    for ci, cam in enumerate(graph.cameras):
        vis = _count_visible_anchors(cam, object_anchors)
        counts.append((ci, vis.sum()))

    counts.sort(key=lambda x: x[1], reverse=True)
    top = counts[:k]

    result = []
    for ci, cnt in top:
        if cnt > 0:
            result.append(graph.cameras[ci])

    logger.info(
        f"Top-{k} views for object: "
        + ", ".join(f"cam {graph.cameras[ci]['id']}({cnt})" for ci, cnt in top[:k])
    )
    return result


def select_anchor_views(
    graph: PerspectiveGraph,
    defect_center: np.ndarray,
    k: int = 4,
    defect_normal: np.ndarray = None,
) -> list:
    """Select the best anchor camera + k neighbors for a defect region.

    Scoring:
      - alignment: camera forward direction aligned with to-defect vector
      - proximity: inverse distance to defect
      - frontality: how well the camera faces the defect surface normal
        (if provided, prefers cameras that see the defect head-on)

    Returns:
        List of camera param dicts (best first), length up to k+1.
    """
    scores = np.zeros(graph._n, dtype=np.float32)
    for i, cam in enumerate(graph.cameras):
        pos = cam['position']
        to_defect = defect_center - pos
        dist = np.linalg.norm(to_defect)
        if dist < 1e-8:
            continue
        to_defect_norm = to_defect / dist

        forward = cam['R'][2, :]
        forward = forward / (np.linalg.norm(forward) + 1e-8)

        alignment = max(0, np.dot(forward, to_defect_norm))
        proximity = 1.0 / (1.0 + dist)

        # Frontality: prefer cameras that see the defect surface head-on
        frontality = 1.0
        if defect_normal is not None:
            # Camera-to-defect direction should be anti-parallel to surface normal
            # (camera looks at the front face)
            cam_to_defect = -to_defect_norm  # direction FROM camera TO defect
            frontality = max(0, np.dot(cam_to_defect, defect_normal))

        scores[i] = alignment * proximity * (0.5 + 0.5 * frontality)

    best_idx = int(np.argmax(scores))
    if scores[best_idx] <= 0:
        logger.warning("No camera has good coverage of defect center. Using closest.")
        dists = np.linalg.norm(graph.positions - defect_center, axis=1)
        best_idx = int(np.argmin(dists))

    neighbor_indices = graph.get_neighbors(best_idx, k=k * 3)  # fetch extra to filter

    result = [graph.cameras[best_idx]]
    for ni in neighbor_indices:
        cam = graph.cameras[ni]
        # Validate overlap: neighbor must be able to see the defect center
        R, T = cam['R'], cam['T']
        K, W, H = cam['K'], cam['width'], cam['height']
        cam_pt = R @ defect_center + T.flatten()
        z = cam_pt[2]
        if z > 0:
            u = K[0, 0] * cam_pt[0] / z + K[0, 2]
            v = K[1, 1] * cam_pt[1] / z + K[1, 2]
            if 0 <= u < W and 0 <= v < H:
                result.append(cam)
        
        if len(result) >= k + 1:
            break

    logger.info(
        f"Selected anchor view: cam {graph.cameras[best_idx]['id']} "
        f"(score={scores[best_idx]:.3f}), "
        f"{len(neighbor_indices)} neighbors"
    )
    return result
