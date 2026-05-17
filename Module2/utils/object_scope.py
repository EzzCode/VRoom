"""


"""
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml
from gstrain.vroom_core import GaussianModel
from gstrain.vroom_core.models.semantics import SemanticCodec

import logging
logger = logging.getLogger(__name__)


# ── Model / anchor helpers ────────────────────────────────────────────────────

def get_anchor_positions(gaussians: GaussianModel) -> np.ndarray:
    """Return anchor world positions as (N, 3) float32 numpy."""
    return gaussians.get_anchor.detach().cpu().numpy().astype(np.float32)


def get_label_ids(gaussians: GaussianModel) -> np.ndarray:
    """Return per-anchor label IDs as (N,) int64 numpy."""
    return gaussians.label_ids.detach().cpu().numpy().reshape(-1).astype(np.int64)

def load_cameras(cameras_json):
    cameras_json = Path(cameras_json)
    if not cameras_json.exists():
        raise FileNotFoundError(f"Expected cameras.json at {cameras_json}")
    with open(cameras_json, "r", encoding="utf-8") as f:
        try:
            cameras = json.load(f)
        except Exception as e:
            raise ValueError(f"Error parsing cameras.json at {cameras_json}: {e}")

    result: list[dict] = []
    for camera in cameras:
        rotation = np.asarray(camera["rotation"], dtype=np.float32)
        position = np.asarray(camera["position"], dtype=np.float32)
        width = int(camera["width"])
        height = int(camera["height"])
        fx = float(camera["fx"])
        fy = float(camera["fy"])
        R = rotation.T                      # camera-to-world → world-to-camera
        T = -R @ position                    # translation vector
        K = np.array(
            [[fx, 0.0, width / 2.0],
             [0.0, fy, height / 2.0],
             [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        result.append({
            "id": camera.get("id"),
            "image_name": camera.get("img_name"),
            "R": R.astype(np.float32),
            "T": T.astype(np.float32),
            "K": K,
            "position": position.astype(np.float32),
            "width": width,
            "height": height,
            "fx": fx,
            "fy": fy,
        })

    logger.info("Loaded %d cameras from %s", len(result), cameras_json)
    return result


def load_gaussians(model_path):
    model_path = Path(model_path)
    config_path = model_path / "config.yaml"

    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            try:
                config = yaml.load(f, Loader=yaml.FullLoader)
            except Exception:
                raise ValueError(f"Error parsing YAML config at {config_path}")
    else:
        raise FileNotFoundError(f"Expected config.yaml at {model_path}")

    model_params = config.get("model_params", {})
    pipeline_params = config.get("pipeline_params", {})

    # ── Load model ──  
    # model_params: { 'model_config': { 'name': 'GaussianModel', 'kwargs': { ... } } }
    model_config = model_params.get("model_config")
    if not model_config:
        raise KeyError("Could not find 'model_config' in model_params")

    kwargs = model_config.get("kwargs", {})
    if kwargs is None:
        raise ValueError(f"model_config.kwargs is missing in {config_path}")

    model = GaussianModel(**kwargs)                      # construct empty model
    model.load_ply(str(model_path / "point_cloud.ply"))  # load anchors + label_ids
    model.load_mlp_checkpoints(str(model_path))          # load MLPs
    if model.label_ids is None:
        raise ValueError(
            f"Model at {model_path} does not have label_ids; cannot compute object scope."
        )
    model.id_encoder = SemanticCodec.from_labels(model.label_ids.view(-1))
    model.explicit_gs = False   # training flag
    model.weed_ratio = 0.0      # disable anchor pruning
    model.set_eval()

    logger.info(
        "Loaded GaussianModel from %s with %d anchors and label IDs: %s",
        model_path, len(model.get_anchor),
        sorted(np.unique(model.label_ids.cpu().numpy()).tolist()),
    )
    return model, pipeline_params


# ── Camera geometry helpers ───────────────────────────────────────────────────

def _count_visible_anchors(cam: dict, points: np.ndarray) -> np.ndarray:
    """Boolean mask of ``points`` that project inside the camera frustum."""
    R, T, K = cam["R"], cam["T"], cam["K"]
    width, height = int(cam["width"]), int(cam["height"])
    cam_pts = (R @ points.T).T + T.reshape(1, 3)
    z = cam_pts[:, 2]
    valid = z > 0.01
    u = K[0, 0] * cam_pts[:, 0] / (z + 1e-8) + K[0, 2]
    v = K[1, 1] * cam_pts[:, 1] / (z + 1e-8) + K[1, 2]
    return valid & (u >= 0) & (u < width) & (v >= 0) & (v < height)


def _normalize(vec: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if not np.isfinite(norm) or norm < 1e-8:
        return np.asarray(fallback, dtype=np.float32)
    return (vec / norm).astype(np.float32)


def _estimate_up(cameras: list[dict]) -> np.ndarray:
    """Estimate world up-vector from the average camera image-up direction."""
    ups = []
    for cam in cameras:
        R = np.asarray(cam["R"], dtype=np.float32)  # world-to-camera
        # Row 1 of R (world-to-cam) is the camera's Y axis in world coords;
        # image up points in the -Y camera direction.
        ups.append(-R[1, :])
    if not ups:
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return _normalize(
        np.mean(np.asarray(ups, dtype=np.float32), axis=0),
        np.array([0.0, 0.0, 1.0], dtype=np.float32),
    )


def _orbit_base_dir(
    cam_centers: np.ndarray,
    centroid: np.ndarray,
    up: np.ndarray,
) -> np.ndarray:
    """Horizontal direction pointing toward the median camera location."""
    up = _normalize(up, np.array([0.0, 0.0, 1.0], dtype=np.float32))
    fallback = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    dirs = np.asarray(cam_centers, dtype=np.float32) - np.asarray(centroid, dtype=np.float32).reshape(1, 3)
    # Project out the up component to stay horizontal
    dirs = dirs - (dirs @ up).reshape(-1, 1) * up.reshape(1, 3)
    norms = np.linalg.norm(dirs, axis=1)
    dirs = dirs[norms > 1e-6]
    base = np.median(dirs, axis=0) if len(dirs) else fallback
    base = base - float(np.dot(base, up)) * up
    if np.linalg.norm(base) < 1e-6:
        alt = fallback if abs(float(np.dot(fallback, up))) < 0.9 else np.array([0.0, 1.0, 0.0], dtype=np.float32)
        base = alt - float(np.dot(alt, up)) * up
    return _normalize(base, fallback)


# ── Data container ────────────────────────────────────────────────────────────

@dataclass
class ObjectScope:
    """Geometric metadata for one object label, consumed by downstream stages."""
    object_label_id: int
    n_anchors: int
    centroid: np.ndarray             # (3,)  float32 — world frame
    aabb_min: np.ndarray             # (3,)  float32 — axis-aligned min corner
    aabb_max: np.ndarray             # (3,)  float32 — axis-aligned max corner
    obb_extents: np.ndarray          # (3,)  float32 — sqrt(PCA eigenvalues), descending
    radius: float                    # orbit radius (world units)
    up: np.ndarray                   # (3,)  float32 — world up vector
    base_dir: np.ndarray             # (3,)  float32 — horizontal base direction
    visible_cam_indices: list        # indices into cameras list
    cam_centers_visible: np.ndarray  # (M, 3) float32
    cameras: list = field(default_factory=list)   # parsed camera dicts


# ── Main entry point ──────────────────────────────────────────────────────────

def compute_object_scope(
    path,
    object_label_id,
    visibility_min_anchors: int = 50,
) -> ObjectScope:
    """Derive :class:`ObjectScope` for ``object_label_id`` from a trained model.

    Args:
        path: directory containing config.yaml, point_cloud.ply, cameras.json.
        object_label_id: integer label to scope.
        visibility_min_anchors: minimum anchors visible to count a camera.

    Returns:
        Populated :class:`ObjectScope`.
    """
    model_path = Path(path)
    cameras_json = model_path / "cameras.json"
    if not cameras_json.exists():
        raise FileNotFoundError(f"Expected cameras.json at {cameras_json}")

    ply_path = model_path / "point_cloud.ply"
    if not ply_path.exists():
        raise FileNotFoundError(f"PLY not found: {ply_path}")

    # ── Load model ──
    gaussians, pipe_config = load_gaussians(str(model_path))
    all_xyz = get_anchor_positions(gaussians)
    label_ids = get_label_ids(gaussians)

    # ── Filter to object anchors ──
    object_mask = (label_ids == int(object_label_id))
    if not object_mask.any():
        raise ValueError(
            f"object_label_id={object_label_id} has no anchors in {model_path}. "
            f"Available labels: {sorted(np.unique(label_ids).tolist())}"
        )
    total_objects = object_mask.sum()
    logger.info(f"Object label {object_label_id} has {total_objects} anchors in {model_path}.")

    anchor_xyz = all_xyz[object_mask].astype(np.float32)
    centroid = anchor_xyz.mean(axis=0).astype(np.float32)
    # claculate axis-aligned bounding box (AABB) of the object anchors
    aabb_min = anchor_xyz.min(axis=0).astype(np.float32)
    aabb_max = anchor_xyz.max(axis=0).astype(np.float32)

    # calculate PCA principal axes of the object anchors to get an oriented bounding box (OBB)
    centered = anchor_xyz - centroid
    eigen_values = np.linalg.eigvalsh(centered.T @ centered / max(len(anchor_xyz) - 1, 1))

    # obb extents are used as a rough size estimate for the object
    # sqrt of eigenvalues gives the standard deviation along each principal axis 
    obb_extents = np.sqrt(np.clip(eigen_values[np.argsort(eigen_values)[::-1]], 0.0, None)).astype(np.float32)

    # ── Camera analysis ──
    cameras = load_cameras(str(cameras_json))

    visible_indices: list[int] = []
    for ci, cam in enumerate(cameras):
        if int(_count_visible_anchors(cam, anchor_xyz).sum()) >= visibility_min_anchors:
            visible_indices.append(ci)

    if not visible_indices:
        # Relaxed fallback: any camera that sees ≥1 anchor
        for ci, cam in enumerate(cameras):
            if int(_count_visible_anchors(cam, anchor_xyz).sum()) >= 1:
                visible_indices.append(ci)
        logger.warning(
            "No camera sees ≥%d anchors of object %s; relaxed to ≥1 (%d cams).",
            visibility_min_anchors, object_label_id, len(visible_indices),
        )
    if not visible_indices:
        raise RuntimeError(f"No training camera sees object {object_label_id} at all.")

    cam_centers_all = np.array([c["position"] for c in cameras], dtype=np.float32)
    cam_centers_vis = cam_centers_all[visible_indices]

    up_W = _estimate_up(cameras)
    base_dir_W = _orbit_base_dir(cam_centers_vis, centroid, up_W)

    # Orbit radius: median distance from visible cameras to centroid
    dists = np.linalg.norm(cam_centers_vis - centroid.reshape(1, 3), axis=1)
    radius = float(np.median(dists)) if len(dists) > 0 else 2.0 * float(np.linalg.norm(aabb_max - aabb_min))
    radius = max(radius, 1e-3)

    scope = ObjectScope(
        object_label_id=int(object_label_id),
        n_anchors=total_objects,
        centroid=centroid,
        aabb_min=aabb_min,
        aabb_max=aabb_max,
        obb_extents=obb_extents,
        radius=radius,
        up=up_W.astype(np.float32),
        base_dir=base_dir_W.astype(np.float32),
        visible_cam_indices=visible_indices,
        cam_centers_visible=cam_centers_vis.astype(np.float32),
        cameras=cameras,
    )

    logger.info(
        "ObjectScope obj=%d: %d anchors | centroid=%s | radius=%.3f | visible_cams=%d/%d",
        scope.object_label_id, scope.n_anchors,
        np.round(scope.centroid, 3).tolist(), scope.radius,
        len(scope.visible_cam_indices), len(cameras),
    )
    return scope