import torch
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union, Any, cast
import numpy as np
from gstrain.vroom_core.config import load_vroom_config
from gstrain.vroom_core.utilities.utils.utils import SemanticsManager
from .gstrain_wrapper import build_vroom_gaussians, load_vroom_checkpoint
from object_refiner.utils.transforms import ObjectFrame
from object_refiner.constants import GAUSSIAN_MODEL_DEFAULTS
from .helpers import normalize

import logging
logger = logging.getLogger(__name__)


def _load_cameras(cameras_json: Union[str, Path]):
    cameras_json = Path(cameras_json)
    if not cameras_json.exists():
        raise FileNotFoundError(f"Expected cameras.json at {cameras_json}")
    with open(cameras_json, "r", encoding="utf-8") as f:
        try:
            cameras = json.load(f)
        except Exception as e:
            raise ValueError(f"Error parsing cameras.json at {cameras_json}: {e}")

    result = []
    for camera in cameras:
        rotation = np.asarray(camera["rotation"], dtype=np.float32)
        position = np.asarray(camera["position"], dtype=np.float32)
        width = int(camera["width"])
        height = int(camera["height"])
        fx = float(camera["fx"])
        fy = float(camera["fy"])
        R = rotation.T                       # camera to world -> world to camera
        T = -R @ position                    # translation vector
        K = np.array(
            [[fx, 0.0, width / 2.0],
             [0.0, fy, height / 2.0],
             [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        result.append({
            "id": camera["id"],
            "image_name": camera["img_name"],
            "position": position,
            "R": R,
            "T": T,
            "K": K,
            "width": width,
            "height": height,
            "fx": fx,
            "fy": fy,
        })

    logger.info("Loaded %d cameras from %s", len(result), cameras_json)
    return result


def load_gaussians(model_path: Union[str, Path], ply_path=None):
    model_path = Path(model_path)
    config_path = model_path / "config.json"

    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found at {config_path}")

    _, model_params, optim_params, _ = load_vroom_config(config_path)
    config = {"optim_params": optim_params}

    if model_params is None or optim_params is None:
        raise ValueError(f"model_params or optim_params missing in config at {config_path}")
    
    # model_params: { 'model_config': { 'name': 'GaussianModel', 'kwargs': { ... } } }
    model_config = model_params.get("model_config")
    if not model_config:
        raise ValueError("Could not find 'model_config' in model_params")

    kwargs = model_config.get("kwargs", {})
    if kwargs is None:
        raise ValueError(f"model_config.kwargs is missing in config file")

    model_kwargs = {}
    for k in GAUSSIAN_MODEL_DEFAULTS:
        model_kwargs[k] = kwargs.get(k, GAUSSIAN_MODEL_DEFAULTS[k])

    if ply_path:
        resolved_ply = Path(ply_path)
    else:
        resolved_ply = model_path / "point_cloud.ply"

    model = build_vroom_gaussians(model_kwargs)
    load_vroom_checkpoint(model, str(resolved_ply), str(resolved_ply.parent))

    if model.anchor_cloud.semantic_labels is None:
        raise ValueError(
            f"Model at {model_path} does not have semantic labels"
        )
    
    labels = torch.unique(model.anchor_cloud.semantic_labels.view(-1))
    object.__setattr__(model, "id_encoder", SemanticsManager(labels))
    model.anchor_cloud.eval()
    model.decoder.eval()
    model.optim_params = config.get("optim_params", {})

    logger.info(
        "Loaded Gaussians from %s with %d anchors and %d unique label IDs: %s",
        model_path, len(model.anchor_cloud.anchors_positions),labels.shape[0],labels.tolist(),
    )
    return model


def count_anchors(cam, points):
    eps = 1e-6
    z_thresh = 0.01
    R, T, K = cam["R"], cam["T"], cam["K"]
    width, height = cam["width"], cam["height"]
    # world to camera rotation then translate
    camera_pts = points @ R.T + T.reshape(1, 3)
    #depth test
    z = camera_pts[:, 2]
    valid = z > z_thresh        
    # u,v projection
    u = K[0, 0] * camera_pts[:, 0] / (z + eps) + K[0, 2] 
    v = K[1, 1] * camera_pts[:, 1] / (z + eps) + K[1, 2]
    mask = valid & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    return mask.sum()



def get_horizontal_vector(cam_centers, centroid, up):
    """Horizontal direction pointing toward the median camera location to make an orbit angle around the object."""
    eps = 1e-6
    # directions from object centroid to cameras
    center_to_cam = cam_centers - centroid.reshape(1, 3)
    # remove the up component. equation: dh = d - (d . u) * u
    dh = center_to_cam - (center_to_cam @ up).reshape(-1, 1) * up.reshape(1, 3)
    # cameras above/below the object will have near-zero horizontal direction
    norm = np.linalg.norm(dh, axis=1)
    dh = dh[norm > eps]
    if len(dh) == 0:
        raise ValueError("Cannot determine horizontal direction. all cameras are aligned with up vector.")
    dh = np.median(dh, axis=0)
    #remove any still existant up component 
    dh = dh - float(np.dot(dh, up)) * up
    if np.linalg.norm(dh) < eps:
        raise ValueError("Cannot determine horizontal direction is too close to up vector.")
    return normalize(dh)


@dataclass
class ObjectScope:
    object_label_id: int
    n_anchors: int
    centroid: np.ndarray
    aabb_min: np.ndarray
    aabb_max: np.ndarray             
    obb_extents: np.ndarray         
    radius: float     
    up: np.ndarray
    base_dir: np.ndarray
    visible_cam_indices: list
    cam_centers_visible: np.ndarray
    cameras: list = field(default_factory=list)
    optim_params: dict = field(default_factory=dict)


def compute_object_scope(path, object_label_id: int, min_anchors: int = 50, ply_path=None):
    model_path = Path(path)
    cameras_json = model_path / "cameras.json"
    if not cameras_json.exists():
        raise FileNotFoundError(f"Expected cameras.json at {cameras_json}")

    resolved_ply = Path(ply_path) if ply_path else model_path / "point_cloud.ply"
    if not resolved_ply.exists():
        raise FileNotFoundError(f"PLY not found: {resolved_ply}")

    gaussians = load_gaussians(str(model_path), ply_path=str(resolved_ply))
    all_anchors = gaussians.anchor_cloud.anchors_positions.detach().cpu().numpy().astype(np.float32)
    label_ids = cast(Any, gaussians.anchor_cloud.semantic_labels).detach().cpu().numpy().reshape(-1).astype(np.int64)

    object_mask = (label_ids == object_label_id)
    if not object_mask.any():
        raise ValueError(
            f"object_label_id={object_label_id} has no anchors in {model_path}. "
            f"Available labels: {sorted(np.unique(label_ids).tolist())}"
        )
    total_objects = object_mask.sum()
    logger.info(f"Object label {object_label_id} has {total_objects} anchors in {model_path}.")

    object_anchor = all_anchors[object_mask]
    centroid = object_anchor.mean(axis=0).astype(np.float32)
    # claculate axis aligned bounding box (AABB) of the anchors
    aabb_min = object_anchor.min(axis=0)
    aabb_max = object_anchor.max(axis=0)

    # calculate PCA principal axes of the anchors to get an oriented bounding box (OBB)
    centered = object_anchor - centroid
    eigen_values = np.linalg.eigvalsh(centered.T @ centered / max(len(object_anchor) - 1, 1))

    # obb extents are used as a rough size estimate for the object
    # sqrt of eigenvalues gives the standard deviation along each principal axis 
    obb_extents = np.sqrt(np.clip(eigen_values[np.argsort(eigen_values)[::-1]], 0.0, None)).astype(np.float32)

    cameras = _load_cameras(cameras_json)

    visible_index = []
    visible_centers = []
    for i, cam in enumerate(cameras):
        if count_anchors(cam, object_anchor) >= min_anchors:
            visible_index.append(i)
            visible_centers.append(cam["position"])

    if not visible_index:
        raise RuntimeError(f"No cameras see at least {min_anchors} anchors of object {object_label_id} in {model_path}.")

    camera_centers = np.asarray(visible_centers, dtype=np.float32)

    up = []
    #from the average camera image up
    for cam in cameras:
        # image up points in the -Y camera direction.
        up.append(-cam["R"][1, :])
    up = normalize(np.mean(up, axis=0))
    
    starting_direction = get_horizontal_vector(camera_centers, centroid, up)

    # radius is median distance from camera to centroid
    distances = np.linalg.norm(camera_centers - centroid.reshape(1, 3), axis=1)
    radius = float(np.median(distances))
    # ensure radius is not too small compared to object size
    radius = max(radius, float(0.1 * np.linalg.norm(obb_extents)))

    scope = ObjectScope(
        object_label_id=object_label_id,
        n_anchors=total_objects,
        centroid=centroid,
        aabb_min=aabb_min,
        aabb_max=aabb_max,
        obb_extents=obb_extents,
        radius=radius,
        up=up,
        base_dir=starting_direction,
        visible_cam_indices=visible_index,
        cam_centers_visible=camera_centers,
        cameras=cameras,
        optim_params=getattr(gaussians, "optim_params", {}),
    )
    
    object_frame = ObjectFrame(
        centroid=scope.centroid,
        up=scope.up,
        base_dir=scope.base_dir,
        radius=scope.radius,
    )

    for ci in visible_index:
        az, el = object_frame.world_to_virtual(cameras[ci]["position"])
        cameras[ci]["azimuth_deg"] = az % 360.0
        cameras[ci]["elevation_deg"] = el
    
    logger.info(
        "ObjectScope obj=%d: %d anchors | centroid=%s | radius=%.3f | visible_cams=%d/%d",
        scope.object_label_id, scope.n_anchors,
        np.round(scope.centroid, 3).tolist(), scope.radius,
        len(scope.visible_cam_indices), len(cameras),
    )
    return scope, object_frame

