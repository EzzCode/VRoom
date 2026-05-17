"""


"""
import json
from pathlib import Path

import numpy as np
import yaml
from gstrain.vroom_core import GaussianModel
from gstrain.vroom_core.models.semantics import SemanticCodec


import logging
logger = logging.getLogger(__name__)

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
        
    cameras: list[dict] = []
    for camera in cameras:
        rotation = np.asarray(camera["rotation"], dtype=np.float32)
        postion = np.asarray(camera["position"], dtype=np.float32)
        width = int(camera.get("width"))
        height = int(camera.get("height"))
        fx = float(camera.get("fx"))
        fy = float(camera.get("fy"))
        R = rotation.T
        T = -R @ postion
        K = np.array([[fx, 0.0, width / 2], [0.0, fy, height / 2], [0.0, 0.0, 1.0]], dtype=np.float32)
        cameras.append({
            "id" : camera.get("id"),
            "image_name": camera.get("image_name"),
            


    return cameras

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
    
    model = GaussianModel(**kwargs)                     # construct empty model with config and uninitialized params
    model.load_ply(str(model_path / "point_cloud.ply")) # load anchors and label_ids from PLY

    model.load_mlp_checkpoints(str(model_path))         # load MLPs from checkpoints
    if model.label_ids is None:
        raise ValueError(f"Model at {model_path} does not have label_ids; cannot compute object scope.")
    model.id_encoder = SemanticCodec.from_labels(model.label_ids.view(-1)) 
    model.explicit_gs = False                           # this is used for training 
    model.weed_ratio = 0.0                              # turn off anchor pruning
    model.set_eval()                                    
    
    logger.info(f"Loaded GaussianModel from {model_path} with {len(model.get_anchor)} anchors and label IDs: {sorted(np.unique(model.label_ids.cpu().numpy()).tolist())}")
    return model, pipeline_params

class ObjectScope:
    object_label_id: int
    n_anchors: int
    centroid: np.ndarray           # (3,)   float32
    aabb_min: np.ndarray           # (3,)   float32
    aabb_max: np.ndarray           # (3,)   float32
    obb_extents: np.ndarray        # (3,)   float32 


def compute_object_scope(path, object_label_id):
    model_path = Path(path) 
    cameras_json = model_path / "cameras.json"
    if not cameras_json.exists():
        raise FileNotFoundError(f"Expected cameras.json at {cameras_json}")
    
    ply_path = model_path / "point_cloud.ply"
    if not ply_path.exists():
        raise FileNotFoundError(f"PLY not found: {ply_path}")
    
    gaussians, pipe_config = load_gaussians(str(model_path))
    all_xyz = get_anchor_positions(gaussians)
    label_ids = get_label_ids(gaussians)

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

    # -- Camera analysis  ──

    scope =  ObjectScope(
        object_label_id=object_label_id,
        n_anchors=total_objects,
        centroid=centroid,
        aabb_min=aabb_min,
        aabb_max=aabb_max,
        obb_extents=obb_extents
    )