"""ObjectGS loading and camera graph helpers for object_isolation."""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml

logger = logging.getLogger(__name__)

_VROOM_ROOT = Path(__file__).resolve().parents[2]
_OBJECTGS_DIR = _VROOM_ROOT / "temp_deps" / "ObjectGS"
if str(_OBJECTGS_DIR) not in sys.path:
    sys.path.insert(0, str(_OBJECTGS_DIR))

from scene.base_model import GaussianModel  # noqa: E402
from utils.general_utils import parse_cfg  # noqa: E402
from utils.semantic_utils import OneHotEncoder  # noqa: E402


@dataclass
class PerspectiveGraph:
    cameras: list[dict]
    adjacency: np.ndarray

    @property
    def positions(self) -> np.ndarray:
        return np.asarray([cam["position"] for cam in self.cameras], dtype=np.float32)


def load_gaussians(model_path: str | Path, iteration: int = -1) -> tuple[GaussianModel, SimpleNamespace]:
    model_path = Path(model_path)
    config_path = model_path / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    lp, _op, pipe_config = parse_cfg(cfg)

    pc_base = model_path / "point_cloud"
    root_model_dir = model_path if (model_path / "point_cloud.ply").exists() else None

    def iter_index(path: Path) -> int | None:
        try:
            return int(path.name.split("_")[-1])
        except ValueError:
            return None

    if int(iteration) == -1:
        iter_dirs = []
        if pc_base.exists():
            iter_dirs = [
                path for path in pc_base.iterdir()
                if path.is_dir() and path.name.startswith("iteration_") and iter_index(path) is not None
            ]
            iter_dirs = sorted(iter_dirs, key=lambda path: int(iter_index(path)))
        if iter_dirs:
            model_dir = iter_dirs[-1]
            loaded_iteration = int(iter_index(model_dir))
        elif root_model_dir is not None:
            model_dir = root_model_dir
            loaded_iteration = -1
        else:
            raise FileNotFoundError(f"No ObjectGS checkpoint found under {pc_base}")
    else:
        candidates = [pc_base / f"iteration_{int(iteration)}", pc_base / f"iteration_{int(iteration):05d}"]
        model_dir = next((path for path in candidates if path.exists()), None)
        if model_dir is None:
            raise FileNotFoundError(f"Iteration {iteration} not found in {pc_base}")
        loaded_iteration = int(iteration)

    ply_path = model_dir / "point_cloud.ply"
    if not ply_path.exists():
        raise FileNotFoundError(f"PLY not found: {ply_path}")

    model_config = lp.model_config
    gaussians = GaussianModel(**model_config["kwargs"])
    gaussians.load_ply(str(ply_path))
    gaussians.load_mlp_checkpoints(str(model_dir))
    gaussians.id_encoder = OneHotEncoder(gaussians.label_ids)
    gaussians.explicit_gs = False
    gaussians.weed_ratio = 0.0
    gaussians.eval()

    logger.info(
        "Loaded ObjectGS model from %s (iteration %s): %d anchors, n_offsets=%d, gs_attr=%s",
        model_dir, loaded_iteration, int(gaussians.get_anchor.shape[0]), int(gaussians.n_offsets), gaussians.gs_attr,
    )
    return gaussians, pipe_config


def get_anchor_positions(gaussians: GaussianModel) -> np.ndarray:
    return gaussians.get_anchor.detach().cpu().numpy().astype(np.float32)


def get_label_ids(gaussians: GaussianModel) -> np.ndarray:
    return gaussians.label_ids.detach().cpu().numpy().reshape(-1).astype(np.int64)


def build_perspective_graph(
    cameras_json_path: str | Path,
    anchor_xyz: np.ndarray | None = None,
    overlap_method: str = "frustum",
) -> PerspectiveGraph:
    path = Path(cameras_json_path)
    if not path.exists():
        raise FileNotFoundError(f"cameras.json not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        raw_cameras = json.load(f)

    cameras: list[dict] = []
    for cam in raw_cameras:
        rot = np.asarray(cam["rotation"], dtype=np.float32)
        pos = np.asarray(cam["position"], dtype=np.float32)
        width = int(cam["width"])
        height = int(cam["height"])
        fx = float(cam["fx"])
        fy = float(cam["fy"])
        R = rot.T
        T = -R @ pos
        K = np.array([[fx, 0.0, width / 2.0], [0.0, fy, height / 2.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        cameras.append({
            "id": cam["id"],
            "img_name": cam.get("img_name", f"cam_{cam['id']}"),
            "R": R.astype(np.float32),
            "T": T.astype(np.float32),
            "K": K,
            "position": pos.astype(np.float32),
            "width": width,
            "height": height,
            "fx": fx,
            "fy": fy,
        })

    adjacency = (
        _compute_visibility_overlap(cameras, anchor_xyz)
        if overlap_method == "visibility" and anchor_xyz is not None
        else _compute_angular_overlap(cameras)
    )
    logger.info("Perspective graph: %d cameras, mean adjacency=%.3f", len(cameras), float(adjacency.mean()))
    return PerspectiveGraph(cameras=cameras, adjacency=adjacency)


def count_visible_anchors(cam: dict, points: np.ndarray) -> np.ndarray:
    R = cam["R"]
    T = cam["T"]
    K = cam["K"]
    width = int(cam["width"])
    height = int(cam["height"])
    cam_pts = (R @ points.T).T + T.reshape(1, 3)
    z = cam_pts[:, 2]
    valid = z > 0.01
    u = K[0, 0] * cam_pts[:, 0] / (z + 1e-8) + K[0, 2]
    v = K[1, 1] * cam_pts[:, 1] / (z + 1e-8) + K[1, 2]
    return valid & (u >= 0) & (u < width) & (v >= 0) & (v < height)


def estimate_scene_up_from_cameras(raw_cameras: list[dict]) -> np.ndarray:
    ups = []
    for cam in raw_cameras:
        R_w2c = np.asarray(cam["rotation"], dtype=np.float32)
        if R_w2c.shape == (3, 3):
            ups.append(-R_w2c[1, :])
    if not ups:
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return _normalize(np.mean(np.asarray(ups, dtype=np.float32), axis=0), np.array([0.0, 0.0, 1.0], dtype=np.float32))


def orbit_base_direction_from_cameras(cam_centers: np.ndarray, object_center: np.ndarray, up_vector: np.ndarray) -> np.ndarray:
    up = _normalize(up_vector, np.array([0.0, 0.0, 1.0], dtype=np.float32))
    fallback = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    dirs = np.asarray(cam_centers, dtype=np.float32) - np.asarray(object_center, dtype=np.float32).reshape(1, 3)
    dirs = dirs - (dirs @ up).reshape(-1, 1) * up.reshape(1, 3)
    norms = np.linalg.norm(dirs, axis=1)
    dirs = dirs[norms > 1e-6]
    base = np.median(dirs, axis=0) if len(dirs) else fallback
    base = base - float(np.dot(base, up)) * up
    if np.linalg.norm(base) < 1e-6:
        alt = fallback if abs(float(np.dot(fallback, up))) < 0.9 else np.array([0.0, 1.0, 0.0], dtype=np.float32)
        base = alt - float(np.dot(alt, up)) * up
    return _normalize(base, fallback)


def _normalize(vec: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if not np.isfinite(norm) or norm < 1e-8:
        return np.asarray(fallback, dtype=np.float32)
    return (vec / norm).astype(np.float32)


def _compute_angular_overlap(cameras: list[dict]) -> np.ndarray:
    positions = np.asarray([cam["position"] for cam in cameras], dtype=np.float32)
    forwards = np.asarray([cam["R"][2, :] for cam in cameras], dtype=np.float32)
    forwards /= np.linalg.norm(forwards, axis=1, keepdims=True) + 1e-8
    angular = (forwards @ forwards.T + 1.0) / 2.0
    dists = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=2)
    proximity = 1.0 - dists / (dists.max() + 1e-8)
    adjacency = (0.6 * angular + 0.4 * proximity).astype(np.float32)
    np.fill_diagonal(adjacency, 1.0)
    return adjacency


def _compute_visibility_overlap(cameras: list[dict], anchor_xyz: np.ndarray) -> np.ndarray:
    visibility = np.zeros((len(cameras), len(anchor_xyz)), dtype=bool)
    for index, cam in enumerate(cameras):
        visibility[index] = count_visible_anchors(cam, anchor_xyz)

    adjacency = np.zeros((len(cameras), len(cameras)), dtype=np.float32)
    for i in range(len(cameras)):
        for j in range(i, len(cameras)):
            inter = float(np.logical_and(visibility[i], visibility[j]).sum())
            union = float(np.logical_or(visibility[i], visibility[j]).sum())
            score = inter / max(union, 1e-8)
            adjacency[i, j] = score
            adjacency[j, i] = score
    np.fill_diagonal(adjacency, 1.0)
    return adjacency