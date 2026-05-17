"""Gaussian loading and camera-graph helpers.

This module wraps the local ``gstrain`` runtime so the rest of
``object_isolation`` can:

    * Load a trained Gaussian model from disk (``load_gaussians``)
        * Read camera poses from ``cameras.json`` for view selection
            (``build_perspective_graph``)
    * Estimate scene up-direction and an orbit base direction from cameras

All public helpers are pure functions over numpy arrays / dataclasses so they
are safe to call from any pipeline stage.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml

logger = logging.getLogger(__name__)

from gstrain.vroom_core.models.facade import GaussianModel
from gstrain.vroom_core.models.semantics import SemanticCodec
from gstrain.vroom_core.training.orchestration import PipelineConfig
from gstrain.vroom_core.utils.checkpoints import CheckpointManager


# ── Data containers ───────────────────────────────────────────────────────────────

@dataclass
class PerspectiveGraph:
    """Camera list parsed from ``cameras.json``."""
    cameras: list[dict]

    @property
    def positions(self) -> np.ndarray:
        """Stack of camera world positions, shape ``(N, 3)``."""
        return np.asarray([cam["position"] for cam in self.cameras], dtype=np.float32)


# ── Model loading ─────────────────────────────────────────────────────────────────

_GSTRAIN_MODEL_KWARGS = {
    "n_offsets",
    "feat_dim",
    "view_dim",
    "appearance_dim",
    "voxel_size",
    "gs_attr",
    "render_mode",
    "tile_size_2dgs",
}


def _pipeline_config_from_yaml(cfg: dict) -> SimpleNamespace:
    values = dict(cfg.get("pipeline_params", {}) or {})
    defaults = PipelineConfig()
    for key, value in vars(defaults).items():
        values.setdefault(key, value)
    return SimpleNamespace(**values)


def _model_kwargs_from_yaml(cfg: dict, model_dir: Path) -> dict:
    model_params = cfg.get("model_params", {}) or {}
    raw_kwargs = ((model_params.get("model_config", {}) or {}).get("kwargs", {}) or {})
    kwargs = {key: raw_kwargs[key] for key in _GSTRAIN_MODEL_KWARGS if key in raw_kwargs}
    if kwargs:
        return kwargs
    return CheckpointManager(GaussianModel()).infer_bundle_kwargs(model_dir)


def load_gaussians(model_path: str | Path, iteration: int = -1) -> tuple[GaussianModel, SimpleNamespace]:
    """Load a trained ``gstrain`` ``GaussianModel`` and its pipeline config.

    If ``iteration`` is ``-1`` (default), the latest ``iteration_*`` checkpoint
    under ``<model_path>/point_cloud/`` is used. Falls back to a flat
    ``<model_path>/point_cloud.ply`` if no iterations are present.
    """
    model_path = Path(model_path)
    config_path = model_path / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    pipe_config = _pipeline_config_from_yaml(cfg)

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
            raise FileNotFoundError(f"No gstrain checkpoint found under {pc_base}")
    else:
        candidates = [pc_base / f"iteration_{int(iteration)}", pc_base / f"iteration_{int(iteration):05d}"]
        model_dir = next((path for path in candidates if path.exists()), None)
        if model_dir is None:
            raise FileNotFoundError(f"Iteration {iteration} not found in {pc_base}")
        loaded_iteration = int(iteration)

    ply_path = model_dir / "point_cloud.ply"
    if not ply_path.exists():
        raise FileNotFoundError(f"PLY not found: {ply_path}")

    gaussians = GaussianModel(**_model_kwargs_from_yaml(cfg, model_dir))
    gaussians.load_ply(str(ply_path))
    gaussians.load_mlp_checkpoints(str(model_dir))
    if gaussians.label_ids is not None:
        gaussians.id_encoder = SemanticCodec.from_labels(gaussians.label_ids.view(-1))
    gaussians.explicit_gs = False
    gaussians.weed_ratio = 0.0
    gaussians.set_eval()

    logger.info(
        "Loaded gstrain model from %s (iteration %s): %d anchors, n_offsets=%d, gs_attr=%s",
        model_dir, loaded_iteration, int(gaussians.get_anchor.shape[0]), int(gaussians.n_offsets), gaussians.gs_attr,
    )
    return gaussians, pipe_config


# ── Anchor / label accessors ────────────────────────────────────────────────────────────

def get_anchor_positions(gaussians: GaussianModel) -> np.ndarray:
    """Return anchor world positions as ``(N, 3)`` float32 numpy."""
    return gaussians.get_anchor.detach().cpu().numpy().astype(np.float32)


def get_label_ids(gaussians: GaussianModel) -> np.ndarray:
    """Return per-anchor label IDs as ``(N,)`` int64 numpy."""
    return gaussians.label_ids.detach().cpu().numpy().reshape(-1).astype(np.int64)


# ── Camera graph construction ─────────────────────────────────────────────────────────

def build_perspective_graph(cameras_json_path: str | Path) -> PerspectiveGraph:
    """Read ``cameras.json`` into render-ready camera dictionaries."""
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

    logger.info("Perspective graph: %d cameras", len(cameras))
    return PerspectiveGraph(cameras=cameras)


# ── Geometry helpers ───────────────────────────────────────────────────────────────────

def count_visible_anchors(cam: dict, points: np.ndarray) -> np.ndarray:
    """Boolean mask marking which ``points`` project inside the camera frustum."""
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
    """Estimate a world up-vector from the average camera image up."""
    ups = []
    for cam in raw_cameras:
        R_c2w = np.asarray(cam["rotation"], dtype=np.float32)
        if R_c2w.shape == (3, 3):
            # cameras.json stores camera-to-world rotation. Image up in world
            # is -row1 of R_w2c, equivalently -column1 of R_c2w.
            ups.append(-R_c2w.T[1, :])
    if not ups:
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return _normalize(np.mean(np.asarray(ups, dtype=np.float32), axis=0), np.array([0.0, 0.0, 1.0], dtype=np.float32))


def orbit_base_direction_from_cameras(cam_centers: np.ndarray, object_center: np.ndarray, up_vector: np.ndarray) -> np.ndarray:
    """Pick a robust horizontal direction toward the typical camera location."""
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


# ── Internal utilities ─────────────────────────────────────────────────────────────────

def _normalize(vec: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    """Return ``vec`` normalized; substitute ``fallback`` if degenerate."""
    vec = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if not np.isfinite(norm) or norm < 1e-8:
        return np.asarray(fallback, dtype=np.float32)
    return (vec / norm).astype(np.float32)


