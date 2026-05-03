"""Phase 8 — Final export + before/after verification renders.

For legacy seeded runs the parent ``ObjectGS`` model may be mutated in-place.
For object training the parent and object models stay separate because
their MLP checkpoints are independent. This module is responsible for:

1. Building a fixed orbit of comparison cameras around an object's centroid
   so the *same* viewpoints can be rendered before and after Phase 7.
2. Saving either a mutated parent model or a scene package that points
    at the reference scene plus per-object checkpoints.
3. Writing a ``reintegration_metadata.json`` summary listing every object
   that was replenished, its anchor counts, and scene-wide totals.

"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Comparison cameras
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _OrbitCam:
    index: int
    azimuth_deg: float
    R: np.ndarray  # (3,3) R_w2c
    T: np.ndarray  # (3,)
    K: np.ndarray  # (3,3)
    width: int
    height: int


def _look_at(cam_pos: np.ndarray, target: np.ndarray, up: np.ndarray):
    """COLMAP-convention look-at used by the comparison orbit renders."""
    forward = target - cam_pos
    forward = forward / max(np.linalg.norm(forward), 1e-8)
    right = np.cross(up, forward)
    right = right / max(np.linalg.norm(right), 1e-8)
    cam_up = np.cross(forward, right)
    cam_up = cam_up / max(np.linalg.norm(cam_up), 1e-8)
    R = np.vstack((right, -cam_up, forward)).astype(np.float32)
    T = (-R @ cam_pos).astype(np.float32)
    return R, T


def build_orbit_cameras(
    *,
    center: np.ndarray,
    radius: float,
    orbit_radius: float,
    up: np.ndarray,
    ref_cam_position: np.ndarray,
    n_views: int = 8,
    width: int = 512,
    height: int = 512,
) -> List[_OrbitCam]:
    """Build a stable orbit of comparison cameras around ``center``."""
    up = up.astype(np.float32) / max(float(np.linalg.norm(up)), 1e-9)
    dist = max(float(orbit_radius) * 0.9, float(radius) * 2.5, 0.5)

    angular = 2.0 * np.arctan(float(radius) / max(dist, 1e-6))
    fov = float(np.clip(angular / 0.55, np.radians(30.0), np.radians(100.0)))
    fx = (width / 2.0) / np.tan(fov / 2.0)
    fy = (height / 2.0) / np.tan(fov / 2.0)
    K = np.array([[fx, 0.0, width / 2.0],
                  [0.0, fy, height / 2.0],
                  [0.0, 0.0, 1.0]], dtype=np.float32)

    ref = ref_cam_position.astype(np.float32) - center.astype(np.float32)
    vertical = float(np.dot(ref, up))
    horizontal = ref - vertical * up
    if np.linalg.norm(horizontal) < 1e-6:
        horizontal = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if abs(np.dot(horizontal, up)) > 0.9:
            horizontal = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    basis_h = horizontal / max(float(np.linalg.norm(horizontal)), 1e-8)
    basis_v = np.cross(up, basis_h)
    basis_v = basis_v / max(float(np.linalg.norm(basis_v)), 1e-8)
    z_off = float(np.clip(vertical, -0.3 * dist, 0.3 * dist))

    cams: List[_OrbitCam] = []
    for i in range(int(n_views)):
        a = 2.0 * np.pi * i / int(n_views)
        cam_pos = (
            center.astype(np.float32)
            + dist * np.cos(a) * basis_h
            + dist * np.sin(a) * basis_v
            + z_off * up
        )
        R, T = _look_at(cam_pos, center.astype(np.float32), up)
        cams.append(_OrbitCam(
            index=i, azimuth_deg=float(np.degrees(a)),
            R=R, T=T, K=K.copy(), width=int(width), height=int(height),
        ))
    return cams


def render_with_orbit(gaussians, pipe_config, cams: List[_OrbitCam],
                      object_label_id: Optional[int] = None,
                      exclude_object_label_id: Optional[int] = None,
                      bg_white: bool = True) -> List[np.ndarray]:
    """Render each orbit camera against the current gaussians state.

    If ``object_label_id`` is given, only anchors with that label are rendered
    (object-isolated view). Otherwise the full scene is rendered."""
    from .gs_renderer import create_camera, render_rgba
    frames: List[np.ndarray] = []
    for c in cams:
        cam = create_camera(c.R, c.T, c.K, c.width, c.height)
        pkg = render_rgba(
            gaussians,
            cam,
            pipe_config,
            bg_white=bool(bg_white),
            object_label_id=object_label_id,
            exclude_object_label_id=exclude_object_label_id,
        )
        rgb = (pkg["rgb"].detach().permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
        frames.append(rgb)
    return frames


def render_composited_with_orbit(
    parent_gaussians,
    obj_gaussians,
    pipe_config,
    cams: List[_OrbitCam],
    *,
    object_label_id: int,
) -> List[np.ndarray]:
    """Render parent scene with old object hidden and trained object alpha-composited."""
    from .gs_renderer import create_camera, render_rgba

    frames: List[np.ndarray] = []
    for c in cams:
        cam = create_camera(c.R, c.T, c.K, c.width, c.height)
        base = render_rgba(
            parent_gaussians,
            cam,
            pipe_config,
            bg_white=True,
            exclude_object_label_id=int(object_label_id),
        )
        obj = render_rgba(
            obj_gaussians,
            cam,
            pipe_config,
            bg_white=False,
        )
        alpha = obj["alpha"].detach().clamp(0.0, 1.0).unsqueeze(0)
        rgb = obj["rgb"].detach().clamp(0.0, 1.0) + base["rgb"].detach().clamp(0.0, 1.0) * (1.0 - alpha)
        rgb_np = (rgb.clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
        frames.append(rgb_np)
    return frames


def save_compare_grid(
    before_frames: List[np.ndarray],
    after_frames: List[np.ndarray],
    out_dir: Path,
    *,
    prefix: str = "view",
) -> None:
    """Save side-by-side before/after PNGs (BGR via cv2)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n = min(len(before_frames), len(after_frames))
    for i in range(n):
        b = before_frames[i]
        a = after_frames[i]
        cv2.imwrite(str(out_dir / f"before_{prefix}_{i:02d}.png"), cv2.cvtColor(b, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out_dir / f"after_{prefix}_{i:02d}.png"), cv2.cvtColor(a, cv2.COLOR_RGB2BGR))
        if b.shape == a.shape:
            sep = np.full((b.shape[0], 6, 3), 255, dtype=np.uint8)
            grid = np.concatenate([b, sep, a], axis=1)
            cv2.imwrite(str(out_dir / f"compare_{prefix}_{i:02d}.png"),
                        cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
            diff = cv2.absdiff(a, b)
            cv2.imwrite(str(out_dir / f"diff_{prefix}_{i:02d}.png"),
                        cv2.cvtColor(diff, cv2.COLOR_RGB2BGR))


# ─────────────────────────────────────────────────────────────────────────────
# Final model export
# ─────────────────────────────────────────────────────────────────────────────

def save_final_model(
    gaussians,
    *,
    output_dir: Path | str,
    reference_model_path: Path | str,
    extra_metadata: Optional[dict] = None,
) -> Path:
    """Save the (mutated) parent gaussians in ObjectGS-compatible layout.

    Creates::

        <output_dir>/final_model/
            point_cloud.ply
            color_mlp.pt cov_mlp.pt opacity_mlp.pt
            point_cloud/iteration_1/
                point_cloud.ply
                color_mlp.pt cov_mlp.pt opacity_mlp.pt
            cameras.json   (copied from reference)
            config.yaml    (copied from reference)
            reintegration_metadata.json
    """
    output_dir = Path(output_dir)
    final_dir = output_dir / "final_model"
    final_dir.mkdir(parents=True, exist_ok=True)

    iter_dir = final_dir / "point_cloud" / "iteration_1"
    iter_dir.mkdir(parents=True, exist_ok=True)

    # Legacy + ObjectGS-loadable layout.
    gaussians.save_ply(str(final_dir / "point_cloud.ply"))
    gaussians.save_mlp_checkpoints(str(final_dir))
    gaussians.save_ply(str(iter_dir / "point_cloud.ply"))
    gaussians.save_mlp_checkpoints(str(iter_dir))

    ref = Path(reference_model_path)
    for name in ("config.yaml", "cameras.json"):
        src = ref / name
        dst = final_dir / name
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)

    if extra_metadata is not None:
        with open(final_dir / "reintegration_metadata.json", "w", encoding="utf-8") as f:
            json.dump(extra_metadata, f, indent=2)

    logger.info("Phase 8: saved final model to %s", final_dir)
    return final_dir


def save_scene_package(
    *,
    output_dir: Path | str,
    reference_model_path: Path | str,
    per_object_summaries: List[dict],
    extra_metadata: Optional[dict] = None,
) -> Path:
    """Save the scene package with per-object trained model references."""
    output_dir = Path(output_dir)
    scene_dir = output_dir / "scene"
    scene_dir.mkdir(parents=True, exist_ok=True)

    ref = Path(reference_model_path)
    for name in ("config.yaml", "cameras.json"):
        src = ref / name
        dst = scene_dir / name
        if src.exists():
            shutil.copy2(src, dst)

    object_models = []
    for summary in per_object_summaries:
        if summary.get("mode") != "object_training":
            continue
        object_id = int(summary.get("object_id", -1))
        object_models.append({
            "object_id": object_id,
            "model_dir": str((Path("..") / f"obj_{object_id}" / "model").as_posix()),
            "n_final_anchors": int(summary.get("n_final_anchors", 0)),
            "source_counts": summary.get("source_counts", {}),
            "final_loss": float(summary.get("final_loss", 0.0)),
        })

    payload = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "mode": "scene_composite",
        "reference_model_path": str(reference_model_path),
        "reference_scene_policy": "Render the parent scene with trained object labels hidden, then alpha-composite the object model.",
        "object_models": object_models,
        "metadata": extra_metadata or {},
    }
    with open(scene_dir / "scene_metadata.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    logger.info("Phase 8: saved scene package to %s", scene_dir)
    return scene_dir


def build_reintegration_metadata(
    *,
    parent_label_counts_pre: dict,
    parent_label_counts_post: dict,
    per_object_summaries: List[dict],
    reference_model_path: str,
) -> dict:
    """Aggregate per-object Phase-7 summaries into a single metadata dict."""
    return {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "reference_model_path": str(reference_model_path),
        "n_anchors_pre": int(sum(parent_label_counts_pre.values())),
        "n_anchors_post": int(sum(parent_label_counts_post.values())),
        "anchors_added": int(sum(parent_label_counts_post.values())
                             - sum(parent_label_counts_pre.values())),
        "label_counts_pre": {str(k): int(v) for k, v in parent_label_counts_pre.items()},
        "label_counts_post": {str(k): int(v) for k, v in parent_label_counts_post.items()},
        "objects": per_object_summaries,
    }


def label_anchor_counts(gaussians) -> dict:
    """Return ``{label_id: n_anchors}`` from the current gaussians state."""
    labels = gaussians.label_ids.squeeze(-1).cpu().numpy().astype(np.int64)
    uniq, counts = np.unique(labels, return_counts=True)
    return {int(k): int(v) for k, v in zip(uniq.tolist(), counts.tolist())}
