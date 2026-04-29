"""
Diagnostics — fixed-pose before/after snapshots, AABB and seed scatter
overlays for visual audit.
"""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import torch

from target_replenishment.core.objectgs_bridge import (
    create_virtual_camera, render_view, get_anchor_positions,
)

logger = logging.getLogger(__name__)


def _to_uint8(rgb_chw: torch.Tensor) -> np.ndarray:
    img = rgb_chw.detach().clamp(0.0, 1.0).cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    return (img * 255.0 + 0.5).astype(np.uint8)


def render_and_save(
    gaussians, pipe_config, cam, out_path: Path,
    bg_white: bool = True, object_label_id: int | None = None,
) -> np.ndarray:
    bg = torch.ones(3, dtype=torch.float32, device="cuda") if bg_white \
        else torch.zeros(3, dtype=torch.float32, device="cuda")
    res = render_view(gaussians, cam, pipe_config, bg, object_label_id=object_label_id)
    img = _to_uint8(res["rgb"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return img


def overlay_aabb(
    img: np.ndarray, cam, bounds_min: np.ndarray, bounds_max: np.ndarray,
    color=(0, 255, 0), thickness: int = 1,
) -> np.ndarray:
    bm, bM = np.asarray(bounds_min, np.float32), np.asarray(bounds_max, np.float32)
    corners = np.array([
        [bm[0], bm[1], bm[2]], [bm[0], bm[1], bM[2]], [bm[0], bM[1], bm[2]], [bm[0], bM[1], bM[2]],
        [bM[0], bm[1], bm[2]], [bM[0], bm[1], bM[2]], [bM[0], bM[1], bm[2]], [bM[0], bM[1], bM[2]],
    ], dtype=np.float32)
    R = cam.R if isinstance(cam.R, np.ndarray) else np.asarray(cam.R)
    T = cam.T if isinstance(cam.T, np.ndarray) else np.asarray(cam.T)
    cam_pts = (R @ corners.T).T + T.reshape(1, 3)
    z = cam_pts[:, 2]
    valid = z > 1e-3
    pix = np.full((8, 2), -1, dtype=np.int32)
    if valid.any():
        u = cam.fx * cam_pts[valid, 0] / z[valid] + img.shape[1] / 2.0
        v = cam.fy * cam_pts[valid, 1] / z[valid] + img.shape[0] / 2.0
        pix[valid, 0] = np.clip(u.round(), -10000, 10000).astype(np.int32)
        pix[valid, 1] = np.clip(v.round(), -10000, 10000).astype(np.int32)

    edges = [(0, 1), (0, 2), (0, 4), (3, 1), (3, 2), (3, 7),
             (5, 1), (5, 4), (5, 7), (6, 2), (6, 4), (6, 7)]
    out = img.copy()
    for a, b in edges:
        if pix[a, 0] == -1 or pix[b, 0] == -1:
            continue
        cv2.line(out, tuple(pix[a]), tuple(pix[b]), color, thickness, cv2.LINE_AA)
    return out


def make_compare(before: np.ndarray, after: np.ndarray, out_path: Path):
    h = max(before.shape[0], after.shape[0])
    w = before.shape[1] + after.shape[1] + 4
    canvas = np.full((h, w, 3), 255, np.uint8)
    canvas[: before.shape[0], : before.shape[1]] = before
    canvas[: after.shape[0], before.shape[1] + 4 :] = after
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def make_diff(before: np.ndarray, after: np.ndarray, out_path: Path) -> float:
    a = before.astype(np.int16)
    b = after.astype(np.int16)
    diff = np.abs(a - b).astype(np.uint8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(diff, cv2.COLOR_RGB2BGR))
    return float(diff.mean())


def make_contact_sheet(image_paths: list[Path], out_path: Path, columns: int = 2):
    images = []
    for path in image_paths:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is not None:
            images.append(img)
    if not images:
        return

    cell_h = max(img.shape[0] for img in images)
    cell_w = max(img.shape[1] for img in images)
    columns = max(1, int(columns))
    rows = int(np.ceil(len(images) / columns))
    canvas = np.full((rows * cell_h, columns * cell_w, 3), 255, np.uint8)
    for idx, img in enumerate(images):
        row, col = divmod(idx, columns)
        y0, x0 = row * cell_h, col * cell_w
        canvas[y0:y0 + img.shape[0], x0:x0 + img.shape[1]] = img
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)


def _look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray):
    forward = target - eye
    forward = forward / max(np.linalg.norm(forward), 1e-8)
    right = np.cross(up, forward)
    right = right / max(np.linalg.norm(right), 1e-8)
    new_up = np.cross(forward, right)
    R = np.vstack((right, -new_up, forward)).astype(np.float32)
    T = (-R @ eye).astype(np.float32)
    return R, T


def _normalize(v: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    norm = float(np.linalg.norm(v))
    if not np.isfinite(norm) or norm < 1e-8:
        return np.asarray(fallback, dtype=np.float32)
    return (v / norm).astype(np.float32)


def _snap_up_to_dominant_axis(up_vector: np.ndarray) -> np.ndarray:
    up = _normalize(up_vector, np.array([0.0, 0.0, 1.0], dtype=np.float32))
    axes = np.eye(3, dtype=np.float32)
    dots = axes @ up
    idx = int(np.argmax(np.abs(dots)))
    snapped = axes[idx] * (1.0 if dots[idx] >= 0.0 else -1.0)
    return snapped.astype(np.float32)


def camera_centers_from_cameras_json(cam_data: list) -> np.ndarray:
    """Return world-space camera centers from ObjectGS/Colmap cameras.json.

    This repository uses the same convention as ObjectGS rendering cameras:
    ``rotation`` is world-to-camera and ``position`` is translation. The world
    camera center is therefore ``C = -R.T @ T``.
    """
    centers = []
    for c in cam_data:
        R_w2c = np.asarray(c["rotation"], dtype=np.float32)
        T_w2c = np.asarray(c["position"], dtype=np.float32)
        centers.append(-R_w2c.T @ T_w2c)
    return np.asarray(centers, dtype=np.float32)


def estimate_scene_up_from_cameras(cam_data: list) -> np.ndarray:
    """Estimate world up from cameras.json world-to-camera rotations.

    COLMAP/OpenCV camera coordinates use +Y downward. Since each row of
    ``R_w2c`` is a camera axis in world coordinates, camera up is
    ``-R_w2c[1, :]``. Averaging those axes gives a stable roll/up reference.
    """
    ups = []
    for c in cam_data:
        R_w2c = np.asarray(c["rotation"], dtype=np.float32)
        if R_w2c.shape == (3, 3):
            ups.append(-R_w2c[1, :])
    if not ups:
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)
    up = _normalize(np.mean(np.asarray(ups, dtype=np.float32), axis=0), np.array([0.0, 0.0, 1.0]))
    return up.astype(np.float32)


def orbit_base_direction_from_cameras(
    cam_centers: np.ndarray | None,
    object_center: np.ndarray,
    up_vector: np.ndarray,
) -> np.ndarray:
    """Use the median training-camera direction as the orbit's zero azimuth."""
    up = _normalize(up_vector, np.array([0.0, 0.0, 1.0], dtype=np.float32))
    fallback = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    if cam_centers is None or len(cam_centers) == 0:
        base = fallback
    else:
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


def build_orbit_camera(
    object_xyz: np.ndarray, cam_template, azimuth_deg: float = 0.0,
    elevation_deg: float = 15.0, dist_factor: float = 0.5,
    cam_centers: np.ndarray | None = None,
    radius_elevation_ratio: float = 0.4,
    zoom_scale: float = 1.0,
    up_vector: np.ndarray | None = None,
    base_direction: np.ndarray | None = None,
):
    """Build an orbit camera around the object centroid.

    Uses median(cam-to-object dist) * dist_factor for radius if
    ``cam_centers`` provided; falls back to extent-based radius. Up = +Z.
    """
    center = object_xyz.mean(axis=0).astype(np.float32)
    up = _snap_up_to_dominant_axis(_normalize(
        np.array([0.0, 0.0, 1.0], dtype=np.float32) if up_vector is None else up_vector,
        np.array([0.0, 0.0, 1.0], dtype=np.float32),
    ))
    if cam_centers is not None and len(cam_centers) > 0:
        dists = np.linalg.norm(cam_centers - center.reshape(1, 3), axis=1)
        avg_dist = float(np.median(dists))
        radius = max(avg_dist * dist_factor, 0.1)
        elev_lift = radius * radius_elevation_ratio
    else:
        extent = float(np.linalg.norm(object_xyz.max(axis=0) - object_xyz.min(axis=0)))
        radius = max(extent * 0.5, 0.1) * max(dist_factor, 1.0)
        elev_lift = radius * np.sin(np.deg2rad(elevation_deg))
    az = np.deg2rad(azimuth_deg)
    base = orbit_base_direction_from_cameras(cam_centers, center, up) if base_direction is None else _normalize(base_direction, np.array([1.0, 0.0, 0.0], dtype=np.float32))
    side = _normalize(np.cross(up, base), np.array([0.0, 1.0, 0.0], dtype=np.float32))
    radial = np.cos(az) * base + np.sin(az) * side
    eye = center + radius * radial.astype(np.float32) + elev_lift * up
    R, T = _look_at(eye, center, up)
    K = np.array([
        [cam_template.fx * zoom_scale, 0.0, cam_template.image_width / 2.0],
        [0.0, cam_template.fy * zoom_scale, cam_template.image_height / 2.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)
    return create_virtual_camera(R, T, K, cam_template.image_width, cam_template.image_height)


def build_orbit_cameras(
    object_xyz: np.ndarray, cam_template, n_views: int,
    start_azimuth_deg: float = 0.0, elevation_deg: float = 15.0,
    dist_factor: float = 0.75, cam_centers: np.ndarray | None = None,
    radius_elevation_ratio: float = 0.25, zoom_scale: float = 0.85,
    up_vector: np.ndarray | None = None,
):
    cams = []
    n_views = max(1, int(n_views))
    center = object_xyz.mean(axis=0).astype(np.float32)
    up = _snap_up_to_dominant_axis(_normalize(
        np.array([0.0, 0.0, 1.0], dtype=np.float32) if up_vector is None else up_vector,
        np.array([0.0, 0.0, 1.0], dtype=np.float32),
    ))
    base = orbit_base_direction_from_cameras(cam_centers, center, up)
    for idx in range(n_views):
        azimuth = start_azimuth_deg + 360.0 * idx / n_views
        cams.append(build_orbit_camera(
            object_xyz, cam_template,
            azimuth_deg=azimuth,
            elevation_deg=elevation_deg,
            dist_factor=dist_factor,
            cam_centers=cam_centers,
            radius_elevation_ratio=radius_elevation_ratio,
            zoom_scale=zoom_scale,
            up_vector=up,
            base_direction=base,
        ))
    return cams


def pick_best_training_camera(
    cam_data: list, object_xyz: np.ndarray,
    require_in_front: bool = True,
):
    """Pick the training camera whose projection of the object centroid lands
    closest to the image center (with the object in front). Returns the
    cam-dict entry from cameras.json.

    Uses ObjectGS/Colmap convention: cameras.json stores world-to-camera
    rotation and translation directly.
    """
    centroid = object_xyz.mean(axis=0).astype(np.float32)
    best = None
    best_score = float("inf")
    for c in cam_data:
        R_w2c = np.array(c["rotation"], dtype=np.float32)
        T_w2c = np.array(c["position"], dtype=np.float32)
        cam_pt = R_w2c @ centroid + T_w2c
        z = float(cam_pt[2])
        if require_in_front and z <= 0.05:
            continue
        u = float(c["fx"]) * cam_pt[0] / z
        v = float(c["fy"]) * cam_pt[1] / z
        # offset from principal point (image center)
        score = float(np.hypot(u, v))
        # mild distance preference: prefer mid-range, not too close not too far
        if score < best_score:
            best_score = score
            best = c
    return best


def build_camera_from_entry(cam_entry: dict, zoom_scale: float = 1.0):
    """Construct a VirtualCamera from a cameras.json dict entry."""
    R_w2c = np.array(cam_entry["rotation"], dtype=np.float32)
    T_w2c = np.array(cam_entry["position"], dtype=np.float32)
    w = int(cam_entry["width"]); h = int(cam_entry["height"])
    fx = float(cam_entry["fx"]) * zoom_scale; fy = float(cam_entry["fy"]) * zoom_scale
    K = np.array([[fx, 0.0, w / 2.0], [0.0, fy, h / 2.0], [0.0, 0.0, 1.0]], np.float32)
    return create_virtual_camera(R_w2c, T_w2c, K, w, h)
