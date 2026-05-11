"""Geometry and image helpers for VRoom."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import json

import numpy as np
import torch


@dataclass(frozen=True)
class PointCloudSample:
    points: np.ndarray
    colors: np.ndarray
    normals: np.ndarray
    label_ids: np.ndarray


@dataclass(frozen=True)
class SceneTransform:
    offset: np.ndarray
    scale: float = 1.0
    units: str = "scene_units"
    up_axis: str = "y"
    handedness: str = "right"

    def to_dict(self) -> dict:
        return {
            "offset": np.asarray(self.offset, dtype=np.float32).tolist(),
            "scale": float(self.scale),
            "units": self.units,
            "up_axis": self.up_axis,
            "handedness": self.handedness,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "SceneTransform":
        return cls(
            offset=np.asarray(payload.get("offset", [0.0, 0.0, 0.0]), dtype=np.float32),
            scale=float(payload.get("scale", 1.0)),
            units=str(payload.get("units", "scene_units")),
            up_axis=str(payload.get("up_axis", "y")),
            handedness=str(payload.get("handedness", "right")),
        )


ARCORE_TO_COLMAP_CAMERA = np.diag([1.0, -1.0, -1.0]).astype(np.float64)


def pil_image_to_tensor(pil_image, resolution: Tuple[int, int]) -> torch.Tensor:
    resized = pil_image.resize(resolution)
    array = np.array(resized, dtype=np.float32, copy=True) / 255.0
    if array.ndim == 2:
        array = array[..., None]
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def focal_to_fov(focal: float, pixels: int) -> float:
    return 2.0 * math.atan(pixels / (2.0 * focal))


def fov_to_focal(fov: float, pixels: int) -> float:
    return pixels / (2.0 * math.tan(fov / 2.0))


def world_to_view_matrix(
    rotation: np.ndarray,
    translation: np.ndarray,
    offset: np.ndarray | None = None,
    scale: float = 1.0,
) -> np.ndarray:
    offset = np.zeros(3, dtype=np.float32) if offset is None else np.asarray(offset, dtype=np.float32)
    view = np.eye(4, dtype=np.float32)
    view[:3, :3] = rotation.T
    view[:3, 3] = np.asarray(translation, dtype=np.float32)

    camera_to_world = np.linalg.inv(view)
    camera_to_world[:3, 3] = (camera_to_world[:3, 3] + offset) * scale
    return np.linalg.inv(camera_to_world).astype(np.float32)


def projection_matrix(znear: float, zfar: float, fov_x: float, fov_y: float) -> torch.Tensor:
    half_x = math.tan(fov_x / 2.0) * znear
    half_y = math.tan(fov_y / 2.0) * znear
    left, right = -half_x, half_x
    bottom, top = -half_y, half_y

    matrix = torch.zeros(4, 4)
    matrix[0, 0] = 2.0 * znear / (right - left)
    matrix[1, 1] = 2.0 * znear / (top - bottom)
    matrix[0, 2] = (right + left) / (right - left)
    matrix[1, 2] = (top + bottom) / (top - bottom)
    matrix[2, 2] = zfar / (zfar - znear)
    matrix[2, 3] = -(zfar * znear) / (zfar - znear)
    matrix[3, 2] = 1.0
    return matrix


def apply_scene_transform(points: np.ndarray, offset: np.ndarray | None = None, scale: float = 1.0) -> np.ndarray:
    offset = np.zeros(3, dtype=np.float32) if offset is None else np.asarray(offset, dtype=np.float32)
    return (points + offset) * float(scale)


def invert_scene_transform(points: np.ndarray, offset: np.ndarray | None = None, scale: float = 1.0) -> np.ndarray:
    offset = np.zeros(3, dtype=np.float32) if offset is None else np.asarray(offset, dtype=np.float32)
    scale = float(scale)
    if abs(scale) < 1e-12:
        raise ValueError("Scene scale must be non-zero.")
    return np.asarray(points, dtype=np.float32) / scale - offset


def save_scene_transform(path: str | Path, transform: SceneTransform) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "w", encoding="utf-8") as handle:
        json.dump(transform.to_dict(), handle, indent=2)


def load_scene_transform(path: str | Path) -> SceneTransform:
    with open(path, "r", encoding="utf-8") as handle:
        return SceneTransform.from_dict(json.load(handle))


def rotation_matrix_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"Expected a 3x3 rotation matrix, got {matrix.shape}.")

    rxx, ryx, rzx, rxy, ryy, rzy, rxz, ryz, rzz = matrix.flat
    k_matrix = np.array(
        [
            [rxx - ryy - rzz, 0.0, 0.0, 0.0],
            [ryx + rxy, ryy - rxx - rzz, 0.0, 0.0],
            [rzx + rxz, rzy + ryz, rzz - rxx - ryy, 0.0],
            [ryz - rzy, rzx - rxz, rxy - ryx, rxx + ryy + rzz],
        ],
        dtype=np.float64,
    ) / 3.0
    eigenvalues, eigenvectors = np.linalg.eigh(k_matrix)
    quaternion = eigenvectors[[3, 0, 1, 2], np.argmax(eigenvalues)]
    if quaternion[0] < 0.0:
        quaternion *= -1.0
    return quaternion


def arcore_camera_to_world_to_colmap_extrinsics(camera_to_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    transform = np.asarray(camera_to_world, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 camera_to_world matrix, got {transform.shape}.")

    rotation_world_from_camera = transform[:3, :3]
    translation_world_from_camera = transform[:3, 3]

    rotation_camera_from_world = ARCORE_TO_COLMAP_CAMERA @ rotation_world_from_camera.T
    translation_camera_from_world = -rotation_camera_from_world @ translation_world_from_camera
    return rotation_camera_from_world.astype(np.float64), translation_camera_from_world.astype(np.float64)
