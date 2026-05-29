from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch


@dataclass(frozen=True)
class PointCloudSample:
    points: np.ndarray
    colors: np.ndarray
    normals: np.ndarray
    label_ids: np.ndarray


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

