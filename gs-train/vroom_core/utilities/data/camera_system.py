"""Camera records and render-facing camera objects."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn

from vroom_core.utilities.utils.geometry import pil_image_to_tensor, projection_matrix, world_to_view_matrix


@dataclass(frozen=True)
class FrameRecord:
    uid: int
    rotation: np.ndarray
    translation: np.ndarray
    fov_y: float
    fov_x: float
    cx: float
    cy: float
    image: Image.Image
    image_path: str
    image_name: str
    width: int
    height: int
    alpha_mask: Optional[Image.Image] = None
    depth: Optional[np.ndarray] = None
    depth_params: Optional[dict] = None


class RenderCamera(nn.Module):
    def __init__(
        self,
        record: FrameRecord,
        resolution: tuple[int, int],
        resolution_scale: float,
        data_device: str = "cuda",
        data_format: str = "colmap",
        scene_translation: np.ndarray | None = None,
        scene_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.uid = record.uid
        self.colmap_id = record.uid
        self.R = record.rotation
        self.T = record.translation
        self.FoVx = record.fov_x
        self.FoVy = record.fov_y
        self.image_name = record.image_name
        self.image_path = record.image_path
        self.resolution_scale = resolution_scale
        self.width = record.width
        self.height = record.height
        self.data_device = torch.device(data_device)
        self.znear = 0.01
        self.zfar = 100.0

        rgba = pil_image_to_tensor(record.image, resolution).to(self.data_device)
        self.original_image = rgba[:3].clamp(0.0, 1.0)
        self.alpha_mask = self._resolve_alpha_mask(record, resolution, rgba)
        self.image_width = int(self.original_image.shape[2])
        self.image_height = int(self.original_image.shape[1])
        self.invdepthmap = None
        self.depth_mask = None

        translation = np.zeros(3, dtype=np.float32) if scene_translation is None else np.asarray(scene_translation, dtype=np.float32)
        world_view = world_to_view_matrix(self.R, self.T, translation, float(scene_scale))
        self.world_view_transform = torch.tensor(world_view, dtype=torch.float32, device=self.data_device).transpose(0, 1)
        self.projection_matrix = projection_matrix(self.znear, self.zfar, self.FoVx, self.FoVy).transpose(0, 1).to(self.data_device)
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0) @ self.projection_matrix.unsqueeze(0)).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

        self.cx = record.cx * resolution[0] / record.image.size[0]
        self.cy = record.cy * resolution[1] / record.image.size[1]
        self.fx = self.image_width / (2.0 * np.tan(self.FoVx * 0.5))
        self.fy = self.image_height / (2.0 * np.tan(self.FoVy * 0.5))
        self.c2w = self.world_view_transform.transpose(0, 1).inverse()
        self.object_mask = self._load_object_mask(resolution)

    def _resolve_alpha_mask(self, record: FrameRecord, resolution: tuple[int, int], rgba: torch.Tensor) -> torch.Tensor:
        if record.alpha_mask is not None:
            return pil_image_to_tensor(record.alpha_mask, resolution).to(self.data_device)
        if rgba.shape[0] == 4:
            return rgba[3:4]
        return torch.ones_like(rgba[:1])

    def _load_object_mask(self, resolution):
        source = Path(self.image_path)
        candidates = [
            Path(str(source).replace("images", "object_mask_deva")).with_suffix(".png"),
            Path(str(source).replace("images_all", "object_mask")).with_suffix(".png"),
            Path(str(source).replace("images", "object_mask")).with_suffix(".png"),
        ]
        for candidate in candidates:
            if candidate == source or not candidate.exists():
                continue
            image = Image.open(candidate).convert("L")
            array = np.array(image.resize(resolution), dtype=np.uint8, copy=True)
            return torch.from_numpy(array)
        return torch.zeros((resolution[1], resolution[0]), dtype=torch.uint8)
