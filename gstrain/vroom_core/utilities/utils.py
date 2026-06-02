from __future__ import annotations
import os
import math
import random
from dataclasses import dataclass
from typing import Tuple, Callable
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from plyfile import PlyData, PlyElement


def ensure_directory(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def seed_everything(seed: int = 0, quiet: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def exponential_lr_schedule(
    lr_init: float,
    lr_final: float,
    lr_delay_steps: int = 0,
    lr_delay_mult: float = 1.0,
    max_steps: int = 1_000_000,
    warmup_steps: int = 0,
) -> Callable[[int], float]:
    def schedule(step: int) -> float:
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            return 0.0
        if warmup_steps > 0 and step < warmup_steps:
            alpha = step / warmup_steps
            return lr_init * (lr_delay_mult + (1.0 - lr_delay_mult) * alpha)
        if lr_delay_steps > 0:
            delay = lr_delay_mult + (1.0 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay = 1.0
        t = np.clip(step / max_steps, 0, 1)
        interpolated = np.exp(
            np.log(max(lr_init, 1e-20)) * (1.0 - t) + np.log(max(lr_final, 1e-20)) * t
        )
        if lr_init == 0.0 and lr_final == 0.0:
            return 0.0
        return float(delay * interpolated)

    return schedule


get_expon_lr_func = exponential_lr_schedule


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
    offset = (
        np.zeros(3, dtype=np.float32)
        if offset is None
        else np.asarray(offset, dtype=np.float32)
    )
    view = np.eye(4, dtype=np.float32)
    view[:3, :3] = rotation.T
    view[:3, 3] = np.asarray(translation, dtype=np.float32)

    camera_to_world = np.linalg.inv(view)
    camera_to_world[:3, 3] = (camera_to_world[:3, 3] + offset) * scale
    return np.linalg.inv(camera_to_world).astype(np.float32)


def projection_matrix(
    znear: float, zfar: float, fov_x: float, fov_y: float
) -> torch.Tensor:
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


def compute_anchors_scale_and_rotation(
    anchors_positions: torch.Tensor,
    distances: torch.Tensor,
    voxel_size: float,
    device: torch.device | str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute scale and rotation for anchors based on distances and voxel size"""
    area_of_effect = distances[:, 1:].mean(dim=-1).clamp(min=max(voxel_size, 1e-6))
    log_scaling = torch.log(area_of_effect.sqrt()).unsqueeze(-1).repeat(1, 6)
    rotations = torch.zeros(
        (anchors_positions.shape[0], 4),
        dtype=torch.float32,
        device=device,
    )
    rotations[:, 0] = 1.0  # identity quaternion
    return log_scaling, rotations


def estimate_voxel_size(knn_distances: torch.Tensor, min_size: float = 1e-6) -> float:
    """
    Estimates a voxel size for a uniform grid using knn distances
    """
    voxel_size = torch.median(knn_distances[:, 1:]).item()
    print(f"voxel size calculated = {voxel_size}")
    return max(voxel_size, min_size)


def calc_volumetric_loss(scales: torch.Tensor, volume_lambda: float) -> torch.Tensor:
    """Calculates volumetric loss from scaling factors."""
    volumes = torch.prod(
        scales, dim=1
    )  # multiply the x, y and z scales to get the volumes of each gaussian
    return torch.mean(volumes) * volume_lambda


class CheckpointManager:
    def __init__(self, anchor_cloud, decoder) -> None:
        self.anchor_cloud = anchor_cloud
        self.decoder = decoder
        self.device = anchor_cloud.device

    def save_anchor_cloud(self, path: str) -> None:
        field = self.anchor_cloud
        ensure_directory(os.path.dirname(path))
        anchor = field.anchors_positions.detach().cpu().numpy()
        offsets = (
            field.gaussians_offsets.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        features = field.anchor_features.detach().cpu().numpy()
        scales = field.anchors_log_scales.detach().cpu().numpy()
        rotations = field.anchors_rotations.detach().cpu().numpy()
        labels = (
            field.semantic_labels.detach().cpu().numpy().reshape(-1, 1)
            if field.semantic_labels is not None
            else np.zeros((anchor.shape[0], 1), dtype=np.uint8)
        )

        names = ["x", "y", "z"]
        names += [f"f_offset_{idx}" for idx in range(offsets.shape[1])]
        names += [f"f_anchor_feat_{idx}" for idx in range(features.shape[1])]
        names += [f"scale_{idx}" for idx in range(scales.shape[1])]
        names += [f"rot_{idx}" for idx in range(rotations.shape[1])]
        names.append("label")
        dtype = [(name, "f4") for name in names]
        rows = np.concatenate(
            [anchor, offsets, features, scales, rotations, labels], axis=1
        )
        elements = np.empty(anchor.shape[0], dtype=dtype)
        elements[:] = list(map(tuple, rows))
        PlyData(
            [PlyElement.describe(elements, "vertex")],
            obj_info=[f"num_anchor {anchor.shape[0]:.6f}"],
        ).write(path)

    def load_anchor_field(self, path: str) -> dict:
        ply = PlyData.read(path).elements[0]
        anchor = np.stack(
            [np.asarray(ply["x"]), np.asarray(ply["y"]), np.asarray(ply["z"])], axis=1
        ).astype(np.float32)
        features = self._stack_prefixed(ply, "f_anchor_feat_")
        offset_flat = self._stack_prefixed(ply, "f_offset_")
        scales = self._stack_prefixed(ply, "scale_")
        rotations = self._stack_prefixed(ply, "rot_")
        labels = (
            np.asarray(ply["label"])[..., None].astype(np.int64)
            if "label" in [prop.name for prop in ply.properties]
            else None
        )
        return {
            "anchor": torch.tensor(anchor, dtype=torch.float32, device=self.device),
            "offset": torch.tensor(
                offset_flat.reshape(anchor.shape[0], 3, -1),
                dtype=torch.float32,
                device=self.device,
            )
            .transpose(1, 2)
            .contiguous(),
            "feature": torch.tensor(features, dtype=torch.float32, device=self.device),
            "log_scaling": torch.tensor(
                scales, dtype=torch.float32, device=self.device
            ),
            "rotation": torch.tensor(
                rotations, dtype=torch.float32, device=self.device
            ),
            "labels": None
            if labels is None
            else torch.tensor(labels, dtype=torch.long, device=self.device),
        }

    def save_decoder(
        self,
        path: str,
        gaussian_type: str = "3D",
        render_mode: str = "RGB+ED",
        tile_size_2dgs: int = 8,
    ) -> None:
        ensure_directory(path)
        self.decoder.eval()
        feat_dim = self.decoder.feature_dim
        view_dim = 3
        appearance_dim = 0

        opacity_mlp = self.decoder.opacity_network
        covariance_mlp = self.decoder.covariance_network
        color_mlp = self.decoder.color_network

        input_dim = feat_dim + view_dim

        torch.jit.trace(opacity_mlp, torch.rand(1, input_dim, device=self.device)).save(
            os.path.join(path, "opacity_mlp.pt")
        )
        torch.jit.trace(
            covariance_mlp, torch.rand(1, input_dim, device=self.device)
        ).save(os.path.join(path, "cov_mlp.pt"))
        torch.jit.trace(
            color_mlp,
            torch.rand(1, input_dim + appearance_dim, device=self.device),
        ).save(os.path.join(path, "color_mlp.pt"))

        torch.save(
            {
                "n_offsets": self.decoder.number_gaussians_per_anchor,
                "feat_dim": feat_dim,
                "view_dim": view_dim,
                "appearance_dim": appearance_dim,
                "gs_attr": gaussian_type,
                "render_mode": render_mode,
                "tile_size_2dgs": tile_size_2dgs,
            },
            os.path.join(path, "vroom_bundle.pt"),
        )
        self.decoder.train()

    def load_decoder(self, path: str) -> None:
        self.decoder.opacity_network = torch.jit.load(
            os.path.join(path, "opacity_mlp.pt")
        ).to(self.device)
        self.decoder.covariance_network = torch.jit.load(
            os.path.join(path, "cov_mlp.pt")
        ).to(self.device)
        self.decoder.color_network = torch.jit.load(
            os.path.join(path, "color_mlp.pt")
        ).to(self.device)

    def infer_bundle_kwargs(self, iteration_dir: Path) -> dict:
        bundle_path = iteration_dir / "vroom_bundle.pt"
        if bundle_path.exists():
            return torch.load(bundle_path, map_location="cpu")

        temp = self.load_anchor_field(str(iteration_dir / "point_cloud.ply"))
        opacity_mlp = torch.jit.load(
            str(iteration_dir / "opacity_mlp.pt"), map_location="cpu"
        )
        color_mlp = torch.jit.load(
            str(iteration_dir / "color_mlp.pt"), map_location="cpu"
        )
        opacity_params = dict(opacity_mlp.named_parameters())
        color_params = dict(color_mlp.named_parameters())
        first_opacity_weight = next(
            param for name, param in opacity_params.items() if name.endswith("weight")
        )
        last_opacity_weight = list(opacity_params.values())[-2]
        first_color_weight = next(
            param for name, param in color_params.items() if name.endswith("weight")
        )
        feat_dim = temp["feature"].shape[1]
        view_dim = first_opacity_weight.shape[1] - feat_dim
        appearance_dim = first_color_weight.shape[1] - feat_dim - view_dim
        return {
            "n_offsets": int(last_opacity_weight.shape[0]),
            "feat_dim": int(feat_dim),
            "view_dim": int(view_dim),
            "appearance_dim": int(appearance_dim),
            "gs_attr": "3D",
            "render_mode": "RGB+ED",
            "tile_size_2dgs": 8,
        }

    def _stack_prefixed(self, element, prefix: str) -> np.ndarray:
        names = sorted(
            [prop.name for prop in element.properties if prop.name.startswith(prefix)],
            key=lambda name: int(name.split("_")[-1]),
        )
        if not names:
            return np.zeros((len(element["x"]), 0), dtype=np.float32)
        return np.stack([np.asarray(element[name]) for name in names], axis=1).astype(
            np.float32
        )


class SemanticsManager:
    def __init__(self, label_ids):
        # Ensure 0 (which represents unknown or background) is always in label_ids for saftey
        unique_labels = label_ids.view(-1)
        if 0 not in unique_labels:
            unique_labels = torch.cat(
                [
                    torch.tensor(
                        [0], dtype=unique_labels.dtype, device=unique_labels.device
                    ),
                    unique_labels,
                ]
            )
        self.label_ids, _ = torch.sort(unique_labels)
        self.num_classes = len(self.label_ids)

    def build_lookup_table(self, labels):
        """
        map each unique label to an index for one hot encoding
        """
        self.label_ids = self.label_ids.to(labels.device)
        labels_indices = torch.bucketize(labels, self.label_ids)

        # use clamping to make sure the incoming label is within the existing labels
        # if it isnt we clamp it
        clamped_indices = torch.clamp(labels_indices, 0, self.label_ids.size(0) - 1)
        # after clamping we check if this label is truly in the correct index
        # if it isnt its false
        mask_match = self.label_ids[clamped_indices] == labels
        # set unknown labels to zero
        labels_indices = torch.where(
            mask_match, labels_indices, torch.zeros_like(labels_indices)
        )

        return labels_indices

    def one_hot_encode(self, labels_indices):
        return F.one_hot(labels_indices, num_classes=self.num_classes)

    def one_hot_decode(self, one_hot, num_classes):
        indices = torch.argmax(one_hot, dim=1)
        self.label_ids = self.label_ids.to(one_hot.device)
        return self.label_ids[indices]

    def update_current_num_classes(self, labels):
        self.num_classes = len(torch.unique(labels))
