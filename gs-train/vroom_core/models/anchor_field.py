"""Anchor field storage and initialization (Density-Aware Custom Implementation)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from vroom_core.utils.geometry import PointCloudSample
from vroom_core.models.semantics import SemanticCodec


@dataclass
class AnchorSeeds:
    anchors: torch.Tensor
    offsets: torch.Tensor
    features: torch.Tensor
    log_scaling: torch.Tensor
    rotations: torch.Tensor
    labels: Optional[torch.Tensor]
    codec: Optional[SemanticCodec]
    voxel_size: float  # Kept for compatibility, stores the mean dynamic size


class AnchorSeedBuilder:
    def __init__(
        self,
        n_offsets: int,
        feat_dim: int,
        voxel_size: float = -1.0,  # Legacy parameter, overridden by dynamic logic
        device: str = "cuda",
    ) -> None:
        self.n_offsets = n_offsets
        self.feat_dim = feat_dim
        self.device = torch.device(device)

    def build(
        self, points: torch.Tensor, labels: Optional[torch.Tensor] = None, logger=None
    ) -> AnchorSeeds:
        points = points.to(self.device, dtype=torch.float32)
        labels = (
            None
            if labels is None
            else labels.to(self.device, dtype=torch.long).view(-1)
        )

        # 1. Dynamically estimate the required grid size for every point based on local density
        voxel_sizes = self._estimate_voxel_sizes(points)

        # 2. Voxelize the points and return the winning anchors, their labels, and their specific sizes
        anchors, voxel_labels, anchor_sizes = self._voxelize(
            points, labels, voxel_sizes
        )

        # 3. Create the offsets and features tensors
        offsets = torch.zeros(
            (anchors.shape[0], self.n_offsets, 3),
            dtype=torch.float32,
            device=self.device,
        )
        features = torch.zeros(
            (anchors.shape[0], self.feat_dim), dtype=torch.float32, device=self.device
        )

        # 4. Mathematically deduce geometry based on the exact grid size the anchor lives in
        log_scaling, rotations = self._seed_geometry(anchors, anchor_sizes)

        # 5. Fit the semantic codec
        codec = (
            SemanticCodec.from_labels(voxel_labels)
            if voxel_labels is not None
            else None
        )

        if logger is not None:
            logger.info(f"Initial Point Count: {points.shape[0]}")
            logger.info(f"Generated Anchor Count: {anchors.shape[0]}")
            logger.info(
                f"Dynamic Voxel Size Range: [{voxel_sizes.min().item():.6f} -> {voxel_sizes.max().item():.6f}]"
            )

        return AnchorSeeds(
            anchors=anchors,
            offsets=offsets,
            features=features,
            log_scaling=log_scaling,
            rotations=rotations,
            labels=None if voxel_labels is None else voxel_labels.view(-1, 1),
            codec=codec,
            voxel_size=float(voxel_sizes.mean().item()),
        )

    def sample_point_cloud(
        self, point_cloud: PointCloudSample, ratio: int
    ) -> PointCloudSample:
        stride = max(int(ratio), 1)
        return PointCloudSample(
            points=point_cloud.points[::stride],
            colors=point_cloud.colors[::stride],
            normals=point_cloud.normals[::stride],
            label_ids=point_cloud.label_ids[::stride],
        )

    def _estimate_voxel_sizes(self, points: torch.Tensor) -> torch.Tensor:
        """Calculates dynamic voxel sizes based on 3D hash-grid point density."""
        scene_min = points.min(dim=0).values
        scene_max = points.max(dim=0).values
        scene_extent = (scene_max - scene_min).max().item()

        coarse_size = scene_extent / 32.0
        coarse_idx = torch.floor((points - scene_min) / coarse_size).long().clamp(0, 31)

        # Flatten 3D index to a 1D key
        cell_keys = coarse_idx[:, 0] * 1024 + coarse_idx[:, 1] * 32 + coarse_idx[:, 2]

        # Count points per cell
        counts = torch.bincount(cell_keys, minlength=32**3).float()
        counts = counts.clamp(min=1)

        # Dense cell → small voxel, sparse cell → large voxel
        per_point_counts = counts[cell_keys]
        max_count = counts.max()
        voxel_sizes = coarse_size * (1.0 - 0.75 * (per_point_counts / max_count))

        return voxel_sizes  # (N,) range: [coarse/4, coarse]

    def _voxelize(
        self,
        points: torch.Tensor,
        labels: Optional[torch.Tensor],
        voxel_sizes: torch.Tensor,
    ):
        """Quantizes points using their dynamic sizes. Fine details override coarse ones."""
        quantized = torch.round(
            points / voxel_sizes.unsqueeze(1)
        ) * voxel_sizes.unsqueeze(1)

        # Round to 5 decimal places to make floating point matching stable
        quantized_key = (quantized * 1e4).round().long()
        flat_keys = (
            quantized_key[:, 0] * (10**8)
            + quantized_key[:, 1] * (10**4)
            + quantized_key[:, 2]
        )

        # Survival of the finest: Among duplicates, keep the one with the smallest voxel size
        sorted_eps, sorted_idx = voxel_sizes.sort()
        seen = {}
        keep = []
        for i in sorted_idx.tolist():
            k = flat_keys[i].item()
            if k not in seen:
                seen[k] = True
                keep.append(i)

        keep = torch.tensor(keep, dtype=torch.long, device=points.device)
        anchors = quantized[keep]
        anchor_sizes = voxel_sizes[keep]

        if labels is None:
            return anchors, None, anchor_sizes

        anchor_labels = labels[keep]
        return anchors, anchor_labels, anchor_sizes

    def _seed_geometry(self, anchors: torch.Tensor, anchor_sizes: torch.Tensor):
        """Mathematically deduces the perfect starting scale without KNN search."""
        support = anchor_sizes.clamp(min=1e-6).sqrt()
        log_scaling = torch.log(support).unsqueeze(-1).repeat(1, 6)

        rotations = torch.zeros(
            (anchors.shape[0], 4), dtype=torch.float32, device=anchors.device
        )
        rotations[:, 0] = 1.0  # Identity quaternion

        return log_scaling, rotations


class AnchorField(nn.Module):
    """Stores and manages the learnable parameters of the anchor field."""

    def __init__(self, device: str = "cuda") -> None:
        super().__init__()
        self.device = torch.device(device)
        self.anchor = nn.Parameter(
            torch.empty((0, 3), dtype=torch.float32, device=self.device)
        )
        self.offset = nn.Parameter(
            torch.empty((0, 0, 3), dtype=torch.float32, device=self.device)
        )
        self.feature = nn.Parameter(
            torch.empty((0, 0), dtype=torch.float32, device=self.device)
        )
        self.log_scaling = nn.Parameter(
            torch.empty((0, 6), dtype=torch.float32, device=self.device)
        )
        self.raw_rotation = nn.Parameter(
            torch.empty((0, 4), dtype=torch.float32, device=self.device),
            requires_grad=False,
        )
        self.label_ids: Optional[torch.Tensor] = None
        self.codec: Optional[SemanticCodec] = None
        self.visible = torch.empty((0,), dtype=torch.bool, device=self.device)
        self.voxel_size = -1.0

    def replace(self, seeds: AnchorSeeds) -> None:
        self.anchor = nn.Parameter(
            seeds.anchors.clone().detach().to(self.device).requires_grad_(True)
        )
        self.offset = nn.Parameter(
            seeds.offsets.clone().detach().to(self.device).requires_grad_(True)
        )
        self.feature = nn.Parameter(
            seeds.features.clone().detach().to(self.device).requires_grad_(True)
        )
        self.log_scaling = nn.Parameter(
            seeds.log_scaling.clone().detach().to(self.device).requires_grad_(True)
        )
        self.raw_rotation = nn.Parameter(
            seeds.rotations.clone().detach().to(self.device), requires_grad=False
        )
        self.label_ids = (
            None
            if seeds.labels is None
            else seeds.labels.clone().detach().to(self.device)
        )
        self.codec = seeds.codec
        self.visible = torch.ones(
            self.anchor.shape[0], dtype=torch.bool, device=self.device
        )
        self.voxel_size = seeds.voxel_size

    def append(
        self,
        anchors: torch.Tensor,
        offsets: torch.Tensor,
        features: torch.Tensor,
        log_scaling: torch.Tensor,
        rotations: torch.Tensor,
        labels: Optional[torch.Tensor],
    ) -> None:
        self.anchor = nn.Parameter(
            torch.cat([self.anchor, anchors], dim=0).requires_grad_(True)
        )
        self.offset = nn.Parameter(
            torch.cat([self.offset, offsets], dim=0).requires_grad_(True)
        )
        self.feature = nn.Parameter(
            torch.cat([self.feature, features], dim=0).requires_grad_(True)
        )
        self.log_scaling = nn.Parameter(
            torch.cat([self.log_scaling, log_scaling], dim=0).requires_grad_(True)
        )
        self.raw_rotation = nn.Parameter(
            torch.cat([self.raw_rotation, rotations], dim=0), requires_grad=False
        )
        self.visible = torch.ones(
            self.anchor.shape[0], dtype=torch.bool, device=self.device
        )
        if labels is not None:
            self.label_ids = (
                labels
                if self.label_ids is None
                else torch.cat([self.label_ids, labels], dim=0)
            )
            if self.codec is None:
                self.codec = SemanticCodec.from_labels(self.label_ids.view(-1))
            else:
                self.codec.fit(self.label_ids.view(-1))

    def prune(self, prune_mask: torch.Tensor) -> None:
        keep = ~prune_mask
        self.anchor = nn.Parameter(
            self.anchor[keep].detach().clone().requires_grad_(True)
        )
        self.offset = nn.Parameter(
            self.offset[keep].detach().clone().requires_grad_(True)
        )
        self.feature = nn.Parameter(
            self.feature[keep].detach().clone().requires_grad_(True)
        )
        self.log_scaling = nn.Parameter(
            self.log_scaling[keep].detach().clone().requires_grad_(True)
        )
        self.raw_rotation = nn.Parameter(
            self.raw_rotation[keep].detach().clone(), requires_grad=False
        )
        self.visible = torch.ones(
            self.anchor.shape[0], dtype=torch.bool, device=self.device
        )
        if self.label_ids is not None:
            self.label_ids = self.label_ids[keep]
            if self.codec is not None:
                self.codec.fit(self.label_ids.view(-1))
