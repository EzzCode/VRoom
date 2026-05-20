"""Anchor field storage and initialization for VRoom."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from vroom_core.utils.geometry import PointCloudSample
from vroom_core.utils.nearest import neighbor_distances
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
    voxel_size: float


class AnchorSeedBuilder:
    def __init__(self, n_offsets: int, feat_dim: int, voxel_size: float = -1.0, device: str = "cuda") -> None:
        self.n_offsets = n_offsets
        self.feat_dim = feat_dim
        self.voxel_size = voxel_size
        self.device = torch.device(device)

    def build(self, points: torch.Tensor, labels: Optional[torch.Tensor] = None, logger=None) -> AnchorSeeds:
        points = points.to(self.device, dtype=torch.float32)
        labels = None if labels is None else labels.to(self.device, dtype=torch.long).view(-1)

        voxel_size = self.voxel_size if self.voxel_size > 0 else self._estimate_voxel_size(points)
        anchors, voxel_labels = self._voxelize(points, labels, voxel_size)
        offsets = torch.zeros((anchors.shape[0], self.n_offsets, 3), dtype=torch.float32, device=self.device)
        features = torch.zeros((anchors.shape[0], self.feat_dim), dtype=torch.float32, device=self.device)
        log_scaling, rotations = self._seed_geometry(anchors, voxel_size)
        codec = SemanticCodec.from_labels(voxel_labels) if voxel_labels is not None else None

        if logger is not None:
            logger.info(f"Initial Point Count: {points.shape[0]}")
            logger.info(f"Initial Anchor Count: {anchors.shape[0]}")
            logger.info(f"Voxel Size: {voxel_size:.6f}")

        return AnchorSeeds(
            anchors=anchors,
            offsets=offsets,
            features=features,
            log_scaling=log_scaling,
            rotations=rotations,
            labels=None if voxel_labels is None else voxel_labels.view(-1, 1),
            codec=codec,
            voxel_size=float(voxel_size),
        )

    def sample_point_cloud(self, point_cloud: PointCloudSample, ratio: int) -> PointCloudSample:
        stride = max(int(ratio), 1)
        return PointCloudSample(
            points=point_cloud.points[::stride],
            colors=point_cloud.colors[::stride],
            normals=point_cloud.normals[::stride],
            label_ids=point_cloud.label_ids[::stride],
        )

    def _estimate_voxel_size(self, points: torch.Tensor) -> float:
        distances = neighbor_distances(points, 4)
        local_scale = distances[:, 1:].pow(2).mean(dim=-1)
        midpoint = max(int(local_scale.numel() * 0.5), 1)
        value, _ = torch.kthvalue(local_scale, midpoint)
        return max(float(value.item()), 1e-6)

    def _voxelize(self, points: torch.Tensor, labels: Optional[torch.Tensor], voxel_size: float) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        grid = torch.round(points / voxel_size).to(torch.int64)
        unique_grid, inverse = torch.unique(grid, return_inverse=True, dim=0)
        anchors = unique_grid.to(torch.float32) * voxel_size
        if labels is None:
            return anchors, None

        voxel_labels = torch.zeros(unique_grid.shape[0], dtype=torch.long, device=self.device)
        for voxel_index in range(unique_grid.shape[0]):
            members = labels[inverse == voxel_index]
            counts = torch.bincount(members)
            voxel_labels[voxel_index] = torch.argmax(counts)
        return anchors, voxel_labels

    def _seed_geometry(self, anchors: torch.Tensor, voxel_size: float) -> tuple[torch.Tensor, torch.Tensor]:
        distances = neighbor_distances(anchors, 4)
        support = distances[:, 1:].mean(dim=-1).clamp(min=max(voxel_size, 1e-6))
        log_scaling = torch.log(support.sqrt()).unsqueeze(-1).repeat(1, 6)
        rotations = torch.zeros((anchors.shape[0], 4), dtype=torch.float32, device=self.device)
        rotations[:, 0] = 1.0
        return log_scaling, rotations


class AnchorField(nn.Module):
    def __init__(self, device: str = "cuda") -> None:
        super().__init__()
        self.device = torch.device(device)
        self.anchor = nn.Parameter(torch.empty((0, 3), dtype=torch.float32, device=self.device))
        self.offset = nn.Parameter(torch.empty((0, 0, 3), dtype=torch.float32, device=self.device))
        self.feature = nn.Parameter(torch.empty((0, 0), dtype=torch.float32, device=self.device))
        self.log_scaling = nn.Parameter(torch.empty((0, 6), dtype=torch.float32, device=self.device))
        self.raw_rotation = nn.Parameter(torch.empty((0, 4), dtype=torch.float32, device=self.device), requires_grad=False)
        self.label_ids: Optional[torch.Tensor] = None
        self.codec: Optional[SemanticCodec] = None
        self.visible = torch.empty((0,), dtype=torch.bool, device=self.device)
        self.voxel_size = -1.0

    def replace(self, seeds: AnchorSeeds) -> None:
        self.anchor = nn.Parameter(seeds.anchors.clone().detach().to(self.device).requires_grad_(True))
        self.offset = nn.Parameter(seeds.offsets.clone().detach().to(self.device).requires_grad_(True))
        self.feature = nn.Parameter(seeds.features.clone().detach().to(self.device).requires_grad_(True))
        self.log_scaling = nn.Parameter(seeds.log_scaling.clone().detach().to(self.device).requires_grad_(True))
        self.raw_rotation = nn.Parameter(seeds.rotations.clone().detach().to(self.device), requires_grad=False)
        self.label_ids = None if seeds.labels is None else seeds.labels.clone().detach().to(self.device)
        self.codec = seeds.codec
        self.visible = torch.ones(self.anchor.shape[0], dtype=torch.bool, device=self.device)
        self.voxel_size = seeds.voxel_size

    def append(self, anchors: torch.Tensor, offsets: torch.Tensor, features: torch.Tensor, log_scaling: torch.Tensor, rotations: torch.Tensor, labels: Optional[torch.Tensor]) -> None:
        self.anchor = nn.Parameter(torch.cat([self.anchor, anchors], dim=0).requires_grad_(True))
        self.offset = nn.Parameter(torch.cat([self.offset, offsets], dim=0).requires_grad_(True))
        self.feature = nn.Parameter(torch.cat([self.feature, features], dim=0).requires_grad_(True))
        self.log_scaling = nn.Parameter(torch.cat([self.log_scaling, log_scaling], dim=0).requires_grad_(True))
        self.raw_rotation = nn.Parameter(torch.cat([self.raw_rotation, rotations], dim=0), requires_grad=False)
        self.visible = torch.ones(self.anchor.shape[0], dtype=torch.bool, device=self.device)
        if labels is not None:
            self.label_ids = labels if self.label_ids is None else torch.cat([self.label_ids, labels], dim=0)
            if self.codec is None:
                self.codec = SemanticCodec.from_labels(self.label_ids.view(-1))
            else:
                self.codec.fit(self.label_ids.view(-1))

    def prune(self, prune_mask: torch.Tensor) -> None:
        keep = ~prune_mask
        self.anchor = nn.Parameter(self.anchor[keep].detach().clone().requires_grad_(True))
        self.offset = nn.Parameter(self.offset[keep].detach().clone().requires_grad_(True))
        self.feature = nn.Parameter(self.feature[keep].detach().clone().requires_grad_(True))
        self.log_scaling = nn.Parameter(self.log_scaling[keep].detach().clone().requires_grad_(True))
        self.raw_rotation = nn.Parameter(self.raw_rotation[keep].detach().clone(), requires_grad=False)
        self.visible = torch.ones(self.anchor.shape[0], dtype=torch.bool, device=self.device)
        if self.label_ids is not None:
            self.label_ids = self.label_ids[keep]
            if self.codec is not None:
                self.codec.fit(self.label_ids.view(-1))

