"""Anchor field storage and initialization for VRoom."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors

from vroom_core.utils.geometry import PointCloudSample
from vroom_core.models.semantics import SemanticCodec


@dataclass
class AnchorCloudData:
    anchors_positions: torch.Tensor
    anchor_features: torch.Tensor
    anchors_log_scales: torch.Tensor
    anchors_rotations: torch.Tensor
    labels: Optional[torch.Tensor]
    codec: Optional[SemanticCodec]
    gaussians_offsets: torch.Tensor
    voxel_size: float


class AnchorCloud(nn.Module):
    """
    Anchor cloud is the memory bank of the anchors, features, and gaussians.
    """

    def __init__(
        self,
        point_cloud=None,
        voxel_size=None,
        density_mode=False,
        gaussians_per_anchor=5,
        feature_dim=32,
        codec=None,
        device="cuda",
    ):
        super().__init__()
        self.device = device
        self.gaussians_per_anchor = gaussians_per_anchor
        self.voxel_size = voxel_size
        self.density_mode = density_mode
        self.visibility_mask = torch.empty((0,), dtype=torch.bool, device=self.device)
        self.codec = codec
        self.feature_dim = feature_dim

        self.anchors_positions = nn.Parameter(
            torch.empty((0, 3), dtype=torch.float32, device=self.device)
        )

        self.anchor_features = nn.Parameter(
            torch.empty(
                (0, self.feature_dim),
                dtype=torch.float32,
                device=self.device,
            )
        )

        self.anchors_log_scales = nn.Parameter(
            torch.empty((0, 6), dtype=torch.float32, device=self.device)
        )  # log is taken in case optimizer made scale a negative number

        self.anchors_rotations = nn.Parameter(
            torch.empty((0, 4), dtype=torch.float32, device=self.device),
            requires_grad=False,
        )
        self.semantic_labels: Optional[torch.Tensor] = None
        self.gaussians_offsets = nn.Parameter(
            torch.empty(
                (0, gaussians_per_anchor, 3), dtype=torch.float32, device=self.device
            )
        )

    def _generate_anchors(self, point_cloud, point_labels):
        """
        Intialize the anchor cloud from the sparse point cloud by quantization
        """

        if self.density_mode:
            self.voxel_size = self._estimate_voxel_size(point_cloud)
        else:
            self.voxel_size = self._estimate_voxel_size(point_cloud)

        # voxelize the point cloud
        unique_voxels, inversed_indices = self._quantize_cloud(
            point_cloud, self.voxel_size
        )  # updates anchors tensor

        self.anchors_positions = nn.Parameter(
            (unique_voxels * self.voxel_size).float()
        )  # quantized points (anchors)

        # resolve label for each anchor based on majority vote of the points in the voxel
        self.anchor_labels = self._majority_vote(
            unique_voxels, inversed_indices, point_labels
        )

        # set scale and rotation for each anchor
        log_scales, rotations = self._set_anchors_scale_and_rotation()
        self.anchors_log_scales = nn.Parameter(log_scales)
        self.anchors_rotations = nn.Parameter(rotations, requires_grad=False)

        self.anchor_features = nn.Parameter(
            torch.zeros(
                (self.anchors_positions.shape[0], self.feature_dim),
                dtype=torch.float32,
                device=self.device,
            )
        )

        self.codec = SemanticCodec.from_labels(self.anchor_labels)

        self.gaussians_offsets = nn.Parameter(
            torch.zeros(
                (self.anchors_positions.shape[0], self.gaussians_per_anchor, 3),
                dtype=torch.float32,
                device=self.device,
            )
        )
        print(f"Generated {self.anchors_positions.shape[0]} anchors")
        print(f"voxel size = {self.voxel_size}")

        return AnchorCloudData(
            anchors_positions=self.anchors_positions,
            anchor_features=self.anchor_features,
            anchors_log_scales=self.anchors_log_scales,
            anchors_rotations=self.anchors_rotations,
            labels=self.anchor_labels,
            codec=self.codec,
            gaussians_offsets=self.gaussians_offsets,
            voxel_size=self.voxel_size,
        )

    def _estimate_voxel_size(self, point_cloud):
        """
        if density_mode mode is off we use knn to esitmate a voxel size for a uniform grid
        """
        distances = self._knn(point_cloud, k=4)
        voxel_size = torch.median(distances[:, 1:]).item()
        return max(voxel_size, 1e-6)

    def _knn(self, point_cloud, k):
        """finds the k nearest neighbors of each point in the point cloud"""
        knn_model = NearestNeighbors(n_neighbors=k, algorithm="ball_tree").fit(
            point_cloud.detach().cpu().numpy()
        )
        distances, _ = knn_model.kneighbors(point_cloud.detach().cpu().numpy())
        return torch.from_numpy(distances).float().to(self.device)

    def _quantize_cloud(self, point_cloud, voxel_size):
        """
        Quantize the point cloud based on the voxel size
        """
        quantized_grid = (point_cloud / voxel_size).to(torch.int64)
        # use torch.unique to get the unique voxels and their counts
        unique_voxels, inversed_indices, counts = torch.unique(
            quantized_grid, dim=0, return_counts=True, return_inverse=True
        )

        return (
            unique_voxels,
            inversed_indices,
        )  # inversed ind will map each point to the voxel index

    def _majority_vote(self, unique_voxels, inversed_indices, labels):
        """
        Loops through the voxels and for each voxel I apply the majority rule to choose a label for this voxel's anchor
        """
        if labels is None:
            return None
        anchor_labels = torch.zeros(unique_voxels.shape[0], dtype=torch.long)
        for voxel_index in range(unique_voxels.shape[0]):
            label_for_voxel = labels[inversed_indices == voxel_index]
            if len(label_for_voxel) > 0:
                counts = torch.bincount(label_for_voxel)
                anchor_labels[voxel_index] = torch.argmax(counts)
        return anchor_labels

    def _set_anchors_scale_and_rotation(self):
        """Set scales for anchors and set its rotation to identity"""
        distances = self._knn(self.anchors_positions, k=4)
        area_of_effect = (
            distances[:, 1:].mean(dim=-1).clamp(min=max(self.voxel_size, 1e-6))
        )
        log_scaling = torch.log(area_of_effect.sqrt()).unsqueeze(-1).repeat(1, 6)
        rotations = torch.zeros(
            (self.anchors_positions.shape[0], 4),
            dtype=torch.float32,
            device=self.device,
        )
        rotations[:, 0] = 1.0  # identity quaternion
        return log_scaling, rotations

    def set_anchors_cloud(self, data: AnchorCloudData) -> None:
        """set the anchor cloud"""
        self.anchors_positions = nn.Parameter(
            data.anchors_positions.clone().detach().to(self.device).requires_grad_(True)
        )
        self.gaussians_offsets = nn.Parameter(
            data.gaussians_offsets.clone().detach().to(self.device).requires_grad_(True)
        )
        self.anchor_features = nn.Parameter(
            data.anchor_features.clone().detach().to(self.device).requires_grad_(True)
        )
        self.anchors_log_scales = nn.Parameter(
            data.anchors_log_scales.clone().detach().to(self.device).requires_grad_(True)
        )
        self.anchors_rotations = nn.Parameter(
            data.anchors_rotations.clone().detach().to(self.device), requires_grad=False
        )
        self.semantic_labels = (
            None
            if data.labels is None
            else data.labels.clone().detach().to(self.device)
        )
        self.codec = data.codec
        self.visibility_mask = torch.ones(
            self.anchors_positions.shape[0], dtype=torch.bool, device=self.device
        )
        self.voxel_size = data.voxel_size

    def append(
        self,
        anchors_positions: torch.Tensor,
        gaussians_offsets: torch.Tensor,
        anchor_features: torch.Tensor,
        anchors_log_scales: torch.Tensor,
        anchors_rotations: torch.Tensor,
        labels: Optional[torch.Tensor],
    ) -> None:
        """Appends new anchors to the anchor cloud after growing process"""
        self.anchors_positions = nn.Parameter(
            torch.cat(
                [self.anchors_positions, anchors_positions], dim=0
            ).requires_grad_(True)
        )
        self.gaussians_offsets = nn.Parameter(
            torch.cat(
                [self.gaussians_offsets, gaussians_offsets], dim=0
            ).requires_grad_(True)
        )
        self.anchor_features = nn.Parameter(
            torch.cat([self.anchor_features, anchor_features], dim=0).requires_grad_(
                True
            )
        )
        self.anchors_log_scales = nn.Parameter(
            torch.cat(
                [self.anchors_log_scales, anchors_log_scales], dim=0
            ).requires_grad_(True)
        )
        self.anchors_rotations = nn.Parameter(
            torch.cat([self.anchors_rotations, anchors_rotations], dim=0),
            requires_grad=False,
        )
        self.visibility_mask = torch.ones(
            self.anchors_positions.shape[0], dtype=torch.bool, device=self.device
        )
        # if labels are
        if labels is not None:
            self.semantic_labels = (
                labels.view(-1)
                if self.semantic_labels is None
                else torch.cat([self.semantic_labels.view(-1), labels.view(-1)], dim=0)
            )
            if self.codec is None:
                self.codec = SemanticCodec.from_labels(self.semantic_labels.view(-1))
            else:
                self.codec.fit(self.semantic_labels.view(-1))

    def prune(self, prune_mask: torch.Tensor) -> None:
        """Prunes the anchor cloud after pruning process"""
        keep = ~prune_mask
        self.anchors_positions = nn.Parameter(
            self.anchors_positions[keep].detach().clone().requires_grad_(True)
        )
        self.gaussians_offsets = nn.Parameter(
            self.gaussians_offsets[keep].detach().clone().requires_grad_(True)
        )
        self.anchor_features = nn.Parameter(
            self.anchor_features[keep].detach().clone().requires_grad_(True)
        )
        self.anchors_log_scales = nn.Parameter(
            self.anchors_log_scales[keep].detach().clone().requires_grad_(True)
        )
        self.anchors_rotations = nn.Parameter(
            self.anchors_rotations[keep].detach().clone(), requires_grad=False
        )
        self.visibility_mask = torch.ones(
            self.anchors_positions.shape[0], dtype=torch.bool, device=self.device
        )
        if self.semantic_labels is not None:
            self.semantic_labels = self.semantic_labels[keep]
            if self.codec is not None:
                self.codec.fit(self.semantic_labels.view(-1))
