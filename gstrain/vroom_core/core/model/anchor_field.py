from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from gstrain.vroom_core.utilities.utils import (
    compute_anchors_scale_and_rotation,
    SemanticsManager,
)


@dataclass
class AnchorCloudData:
    anchors_positions: torch.Tensor
    anchor_features: torch.Tensor
    anchors_log_scales: torch.Tensor
    anchors_rotations: torch.Tensor
    labels: Optional[torch.Tensor]
    semantic_manager: Optional[SemanticsManager]
    gaussians_offsets: torch.Tensor
    quantization_size: float


class AnchorCloud(nn.Module):
    """
    Anchor cloud is the memory bank of the anchors, features, and gaussians
    """

    def __init__(
        self,
        gaussians_per_anchor,
        feature_dim,
        knn_k,
        knn_chunk_size,
        min_quantization_size,
        point_cloud=None,
        quantization_size=None,
        density_mode=False,
        semantic_manager=None,
        device="cuda",
    ):
        super().__init__()
        self.device = device
        self.gaussians_per_anchor = gaussians_per_anchor
        self.quantization_size = quantization_size
        self.density_mode = density_mode
        self.visibility_mask = torch.empty((0,), dtype=torch.bool, device=self.device)
        self.semantic_manager = semantic_manager
        self.feature_dim = feature_dim
        self.knn_k = knn_k
        self.knn_chunk_size = knn_chunk_size
        self.min_quantization_size = min_quantization_size

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

    @property
    def num_anchors(self):
        return self.anchors_positions.shape[0]

    def initialize_anchors(self, point_cloud_sampled):
        if hasattr(point_cloud_sampled, "points"):
            points = (
                torch.from_numpy(point_cloud_sampled.points).float().to(self.device)
            )
            labels = (
                torch.from_numpy(point_cloud_sampled.label_ids).long().to(self.device)
                if getattr(point_cloud_sampled, "label_ids", None) is not None
                else None
            )
        else:
            raise ValueError("no points in point cloud")

        anchor_cloud = self._generate_anchors(points, labels)
        self.set_anchors_cloud(anchor_cloud)

    def estimate_quantization_size(self, knn_distances, min_size):
        """
        Estimates a quantization size for a uniform grid using knn distances
        """
        print("estimating quantization size enabled")
        quantization_size = torch.median(knn_distances[:, 1:]).item()
        return max(quantization_size, min_size)

    def _generate_anchors(self, point_cloud, point_labels):
        """
        Intialize the anchor cloud from the sparse point cloud by quantization
        """
        distances_between_points = self._knn(
            point_cloud, k=self.knn_k, chunk_size=self.knn_chunk_size
        )

        if self.quantization_size is None:
            self.quantization_size = self.estimate_quantization_size(
                distances_between_points, self.min_quantization_size
            )

        # voxelize the point cloud
        unique_voxels, inversed_indices = self._quantize_cloud(
            point_cloud, self.quantization_size
        )  # updates anchors tensor

        self.anchors_positions = nn.Parameter(
            (unique_voxels * self.quantization_size).float()
        )  # quantized points (anchors)

        distances_between_anchors = self._knn(
            self.anchors_positions, k=self.knn_k, chunk_size=self.knn_chunk_size
        )

        # resolve label for each anchor based on majority vote of the points in the voxel
        self.anchor_labels = self._majority_vote(
            unique_voxels, inversed_indices, point_labels
        )

        # set scale and rotation for each anchor
        log_scales, rotations = compute_anchors_scale_and_rotation(
            self.anchors_positions,
            distances_between_anchors,
            self.quantization_size,
            self.device,
        )
        self.anchors_log_scales = nn.Parameter(log_scales)
        self.anchors_rotations = nn.Parameter(rotations, requires_grad=False)

        self.anchor_features = nn.Parameter(
            torch.zeros(
                (self.anchors_positions.shape[0], self.feature_dim),
                dtype=torch.float32,
                device=self.device,
            )
        )

        self.semantic_manager = (
            SemanticsManager(torch.unique(self.anchor_labels))
            if self.anchor_labels is not None
            else None
        )

        self.gaussians_offsets = nn.Parameter(
            torch.zeros(
                (self.anchors_positions.shape[0], self.gaussians_per_anchor, 3),
                dtype=torch.float32,
                device=self.device,
            )
        )

        return AnchorCloudData(
            anchors_positions=self.anchors_positions,
            anchor_features=self.anchor_features,
            anchors_log_scales=self.anchors_log_scales,
            anchors_rotations=self.anchors_rotations,
            labels=self.anchor_labels,
            semantic_manager=self.semantic_manager,
            gaussians_offsets=self.gaussians_offsets,
            quantization_size=self.quantization_size,
        )

    def _knn(self, point_cloud, k, chunk_size):
        """finds the k nearest neighbors of each point in the point cloud"""
        with torch.no_grad():
            final_distances = torch.empty((point_cloud.shape[0], k), device=self.device)
            for i in range(0, point_cloud.shape[0], chunk_size):
                end = min(i + chunk_size, point_cloud.shape[0])
                chunk = point_cloud[i:end]
                diff = chunk.unsqueeze(1) - point_cloud.unsqueeze(0)
                dist_matrix = torch.norm(diff, dim=-1)
                chunk_distances = torch.topk(dist_matrix, k + 1, largest=False).values
                final_distances[i:end] = chunk_distances[:, 1:]
        return final_distances

    def _quantize_cloud(self, point_cloud, quantization_size):
        """
        Quantize the point cloud based on the quantization size
        """
        quantized_grid = (point_cloud / quantization_size).to(torch.int64)
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
            label_for_voxel = labels[
                inversed_indices == voxel_index
            ]  # contains the labels for the points at a certain voxel
            if len(label_for_voxel) > 0:
                counts = torch.bincount(label_for_voxel)
                anchor_labels[voxel_index] = torch.argmax(
                    counts
                )  # returns the highest voted label for the voxel
        return anchor_labels

    def set_anchors_cloud(self, data: AnchorCloudData):
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
            data.anchors_log_scales.clone()
            .detach()
            .to(self.device)
            .requires_grad_(True)
        )
        self.anchors_rotations = nn.Parameter(
            data.anchors_rotations.clone().detach().to(self.device), requires_grad=False
        )
        self.semantic_labels = (
            None
            if data.labels is None
            else data.labels.clone().detach().to(self.device)
        )
        self.semantic_manager = data.semantic_manager
        self.visibility_mask = torch.ones(
            self.anchors_positions.shape[0], dtype=torch.bool, device=self.device
        )
        self.quantization_size = data.quantization_size

    def append(
        self,
        anchors_positions,
        gaussians_offsets,
        anchor_features,
        anchors_log_scales,
        anchors_rotations,
        labels,
    ):
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
            if self.semantic_manager is None:
                self.semantic_manager = SemanticsManager(
                    torch.unique(self.semantic_labels.view(-1))
                )
            else:
                self.semantic_manager.update_current_num_classes(
                    self.semantic_labels.view(-1)
                )
                self.semantic_manager.label_ids, _ = torch.sort(
                    torch.unique(self.semantic_labels.view(-1))
                )

    def prune(self, prune_mask):
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
            if self.semantic_manager is not None:
                self.semantic_manager.update_current_num_classes(
                    self.semantic_labels.view(-1)
                )
                self.semantic_manager.label_ids, _ = torch.sort(
                    torch.unique(self.semantic_labels.view(-1))
                )

    def instantiate_gaussian_positions(self, visible_mask, negative_opacity_filter):
        """Calculate the positions of the visible Gaussians"""
        anchor_positions = self.anchors_positions[visible_mask]
        num_visible = anchor_positions.shape[0]
        gaussian_offsets = self.gaussians_offsets[visible_mask]
        anchor_scales = torch.exp(self.anchors_log_scales[visible_mask])

        valid_scales = anchor_scales.unsqueeze(1).expand(
            -1, self.gaussians_per_anchor, -1
        )[negative_opacity_filter][:, :3]
        valid_offsets = gaussian_offsets.view(
            num_visible, self.gaussians_per_anchor, 3
        )[negative_opacity_filter]
        valid_anchors = anchor_positions.unsqueeze(1).expand(
            -1, self.gaussians_per_anchor, -1
        )[negative_opacity_filter]
        return valid_anchors + (valid_offsets * valid_scales)
