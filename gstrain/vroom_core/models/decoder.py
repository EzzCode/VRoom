"""Neural decoding of anchor fields into renderable Gaussians."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .semantics import SemanticCodec


class AppearanceTable(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, init_std: float | None = None) -> None:
        super().__init__()
        self.table = nn.Embedding(num_embeddings, embedding_dim)
        self._table = self.table
        if init_std is not None:
            nn.init.normal_(self.table.weight, mean=0.0, std=init_std)

    def mean(self, dim: int = 0) -> torch.Tensor:
        return self.table.weight.mean(dim)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        return self.table(indices)


@dataclass
class DecodedGaussians:
    xyz: torch.Tensor
    offsets: torch.Tensor
    color: torch.Tensor
    opacity: torch.Tensor
    scaling: torch.Tensor
    rotation: torch.Tensor
    selection_mask: torch.Tensor
    semantics: torch.Tensor


class GaussianDecoder(nn.Module):
    def __init__(self, feat_dim: int, view_dim: int, appearance_dim: int, n_offsets: int, device: str = "cuda") -> None:
        super().__init__()
        self.feat_dim = feat_dim
        self.view_dim = view_dim
        self.appearance_dim = appearance_dim
        self.n_offsets = n_offsets
        self.device = torch.device(device)

        local_dim = feat_dim + view_dim
        self.opacity_head = nn.Sequential(
            nn.Linear(local_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, n_offsets),
            nn.Tanh(),
        ).to(self.device)
        self.covariance_head = nn.Sequential(
            nn.Linear(local_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, 7 * n_offsets),
        ).to(self.device)
        self.color_head = nn.Sequential(
            nn.Linear(local_dim + appearance_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, 3 * n_offsets),
        ).to(self.device)
        self.appearance: Optional[AppearanceTable] = None

    def configure_appearance(self, num_cameras: int) -> None:
        if self.appearance_dim > 0:
            self.appearance = AppearanceTable(max(num_cameras, 1), self.appearance_dim).to(self.device)
        else:
            self.appearance = None

    def decode(self, field, viewpoint_camera, visible_mask: torch.Tensor, training: bool) -> DecodedGaussians:
        anchors = field.anchor[visible_mask]
        features = field.feature[visible_mask]
        offsets = field.offset[visible_mask]
        scaling = torch.exp(field.log_scaling[visible_mask])
        semantics = self._semantic_vectors(field.codec, field.label_ids, visible_mask, anchors.shape[0], anchors.device)

        observer = viewpoint_camera.camera_center.to(anchors.device)
        if self.view_dim > 0:
            view_direction = F.normalize(observer - anchors, dim=-1)
            local_code = torch.cat([features, view_direction], dim=-1)
        else:
            local_code = features

        if self.appearance is not None:
            camera_id = int(viewpoint_camera.uid if training else 0)
            camera_ids = torch.full((anchors.shape[0],), camera_id, dtype=torch.long, device=anchors.device)
            appearance = self.appearance(camera_ids)
            color_input = torch.cat([local_code, appearance], dim=-1)
        else:
            color_input = local_code

        opacity_logits = self.opacity_head(local_code).reshape(-1, 1)
        color = self.color_head(color_input).reshape(anchors.shape[0] * self.n_offsets, 3)
        covariance = self.covariance_head(local_code).reshape(anchors.shape[0] * self.n_offsets, 7)

        selection_mask = opacity_logits.squeeze(-1) > 0.0
        repeated_anchor = anchors.repeat_interleave(self.n_offsets, dim=0)
        repeated_scale = scaling.repeat_interleave(self.n_offsets, dim=0)
        repeated_semantics = semantics.repeat_interleave(self.n_offsets, dim=0)

        offset_xyz = offsets.reshape(-1, 3)
        scaled_offsets = offset_xyz[selection_mask] * repeated_scale[selection_mask, :3]
        xyz = repeated_anchor[selection_mask] + scaled_offsets
        final_scaling = repeated_scale[selection_mask, 3:] * torch.sigmoid(covariance[selection_mask, :3])
        final_rotation = F.normalize(covariance[selection_mask, 3:7], dim=-1)

        return DecodedGaussians(
            xyz=xyz,
            offsets=scaled_offsets,
            color=color[selection_mask],
            opacity=opacity_logits[selection_mask],
            scaling=final_scaling,
            rotation=final_rotation,
            selection_mask=selection_mask,
            semantics=repeated_semantics[selection_mask],
        )

    def _semantic_vectors(self, codec: Optional[SemanticCodec], labels: Optional[torch.Tensor], visible_mask: torch.Tensor, count: int, device) -> torch.Tensor:
        if codec is None or labels is None:
            return torch.zeros((count, 1), dtype=torch.float32, device=device)
        return codec.transform(labels.view(-1)[visible_mask])
