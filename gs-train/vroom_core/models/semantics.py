"""Semantic label utilities for VRoom."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class SemanticCodec:
    unique_labels: torch.Tensor

    @classmethod
    def from_labels(cls, labels: torch.Tensor) -> "SemanticCodec":
        flattened = labels.view(-1).detach()
        return cls(unique_labels=torch.unique(flattened, sorted=True))

    @property
    def num_classes(self) -> int:
        return int(self.unique_labels.numel())

    def fit(self, labels: torch.Tensor) -> None:
        self.unique_labels = torch.unique(labels.view(-1), sorted=True)

    def label_to_index(self, labels: torch.Tensor) -> torch.Tensor:
        labels = labels.to(self.unique_labels.device)
        indices = torch.zeros_like(labels, dtype=torch.long)
        for index, label in enumerate(self.unique_labels):
            indices[labels == label] = index
        return indices

    def transform(self, labels: torch.Tensor) -> torch.Tensor:
        return F.one_hot(self.label_to_index(labels).long(), num_classes=self.num_classes).float().to(labels.device)

    def index_to_label(self, indices: torch.Tensor) -> torch.Tensor:
        return self.unique_labels[indices.long()]

    def inverse_transform(self, one_hot: torch.Tensor) -> torch.Tensor:
        """Convert rendered one-hot semantic maps back to original label IDs."""
        indices = torch.argmax(one_hot, dim=-1)
        return self.index_to_label(indices)

    def visualize(self, one_hot: torch.Tensor, seed: int = 0) -> torch.Tensor:
        """Convert rendered one-hot semantic maps to an RGB color image (uint8)."""
        import numpy as np
        indices = torch.argmax(one_hot, dim=-1)
        rng = np.random.RandomState(seed)
        colors = rng.randint(0, 256, size=(self.num_classes, 3)).astype(np.uint8)
        colors[0] = [0, 0, 0]  # background is black
        color_map = torch.tensor(colors, dtype=torch.uint8, device=one_hot.device)
        return color_map[indices.long()]

