"""Shared nearest-neighbor backend for VRoom."""

from __future__ import annotations

import torch
from sklearn.neighbors import NearestNeighbors


def neighbor_distances(points: torch.Tensor, k: int) -> torch.Tensor:
    if points.numel() == 0:
        return torch.empty((0, k), dtype=points.dtype, device=points.device)
    cpu_points = points.detach().float().cpu().numpy()
    model = NearestNeighbors(n_neighbors=min(k, len(cpu_points)), metric="euclidean")
    distances, _ = model.fit(cpu_points).kneighbors(cpu_points)
    distances = torch.from_numpy(distances).to(points)
    if distances.shape[1] < k:
        pad = distances[:, -1:].repeat(1, k - distances.shape[1])
        return torch.cat([distances, pad], dim=1)
    return distances
