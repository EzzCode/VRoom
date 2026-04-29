"""
Stage A — Dense Surface Extraction.

Topologically separates the "dense fabric" of an object from sparse floaters
that the original training run produced because no camera looked there.

Algorithm (no quantile heuristics):
  1. kNN graph over object anchors (k=knn_k).
  2. Per-anchor mean-knn-distance r_i; r_med = median(r_i).
  3. Hard isolation filter: drop anchors with r_i > iso_factor * r_med.
  4. Connectivity: edge if dist < edge_factor * r_med. Keep components with
     size >= max(min_component_frac * N_obj, min_component_size). Always
     keep largest.

Output: indices into the original object anchor array that survive.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


@dataclass
class SurfaceExtractionResult:
    survivor_indices: np.ndarray          # (M,) int into local object array
    r_med: float                          # median kNN distance (for downstream)
    n_in: int
    n_after_isolation: int
    n_out: int
    kept_components: int
    total_components: int
    extent_before: np.ndarray             # (3,) bbox extent before
    extent_after: np.ndarray              # (3,) bbox extent after

    def to_dict(self) -> dict:
        return {
            "n_in": int(self.n_in),
            "n_after_isolation": int(self.n_after_isolation),
            "n_out": int(self.n_out),
            "kept_components": int(self.kept_components),
            "total_components": int(self.total_components),
            "r_med": float(self.r_med),
            "extent_before": self.extent_before.tolist(),
            "extent_after": self.extent_after.tolist(),
        }


def extract_dense_surface(
    object_xyz: np.ndarray,
    knn_k: int = 16,
    iso_factor: float = 3.0,
    edge_factor: float = 2.0,
    min_component_frac: float = 0.005,
    min_component_size: int = 8,
    keep_component_min_frac: float = 0.05,
) -> SurfaceExtractionResult:
    """Topological floater removal. See module docstring."""
    object_xyz = np.asarray(object_xyz, dtype=np.float32)
    n = object_xyz.shape[0]
    if n == 0:
        return SurfaceExtractionResult(
            survivor_indices=np.zeros((0,), dtype=np.int64),
            r_med=0.0, n_in=0, n_after_isolation=0, n_out=0,
            kept_components=0, total_components=0,
            extent_before=np.zeros(3, dtype=np.float32),
            extent_after=np.zeros(3, dtype=np.float32),
        )

    extent_before = (object_xyz.max(axis=0) - object_xyz.min(axis=0)).astype(np.float32)

    if n <= max(knn_k + 1, 4):
        return SurfaceExtractionResult(
            survivor_indices=np.arange(n, dtype=np.int64),
            r_med=0.0, n_in=n, n_after_isolation=n, n_out=n,
            kept_components=1, total_components=1,
            extent_before=extent_before, extent_after=extent_before,
        )

    k = min(knn_k, n - 1)
    tree = cKDTree(object_xyz)
    # query k+1 to drop self
    dists, idxs = tree.query(object_xyz, k=k + 1)
    knn_dists = dists[:, 1:]                 # (n, k)
    mean_knn = knn_dists.mean(axis=1)        # (n,)
    r_med = float(np.median(mean_knn))
    if not np.isfinite(r_med) or r_med <= 0.0:
        return SurfaceExtractionResult(
            survivor_indices=np.arange(n, dtype=np.int64),
            r_med=r_med, n_in=n, n_after_isolation=n, n_out=n,
            kept_components=1, total_components=1,
            extent_before=extent_before, extent_after=extent_before,
        )

    iso_keep = mean_knn <= (iso_factor * r_med)
    kept_iso_idx = np.where(iso_keep)[0]
    n_after_iso = int(kept_iso_idx.size)
    if n_after_iso < 2:
        return SurfaceExtractionResult(
            survivor_indices=kept_iso_idx.astype(np.int64),
            r_med=r_med, n_in=n, n_after_isolation=n_after_iso,
            n_out=n_after_iso, kept_components=int(n_after_iso > 0),
            total_components=int(n_after_iso > 0),
            extent_before=extent_before, extent_after=extent_before,
        )

    # Build sparse adjacency on the iso-kept subset using a radius query.
    sub_xyz = object_xyz[kept_iso_idx]
    sub_tree = cKDTree(sub_xyz)
    pairs = sub_tree.query_pairs(r=edge_factor * r_med, output_type="ndarray")
    if pairs.size == 0:
        rows = np.arange(n_after_iso)
        cols = np.arange(n_after_iso)
        data = np.zeros(n_after_iso, dtype=np.uint8)
    else:
        rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
        cols = np.concatenate([pairs[:, 1], pairs[:, 0]])
        data = np.ones(rows.size, dtype=np.uint8)
    adj = csr_matrix((data, (rows, cols)), shape=(n_after_iso, n_after_iso))
    n_components, labels = connected_components(adj, directed=False)

    comp_sizes = np.bincount(labels)
    largest = int(np.argmax(comp_sizes))
    min_size = max(int(min_component_frac * n), int(min_component_size))
    keep_min = max(min_size, int(keep_component_min_frac * comp_sizes[largest]))

    keep_components = np.zeros(n_components, dtype=bool)
    keep_components[largest] = True
    for c in range(n_components):
        if comp_sizes[c] >= keep_min:
            keep_components[c] = True

    sub_keep_mask = keep_components[labels]
    survivor_local = kept_iso_idx[sub_keep_mask]
    survivor_indices = np.sort(survivor_local).astype(np.int64)

    if survivor_indices.size > 0:
        survivors = object_xyz[survivor_indices]
        extent_after = (survivors.max(axis=0) - survivors.min(axis=0)).astype(np.float32)
    else:
        extent_after = np.zeros(3, dtype=np.float32)

    return SurfaceExtractionResult(
        survivor_indices=survivor_indices,
        r_med=r_med,
        n_in=int(n),
        n_after_isolation=int(n_after_iso),
        n_out=int(survivor_indices.size),
        kept_components=int(keep_components.sum()),
        total_components=int(n_components),
        extent_before=extent_before,
        extent_after=extent_after,
    )
