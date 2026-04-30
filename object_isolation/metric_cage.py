"""Phase 3.5 — Metric cage construction.

Two purposes:

1. **Floater cleanup.** ObjectGS label_ids are noisy: a small fraction of
   anchors carry the wrong label and sit far from the actual object. Naive
   "all anchors with label_id == k" gives a radius dominated by these
   outliers. We DBSCAN the anchor cloud and keep only the largest
   connected cluster.

2. **Metric cage = hard 3D bound for standalone-2DGS training.** The
   downstream training (Phase 4) uses this AABB as a clamping region for
   the densified Gaussian centers, so Zero123++'s "depth-stretching"
   hallucinations cannot run away in metric units.

The cage is *mirror-extended* along the world-up axis to account for the
fact that the visible-side anchors only cover the camera-facing half of
the object. Without this, the back side of the object would be clipped
during 2DGS training.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MetricCage:
    """Persisted metric cage (world-frame AABB)."""

    object_id: int
    object_center_clean: list  # length-3 (median of cleaned cluster)
    object_radius_clean: float
    aabb_visible: list         # [[xmin..],[xmax..]] from clean anchors
    aabb_full: list            # mirror-extended along up axis
    object_up_world: list
    n_anchors_total: int
    n_anchors_kept: int
    dbscan_eps: float
    dbscan_min_samples: int


# ── Anchor cleanup via DBSCAN ───────────────────────────────────────────────


def _auto_eps(points: np.ndarray, k: int = 12, percentile: float = 90.0) -> float:
    """Estimate a good DBSCAN ``eps`` from the k-NN distance distribution.

    Heuristic: take the k-th nearest-neighbor distance for every point and
    return the chosen percentile. This adapts to the anchor density without
    requiring the caller to know the object's metric scale.
    """
    from sklearn.neighbors import NearestNeighbors
    n = points.shape[0]
    k_eff = min(k, n - 1)
    if k_eff < 1:
        return 0.05
    nbr = NearestNeighbors(n_neighbors=k_eff + 1).fit(points)
    d, _ = nbr.kneighbors(points)
    return float(np.percentile(d[:, -1], percentile))


def clean_anchors(
    anchor_xyz: np.ndarray,
    eps: Optional[float] = None,
    min_samples: int = 8,
) -> tuple[np.ndarray, dict]:
    """DBSCAN-cluster ``anchor_xyz`` and keep the largest cluster.

    Returns the kept points and a stats dict with ``eps``, ``n_clusters``,
    ``n_noise``, and ``kept_label``. If clustering returns no clusters
    (all noise) we fall back to returning the original points unchanged.
    """
    from sklearn.cluster import DBSCAN

    if anchor_xyz.shape[0] < min_samples:
        return anchor_xyz, {"eps": 0.0, "n_clusters": 1, "n_noise": 0, "kept_label": 0,
                            "fallback": "too_few_points"}

    eps_used = float(eps) if eps is not None else _auto_eps(anchor_xyz)
    db = DBSCAN(eps=eps_used, min_samples=int(min_samples)).fit(anchor_xyz)
    labels = db.labels_
    unique = [u for u in np.unique(labels) if u != -1]
    if not unique:
        logger.warning("DBSCAN found 0 clusters (all noise); skipping cleanup")
        return anchor_xyz, {"eps": eps_used, "n_clusters": 0,
                            "n_noise": int((labels == -1).sum()),
                            "kept_label": -1, "fallback": "all_noise"}
    sizes = [(int(u), int((labels == u).sum())) for u in unique]
    best_label, _ = max(sizes, key=lambda x: x[1])
    mask = labels == best_label
    return anchor_xyz[mask], {
        "eps": eps_used,
        "n_clusters": int(len(unique)),
        "n_noise": int((labels == -1).sum()),
        "kept_label": int(best_label),
        "kept_count": int(mask.sum()),
        "cluster_sizes": sizes,
    }


# ── Cage construction ──────────────────────────────────────────────────────


def _mirror_extend_aabb(
    aabb_visible: np.ndarray,
    center: np.ndarray,
    up_world: np.ndarray,
) -> np.ndarray:
    """Mirror-extend the visible-side AABB across the plane through
    ``center`` perpendicular to ``up_world``.

    Implementation:
        1. Express the 8 AABB corners in a local frame whose +Z = up.
        2. Reflect their "horizontal" extents (perpendicular to up) about
           the origin to cover the unseen back half.
        3. Take the per-axis world-frame min/max of the union.

    Equivalent (and the implementation we use): reflect each corner across
    the plane (point=center, normal=up) — i.e. ``p' = p - 2 * dot(p-center,
    horiz_axis) * horiz_axis`` for both horizontal axes — and union with the
    original corners. Simpler still: the AABB is axis-aligned, so we can
    symmetrically widen it about ``center`` along all axes perpendicular to
    ``up`` by the larger of (center - min, max - center).
    """
    aabb = np.asarray(aabb_visible, dtype=np.float64).copy()  # (2, 3)
    c = np.asarray(center, dtype=np.float64)
    up = np.asarray(up_world, dtype=np.float64)
    up = up / max(float(np.linalg.norm(up)), 1e-8)

    # For each world axis, decide whether it's "mostly horizontal" (project
    # to plane perp to up) or "mostly vertical" (along up).
    new_min = aabb[0].copy()
    new_max = aabb[1].copy()
    for axis in range(3):
        e = np.zeros(3); e[axis] = 1.0
        horiz = float(np.linalg.norm(e - (e @ up) * up))
        if horiz < 0.5:
            # axis is roughly aligned with up — leave it (we trust that the
            # object's height is fully captured by the visible anchors)
            continue
        # Symmetrize around center
        d_min = float(c[axis] - aabb[0, axis])
        d_max = float(aabb[1, axis] - c[axis])
        d = max(d_min, d_max)
        new_min[axis] = c[axis] - d
        new_max[axis] = c[axis] + d
    return np.stack([new_min, new_max], axis=0)


def build_metric_cage(
    obj_dir: str,
    eps: Optional[float] = None,
    min_samples: int = 8,
    radius_percentile: float = 99.0,
) -> dict:
    """Read object_anchors.ply (written by Phase 1) + extraction_summary.json,
    DBSCAN-clean, build visible AABB + mirror-extended full AABB, and write
    ``<obj_dir>/metric_cage.json``.

    The summary's ``object_center`` and ``object_radius`` are recomputed
    on the cleaned cluster and surfaced in the cage JSON. Phase 4 should
    prefer ``object_center_clean`` and ``object_radius_clean`` over the
    raw values from the extraction summary.
    """
    obj_dir_p = Path(obj_dir)
    ply_path = obj_dir_p / "object_anchors.ply"
    summary_path = obj_dir_p / "extraction_summary.json"
    if not ply_path.exists():
        raise FileNotFoundError(f"object_anchors.ply missing at {ply_path}")
    if not summary_path.exists():
        raise FileNotFoundError(f"extraction_summary.json missing at {summary_path}")

    pts = _read_simple_ply(ply_path)
    n_total = pts.shape[0]
    if n_total == 0:
        raise ValueError(f"object_anchors.ply at {ply_path} contains zero points")

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    object_id = int(summary["object_frame"]["object_id"])
    up_world = np.asarray(summary["object_frame"]["object_up_world"], dtype=np.float64)

    # 1. DBSCAN cleanup
    kept, stats = clean_anchors(pts, eps=eps, min_samples=min_samples)
    n_kept = kept.shape[0]
    if n_kept == 0:
        raise RuntimeError("No anchors survived DBSCAN cleanup")

    # 2. Stats on the cleaned cluster
    center = np.median(kept, axis=0)
    radii = np.linalg.norm(kept - center, axis=1)
    radius = float(np.percentile(radii, radius_percentile))
    aabb_visible = np.stack([kept.min(axis=0), kept.max(axis=0)], axis=0)

    # 3. Mirror-extend along the up plane
    aabb_full = _mirror_extend_aabb(aabb_visible, center, up_world)

    cage = MetricCage(
        object_id=object_id,
        object_center_clean=center.tolist(),
        object_radius_clean=radius,
        aabb_visible=aabb_visible.tolist(),
        aabb_full=aabb_full.tolist(),
        object_up_world=up_world.tolist(),
        n_anchors_total=int(n_total),
        n_anchors_kept=int(n_kept),
        dbscan_eps=float(stats["eps"]),
        dbscan_min_samples=int(min_samples),
    )
    out_path = obj_dir_p / "metric_cage.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({**asdict(cage), "dbscan_stats": stats}, f, indent=2)
    logger.info(
        "Metric cage: kept %d/%d anchors (%.1f%%), radius_clean=%.3f (was %.3f), "
        "dbscan_eps=%.4f, n_clusters=%d",
        n_kept, n_total, 100.0 * n_kept / n_total, radius,
        summary["object_frame"]["object_radius"], stats["eps"], stats["n_clusters"],
    )
    return asdict(cage)


# ── Tiny PLY reader (matches the writer in extraction.py) ──────────────────


def _read_simple_ply(path: Path) -> np.ndarray:
    """Read the binary little-endian XYZ PLY produced by ``extraction._write_simple_ply``."""
    with open(path, "rb") as f:
        header_bytes = b""
        while True:
            line = f.readline()
            if not line:
                raise IOError(f"Unexpected EOF reading PLY header at {path}")
            header_bytes += line
            if line.strip() == b"end_header":
                break
        n = 0
        for ln in header_bytes.split(b"\n"):
            if ln.startswith(b"element vertex"):
                n = int(ln.split()[-1])
                break
        if n == 0:
            return np.zeros((0, 3), dtype=np.float32)
        data = np.frombuffer(f.read(n * 12), dtype=np.float32).reshape(n, 3)
    return data.astype(np.float64)
