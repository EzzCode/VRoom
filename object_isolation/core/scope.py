"""Phase 1 — object scope discovery for object_isolation."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import logging

import numpy as np


from object_isolation.core.coordinate_frames import WorldLocal, LocalSV3D
from object_isolation.core.objectgs_model import (
    build_perspective_graph,
    count_visible_anchors,
    estimate_scene_up_from_cameras,
    get_anchor_positions,
    get_label_ids,
    load_gaussians,
    orbit_base_direction_from_cameras,
)


logger = logging.getLogger(__name__)


@dataclass
class ObjectScope:
    """Everything Phase 2+ needs about an object.

    Coordinates are in world frame W unless suffixed `_L`.
    """
    object_label_id: int
    n_anchors: int
    anchor_xyz_W: np.ndarray         # (N, 3) float32
    centroid_W: np.ndarray           # (3,)   float32
    aabb_min_W: np.ndarray           # (3,)   float32
    aabb_max_W: np.ndarray           # (3,)   float32
    principal_axes_W: np.ndarray     # (3, 3) float32 columns are PCA axes (largest first)
    principal_extents: np.ndarray    # (3,)   float32 sqrt of PCA eigenvalues
    radius: float                    # orbit radius in world units
    up_W: np.ndarray                 # (3,)   float32
    base_dir_W: np.ndarray           # (3,)   float32
    visible_cam_indices: list        # indices into cameras list
    cam_centers_visible_W: np.ndarray  # (M, 3)
    azimuth_histogram_V: dict = field(default_factory=dict)
    # Bookkeeping
    cameras: list = field(default_factory=list)   # raw perspective_graph camera dicts


def discover_object_scope(
    model_path: str,
    object_label_id: int,
    visibility_min_anchors: int = 50,
    azimuth_bin_deg: float = 10.0,
) -> tuple[ObjectScope, "WorldLocal", "LocalSV3D", object, object]:
    """Run Phase-1 discovery.

    Returns:
        scope, world_local, local_sv3d, gaussians, pipe_config
    The gaussians and pipe_config are returned so downstream phases avoid a
    second load.
    """
    model_path = Path(model_path)
    cameras_json = model_path / "cameras.json"
    if not cameras_json.exists():
        raise FileNotFoundError(f"cameras.json not found: {cameras_json}")

    # ── Load model ──
    gaussians, pipe_config = load_gaussians(str(model_path))
    all_xyz = get_anchor_positions(gaussians)
    label_ids = get_label_ids(gaussians)

    # ── Object anchors ──
    obj_mask = (label_ids == int(object_label_id))
    n_obj = int(obj_mask.sum())
    if n_obj == 0:
        raise ValueError(
            f"object_label_id={object_label_id} has no anchors in {model_path}. "
            f"Available labels: {sorted(np.unique(label_ids).tolist())}"
        )
    anchor_xyz = all_xyz[obj_mask].astype(np.float32)
    centroid = anchor_xyz.mean(axis=0).astype(np.float32)
    aabb_min = anchor_xyz.min(axis=0).astype(np.float32)
    aabb_max = anchor_xyz.max(axis=0).astype(np.float32)

    # ── PCA principal axes ──
    centered = anchor_xyz - centroid
    cov = (centered.T @ centered) / max(len(anchor_xyz) - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)            # ascending
    order = np.argsort(eigvals)[::-1]
    principal_axes = eigvecs[:, order].astype(np.float32)
    principal_extents = np.sqrt(np.clip(eigvals[order], 0.0, None)).astype(np.float32)

    # ── Camera analysis (reuse perspective_graph) ──
    pgraph = build_perspective_graph(str(cameras_json), anchor_xyz=None,
                                     overlap_method='frustum')
    cameras = pgraph.cameras

    visible_indices: list = []
    for ci, cam in enumerate(cameras):
        vis = count_visible_anchors(cam, anchor_xyz)
        if int(vis.sum()) >= visibility_min_anchors:
            visible_indices.append(ci)
    if not visible_indices:
        # Relaxed fallback: any camera that sees ≥1 anchor.
        for ci, cam in enumerate(cameras):
            vis = count_visible_anchors(cam, anchor_xyz)
            if int(vis.sum()) >= 1:
                visible_indices.append(ci)
        logger.warning(
            "No camera sees ≥%d anchors of object %s; relaxed to ≥1 (%d cams).",
            visibility_min_anchors, object_label_id, len(visible_indices),
        )
    if not visible_indices:
        raise RuntimeError(f"No training camera sees object {object_label_id} at all.")

    cam_centers_all = np.array([c['position'] for c in cameras], dtype=np.float32)
    cam_centers_vis = cam_centers_all[visible_indices]

    # ── World up + orbit base direction (numpy-only helpers from diagnostics) ──
    raw_cam_data = _read_cameras_json_raw(cameras_json)
    up_W = estimate_scene_up_from_cameras(raw_cam_data)
    base_dir_W = orbit_base_direction_from_cameras(
        cam_centers_vis, centroid, up_W,
    )

    # ── Orbit radius: median(visible cam → centroid) ──
    dists = np.linalg.norm(cam_centers_vis - centroid.reshape(1, 3), axis=1)
    radius = float(np.median(dists)) if len(dists) > 0 else 2.0 * float(np.linalg.norm(aabb_max - aabb_min))
    radius = max(radius, 1e-3)

    # ── Build coordinate frames ──
    world_local = WorldLocal(
        centroid_W=centroid.astype(np.float64),
        up_W=up_W.astype(np.float64),
        base_dir_W=base_dir_W.astype(np.float64),
        radius=radius,
    )
    local_sv3d = LocalSV3D(world_local=world_local)

    # ── Azimuth histogram in V-frame for visible cameras ──
    azimuth_hist: dict = {}
    bin_deg = float(azimuth_bin_deg)
    for ci in visible_indices:
        C_W = cameras[ci]['position']
        az_V, el_V = local_sv3d.world_camera_to_sv3d_view(C_W)
        # Normalize to [0, 360)
        az_norm = float(az_V) % 360.0
        bin_idx = int(az_norm // bin_deg)
        azimuth_hist[bin_idx] = azimuth_hist.get(bin_idx, 0) + 1
        # Tag the camera dict so downstream phases can read it directly.
        cameras[ci]['azimuth_V_deg'] = az_norm
        cameras[ci]['elevation_V_deg'] = float(el_V)

    scope = ObjectScope(
        object_label_id=int(object_label_id),
        n_anchors=int(n_obj),
        anchor_xyz_W=anchor_xyz,
        centroid_W=centroid,
        aabb_min_W=aabb_min,
        aabb_max_W=aabb_max,
        principal_axes_W=principal_axes,
        principal_extents=principal_extents,
        radius=radius,
        up_W=up_W.astype(np.float32),
        base_dir_W=base_dir_W.astype(np.float32),
        visible_cam_indices=visible_indices,
        cam_centers_visible_W=cam_centers_vis.astype(np.float32),
        azimuth_histogram_V=azimuth_hist,
        cameras=cameras,
    )

    logger.info(
        "Scope obj=%d: %d anchors | centroid=%s | radius=%.3f | "
        "up=%s | base_dir=%s | visible_cams=%d/%d | az_bins_covered=%d",
        scope.object_label_id, scope.n_anchors,
        np.round(scope.centroid_W, 3).tolist(), scope.radius,
        np.round(scope.up_W, 3).tolist(), np.round(scope.base_dir_W, 3).tolist(),
        len(scope.visible_cam_indices), len(cameras), len(scope.azimuth_histogram_V),
    )

    return scope, world_local, local_sv3d, gaussians, pipe_config


def _read_cameras_json_raw(path: Path) -> list:
    """Diagnostics helpers expect the raw cameras.json list (with 'rotation', 'position')."""
    import json
    with open(path) as f:
        return json.load(f)


def find_uncovered_azimuth_sectors(
    scope: ObjectScope,
    bin_deg: float = 10.0,
    min_gap_deg: float = 30.0,
) -> list[tuple[float, float]]:
    """Return list of (start_deg, end_deg) sectors in V-frame where no training
    camera lies within `min_gap_deg` of any covered bin.

    Used by hallucination phase to pick which SV3D azimuths to keep.
    """
    n_bins = int(round(360.0 / bin_deg))
    covered = np.zeros(n_bins, dtype=bool)
    for b in scope.azimuth_histogram_V.keys():
        covered[b % n_bins] = True

    # Dilate covered bins by min_gap (so a bin is "near coverage" if any
    # covered bin lies within ceil(min_gap/bin_deg) bins, with wraparound).
    halo = int(np.ceil(min_gap_deg / bin_deg))
    if halo > 0:
        dilated = covered.copy()
        for off in range(1, halo + 1):
            dilated |= np.roll(covered, off)
            dilated |= np.roll(covered, -off)
    else:
        dilated = covered.copy()

    sectors: list[tuple[float, float]] = []
    in_gap = False
    start = 0
    for b in range(n_bins):
        if not dilated[b] and not in_gap:
            in_gap = True
            start = b
        elif dilated[b] and in_gap:
            sectors.append((start * bin_deg, b * bin_deg))
            in_gap = False
    if in_gap:
        sectors.append((start * bin_deg, n_bins * bin_deg))
    return sectors
