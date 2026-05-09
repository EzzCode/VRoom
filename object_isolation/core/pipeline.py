"""Phases 6–7 — Supervision alignment and object training.

Uses ONLY ``object_isolation`` internals — no dependency on
``target_replenishment``.

Phases driven here
------------------
Phase 6  : ``dataset_builder.build_joint_supervision_views``
Phase 7  : ``trainer.train_object``

Output layout (per object_id)::

    <output_dir>/obj_<id>/
        supervision_manifest.json
        training_summary.json
        model/
            point_cloud.ply
            color_mlp.pt  cov_mlp.pt  opacity_mlp.pt
            object_model.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

# SV3D native output resolution (square).  Matches SV3DBackend.native_resolution.
_SV3D_RESOLUTION: int = 576

logger = logging.getLogger(__name__)


def _signed_angle_delta_deg(a: float, b: float) -> float:
    """Shortest signed angular difference a-b in degrees."""
    return float(((float(a) - float(b) + 180.0) % 360.0) - 180.0)


def run_pipeline(
    *,
    model_path: str,
    object_label_id: int,
    halluc_index_path: str | Path,
    output_dir: str | Path,
    gaussians=None,
    pipe_config=None,
    scope=None,
    local_sv3d=None,
    iterations: int = 1200,
    lr_scale: float = 1.0,
    hallucination_weight: float = 1.0,
    real_weight: float = 1.0,
    novel_rgb_weight: float = 1.0,
    hallucination_rgb_scale: float = 1.0,
    depth_weight: float = 0.1,
    depth_start_iter: int = 100,
    depth_front_weight: float = 1.0,
    depth_back_weight: float = 0.15,
    colmap_init_target_points: int = 8000,
    enable_densification: bool = False,
    max_anchor_count: int = 20000,
    densify_grad_threshold: float = 0.00005,
    densify_extra_ratio: float = 0.08,
    use_cond_cam_up: bool = True,
    fov_y_deg: float = 50.0,
) -> dict:
    """Train a fresh object-only ObjectGS model for one object.

    Either pass pre-loaded ``gaussians/pipe_config/scope/local_sv3d`` or
    omit them — they will be (re-)discovered from ``model_path``.

    Returns a summary dict (also written to disk).
    """
    from .dataset_builder import build_joint_supervision_views, save_supervision_manifest, write_projection_overlays
    from .object_scope import discover_object_scope
    from .trainer import train_object

    out_dir = Path(output_dir)
    obj_id = int(object_label_id)
    obj_dir = out_dir / f"obj_{obj_id}"
    (obj_dir / "model").mkdir(parents=True, exist_ok=True)

    # ── Load (or reuse) parent model + scope ──────────────────────────────
    if gaussians is None or pipe_config is None or scope is None or local_sv3d is None:
        logger.info("Rediscovering scope for obj %d at %s", obj_id, model_path)
        scope, _world_local, local_sv3d, gaussians, pipe_config = discover_object_scope(
            model_path, obj_id,
        )

    # ── Pull cond cam up (matches Phase 5 reference renders) ──────────────
    with open(halluc_index_path) as f:
        manifest = json.load(f)
    cam_idx = int(manifest.get("conditioning", {}).get("cam_index", -1))
    if cam_idx < 0 or cam_idx >= len(scope.cameras):
        raise RuntimeError(
            f"halluc_index 'conditioning.cam_index'={cam_idx} is out of range "
            f"(scope has {len(scope.cameras)} cameras).  Re-run Phase 5."
        )

    manifest_cond = manifest.get("conditioning", {}) or {}
    current_az, current_el = local_sv3d.world_camera_to_sv3d_view(scope.cameras[cam_idx]["position"])
    current_az = ((float(current_az) + 180.0) % 360.0) - 180.0
    current_el = float(current_el)
    manifest_az = float(manifest_cond.get("azimuth_V_deg", current_az))
    manifest_el = float(manifest_cond.get("elevation_V_deg", current_el))
    stale_az = abs(_signed_angle_delta_deg(manifest_az, current_az))
    stale_el = abs(float(manifest_el) - current_el)
    if stale_az > 0.5 or stale_el > 0.5:
        raise RuntimeError(
            "hallucination_index.json was generated with a different camera frame/up-vector: "
            f"manifest az/el=({manifest_az:.2f}, {manifest_el:.2f}), "
            f"current az/el=({current_az:.2f}, {current_el:.2f}) for conditioning cam {cam_idx}. "
            "Re-run Phase 5 after the coordinate-frame fix before training."
        )
    cond_cam_up_W: np.ndarray
    if use_cond_cam_up:
        R_cond = np.asarray(scope.cameras[cam_idx]["R"], dtype=np.float64)
        cond_cam_up_W = -R_cond[1]  # camera up in world = -row1 of R_w2c
        ang = float(np.degrees(np.arccos(np.clip(
            cond_cam_up_W @ scope.up_W /
            (np.linalg.norm(cond_cam_up_W) * max(np.linalg.norm(scope.up_W), 1e-9)),
            -1.0, 1.0,
        ))))
        logger.info("Cond cam %d up vector (%.2f\u00b0 from scope.up_W).", cam_idx, ang)
    else:
        cond_cam_up_W = np.asarray(scope.up_W, dtype=np.float64)

    extraction_index_path = Path(halluc_index_path).parents[1] / "phase3" / "extraction_index.json"

    # ── Load COLMAP seed points for Phase 6 world-scale computation ──────
    from .colmap_init import load_colmap_object_point_cloud
    pcd, _meta = load_colmap_object_point_cloud(
        model_path=model_path,
        object_id=obj_id,
        scope=scope,
        extraction_index_path=extraction_index_path,
        max_points=20000,
        target_points=8000,
    )
    seed_points_W = np.asarray(pcd.points, dtype=np.float32)
    logger.info("Loaded %d COLMAP seed points (obj %d).", len(seed_points_W), obj_id)

    # ── Phase 6: build real + hallucinated supervision views ────────────
    supervision_views = build_joint_supervision_views(
        halluc_index_path=halluc_index_path,
        extraction_index_path=extraction_index_path,
        scope=scope,
        local_sv3d=local_sv3d,
        seed_points_W=seed_points_W,
        real_weight=real_weight,
        hallucination_weight=hallucination_weight,
        fov_y_deg=fov_y_deg,
        hallucination_resolution=_SV3D_RESOLUTION,
        real_target_long_edge=_SV3D_RESOLUTION,
        up_W_override=cond_cam_up_W,
    )
    if not supervision_views:
        raise RuntimeError(f"No supervision views produced for obj {obj_id}.")

    save_supervision_manifest(supervision_views, obj_dir / "supervision_manifest.json")
    # Always write projection overlays immediately after building supervision views.
    # If the red dots don't trace the object the camera coordinate frame is wrong.
    write_projection_overlays(
        seed_points_W,
        supervision_views,
        obj_dir / "phase6_projection_audit",
    )
    n_real = sum(1 for v in supervision_views if v.get("source") == "real")
    n_hall = len(supervision_views) - n_real
    logger.info("%d supervision views for obj %d (real=%d hallucinated=%d).",
                len(supervision_views), obj_id, n_real, n_hall)

    if gaussians is not None:
        labels = gaussians.label_ids.squeeze(-1).cpu().numpy()
        n_parent_anchors = int(gaussians._anchor.shape[0])
        n_parent_obj_anchors = int((labels == obj_id).sum())
    else:
        n_parent_anchors = n_parent_obj_anchors = 0

    logger.info("Training obj %d for %d iters from COLMAP seed points.", obj_id, iterations)
    scratch = train_object(
        supervision_views=supervision_views,
        scope=scope,
        object_id=obj_id,
        model_path=model_path,
        output_dir=obj_dir,
        n_iterations=iterations,
        extraction_index_path=extraction_index_path,
        parent_gaussians=gaussians,
        pipe_config=pipe_config,
        lr_scale=lr_scale,
        colmap_init_target_points=colmap_init_target_points,
        rgb_weight=novel_rgb_weight,
        hallucination_rgb_scale=hallucination_rgb_scale,
        depth_weight=depth_weight,
        depth_start_iter=depth_start_iter,
        depth_front_weight=depth_front_weight,
        depth_back_weight=depth_back_weight,
        enable_densification=enable_densification,
        max_anchor_count=max_anchor_count,
        densify_grad_threshold=densify_grad_threshold,
        densify_extra_ratio=densify_extra_ratio,
    )
    summary = dict(scratch["summary"])
    summary.update({
        "n_real_supervision_views": n_real,
        "n_hallucinated_supervision_views": n_hall,
        "n_parent_anchors": n_parent_anchors,
        "n_parent_obj_anchors": n_parent_obj_anchors,
        "halluc_index_path": str(halluc_index_path),
        "extraction_index_path": str(extraction_index_path),
        "model_path": str(model_path),
    })

    with open(obj_dir / "model" / "object_model.json", "w", encoding="utf-8") as f:
        json.dump({
            "object_id": obj_id,
            "mode": "object_training",
            "n_parent_obj_anchors": n_parent_obj_anchors,
            "n_final_anchors": summary.get("n_final_anchors", 0),
        }, f, indent=2)
    with open(obj_dir / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(
        "Phase 7 done: obj %d anchors=%d final_loss=%.5f",
        obj_id, summary.get("n_final_anchors", 0), summary.get("final_loss", 0.0),
    )
    summary["_gaussians"] = scratch["gaussians"]
    return summary
