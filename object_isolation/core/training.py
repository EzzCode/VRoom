"""Phase 7 — Fresh object training using aligned real + SV3D supervision.

Uses ONLY ``object_isolation`` internals — no dependency on
``target_replenishment``.

Phases driven here
------------------
Phase 6  : ``dataset_builder.build_joint_supervision_views``
Phase 7  : ``scratch_trainer.train_scratch_object``

Output layout (per object_id)::

    <output_dir>/obj_<id>/
        supervision_manifest.json
        scratch_training_summary.json
        phase6_alignment_audit.json
        model/
            point_cloud.ply
            color_mlp.pt  cov_mlp.pt  opacity_mlp.pt
            scratch_object.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
logger = logging.getLogger(__name__)


def run_phase7(
    *,
    model_path: str,
    object_label_id: int,
    halluc_index_path: str | Path,
    output_dir: str | Path,
    gaussians=None,
    pipe_config=None,
    scope=None,
    local_sv3d=None,
    scratch_iterations: int = 1200,
    scratch_lr_scale: float = 1.0,
    hallucination_weight: float = 1.0,
    real_weight: float = 1.0,
    novel_rgb_weight: float = 1.0,
    grid_resolution: int = 25,
    visual_hull_min_views: int = 10,
    use_cond_cam_up: bool = True,
    fov_y_deg: float = 50.0,
) -> dict:
    """Train a fresh object-only ObjectGS model for one object.

    Either pass pre-loaded ``gaussians/pipe_config/scope/local_sv3d`` or
    omit them — they will be (re-)discovered from ``model_path``.

    Returns a summary dict (also written to disk).
    """
    from .dataset_builder import build_joint_supervision_views, save_supervision_manifest
    from .scope import discover_object_scope
    from .scratch_trainer import train_scratch_object

    out_dir = Path(output_dir)
    obj_dir = out_dir / f"obj_{int(object_label_id)}"
    (obj_dir / "model").mkdir(parents=True, exist_ok=True)

    # ── Load (or reuse) parent model + scope ──────────────────────────────
    if gaussians is None or pipe_config is None or scope is None or local_sv3d is None:
        logger.info("Phase 7: rediscovering scope for obj %d at %s",
                    object_label_id, model_path)
        scope, _world_local, local_sv3d, gaussians, pipe_config = discover_object_scope(
            model_path, int(object_label_id),
        )

    # ── Pull cond cam up (matches Phase 5 reference renders) ──────────────
    cond_cam_up_W: Optional[np.ndarray] = None
    cond_cam_idx: Optional[int] = None
    try:
        with open(halluc_index_path) as f:
            manifest = json.load(f)
        cond_cam_idx = int(manifest.get("conditioning", {}).get("cam_index", -1))
        if use_cond_cam_up and cond_cam_idx >= 0 and cond_cam_idx < len(scope.cameras):
            R_cond = np.asarray(scope.cameras[cond_cam_idx]["R"], dtype=np.float64)
            cond_cam_up_W = -R_cond[1]  # camera up in world = -row1 of R_w2c
            ang = float(np.degrees(np.arccos(np.clip(
                cond_cam_up_W @ scope.up_W /
                (np.linalg.norm(cond_cam_up_W) * max(np.linalg.norm(scope.up_W), 1e-9)),
                -1.0, 1.0,
            ))))
            logger.info("Phase 7: using cond cam %d up (%.2f deg from scope.up_W).",
                        cond_cam_idx, ang)
    except Exception as e:
        logger.warning("Could not read cond cam up from halluc_index (%s); "
                       "falling back to scope.up_W.", e)
        cond_cam_up_W = None

    extraction_index_path = Path(halluc_index_path).parents[1] / "phase3" / "extraction_index.json"

    # ── Phase 6: build real + hallucinated supervision views ─────────────
    supervision_views = build_joint_supervision_views(
        halluc_index_path=halluc_index_path,
        extraction_index_path=extraction_index_path,
        scope=scope,
        local_sv3d=local_sv3d,
        real_weight=real_weight,
        hallucination_weight=hallucination_weight,
        fov_y_deg=fov_y_deg,
        hallucination_resolution=576,
        real_target_long_edge=576,
        up_W_override=cond_cam_up_W,
        hallucination_alignment_audit_path=obj_dir / "phase6_alignment_audit.json",
    )
    if not supervision_views:
        raise RuntimeError(f"Phase 6 produced no joint supervision views for obj {object_label_id}.")

    save_supervision_manifest(supervision_views, obj_dir / "supervision_manifest.json")
    n_real = sum(1 for v in supervision_views if v.get("source") == "real")
    n_hall = sum(1 for v in supervision_views if v.get("source") == "hallucinated")
    logger.info("Phase 7: %d supervision views queued for obj %d (real=%d hallucinated=%d).",
                len(supervision_views), object_label_id, n_real, n_hall)

    labels = gaussians.label_ids.squeeze(-1).cpu().numpy() if gaussians is not None else np.array([])
    n_parent_anchors = int(gaussians._anchor.shape[0]) if gaussians is not None else 0
    n_parent_obj_anchors = int((labels == int(object_label_id)).sum()) if labels.size else 0

    logger.info(
        "Phase 7: scratch-training obj %d for %d iters from aligned views (no seeding, no target optimizer).",
        object_label_id, int(scratch_iterations),
    )
    scratch = train_scratch_object(
        supervision_views=supervision_views,
        scope=scope,
        object_id=int(object_label_id),
        output_dir=obj_dir,
        n_iterations=int(scratch_iterations),
        parent_gaussians=gaussians,
        pipe_config=pipe_config,
        lr_scale=float(scratch_lr_scale),
        init_grid_resolution=max(16, int(grid_resolution)),
        min_visual_hull_support=max(10, int(visual_hull_min_views)),
        rgb_weight=float(novel_rgb_weight),
    )
    summary = dict(scratch["summary"])
    summary.update({
        "n_real_supervision_views": int(n_real),
        "n_hallucinated_supervision_views": int(n_hall),
        "n_parent_anchors": int(n_parent_anchors),
        "n_parent_obj_anchors": int(n_parent_obj_anchors),
        "halluc_index_path": str(halluc_index_path),
        "extraction_index_path": str(extraction_index_path),
        "model_path": str(model_path),
    })

    with open(obj_dir / "model" / "scratch_object.json", "w", encoding="utf-8") as f:
        json.dump({
            "object_id": int(object_label_id),
            "mode": "scratch_object_training",
            "n_parent_obj_anchors": int(n_parent_obj_anchors),
            "n_final_anchors": int(summary.get("n_final_anchors", 0)),
        }, f, indent=2)
    with open(obj_dir / "scratch_training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(
        "Phase 7 complete for obj %d: scratch anchors=%d final_loss=%.5f",
        object_label_id, int(summary.get("n_final_anchors", 0)), float(summary.get("final_loss", 0.0)),
    )
    summary["_scratch_gaussians"] = scratch["gaussians"]
    return summary
