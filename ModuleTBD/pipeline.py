import json
import logging
import math
from pathlib import Path
from typing import cast, Any

import numpy as np

from .utils.transforms import ObjectFrame
from .utils.scene_analysis import compute_object_scope, load_gaussians
from .trainer import train_object
from .utils.colmap_init import load_colmap_object_point_cloud

logger = logging.getLogger(__name__)


def run_pipeline(
    model_path,
    object_label_id,
    halluc_index_path,
    output_dir,
    halluc_manifest=None,
    gaussians=None,
    pipe_config=None,
    scope=None,
    frame=None,
    iterations=1200,
    lr_scale=1.0,
    generated_weight=1.0,
    real_weight=1.0,
    rgb_weight=1.0,
    generated_rgb_scale=1.0,
    depth_weight=0.1,
    depth_start_iter=100,
    depth_front_weight=1.0,
    depth_back_weight=0.15,
    colmap_init_target_points=8000,
    enable_densification=False,
    max_anchor_count=20000,
    densify_grad_threshold=0.00005,
    densify_extra_ratio=0.08,
    use_cond_cam_up=True,
):
    from .dataset_builder import build_supervision_views

    out_dir = Path(output_dir)
    obj_id = int(object_label_id)
    obj_dir = out_dir / f"obj_{obj_id}"
    obj_dir.mkdir(parents=True, exist_ok=True)

    if scope is None or pipe_config is None:
        logger.info("Computing scope for obj %d from %s", obj_id, model_path)
        scope, _, pipe_config = compute_object_scope(model_path, obj_id)
    if gaussians is None:
        gaussians, _ = load_gaussians(model_path)
    if frame is None:
        frame = ObjectFrame(centroid=scope.centroid, up=scope.up,
                            base_dir=scope.base_dir, radius=scope.radius)

    halluc_path = Path(halluc_index_path)
    if halluc_manifest is not None:
        halluc = halluc_manifest
    else:
        with open(halluc_path) as f:
            halluc = json.load(f)

    cam_idx = int(halluc.get("conditioning", {}).get("cam_index", -1))
    if not (0 <= cam_idx < len(scope.cameras)):
        raise RuntimeError(
            f"generation.json conditioning.cam_index={cam_idx} out of range "
            f"(scope has {len(scope.cameras)} cameras). Re-run hallucination."
        )

    manifest_az = float(halluc.get("conditioning", {}).get("azimuth_deg", float("nan")))
    manifest_el = float(halluc.get("conditioning", {}).get("elevation_deg", float("nan")))
    if math.isfinite(manifest_az) and math.isfinite(manifest_el):
        current_az, current_el = frame.world_to_virtual(
            np.asarray(scope.cameras[cam_idx]["position"], np.float32)
        )
        current_az = ((current_az + 180.0) % 360.0) - 180.0
        delta_az = abs(((manifest_az - current_az + 180.0) % 360.0) - 180.0)
        if delta_az > 0.5 or abs(manifest_el - current_el) > 0.5:
            raise RuntimeError(
                f"Hallucination manifest frame mismatch for obj {obj_id}: "
                f"manifest az/el=({manifest_az:.2f}, {manifest_el:.2f}) vs "
                f"current ({current_az:.2f}, {float(current_el):.2f}). Re-run hallucination."
            )

    if use_cond_cam_up:
        up_override = -np.asarray(scope.cameras[cam_idx]["R"], np.float32)[1]
    else:
        up_override = np.asarray(scope.up, np.float32)

    extraction_index_path = obj_dir / "01_extraction" / "extraction_index.json"

    pcd, _ = load_colmap_object_point_cloud(
        model_path=model_path, object_id=obj_id, scope=scope,
        extraction_index_path=extraction_index_path,
        max_points=20000, target_points=int(colmap_init_target_points),
    )
    seed_points = np.asarray(pcd.points, np.float32)

    supervision_views = build_supervision_views(
        generation_log_path=halluc_path,
        extraction_path=extraction_index_path,
        scope=scope,
        frame=frame,
        cloud_points=seed_points,
        real_weight=float(real_weight),
        generated_weight=float(generated_weight),
        up_override=up_override,
    )
    if not supervision_views:
        raise RuntimeError(f"No supervision views produced for obj {obj_id}.")

    n_real = sum(1 for v in supervision_views if v.get("source") == "real")
    n_generated = len(supervision_views) - n_real
    logger.info("%d supervision views for obj %d (real=%d generated=%d)", len(supervision_views), obj_id, n_real, n_generated)

    n_parent_anchors = int(gaussians._anchor.shape[0]) if gaussians is not None else 0
    n_parent_obj_anchors = 0
    if gaussians is not None and getattr(gaussians, "label_ids", None) is not None:
        labels = cast(Any, gaussians.label_ids).squeeze(-1).cpu().numpy()
        n_parent_obj_anchors = int((labels == obj_id).sum())

    result = train_object(
        supervision_views=supervision_views,
        scope=scope,
        object_id=obj_id,
        model_path=model_path,
        output_dir=obj_dir,
        n_iterations=int(iterations),
        extraction_index_path=extraction_index_path,
        parent_gaussians=gaussians,
        pipe_config=pipe_config,
        lr_scale=float(lr_scale),
        colmap_init_target_points=int(colmap_init_target_points),
        rgb_weight=float(rgb_weight),
        generated_rgb_scale=float(generated_rgb_scale),
        depth_weight=float(depth_weight),
        depth_start_iter=int(depth_start_iter),
        depth_front_weight=float(depth_front_weight),
        depth_back_weight=float(depth_back_weight),
        enable_densification=bool(enable_densification),
        max_anchor_count=int(max_anchor_count),
        densify_grad_threshold=float(densify_grad_threshold),
        densify_extra_ratio=float(densify_extra_ratio),
    )

    summary = dict(result["summary"])
    summary.update({
        "n_real_supervision_views": n_real,
        "n_generated_supervision_views": n_generated,
        "n_parent_anchors": n_parent_anchors,
        "n_parent_obj_anchors": n_parent_obj_anchors,
        "generated_index_path": str(halluc_path),
        "extraction_index_path": str(extraction_index_path),
        "model_path": str(model_path),
    })
    summary["_gaussians"] = result["gaussians"]

    logger.info("obj %d done: anchors=%d final_loss=%.5f",
                obj_id, summary.get("n_final_anchors", 0), summary.get("final_loss", 0.0))
    return summary
