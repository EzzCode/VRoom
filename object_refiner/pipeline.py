import json
import logging
import math
import numpy as np
from pathlib import Path

from .utils.transforms import ObjectFrame
from .utils.scene_analysis import compute_object_scope, load_gaussians
from .trainer import train_object
from .utils.colmap_init import load_colmap_object_point_cloud
from .config import ObjectTrainingConfig
from .dataset_builder import build_views

logger = logging.getLogger(__name__)


def run_pipeline(
    model_path,
    object_id,
    generation_path,
    output_dir,
    generation_log=None,
    gaussians=None,
    scope=None,
    frame=None,
    config = ObjectTrainingConfig(),
):
    colmap_init_target_points = config.colmap_init_target_points
    real_weight = config.real_weight
    generated_weight = config.generated_weight
    use_cond_cam_up = config.use_cond_cam_up

    output_dir = Path(output_dir)
    object_id = int(object_id)
    object_dir = output_dir / f"obj_{object_id}"
    object_dir.mkdir(parents=True, exist_ok=True)

    if scope is None:
        raise ValueError("Scope must be provided to run_pipeline.")
    if frame is None:
        raise ValueError("Frame must be provided to run_pipeline.")
    if gaussians is None:
        raise ValueError("Gaussians must be provided to run_pipeline.")
    if generation_log is None:
        raise ValueError("Generation log path must be provided to run_pipeline.")

    generation_file = Path(generation_path)

    cam_idx = int(generation_log.get("conditioning", {}).get("cam_index", -1))
    if not (0 <= cam_idx < len(scope.cameras)):
        raise RuntimeError(
            f"generation.json conditioning.cam_index={cam_idx} out of range "
            f"(scope has {len(scope.cameras)} cameras). Re-run generation."
        )

    azimuth = float(generation_log.get("conditioning", {}).get("azimuth_deg", float("nan")))
    elevation = float(generation_log.get("conditioning", {}).get("elevation_deg", float("nan")))
    if math.isfinite(azimuth) and math.isfinite(elevation):
        current_az, current_el = frame.world_to_virtual(
            np.asarray(scope.cameras[cam_idx]["position"], np.float32)
        )
        current_az = ((current_az + 180.0) % 360.0) - 180.0
        delta_az = abs(((azimuth - current_az + 180.0) % 360.0) - 180.0)
        if delta_az > 0.5 or abs(elevation - current_el) > 0.5:
            raise RuntimeError(
                f"log frame mismatch for obj {object_id}: "
                f"log az/el=({azimuth:.2f}, {elevation:.2f}) vs "
                f"current ({current_az:.2f}, {float(current_el):.2f}). Re-run generation."
            )

    if use_cond_cam_up:
        up_override = -np.asarray(scope.cameras[cam_idx]["R"], np.float32)[1]
    else:
        up_override = np.asarray(scope.up, np.float32)

    extraction_index_path = object_dir / "01_extraction" / "extraction_index.json"

    point_cloud, _ = load_colmap_object_point_cloud(
        model_path=model_path, object_id=object_id, scope=scope,
        extraction_index_path=extraction_index_path,
        max_points=20000, target_points=colmap_init_target_points,
    )
    seed_points = np.asarray(point_cloud.points, np.float32)

    built_views = build_views(
        generation_log_path=generation_file,
        extraction_path=extraction_index_path,
        scope=scope,
        frame=frame,
        cloud_points=seed_points,
        real_weight=real_weight,
        generated_weight=generated_weight,
        up_override=up_override,
    )
    result = train_object(
        built_views=built_views,
        scope=scope,
        object_id=object_id,
        model_path=model_path,
        output_dir=object_dir,
        extraction_index_path=extraction_index_path,
        parent_gaussians=gaussians,
        config=config,
    )

    summary = dict(result["summary"])
    summary["_gaussians"] = result.get("gaussians") #used in debug

    logger.info("obj %d done: anchors=%d final_loss=%.5f",
                object_id, summary.get("n_final_anchors", 0), summary.get("final_loss", 0.0))
    
    return summary
