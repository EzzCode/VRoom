"""Before/after render comparisons for a trained object model."""
from __future__ import annotations

import json
import logging
import sys
import argparse
from pathlib import Path

import cv2
import numpy as np

_VROOM_ROOT = Path(__file__).resolve().parents[2]
if str(_VROOM_ROOT) not in sys.path:
    sys.path.insert(0, str(_VROOM_ROOT))

from object_refiner.utils.gstrain_wrapper import make_camera, render_rgba
from object_refiner.utils.transforms import R_L2V, look_at, orbit_position

logger = logging.getLogger(__name__)


def _orbit_cameras(scope, frame, halluc_manifest=None, n_views=8, width=512, height=512, fov_y_deg=50.0):
    center = np.asarray(frame.centroid if frame is not None else scope.centroid, np.float32)
    up = np.asarray(frame.up if frame is not None else scope.up, np.float32)
    radius = float(frame.radius if frame is not None else scope.radius)
    distance = max(radius * 2.5, 0.5)
    fov_y = np.deg2rad(float(fov_y_deg))
    focal = (float(height) / 2.0) / np.tan(fov_y / 2.0)
    K = np.array([
        [focal, 0.0, float(width) / 2.0],
        [0.0, focal, float(height) / 2.0],
        [0.0, 0.0, 1.0],
    ], np.float32)

    conditioning = (halluc_manifest or {}).get("conditioning", {})
    start_azimuth = float(conditioning.get("azimuth_deg", 0.0))
    elevation = float(np.clip(conditioning.get("elevation_deg", 0.0), -35.0, 35.0))

    cameras = []
    for index in range(int(n_views)):
        azimuth = start_azimuth + 360.0 * index / max(int(n_views), 1)
        direction_L = R_L2V.T @ orbit_position(azimuth, elevation)
        position = frame.local_to_world((direction_L * distance).reshape(1, 3))[0]
        R_w2c, T_w2c = look_at(position, center, up)
        cameras.append({
            "index": index,
            "azimuth_deg": ((azimuth + 180.0) % 360.0) - 180.0,
            "elevation_deg": elevation,
            "R": R_w2c,
            "T": T_w2c,
            "K": K.copy(),
            "width": int(width),
            "height": int(height),
        })
    return cameras


def _render_rgb(gaussians, pipe_config, camera_spec, object_label_id=None):
    cam = make_camera(
        camera_spec["R"], camera_spec["T"], camera_spec["K"],
        camera_spec["width"], camera_spec["height"], uid=int(camera_spec["index"]),
    )
    result = render_rgba(
        gaussians, cam, pipe_config,
        bg_white=True,
        object_label_id=object_label_id,
        training=False,
    )
    rgb = result["rgb"].detach().clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
    return (rgb * 255.0 + 0.5).astype(np.uint8)


def generate_debug_artifacts(
    *,
    scope,
    frame,
    parent_gaussians,
    trained_gaussians,
    pipe_config,
    object_id,
    debug_dir,
    halluc_manifest=None,
    n_views=8,
):
    if parent_gaussians is None or trained_gaussians is None or pipe_config is None:
        return {"skipped": True, "reason": "missing parent/trained gaussians or pipe_config"}

    debug_dir = Path(debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)
    cameras = _orbit_cameras(scope, frame, halluc_manifest=halluc_manifest, n_views=n_views)
    files = []

    for camera in cameras:
        before = _render_rgb(parent_gaussians, pipe_config, camera, object_label_id=int(object_id))
        after = _render_rgb(trained_gaussians, pipe_config, camera)
        separator = np.full((before.shape[0], 6, 3), 255, np.uint8)
        compare = np.hstack([before, separator, after])

        index = int(camera["index"])
        before_path = debug_dir / f"before_view_{index:02d}.png"
        after_path = debug_dir / f"after_view_{index:02d}.png"
        compare_path = debug_dir / f"compare_view_{index:02d}.png"
        cv2.imwrite(str(before_path), cv2.cvtColor(before, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(after_path), cv2.cvtColor(after, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(compare_path), cv2.cvtColor(compare, cv2.COLOR_RGB2BGR))
        files.append({
            "index": index,
            "azimuth_deg": camera["azimuth_deg"],
            "elevation_deg": camera["elevation_deg"],
            "before": str(before_path),
            "after": str(after_path),
            "compare": str(compare_path),
        })

    summary = {"n_views": len(files), "files": files}
    with open(debug_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Compare renders saved to: %s", debug_dir)
    return summary


def _load_trained_gaussians(parent_gaussians, model_dir):
    from object_refiner.utils.gstrain_bridge import VRoomModel as GaussianModel
    from object_refiner.utils.config_compat import adapt_legacy_model_config
    from object_refiner.constants import GAUSSIAN_MODEL_DEFAULTS

    model_dir = Path(model_dir)
    kwargs = {
        k: getattr(parent_gaussians, k, GAUSSIAN_MODEL_DEFAULTS[k])
        for k in GAUSSIAN_MODEL_DEFAULTS
    } if parent_gaussians is not None else GAUSSIAN_MODEL_DEFAULTS
    kwargs = adapt_legacy_model_config(kwargs)

    gaussians = GaussianModel(
        gs_attr=str(kwargs.get("gs_attr", "2D")),
        feature_dim=int(kwargs.get("feature_dim", 32)),
        view_dim=int(kwargs.get("view_dim", 3)),
        appearance_dim=int(kwargs.get("appearance_dim", 0)),
        gaussians_per_anchor=int(kwargs.get("gaussians_per_anchor", 10)),
        voxel_size=float(kwargs.get("voxel_size", 0.001)),
        render_mode=str(kwargs.get("render_mode", "RGB+ED")),
        tile_size_2dgs=int(kwargs.get("tile_size_2dgs", 8)),
    )
    gaussians.load_ply(str(model_dir / "point_cloud.ply"))
    gaussians.load_mlp_checkpoints(str(model_dir))
    object.__setattr__(gaussians, "explicit_gs", False)
    gaussians.weed_ratio = 0.0
    gaussians.set_eval()
    return gaussians


def _resolve_ply(model_path, ply_path=None):
    if ply_path:
        return Path(ply_path)
    pc_base = Path(model_path) / "point_cloud"
    iter_dirs = sorted(
        [path for path in pc_base.iterdir() if path.is_dir() and path.name.startswith("iteration_")],
        key=lambda path: int(path.name.split("_")[-1]),
    ) if pc_base.exists() else []
    return (iter_dirs[-1] / "point_cloud.ply") if iter_dirs else (pc_base / "point_cloud.ply")


def main():
    from object_refiner.utils.scene_analysis import compute_object_scope, load_gaussians
    parser = argparse.ArgumentParser(description="Render before/after comparison images for a trained object_refiner object.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--object_id", type=int, required=True)
    parser.add_argument("--obj_dir", default=None)
    parser.add_argument("--output_root", default=None)
    parser.add_argument("--ply_path", default=None)
    parser.add_argument("--n_views", type=int, default=8)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
    obj_dir = Path(args.obj_dir) if args.obj_dir else Path(args.output_root) / f"obj_{args.object_id}" if args.output_root else None
    if obj_dir is None:
        raise SystemExit("--obj_dir or --output_root is required.")

    resolved_ply = _resolve_ply(args.model_path, args.ply_path)
    scope, frame, pipe_config = compute_object_scope(args.model_path, int(args.object_id), ply_path=str(resolved_ply))
    parent_gaussians, _ = load_gaussians(args.model_path, ply_path=str(resolved_ply))
    trained_gaussians = _load_trained_gaussians(parent_gaussians, obj_dir / "06_model")
    hallucination_index = obj_dir / "03_novel_views" / "generation.json"
    halluc_manifest = None
    if hallucination_index.exists():
        with open(hallucination_index) as f:
            halluc_manifest = json.load(f)

    generate_debug_artifacts(
        scope=scope,
        frame=frame,
        parent_gaussians=parent_gaussians,
        trained_gaussians=trained_gaussians,
        pipe_config=pipe_config,
        object_id=int(args.object_id),
        halluc_manifest=halluc_manifest,
        debug_dir=obj_dir / "debug" / "compare",
        n_views=int(args.n_views),
    )


if __name__ == "__main__":
    main()
