"""Projection audit for rebuilt supervision views."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

_VROOM_ROOT = Path(__file__).resolve().parents[2]
if str(_VROOM_ROOT) not in sys.path:
    sys.path.insert(0, str(_VROOM_ROOT))

from object_refiner.constants import SEED_DEPTH_MIN

logger = logging.getLogger(__name__)


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def test_generation_manifest(generation_index_path):
    path = Path(generation_index_path)
    if not path.exists():
        return {"exists": False, "error": f"missing {path}"}

    with open(path) as f:
        manifest = json.load(f)

    frames = manifest.get("frames", [])
    accepted = [fr for fr in frames if fr.get("accepted", False)]
    missing = [fr.get("index") for fr in accepted if not Path(fr.get("rgba_path", "")).exists()]
    return {
        "exists": True,
        "n_frames": len(frames),
        "n_accepted": len(accepted),
        "n_reported_kept": int(manifest.get("n_kept", len(accepted))),
        "missing_accepted_rgba": missing,
    }


def test_point_cloud_geometry(seed_points_W, supervision_views):
    xyz = np.asarray(seed_points_W, np.float64)
    if xyz.size == 0:
        return {"error": "empty seed point cloud"}

    cameras = []
    for view in supervision_views:
        camera = view["camera"]
        if "position" in camera:
            cameras.append(np.asarray(camera["position"], np.float64))

    centroid = xyz.mean(axis=0)
    extent = xyz.max(axis=0) - xyz.min(axis=0)
    camera_distances = [float(np.linalg.norm(cam - centroid)) for cam in cameras]
    return {
        "n_points": xyz.shape[0],
        "centroid": centroid,
        "aabb_min": xyz.min(axis=0),
        "aabb_max": xyz.max(axis=0),
        "extent": extent,
        "camera_distance_min": min(camera_distances) if camera_distances else None,
        "camera_distance_max": max(camera_distances) if camera_distances else None,
        "camera_distance_mean": float(np.mean(camera_distances)) if camera_distances else None,
    }


def test_projection_overlay(seed_points_W, supervision_views, output_dir):
    import cv2

    overlay_dir = Path(output_dir) / "projection_overlay"
    xyz = np.asarray(seed_points_W, np.float64)
    results = []

    for index, view in enumerate(supervision_views):
        camera = view["camera"]
        R = np.asarray(camera["R"], np.float64)
        T = np.asarray(camera["T"], np.float64).reshape(3)
        K = np.asarray(camera["K"], np.float64)
        width = int(camera["width"])
        height = int(camera["height"])
        source = view.get("source", "unknown")

        pts_c = (R @ xyz.T).T + T
        in_front = pts_c[:, 2] > SEED_DEPTH_MIN
        pts_f = pts_c[in_front]
        n_behind = int((~in_front).sum())
        valid = np.zeros((0,), dtype=bool)
        u = np.asarray([], dtype=np.float64)
        v = np.asarray([], dtype=np.float64)

        if len(pts_f) > 0:
            u = K[0, 0] * (pts_f[:, 0] / pts_f[:, 2]) + K[0, 2]
            v = K[1, 1] * (pts_f[:, 1] / pts_f[:, 2]) + K[1, 2]
            valid = (u >= 0) & (u < width) & (v >= 0) & (v < height)

        n_in_frame = int(valid.sum())
        rgb = np.asarray(view["rgb"])
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) if rgb.ndim == 3 else cv2.cvtColor(rgb, cv2.COLOR_GRAY2BGR)
        if bgr.shape[:2] != (height, width):
            bgr = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA)

        depths = pts_f[valid, 2] if len(pts_f) > 0 else np.asarray([])
        if depths.size:
            d_norm = ((depths - depths.min()) / max(float(depths.max() - depths.min()), 1e-6)).clip(0, 1)
            colors = cv2.applyColorMap((d_norm * 255).astype(np.uint8).reshape(-1, 1), cv2.COLORMAP_JET)
            for point_index, (px, py) in enumerate(zip(u[valid].astype(np.int32), v[valid].astype(np.int32))):
                cv2.circle(bgr, (int(px), int(py)), 2, tuple(int(c) for c in colors[point_index, 0]), -1)

        azimuth = float(camera.get("azimuth_deg", 0.0))
        elevation = float(camera.get("elevation_deg", 0.0))
        label = f"#{index} {source} az={azimuth:.0f} el={elevation:.0f} n={n_in_frame}"
        cv2.putText(bgr, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        overlay_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(overlay_dir / f"view_{index:03d}_{source}.png"), bgr)

        results.append({
            "view_idx": index,
            "source": source,
            "azimuth_deg": azimuth,
            "elevation_deg": elevation,
            "n_in_frame": n_in_frame,
            "n_behind": n_behind,
            "mean_depth": float(depths.mean()) if depths.size else 0.0,
            "depth_min": float(depths.min()) if depths.size else 0.0,
            "depth_max": float(depths.max()) if depths.size else 0.0,
        })

    return results


def _run_projection_audit(obj_dir, model_path, object_id, debug_dir, scope=None, frame=None, ply_path=None):
    from object_refiner.dataset_builder import build_views
    from object_refiner.utils.colmap_init import load_colmap_object_point_cloud
    from object_refiner.utils.scene_analysis import compute_object_scope

    obj_dir = Path(obj_dir)
    debug_dir = Path(debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)

    if scope is None or frame is None:
        model_path = Path(model_path)
        if ply_path is None:
            pc_base = model_path / "point_cloud"
            iter_dirs = sorted(
                [d for d in pc_base.iterdir() if d.is_dir() and d.name.startswith("iteration_")],
                key=lambda d: int(d.name.split("_")[-1]),
            ) if pc_base.exists() else []
            resolved_ply = (iter_dirs[-1] / "point_cloud.ply") if iter_dirs else (pc_base / "point_cloud.ply")
        else:
            resolved_ply = Path(ply_path)
        scope, frame = compute_object_scope(str(model_path), int(object_id), ply_path=str(resolved_ply))

    extraction_index = obj_dir / "01_extraction" / "extraction_index.json"
    generation_index = obj_dir / "03_novel_views" / "generation.json"
    point_cloud, _meta = load_colmap_object_point_cloud(
        model_path=str(model_path),
        object_id=int(object_id),
        scope=scope,
        extraction_index_path=extraction_index if extraction_index.exists() else None,
        max_points=20000,
        target_points=8000,
    )
    seed_points_W = np.asarray(point_cloud.points, np.float32)

    # Load conditioning camera up-vector override if available
    up_override = None
    try:
        with open(generation_index) as f:
            gen_manifest = json.load(f)
        cam_idx = gen_manifest.get("conditioning", {}).get("cam_index")
        if cam_idx is not None and 0 <= int(cam_idx) < len(scope.cameras):
            up_override = -np.asarray(scope.cameras[int(cam_idx)]["R"], np.float32)[1]
    except Exception:
        pass

    supervision_views = build_views(
        generation_log_path=generation_index,
        extraction_path=extraction_index,
        scope=scope,
        frame=frame,
        cloud_points=seed_points_W,
        up_override=up_override,
    )

    report = {
        "n_seed_points": seed_points_W.shape[0],
        "n_views": len(supervision_views),
        "generation_manifest": test_generation_manifest(generation_index),
        "point_cloud_geometry": test_point_cloud_geometry(seed_points_W, supervision_views),
        "projection_overlay": test_projection_overlay(seed_points_W, supervision_views, debug_dir),
    }

    report_path = debug_dir / "audit_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=_json_default)
    logger.info("Projection audit report: %s", report_path)
    return report


def generate_debug_artifacts(
    *,
    obj_dir,
    scope=None,
    frame=None,
    model_path=None,
    object_id=None,
):
    if model_path is None or object_id is None:
        return {}
    obj_dir = Path(obj_dir)
    return {
        "projection": _run_projection_audit(
            obj_dir=obj_dir,
            model_path=model_path,
            object_id=object_id,
            debug_dir=obj_dir / "debug" / "projection_audit",
            scope=scope,
            frame=frame,
        )
    }


def _parse_args():
    parser = argparse.ArgumentParser(
        description="object_refiner projection audit.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--obj_dir", default=None)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_root", default=None)
    parser.add_argument("--object_id", type=int, required=True)
    parser.add_argument("--ply_path", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    args = _parse_args()
    obj_dir = args.obj_dir
    if obj_dir is None and args.output_root is not None:
        obj_dir = str(Path(args.output_root) / f"obj_{args.object_id}")
    if obj_dir is None:
        raise SystemExit("--obj_dir or --output_root is required.")

    _run_projection_audit(
        obj_dir=obj_dir,
        model_path=args.model_path,
        object_id=args.object_id,
        debug_dir=Path(obj_dir) / "debug" / "projection_audit",
        ply_path=args.ply_path,
    )
