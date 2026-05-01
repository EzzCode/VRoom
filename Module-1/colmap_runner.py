"""
COLMAP automation pipeline for VRoom / ObjectGS.

Supports both:
- standard scale-ambiguous SfM with COLMAP mapper
- metric known-pose triangulation from ARCore mobile scene bundles
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import struct
import subprocess
import sys
from pathlib import Path

from metric_bundle import export_known_pose_colmap_workspace, load_metric_bundle
from sim3_alignment import compute_metric_alignment


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def check_colmap_installed():
    try:
        subprocess.run(["colmap", "help"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("COLMAP is not installed or not in the system PATH.")
        logger.error("Please install COLMAP: https://colmap.github.io/install.html")
        sys.exit(1)


def run_step(cmd, step_name):
    logger.info(f"--- Starting {step_name} ---")
    logger.debug("CMD: %s", " ".join(str(x) for x in cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        logger.error("Fatal error during %s. Exit code: %s", step_name, result.returncode)
        sys.exit(result.returncode)
    logger.info("--- Finished %s ---\n", step_name)


def _count_points3d(path: Path) -> int:
    if not path.exists():
        return 0
    if path.suffix == ".bin":
        with open(path, "rb") as handle:
            payload = handle.read(8)
            if len(payload) != 8:
                return 0
            return int(struct.unpack("<Q", payload)[0])
    count = 0
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                count += 1
    return count


def _read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _feature_extract(database_path: Path, image_dir: Path, args, camera_model: str | None = None) -> None:
    extract_cmd = [
        "colmap",
        "feature_extractor",
        "--database_path",
        str(database_path),
        "--image_path",
        str(image_dir),
        "--ImageReader.camera_model",
        camera_model or args.camera_model,
        "--ImageReader.single_camera",
        "1" if args.single_camera else "0",
    ]
    run_step(extract_cmd, "Feature Extraction")


def _feature_match(database_path: Path, args) -> None:
    match_cmd = [
        "colmap",
        f"{args.matcher_type}_matcher",
        "--database_path",
        str(database_path),
    ]
    run_step(match_cmd, "Feature Matching")


def _update_summary(output_path: Path, update: dict) -> None:
    summary_path = output_path / "reconstruction_summary.json"
    summary = _read_json(summary_path) if summary_path.exists() else {}
    summary.update(update)
    _write_json(summary_path, summary)


def run_standard_colmap_pipeline(args, data_path, output_path):
    image_dir = data_path / "images"
    if not image_dir.exists():
        image_dir = data_path / "frames"
    
    database_path = output_path / "database.db"
    sparse_dir = output_path / "sparse" / "0"

    if not image_dir.exists() or not any(image_dir.iterdir()):
        logger.error("Image directory not found or empty: %s", image_dir)
        sys.exit(1)

    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)
    
    # Symlink images if output_path is different from data_path
    if output_path != data_path:
        out_images_dir = output_path / "images"
        out_images_dir.mkdir(parents=True, exist_ok=True)
        from metric_bundle import _symlink_or_copy
        for img_path in image_dir.iterdir():
            if img_path.is_file():
                _symlink_or_copy(img_path, out_images_dir / img_path.name)
        image_dir = out_images_dir

    sparse_dir.mkdir(parents=True, exist_ok=True)
    check_colmap_installed()

    if database_path.exists() and not args.force:
        logger.info("Database %s already exists. Skipping extraction (use --force to overwrite).", database_path)
    else:
        if args.force and database_path.exists():
            database_path.unlink()
        _feature_extract(database_path, image_dir, args)

    _feature_match(database_path, args)

    if (sparse_dir / "cameras.bin").exists() and not args.force:
        logger.info("Sparse model already exists in %s. Skipping mapping.", sparse_dir)
    else:
        map_cmd = [
            "colmap",
            "mapper",
            "--database_path",
            str(database_path),
            "--image_path",
            str(image_dir),
            "--output_path",
            str(sparse_dir.parent),
        ]
        run_step(map_cmd, "Sparse Mapping")

    point_path = sparse_dir / "points3D.bin"
    if not point_path.exists():
        point_path = sparse_dir / "points3D.txt"

    _write_json(
        output_path / "reconstruction_summary.json",
        {
            "reconstruction_mode": "standard_sfm",
            "input_frame_count": len(list(image_dir.iterdir())),
            "valid_frame_count": len(list(image_dir.iterdir())),
            "rejected_frame_count": 0,
            "rejected_frame_ids": [],
            "camera_path_length_m": None,
            "angular_coverage_deg": None,
            "contiguous_tracking_runs": None,
            "tracking_quality_summary": "unknown",
            "triangulated_point_count": _count_points3d(point_path),
            "colmap_workspace": str(output_path),
            "frames_root": str(image_dir),
        },
    )

    if not ((sparse_dir / "cameras.bin").exists() or (sparse_dir / "cameras.txt").exists()):
        logger.error("COLMAP Pipeline finished, but no sparse model was generated.")
        sys.exit(1)

    logger.info("COLMAP Pipeline completed successfully.")
    logger.info("Data is ready for the voting script at: %s", sparse_dir)

    # ## Metric alignment: if ARCore data exists, align SfM → meters ####################
    if (data_path / "poses.json").exists() and (data_path / "tracking.json").exists():
        logger.info("ARCore data detected — running Sim(3) metric alignment.")
        alignment = compute_metric_alignment(output_path, data_path)
        if alignment is not None:
            _update_summary(output_path, {
                "metric_alignment": {
                    "method": "umeyama_sim3",
                    "scale": alignment.scale,
                    "rmse_meters": alignment.rmse,
                    "num_correspondences": alignment.num_correspondences,
                },
                "units": "meters",
            })
            logger.info(
                "Model aligned to metric scale: scale=%.6f, RMSE=%.4fm",
                alignment.scale, alignment.rmse,
            )
        else:
            logger.warning("Metric alignment failed or was skipped; model remains in arbitrary scale.")


def run_known_pose_pipeline(args, data_path, output_path):
    database_path = output_path / "database.db"
    sparse_dir = output_path / "sparse" / "0"

    check_colmap_installed()
    bundle = load_metric_bundle(data_path)
    
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)
    export_known_pose_colmap_workspace(bundle, output_path)

    if args.force and database_path.exists():
        database_path.unlink()

    if not database_path.exists():
        # Known pose triangulation uses the symlinked images in output_path/images
        _feature_extract(database_path, output_path / "images", args, camera_model="PINHOLE")
    else:
        logger.info("Database %s already exists. Reusing extracted features.", database_path)

    _feature_match(database_path, args)

    triangulate_cmd = [
        "colmap",
        "point_triangulator",
        "--database_path",
        str(database_path),
        "--image_path",
        str(output_path / "images"),
        "--input_path",
        str(sparse_dir),
        "--output_path",
        str(sparse_dir),
    ]
    run_step(triangulate_cmd, "Known-Pose Triangulation")

    point_path = sparse_dir / "points3D.bin"
    if not point_path.exists():
        point_path = sparse_dir / "points3D.txt"
    _update_summary(
        output_path,
        {
            "triangulated_point_count": _count_points3d(point_path),
            "valid_frame_ids": bundle.valid_frame_ids,
            "rejected_frame_ids": bundle.rejected_frame_ids,
            "colmap_workspace": str(output_path),
            "frames_root": str(output_path / "images"),
        },
    )
    logger.info("Known-pose COLMAP pipeline completed successfully.")
    logger.info("Data is ready for the voting script at: %s", sparse_dir)


def run_colmap_pipeline(args):
    data_path = Path(args.data_path).resolve()
    output_path = Path(args.output_path).resolve() if args.output_path else data_path
    
    if args.reconstruction_mode == "known_pose_triangulation":
        run_known_pose_pipeline(args, data_path, output_path)
    else:
        run_standard_colmap_pipeline(args, data_path, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated COLMAP pipeline for VRoom")
    parser.add_argument("--data_path", required=True, help="Root folder containing images/ or a metric bundle manifest")
    parser.add_argument("--output_path", help="Path to save outputs (defaults to data_path)")
    parser.add_argument(
        "--camera_model",
        default="OPENCV",
        choices=["PINHOLE", "OPENCV", "SIMPLE_RADIAL", "RADIAL"],
        help="Camera lens model used by COLMAP feature extraction",
    )
    parser.add_argument(
        "--matcher_type",
        default="sequential",
        choices=["exhaustive", "sequential", "spatial"],
        help="Use 'sequential' if images were extracted directly from a video",
    )
    parser.add_argument(
        "--single_camera",
        action="store_true",
        default=True,
        help="Assume all images were taken with the exact same camera lens",
    )
    parser.add_argument("--force", action="store_true", help="Force overwrite of existing database and model artifacts")
    parser.add_argument(
        "--reconstruction_mode",
        default="standard_sfm",
        choices=["standard_sfm", "known_pose_triangulation"],
        help="Use known_pose_triangulation for ARCore metric bundles",
    )
    args = parser.parse_args()
    run_colmap_pipeline(args)
