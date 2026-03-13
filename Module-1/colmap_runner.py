"""
COLMAP Automation Pipeline for VRoom / ObjectGS

Automates the Structure-from-Motion (SfM) pipeline to extract camera poses 
and sparse point clouds from a directory of images.

Usage:
    python colmap_runner.py --data_path data/room_scene --camera_model OPENCV
"""

import os
import sys
import subprocess
import argparse
import logging
from pathlib import Path

# Configure professional logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

def check_colmap_installed():
    """Verify COLMAP is accessible in the system path."""
    try:
        subprocess.run(["colmap", "help"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("COLMAP is not installed or not in the system PATH.")
        logger.error("Please install COLMAP: https://colmap.github.io/install.html")
        sys.exit(1)

def run_step(cmd, step_name):
    """
    Run a subprocess command by delegating to the terminal (no capture),
    so the terminal can show native formatting, colors, and progress bars.
    """
    logger.info(f"--- Starting {step_name} ---")
    
    # Print the command for reproducibility
    cmd_str = ' '.join(str(x) for x in cmd)
    logger.debug(f"[CMD] {cmd_str}")

    try:
        # By omitting stdout and stderr, the child process inherits the parent's 
        # terminal handles natively. This preserves COLMAP's \r line-replacement!
        result = subprocess.run(cmd, check=False)
        
        if result.returncode != 0:
            logger.error(f"Fatal error during {step_name}. Exit code: {result.returncode}")
            sys.exit(1)
            
        logger.info(f"--- Finished {step_name} ---\n")
        
    except Exception as e:
        logger.error(f"Command execution failed: {e}")
        sys.exit(1)

def run_colmap_pipeline(args):
    """Orchestrates the COLMAP feature extraction and mapping process."""
    data_path = Path(args.data_path)
    image_dir = data_path / "images"
    database_path = data_path / "database.db"
    sparse_dir = data_path / "sparse" / "0"

    # 1. Validation
    if not image_dir.exists() or not any(image_dir.iterdir()):
        logger.error(f"Image directory not found or empty: {image_dir}")
        sys.exit(1)

    sparse_dir.mkdir(parents=True, exist_ok=True)
    check_colmap_installed()

    # 2. Feature Extraction
    if database_path.exists() and not args.force:
        logger.info(f"Database {database_path} already exists. Skipping extraction (use --force to overwrite).")
    else:
        if args.force and database_path.exists():
            database_path.unlink() # Delete old database
            
        extract_cmd = [
            "colmap", "feature_extractor",
            "--database_path", str(database_path),
            "--image_path", str(image_dir),
            "--ImageReader.camera_model", args.camera_model,
            "--ImageReader.single_camera", "1" if args.single_camera else "0"
        ]
        run_step(extract_cmd, "Feature Extraction")

    # 3. Feature Matching
    # Exhaustive is better for unordered photos; Sequential is faster for video frames
    match_cmd = [
        "colmap", args.matcher_type + "_matcher",
        "--database_path", str(database_path)
    ]
    run_step(match_cmd, "Feature Matching")

    # 4. Mapper (Sparse Reconstruction)
    # Check if we already have a successful reconstruction
    if (sparse_dir / "cameras.bin").exists() and not args.force:
        logger.info(f"Sparse model already exists in {sparse_dir}. Skipping mapping.")
    else:
        map_cmd = [
            "colmap", "mapper",
            "--database_path", str(database_path),
            "--image_path", str(image_dir),
            "--output_path", str(sparse_dir.parent) # COLMAP creates the '0' subfolder automatically
        ]
        run_step(map_cmd, "Sparse Mapping")

    # 5. Summary & Hand-off
    cameras_file = sparse_dir / "cameras.bin"
    if cameras_file.exists():
        logger.info("COLMAP Pipeline completed successfully! 🎉")
        logger.info(f"Data is ready for the Voting Script at: {sparse_dir}")
    else:
        logger.error("COLMAP Pipeline finished, but no cameras.bin was generated. COLMAP failed to reconstruct the scene.")
        logger.error("Try using more images, ensuring better overlap, or checking for blurry frames.")

if __name__ == "__main__":
    check_colmap_installed() # Early check before parsing arguments
    parser = argparse.ArgumentParser(description="Automated COLMAP Pipeline for VRoom")
    parser.add_argument("--data_path", required=True, help="Root folder containing the 'images' directory")
    parser.add_argument("--camera_model", default="OPENCV", choices=["PINHOLE", "OPENCV", "SIMPLE_RADIAL", "RADIAL"], 
                        help="Camera lens model (Gaussian Splatting highly prefers OPENCV or PINHOLE)")
    parser.add_argument("--matcher_type", default="sequential", choices=["exhaustive", "sequential", "spatial"],
                        help="Use 'sequential' if images were extracted directly from a video")
    parser.add_argument("--single_camera", action="store_true", default=True,
                        help="Assume all images were taken with the exact same camera lens (Recommended)")
    parser.add_argument("--force", action="store_true", 
                        help="Force overwrite of existing database and models")
    
    args = parser.parse_args()
    run_colmap_pipeline(args)