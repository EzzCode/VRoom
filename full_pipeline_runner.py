"""
VRoom Full Pipeline Runner

This script coordinates the entire pipeline:
1) sfm (Structure-from-Motion via COLMAP)
2) masks_and_tracking (SAM3 segmentation, tracking, and 3D voting)
3) gstrain (Gaussian Splatting Training)
4) mesh_generation (Extract RGB/Depth/Semantics and generate Meshes)

Usage:
    python full_pipeline_runner.py --data_path /path/to/scene_folder
"""

import argparse
import shlex
import subprocess
import sys
import json
import os
from pathlib import Path
from datetime import datetime

from masks_and_tracking.tracker_defaults import TRACKING_DEFAULTS


def _quote_for_log(parts):
    """Return a shell-safe command string for readable logging output."""
    return " ".join(shlex.quote(str(p)) for p in parts)


def run_step(step_name, cmd, dry_run=False, conda_env=None):
    """Run one pipeline stage command and raise on non-zero exit."""
    # Ensure Python output is unbuffered so progress bars stream in real-time
    if cmd and cmd[0] == "python":
        cmd = ["python", "-u"] + cmd[1:]

    if conda_env:
        # Prepend conda run with --no-capture-output to disable conda-side buffering
        cmd = ["conda", "run", "--no-capture-output", "-n", conda_env] + cmd

    print(f"\n=== {step_name} ===")
    if conda_env:
        print(f"[Environment: {conda_env}]")
    print(_quote_for_log(cmd))
    
    if dry_run:
        return

    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"Step failed: {step_name} (exit code {result.returncode})")

def get_latest_directory(parent_dir):
    """Finds the latest directory inside parent_dir."""
    if not os.path.exists(parent_dir):
        return None
    dirs = [os.path.join(parent_dir, d) for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d))]
    if not dirs:
        return None
    latest_dir = max(dirs, key=os.path.getctime)
    return latest_dir

def setup_images_dir(src_images_dir, dst_images_dir):
    """Link or copy images directory from source to destination."""
    src = Path(src_images_dir).absolute()
    dst = Path(dst_images_dir).absolute()
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    
    # Try directory junction (Windows specific, does not require admin)
    try:
        import subprocess
        subprocess.run(f'cmd /c mklink /J "{dst}" "{src}"', check=True, shell=True, stdout=subprocess.DEVNULL)
        print(f"Created directory junction: {dst} -> {src}")
        return
    except Exception:
        pass

    # Try symlink
    try:
        os.symlink(src, dst, target_is_directory=True)
        print(f"Created symbolic link: {dst} -> {src}")
        return
    except Exception:
        pass

    # Fallback to copy
    import shutil
    print(f"Copying images from {src} to {dst} (this may take a few seconds)...")
    shutil.copytree(src, dst)
    print("Images copied successfully.")

def main():
    parser = argparse.ArgumentParser(description="Full VRoom Pipeline Runner")
    parser.add_argument("--data_path", required=True, help="Scene folder containing images/")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without executing")

    # Conda Environments
    parser.add_argument("--gp_env", default="GP", help="Conda environment for Masks & Tracking")
    parser.add_argument("--objectgs_env", default="objectgs", help="Conda environment for SfM, Training, and Mesh Generation")

    # Skipping flags
    parser.add_argument("--skip_colmap", action="store_true", help="Skip COLMAP step")
    parser.add_argument("--skip_masks", action="store_true", help="Skip mask generation step")
    parser.add_argument("--skip_tracking", action="store_true", help="Skip object tracking step")
    parser.add_argument("--skip_voting", action="store_true", help="Skip 3D voting step")
    parser.add_argument("--skip_training", action="store_true", help="Skip Gaussian Splatting training step")
    parser.add_argument("--skip_mesh_gen", action="store_true", help="Skip Mesh Generation step")

    # Small run flag
    parser.add_argument("--small_run", action="store_true", help="Limit training iterations and frames for a quick test run")
    parser.add_argument("--num_iterations", type=int, default=None, help="Limit training iterations to a custom number")

    # Unified output directory
    parser.add_argument("--out_base_dir", type=str, default=None, help="If set, all pipeline outputs will be saved in this directory.")

    # COLMAP arguments
    parser.add_argument("--force_colmap", action="store_true", help="Force COLMAP to run from scratch")
    parser.add_argument("--camera_model", default="OPENCV", choices=["PINHOLE", "OPENCV", "SIMPLE_RADIAL", "RADIAL"])
    parser.add_argument("--matcher_type", default="sequential", choices=["exhaustive", "sequential", "spatial"])

    # Masks/Tracking arguments (passed to masks_and_tracking runner)
    parser.add_argument("--sam_ckpt", default="masks_and_tracking/models/sam3.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ultralytics_home", default="")
    parser.add_argument("--text_prompts", nargs="+", default=["chair", "table", "sofa", "bed", "desk", "cabinet"])
    parser.add_argument("--min_mask_area", type=int, default=120)
    parser.add_argument("--max_area_ratio", type=float, default=0.50)
    parser.add_argument("--border_threshold", type=float, default=0.35)
    parser.add_argument("--merge_thresh", type=float, default=0.78)
    parser.add_argument("--proximity_gap", type=int, default=20)
    parser.add_argument("--proximity_color_thresh", type=float, default=0.32)
    parser.add_argument("--no_split_disconnected", action="store_true")

    # Tracker arguments
    parser.add_argument("--iou_w", type=float, default=TRACKING_DEFAULTS["iou_w"])
    parser.add_argument("--color_w", type=float, default=TRACKING_DEFAULTS["color_w"])
    parser.add_argument("--texture_w", type=float, default=TRACKING_DEFAULTS["texture_w"])
    parser.add_argument("--bbox_w", type=float, default=TRACKING_DEFAULTS["bbox_w"])
    parser.add_argument("--match_threshold", type=float, default=TRACKING_DEFAULTS["match_threshold"])
    parser.add_argument("--patience", type=int, default=TRACKING_DEFAULTS["patience"])
    parser.add_argument("--smoothing_factor", type=float, default=TRACKING_DEFAULTS["smoothing_factor"])
    parser.add_argument("--reid_threshold", type=float, default=TRACKING_DEFAULTS["reid_threshold"])
    parser.add_argument("--disable_motion_comp", action="store_true")
    parser.add_argument("--consensus_window", type=int, default=TRACKING_DEFAULTS["consensus_window"])
    parser.add_argument("--consensus_tie_margin", type=float, default=TRACKING_DEFAULTS["consensus_tie_margin"])
    parser.add_argument("--use_opencv", action="store_true")

    # Voting arguments
    parser.add_argument("--output_dir", default="labeled_output")
    parser.add_argument("--min_points", type=int, default=10)
    parser.add_argument("--disable_alias_merge", action="store_true")
    parser.add_argument("--alias_iou_thresh", type=float, default=0.40)
    parser.add_argument("--alias_min_covisibility", type=int, default=15)

    args, unknown = parser.parse_known_args()

    script_dir = Path(__file__).resolve().parent

    # Assume `python` command points to the correct executable inside conda run
    py = "python"

    pipeline_data_path = Path(args.data_path)
    if args.out_base_dir:
        out_base = Path(args.out_base_dir).absolute()
        if not args.dry_run:
            out_base.mkdir(parents=True, exist_ok=True)
            src_images = Path(args.data_path) / "images"
            dst_images = out_base / "images"
            setup_images_dir(src_images, dst_images)
        else:
            print(f"[Dry Run] Would setup images directory: {out_base / 'images'} -> {Path(args.data_path) / 'images'}")
        pipeline_data_path = out_base

    # ── Stage 1: sfm (COLMAP) ──
    if not args.skip_colmap:
        colmap_cmd = [
            py,
            str(script_dir / "sfm" / "colmap_runner.py"),
            "--data_path", str(pipeline_data_path),
            "--camera_model", args.camera_model,
            "--matcher_type", args.matcher_type,
        ]
        if args.force_colmap:
            colmap_cmd.append("--force")
        run_step("COLMAP Reconstruction (sfm)", colmap_cmd, dry_run=args.dry_run, conda_env=args.objectgs_env)

    # ── Stage 2: masks_and_tracking ──
    if not (args.skip_masks and args.skip_tracking and args.skip_voting):
        tracking_cmd = [
            py,
            "-m", "masks_and_tracking.runner",
            "--data_path", str(pipeline_data_path),
            "--sam_ckpt", args.sam_ckpt,
            "--device", args.device,
            "--ultralytics_home", args.ultralytics_home,
            "--text_prompts", *args.text_prompts,
            "--min_mask_area", str(args.min_mask_area),
            "--max_area_ratio", str(args.max_area_ratio),
            "--border_threshold", str(args.border_threshold),
            "--merge_thresh", str(args.merge_thresh),
            "--proximity_gap", str(args.proximity_gap),
            "--proximity_color_thresh", str(args.proximity_color_thresh),
            "--iou_w", str(args.iou_w),
            "--color_w", str(args.color_w),
            "--texture_w", str(args.texture_w),
            "--bbox_w", str(args.bbox_w),
            "--match_threshold", str(args.match_threshold),
            "--patience", str(args.patience),
            "--smoothing_factor", str(args.smoothing_factor),
            "--reid_threshold", str(args.reid_threshold),
            "--consensus_window", str(args.consensus_window),
            "--consensus_tie_margin", str(args.consensus_tie_margin),
            "--output_dir", args.output_dir,
            "--min_points", str(args.min_points),
            "--alias_iou_thresh", str(args.alias_iou_thresh),
            "--alias_min_covisibility", str(args.alias_min_covisibility),
        ]
        
        # Override output_dir if unified base directory is given
        if args.out_base_dir:
            unified_tracking_out = Path(args.out_base_dir).absolute() / "labeled_output"
            # Replace the default output_dir in the command
            try:
                idx = tracking_cmd.index("--output_dir")
                tracking_cmd[idx + 1] = str(unified_tracking_out)
            except ValueError:
                tracking_cmd.extend(["--output_dir", str(unified_tracking_out)])
        
        if args.skip_masks:
            tracking_cmd.append("--skip_masks")
        if args.skip_tracking:
            tracking_cmd.append("--skip_tracking")
        if args.skip_voting:
            tracking_cmd.append("--skip_voting")
        if args.no_split_disconnected:
            tracking_cmd.append("--no_split_disconnected")
        if args.disable_motion_comp:
            tracking_cmd.append("--disable_motion_comp")
        if args.use_opencv:
            tracking_cmd.append("--use_opencv")
        if args.disable_alias_merge:
            tracking_cmd.append("--disable_alias_merge")

        run_step("Masks & Tracking Pipeline", tracking_cmd, dry_run=args.dry_run, conda_env=args.gp_env)

    # ── Stage 3: gstrain (Training) ──
    scene_name = Path(args.data_path).name
    if not args.skip_training:
        print("\n=== Generating Training Config ===")
        # Load the base 2DGS replica config
        base_config_path = script_dir / "gstrain" / "config" / "vroom" / "2d" / "replica" / "config.json"
        
        with open(base_config_path, "r") as f:
            train_config = json.load(f)
            
        # Update necessary fields
        train_config["experiment"]["dataset_name"] = scene_name
        train_config["experiment"]["dataset_path"] = str(pipeline_data_path.absolute())
        train_config["experiment"]["masks"] = "labeled_output/tracked/id_maps"

        if args.out_base_dir:
            # We use a path traversal trick to escape the hardcoded "output" folder in trainer.py
            # trainer.py does: os.path.join("output", dataset_name, exp_name, timestamp)
            # If dataset_name = "../test_run_outputs", it will resolve to test_run_outputs
            out_base_abs = Path(args.out_base_dir).absolute()
            # Calculate relative path from "output" (which is at script_dir/output) to out_base_abs
            # A simpler way is just to pass absolute path, but os.path.join("output", "/abs/path") 
            # in Python replaces the whole path if an absolute path is encountered.
            # Wait, os.path.join("output", "/path/to/...") -> "/path/to/..."
            # Let's use absolute path directly!
            train_config["experiment"]["dataset_name"] = str(out_base_abs / "training")
            train_config["experiment"]["save_dir"] = "gs_model"

        iterations = args.num_iterations
        if args.small_run:
            print("[Small Run] Limiting training iterations to 1000")
            iterations = 1000

        if iterations is not None:
            print(f"Limiting training iterations to {iterations}")
            train_config["optimization"]["num_iterations"] = iterations
            train_config["pipeline"]["save_iterations"] = [iterations]
            if "densifier" in train_config:
                train_config["densifier"]["desification_end"] = iterations
                current_start = train_config["densifier"].get("desification_start", 1500)
                train_config["densifier"]["desification_start"] = min(current_start, iterations // 2 if iterations > 200 else 100)

        # Save temp config
        temp_config_path = script_dir / f"temp_train_config_{scene_name}.json"
        if not args.dry_run:
            with open(temp_config_path, "w") as f:
                json.dump(train_config, f, indent=4)
            print(f"Created temporary training config: {temp_config_path}")

        training_cmd = [
            py,
            str(script_dir / "gstrain" / "trainer.py"),
            "--config", str(temp_config_path)
        ]
        
        run_step("Gaussian Splatting Training (gstrain)", training_cmd, dry_run=args.dry_run, conda_env=args.objectgs_env)

    # ── Stage 4: Mesh Generation ──
    if not args.skip_mesh_gen:
        print("\n=== Locating Trained Model ===")
        # Find the latest output directory for this dataset
        if args.out_base_dir:
            model_parent_dir = Path(args.out_base_dir) / "training" / "gs_model"
        else:
            save_dir_name = "saved_results" # matches the default in replica config
            model_parent_dir = script_dir / "output" / scene_name / save_dir_name
            
        latest_model_path = get_latest_directory(model_parent_dir)
        
        if not latest_model_path and not args.dry_run:
            print(f"Could not find trained model in {model_parent_dir}. Have you run training?")
            sys.exit(1)
            
        print(f"Found latest model: {latest_model_path}")
        
        # 1. Extract mesh inputs
        if args.out_base_dir:
            mesh_inputs_dir = Path(args.out_base_dir).absolute() / "mesh_inputs"
        else:
            mesh_inputs_dir = script_dir / "mesh_generation" / "inputs" / scene_name
            
        extract_inputs_cmd = [
            py,
            str(script_dir / "mesh_generation" / "extract_mesh_inputs.py"),
            "--model_path", str(latest_model_path) if latest_model_path else "DUMMY_PATH",
            "--output_dir", str(mesh_inputs_dir)
        ]
        

            
        run_step("Extract Mesh Inputs", extract_inputs_cmd, dry_run=args.dry_run, conda_env=args.objectgs_env)
        
        # 2. Generate meshes
        if args.out_base_dir:
            mesh_output_dir = Path(args.out_base_dir).absolute() / "mesh_objects"
        else:
            mesh_output_dir = script_dir / "mesh_generation" / "objects" / scene_name
            
        extract_meshes_cmd = [
            py,
            str(script_dir / "mesh_generation" / "extract_object_meshes.py"),
            "--input_dir", str(mesh_inputs_dir),
            "--output_dir", str(mesh_output_dir)
        ]
        run_step("Generate Object Meshes", extract_meshes_cmd, dry_run=args.dry_run, conda_env=args.objectgs_env)


    print("\nFull VRoom pipeline finished.")


if __name__ == "__main__":
    main()
