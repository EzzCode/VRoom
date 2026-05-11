"""
Module-1 End-to-End Runner

This entrypoint coordinates the four-stage processing pipeline:
1) COLMAP reconstruction
2) SAM3 mask generation + postprocessing
3) multi-object tracking from NPZ masks
4) 3D voting/projection

Usage:
    python module1_runner.py --data_path /path/to/scene_folder --output_path /path/to/writable_folder
"""

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


def _quote_for_log(parts):
    """Return a shell-safe command string for readable logging output."""
    return " ".join(shlex.quote(str(p)) for p in parts)


def run_step(step_name, cmd, dry_run=False):
    """Run one pipeline stage command and raise on non-zero exit."""
    print(f"\n=== {step_name} ===")
    print(_quote_for_log(cmd))
    if dry_run:
        return

    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"Step failed: {step_name} (exit code {result.returncode})")


def build_paths(data_path, output_path):
    """Build canonical per-stage filesystem paths for a scene folder."""
    image_root = data_path / "images"
    if not image_root.exists() and (data_path / "frames").exists():
        image_root = data_path / "frames"
    
    # If output_path is different from data_path, we might find images in output_path/images
    # (e.g. symlinked by colmap_runner.py in known_pose mode)
    if not image_root.exists() and (output_path / "images").exists():
        image_root = output_path / "images"

    return {
        "images": image_root,
        "sam_output": output_path / "sam_output",
        "sam_masks": output_path / "sam_output" / "masks",
        "tracked": output_path / "tracked",
        "tracked_id_maps": output_path / "tracked" / "id_maps",
    }


def main():
    """Parse CLI arguments, build stage commands, and execute pipeline."""
    parser = argparse.ArgumentParser(description="Run the full Module-1 pipeline")
    parser.add_argument("--data_path", required=True, help="Scene folder containing images/ (read-only input)")
    parser.add_argument("--output_path", help="Path to save all outputs (defaults to data_path)")

    parser.add_argument("--dry_run", action="store_true", help="Print commands without executing")
    parser.add_argument("--force_colmap", action="store_true", help="Pass --force to colmap_runner.py")
    parser.add_argument("--skip_colmap", action="store_true", help="Skip COLMAP step")
    parser.add_argument("--skip_masks", action="store_true", help="Skip mask_processor step")
    parser.add_argument("--skip_tracking", action="store_true", help="Skip object_tracker step")
    parser.add_argument("--skip_voting", action="store_true", help="Skip vote step")
    parser.add_argument(
        "--reconstruction_mode",
        default=None,
        choices=["standard_sfm", "known_pose_triangulation"],
        help="Override reconstruction mode. Defaults to known_pose_triangulation when manifest.json is present.",
    )

    # COLMAP
    parser.add_argument("--camera_model", default="OPENCV", choices=["PINHOLE", "OPENCV", "SIMPLE_RADIAL", "RADIAL"])
    parser.add_argument("--matcher_type", default="sequential", choices=["exhaustive", "sequential", "spatial"])

    # Mask generation
    parser.add_argument("--sam_ckpt", default="Module-1/models/sam3.pt", help="SAM3 checkpoint (e.g., sam3.pt, sam3_b.pt)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ultralytics_home", default="", help="Directory for Ultralytics checkpoints/cache")
    parser.add_argument("--text_prompts", nargs="+", default=["object"])
    parser.add_argument("--min_mask_area", type=int, default=120)
    parser.add_argument("--max_area_ratio", type=float, default=0.50)
    parser.add_argument("--border_touch_threshold", type=float, default=0.35)
    parser.add_argument("--merge_thresh", type=float, default=0.78)
    parser.add_argument("--proximity_gap", type=int, default=20)
    parser.add_argument("--proximity_color_thresh", type=float, default=0.32)
    parser.add_argument("--no_split_disconnected", action="store_true")

    # Tracking
    parser.add_argument("--alpha", type=float, default=0.35)
    parser.add_argument("--beta", type=float, default=0.30)
    parser.add_argument("--gamma", type=float, default=0.15)
    parser.add_argument("--delta", type=float, default=0.20)
    parser.add_argument("--match_threshold", type=float, default=0.74)
    parser.add_argument("--patience", type=int, default=28)
    parser.add_argument("--ema", type=float, default=0.70)
    parser.add_argument("--reid_threshold", type=float, default=0.50)
    parser.add_argument("--disable_motion_comp", action="store_true", help="Disable global camera-motion compensation in tracker")
    parser.add_argument("--consensus_window", type=int, default=8, help="Temporal window length for tracker consensus")
    parser.add_argument("--consensus_tie_margin", type=float, default=0.05, help="IoU vote margin for appearance tie-break")

    # Voting
    parser.add_argument("--algorithm", default="majority", choices=["majority", "prob", "corr"])
    parser.add_argument("--output_dir", default="labeled_output", help="Output dir name inside output_path for vote.py")
    parser.add_argument("--min_points", type=int, default=10)
    parser.add_argument("--disable_alias_merge", action="store_true", help="Disable correspondence-based alias merging in vote.py")
    parser.add_argument("--alias_iou_thresh", type=float, default=0.40, help="Min 3D IoU to consider two tracker IDs as aliases")
    parser.add_argument("--alias_min_covisibility", type=int, default=15, help="Min shared views for alias merge candidates")

    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    data_path = Path(args.data_path).resolve()
    output_path = Path(args.output_path).resolve() if args.output_path else data_path

    if not data_path.exists():
        raise FileNotFoundError(f"data_path does not exist: {data_path}")

    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)

    paths = build_paths(data_path, output_path)
    reconstruction_mode = args.reconstruction_mode
    if reconstruction_mode is None:
        reconstruction_mode = "standard_sfm"
        if (data_path / "manifest.json").exists():
            print("ARCore manifest detected — using standard SfM with post-hoc metric alignment.")

    if not paths["images"].exists() and not args.skip_colmap:
         # In known_pose mode, colmap_runner will create output_path/images
         if reconstruction_mode != "known_pose_triangulation":
            raise FileNotFoundError(f"Missing images directory: {paths['images']}")

    py = sys.executable

    # ── Stage 1: COLMAP ──
    if not args.skip_colmap:
        colmap_cmd = [
            py,
            str(script_dir / "colmap_runner.py"),
            "--data_path", str(data_path),
            "--output_path", str(output_path),
            "--camera_model", args.camera_model,
            "--matcher_type", args.matcher_type,
            "--reconstruction_mode", reconstruction_mode,
        ]
        if args.force_colmap:
            colmap_cmd.append("--force")
        run_step("COLMAP Reconstruction", colmap_cmd, dry_run=args.dry_run)
        
        # Refresh paths in case colmap_runner created output_path/images
        paths = build_paths(data_path, output_path)

    # ── Stage 2: Mask Generation ──
    if not args.skip_masks:
        masks_cmd = [
            py,
            str(script_dir / "mask_processor.py"),
            "--input_dir", str(paths["images"]),
            "--output_dir", str(paths["sam_output"]),
            "--sam_ckpt", args.sam_ckpt,
            "--device", args.device,
            "--ultralytics_home", args.ultralytics_home,
            "--text_prompts", *args.text_prompts,
            "--min_mask_area", str(args.min_mask_area),
            "--max_area_ratio", str(args.max_area_ratio),
            "--border_touch_threshold", str(args.border_touch_threshold),
            "--merge_thresh", str(args.merge_thresh),
            "--proximity_gap", str(args.proximity_gap),
            "--proximity_color_thresh", str(args.proximity_color_thresh),
        ]
        if args.no_split_disconnected:
            masks_cmd.append("--no_split_disconnected")
        run_step("Mask Generation", masks_cmd, dry_run=args.dry_run)

    # ── Stage 3: Object Tracking ──
    if not args.skip_tracking:
        tracking_cmd = [
            py,
            str(script_dir / "object_tracker.py"),
            "--input_dir", str(paths["images"]),
            "--mask_dir", str(paths["sam_masks"]),
            "--output_dir", str(paths["tracked"]),
            "--alpha", str(args.alpha),
            "--beta", str(args.beta),
            "--gamma", str(args.gamma),
            "--delta", str(args.delta),
            "--match_threshold", str(args.match_threshold),
            "--patience", str(args.patience),
            "--ema", str(args.ema),
            "--reid_threshold", str(args.reid_threshold),
            "--consensus_window", str(args.consensus_window),
            "--consensus_tie_margin", str(args.consensus_tie_margin),
        ]
        if args.disable_motion_comp:
            tracking_cmd.append("--disable_motion_comp")
        run_step("Object Tracking", tracking_cmd, dry_run=args.dry_run)

    # ── Stage 4: 3D Voting ──
    if not args.skip_voting:
        voting_cmd = [
            py,
            str(script_dir / "vote.py"),
            "--data_path", str(output_path),
            "--sparse_dir", "sparse/0",
            "--mask_dir", "tracked/id_maps",
            "--output_dir", args.output_dir,
            "--algorithm", args.algorithm,
            "--min_points", str(args.min_points),
            "--alias_iou_thresh", str(args.alias_iou_thresh),
            "--alias_min_covisibility", str(args.alias_min_covisibility),
        ]
        if args.disable_alias_merge:
            voting_cmd.append("--disable_alias_merge")
        run_step("3D Voting", voting_cmd, dry_run=args.dry_run)

    print("\nPipeline finished.")


if __name__ == "__main__":
    main()
