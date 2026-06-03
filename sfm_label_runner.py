"""
VRoom SfM & Semantic Labeling Combined Runner

This script coordinates:
1) sfm (Structure-from-Motion via COLMAP)
2) masks_and_tracking (SAM3 segmentation, tracking, and 3D voting)

Usage:
    python sfm_label_runner.py --data_path /path/to/scene_folder
"""

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

from masks_and_tracking.tracker_defaults import TRACKING_DEFAULTS


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


def main():
    parser = argparse.ArgumentParser(description="Combined SfM and Semantic Labeling Runner")
    parser.add_argument("--data_path", required=True, help="Scene folder containing images/")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without executing")

    # Skipping flags
    parser.add_argument("--skip_colmap", action="store_true", help="Skip COLMAP step")
    parser.add_argument("--skip_masks", action="store_true", help="Skip mask generation step")
    parser.add_argument("--skip_tracking", action="store_true", help="Skip object tracking step")
    parser.add_argument("--skip_voting", action="store_true", help="Skip 3D voting step")

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
    py = sys.executable

    # ── Stage 1: sfm (COLMAP) ──
    if not args.skip_colmap:
        colmap_cmd = [
            py,
            str(script_dir / "sfm" / "colmap_runner.py"),
            "--data_path", str(args.data_path),
            "--camera_model", args.camera_model,
            "--matcher_type", args.matcher_type,
        ]
        if args.force_colmap:
            colmap_cmd.append("--force")
        run_step("COLMAP Reconstruction (sfm)", colmap_cmd, dry_run=args.dry_run)

    # ── Stage 2: masks_and_tracking ──
    if not (args.skip_masks and args.skip_tracking and args.skip_voting):
        tracking_cmd = [
            py,
            "-m", "masks_and_tracking.runner",
            "--data_path", str(args.data_path),
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

        run_step("Masks & Tracking Pipeline", tracking_cmd, dry_run=args.dry_run)

    print("\nSfM & Semantic Labeling pipeline finished.")


if __name__ == "__main__":
    main()
