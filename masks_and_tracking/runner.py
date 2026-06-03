"""masks_and_tracking end-to-end runner: mask generation → tracking → 3D voting."""

import argparse
import logging
import shlex
import subprocess
import sys
from pathlib import Path

from masks_and_tracking.tracker_defaults import TRACKING_DEFAULTS

logger = logging.getLogger(__name__)


def _quote_for_log(parts):
    return " ".join(shlex.quote(str(p)) for p in parts)


def run_step(step_name, cmd, dry_run=False):
    """Run one pipeline stage and raise on non-zero exit."""
    logger.info("=== %s ===", step_name)
    logger.info(_quote_for_log(cmd))
    if dry_run:
        return
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"Step failed: {step_name} (exit code {result.returncode})")


def build_paths(data_path, output_dir=None):
    """Build canonical per-stage filesystem paths for a scene folder."""
    base_out = data_path / output_dir if output_dir else data_path
    return {
        "images":         data_path / "images",
        "sam_output":     base_out / "sam_output",
        "sam_masks":      base_out / "sam_output" / "masks",
        "tracked":        base_out / "tracked",
        "tracked_id_maps":base_out / "tracked" / "id_maps",
    }


def main():
    """Parse CLI arguments, build stage commands, and execute pipeline."""
    parser = argparse.ArgumentParser(description="Run the masks_and_tracking pipeline")
    parser.add_argument("--data_path", required=True, help="Scene folder containing images/")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without executing")
    parser.add_argument("--skip_masks", action="store_true", help="Skip mask_processor step")
    parser.add_argument("--skip_tracking", action="store_true", help="Skip object_tracker step")
    parser.add_argument("--skip_voting", action="store_true", help="Skip vote step")

    # Mask generation
    parser.add_argument("--sam_ckpt", default="masks_and_tracking/models/sam3.pt", help="SAM3 checkpoint")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ultralytics_home", default="", help="Directory for Ultralytics checkpoints/cache")
    parser.add_argument("--text_prompts", nargs="+", default=["chair","table","sofa","bed","desk","cabinet"])
    parser.add_argument("--min_mask_area", type=int, default=120)
    parser.add_argument("--max_area_ratio", type=float, default=0.50)
    parser.add_argument("--border_threshold", type=float, default=0.35)
    parser.add_argument("--merge_thresh", type=float, default=0.78)
    parser.add_argument("--proximity_gap", type=int, default=20)
    parser.add_argument("--proximity_color_thresh", type=float, default=0.32)
    parser.add_argument("--no_split_disconnected", action="store_true")

    # Tracking
    parser.add_argument("--iou_w", type=float, default=TRACKING_DEFAULTS["iou_w"])
    parser.add_argument("--color_w", type=float, default=TRACKING_DEFAULTS["color_w"])
    parser.add_argument("--texture_w", type=float, default=TRACKING_DEFAULTS["texture_w"])
    parser.add_argument("--bbox_w", type=float, default=TRACKING_DEFAULTS["bbox_w"])
    parser.add_argument("--match_threshold", type=float, default=TRACKING_DEFAULTS["match_threshold"])
    parser.add_argument("--patience", type=int, default=TRACKING_DEFAULTS["patience"])
    parser.add_argument("--smoothing_factor", type=float, default=TRACKING_DEFAULTS["smoothing_factor"])
    parser.add_argument("--reid_threshold", type=float, default=TRACKING_DEFAULTS["reid_threshold"])
    parser.add_argument("--disable_motion_comp", action="store_true", help="Disable global camera-motion compensation in tracker")
    parser.add_argument("--consensus_window", type=int, default=TRACKING_DEFAULTS["consensus_window"], help="Temporal window length for tracker consensus")
    parser.add_argument("--consensus_tie_margin", type=float, default=TRACKING_DEFAULTS["consensus_tie_margin"], help="IoU vote margin for appearance tie-break")
    parser.add_argument("--use_opencv", action="store_true", help="Use standard OpenCV library in the tracker")

    # Voting
    parser.add_argument("--output_dir", default="labeled_output", help="Output dir name inside data_path for vote.py")
    parser.add_argument("--min_points", type=int, default=10)
    parser.add_argument("--disable_alias_merge", action="store_true", help="Disable correspondence-based alias merging in vote.py")
    parser.add_argument("--alias_iou_thresh", type=float, default=0.40, help="Min 3D IoU to consider two tracker IDs as aliases")
    parser.add_argument("--alias_min_covisibility", type=int, default=15, help="Min shared views for alias merge candidates")

    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    data_path  = Path(args.data_path).resolve()

    if not data_path.exists():
        raise FileNotFoundError(f"data_path does not exist: {data_path}")

    paths = build_paths(data_path, args.output_dir)
    if not paths["images"].exists() and not args.skip_masks and not args.skip_tracking:
        raise FileNotFoundError(f"Missing images directory: {paths['images']}")

    py = sys.executable

    # ── Stage 2: Mask Generation ──
    if not args.skip_masks:
        masks_cmd = [
            py, "-m", "masks_and_tracking.mask_processor",
            "--input_dir", str(paths["images"]),
            "--output_dir", str(paths["sam_output"]),
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
        ]
        if args.no_split_disconnected:
            masks_cmd.append("--no_split_disconnected")
        run_step("Mask Generation", masks_cmd, dry_run=args.dry_run)

    # ── Stage 3: Object Tracking ──
    if not args.skip_tracking:
        tracking_cmd = [
            py, "-m", "masks_and_tracking.object_tracker",
            "--input_dir", str(paths["images"]),
            "--mask_dir", str(paths["sam_masks"]),
            "--output_dir", str(paths["tracked"]),
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
        ]
        if args.disable_motion_comp:
            tracking_cmd.append("--disable_motion_comp")
        if args.use_opencv:
            tracking_cmd.append("--use_opencv")
        run_step("Object Tracking", tracking_cmd, dry_run=args.dry_run)

    # ── Stage 4: 3D Voting ──
    if not args.skip_voting:
        voting_cmd = [
            py, "-m", "masks_and_tracking.vote",
            "--data_path", str(data_path),
            "--sparse_dir", "sparse/0",
            "--mask_dir", str(Path(args.output_dir) / "tracked" / "id_maps") if args.output_dir else "tracked/id_maps",
            "--output_dir", args.output_dir,
            "--min_points", str(args.min_points),
            "--alias_iou_thresh", str(args.alias_iou_thresh),
            "--alias_min_covisibility", str(args.alias_min_covisibility),
        ]
        if args.disable_alias_merge:
            voting_cmd.append("--disable_alias_merge")
        run_step("3D Voting", voting_cmd, dry_run=args.dry_run)

    logger.info("Pipeline finished.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S")
    main()
