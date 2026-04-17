"""
Module-1 End-to-End Runner

This entrypoint coordinates the four-stage processing pipeline:
1) COLMAP reconstruction
2) SAM3 mask generation + postprocessing
3) multi-object tracking from NPZ masks
4) 3D voting/projection

Usage:
    python module1_runner.py --data_path /path/to/scene_folder
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


def build_paths(data_path):
    """Build canonical per-stage filesystem paths for a scene folder."""
    return {
        "images": data_path / "images",
        "sam_output": data_path / "sam_output",
        "sam_masks": data_path / "sam_output" / "masks",
        "tracked": data_path / "tracked",
        "tracked_id_maps": data_path / "tracked" / "id_maps",
    }


def get_profile_overrides(profile):
    """Return profile-specific default parameters used by stages."""
    profile_aliases = {
        "video_balanced": "balanced",
        "video_conservative": "conservative",
        "video_recall": "recall",
    }
    profile = profile_aliases.get(profile, profile)

    presets = {
        "balanced": {
            "min_mask_area": 120,
            "max_area_ratio": 0.50,
            "border_touch_threshold": 0.35,
            "merge_thresh": 0.78,
            "proximity_gap": 20,
            "proximity_color_thresh": 0.32,
            "match_threshold": 0.74,
            "patience": 28,
            "ema": 0.70,
            "reid_threshold": 0.50,
            "min_confidence": 0.35,
            "min_support": 3,
            "temporal_decay": 0.02,
            "alias_min_point_support": 20,
            "alias_min_shared_views": 6,
            "alias_min_weight_support": 0.0,
            "alias_min_support_ratio": 0.12,
            "alias_min_point_balance": 0.25,
            "alias_min_obs_per_label_per_point": 2,
        },
        "conservative": {
            "min_mask_area": 150,
            "max_area_ratio": 0.45,
            "border_touch_threshold": 0.30,
            "merge_thresh": 0.82,
            "proximity_gap": 16,
            "proximity_color_thresh": 0.28,
            "match_threshold": 0.72,
            "patience": 24,
            "ema": 0.68,
            "reid_threshold": 0.45,
            "min_confidence": 0.45,
            "min_support": 4,
            "temporal_decay": 0.03,
            "alias_min_point_support": 28,
            "alias_min_shared_views": 8,
            "alias_min_weight_support": 0.0,
            "alias_min_support_ratio": 0.18,
            "alias_min_point_balance": 0.30,
            "alias_min_obs_per_label_per_point": 3,
        },
        "recall": {
            "min_mask_area": 100,
            "max_area_ratio": 0.60,
            "border_touch_threshold": 0.40,
            "merge_thresh": 0.72,
            "proximity_gap": 24,
            "proximity_color_thresh": 0.36,
            "match_threshold": 0.76,
            "patience": 32,
            "ema": 0.72,
            "reid_threshold": 0.55,
            "min_confidence": 0.30,
            "min_support": 2,
            "temporal_decay": 0.015,
            "alias_min_point_support": 14,
            "alias_min_shared_views": 5,
            "alias_min_weight_support": 0.0,
            "alias_min_support_ratio": 0.08,
            "alias_min_point_balance": 0.20,
            "alias_min_obs_per_label_per_point": 2,
        },
    }
    return presets[profile]


def resolve_param(args, name, profile_values):
    """Resolve one parameter from CLI override or selected profile default."""
    value = getattr(args, name)
    return profile_values[name] if value is None else value


def main():
    """Parse CLI arguments, build stage commands, and execute pipeline."""
    parser = argparse.ArgumentParser(description="Run the full Module-1 pipeline")
    parser.add_argument("--data_path", required=True, help="Scene folder containing images/")

    parser.add_argument("--dry_run", action="store_true", help="Print commands without executing")
    parser.add_argument("--force_colmap", action="store_true", help="Pass --force to colmap_runner.py")
    parser.add_argument("--skip_colmap", action="store_true", help="Skip COLMAP step")
    parser.add_argument("--skip_masks", action="store_true", help="Skip mask_processor step")
    parser.add_argument("--skip_tracking", action="store_true", help="Skip object_tracker step")
    parser.add_argument("--skip_voting", action="store_true", help="Skip vote step")
    parser.add_argument(
        "--profile",
        default="balanced",
        choices=["balanced", "conservative", "recall", "video_balanced", "video_conservative", "video_recall"],
        help="Preset profile for robust one-size behavior",
    )
    parser.add_argument("--video_mode", action="store_true", help="Compatibility flag (no-op)")
    parser.add_argument("--use_correction", action="store_true", help="Compatibility flag (no-op)")
    parser.add_argument("--enable_flow_fallback", action="store_true", help="Compatibility flag (no-op)")

    parser.add_argument("--camera_model", default="OPENCV", choices=["PINHOLE", "OPENCV", "SIMPLE_RADIAL", "RADIAL"])
    parser.add_argument("--matcher_type", default="sequential", choices=["exhaustive", "sequential", "spatial"])

    parser.add_argument("--sam_ckpt", default="Module-1/models/sam3.pt", help="SAM3 checkpoint (e.g., sam3.pt, sam3_b.pt)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ultralytics_home", default="", help="Directory for Ultralytics checkpoints/cache")
    parser.add_argument("--text_prompts", nargs="+", default=["furniture"])
    parser.add_argument("--min_mask_area", type=int, default=None)
    parser.add_argument("--max_area_ratio", type=float, default=None)
    parser.add_argument("--border_touch_threshold", type=float, default=None)
    parser.add_argument("--merge_thresh", type=float, default=None)
    parser.add_argument("--proximity_gap", type=int, default=None)
    parser.add_argument("--proximity_color_thresh", type=float, default=None)
    parser.add_argument("--no_split_disconnected", action="store_true")

    parser.add_argument("--alpha", type=float, default=0.68)
    parser.add_argument("--beta", type=float, default=0.25)
    parser.add_argument("--gamma", type=float, default=0.15)
    parser.add_argument("--delta", type=float, default=0.12)
    parser.add_argument("--match_threshold", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--ema", type=float, default=None)
    parser.add_argument("--reid_threshold", type=float, default=None)
    parser.add_argument("--disable_motion_comp", action="store_true", help="Disable global camera-motion compensation in tracker")
    parser.add_argument("--disable_consensus", action="store_true", help="Disable in-clip temporal consensus in tracker")
    parser.add_argument("--consensus_window", type=int, default=8, help="Temporal window length for tracker consensus")
    parser.add_argument("--consensus_tie_margin", type=float, default=0.05, help="IoU vote margin for appearance tie-break")

    parser.add_argument("--algorithm", default="majority", choices=["majority", "prob", "corr"])
    parser.add_argument("--output_dir", default="labeled_output", help="Output dir name inside data_path for vote.py")
    parser.add_argument("--min_points", type=int, default=10)
    parser.add_argument("--min_confidence", type=float, default=None)
    parser.add_argument("--min_support", type=int, default=None)
    parser.add_argument("--temporal_decay", type=float, default=None)
    parser.add_argument("--disable_alias_merge", action="store_true", help="Disable correspondence-based alias merging in vote.py")

    args = parser.parse_args()
    profile_values = get_profile_overrides(args.profile)

    script_dir = Path(__file__).resolve().parent
    data_path = Path(args.data_path).resolve()

    if not data_path.exists():
        raise FileNotFoundError(f"data_path does not exist: {data_path}")

    paths = build_paths(data_path)
    if not paths["images"].exists() and not args.skip_masks and not args.skip_tracking:
        raise FileNotFoundError(f"Missing images directory: {paths['images']}")

    py = sys.executable

    if not args.skip_colmap:
        colmap_cmd = [
            py,
            str(script_dir / "colmap_runner.py"),
            "--data_path", str(data_path),
            "--camera_model", args.camera_model,
            "--matcher_type", args.matcher_type,
        ]
        if args.force_colmap:
            colmap_cmd.append("--force")
        run_step("COLMAP Reconstruction", colmap_cmd, dry_run=args.dry_run)

    if not args.skip_masks:
        min_mask_area = resolve_param(args, "min_mask_area", profile_values)
        max_area_ratio = resolve_param(args, "max_area_ratio", profile_values)
        border_touch_threshold = resolve_param(args, "border_touch_threshold", profile_values)
        merge_thresh = resolve_param(args, "merge_thresh", profile_values)
        proximity_gap = resolve_param(args, "proximity_gap", profile_values)
        proximity_color_thresh = resolve_param(args, "proximity_color_thresh", profile_values)

        masks_cmd = [
            py,
            str(script_dir / "mask_processor.py"),
            "--input_dir", str(paths["images"]),
            "--output_dir", str(paths["sam_output"]),
            "--sam_ckpt", args.sam_ckpt,
            "--device", args.device,
            "--ultralytics_home", args.ultralytics_home,
            "--text_prompts", *args.text_prompts,
            "--min_mask_area", str(min_mask_area),
            "--max_area_ratio", str(max_area_ratio),
            "--border_touch_threshold", str(border_touch_threshold),
            "--merge_thresh", str(merge_thresh),
            "--proximity_gap", str(proximity_gap),
            "--proximity_color_thresh", str(proximity_color_thresh),
        ]
        if args.no_split_disconnected:
            masks_cmd.append("--no_split_disconnected")
        run_step("Mask Generation", masks_cmd, dry_run=args.dry_run)

    if not args.skip_tracking:
        match_threshold = resolve_param(args, "match_threshold", profile_values)
        patience = resolve_param(args, "patience", profile_values)
        ema = resolve_param(args, "ema", profile_values)
        reid_threshold = resolve_param(args, "reid_threshold", profile_values)

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
            "--match_threshold", str(match_threshold),
            "--patience", str(patience),
            "--ema", str(ema),
            "--reid_threshold", str(reid_threshold),
            "--consensus_window", str(args.consensus_window),
            "--consensus_tie_margin", str(args.consensus_tie_margin),
        ]
        if args.disable_motion_comp:
            tracking_cmd.append("--disable_motion_comp")
        if args.disable_consensus:
            tracking_cmd.append("--disable_consensus")
        run_step("Object Tracking", tracking_cmd, dry_run=args.dry_run)

    if not args.skip_voting:
        min_confidence = resolve_param(args, "min_confidence", profile_values)
        min_support = resolve_param(args, "min_support", profile_values)
        temporal_decay = resolve_param(args, "temporal_decay", profile_values)

        voting_cmd = [
            py,
            str(script_dir / "vote.py"),
            "--data_path", str(data_path),
            "--sparse_dir", "sparse/0",
            "--mask_dir", "tracked/id_maps",
            "--output_dir", args.output_dir,
            "--algorithm", args.algorithm,
            "--min_points", str(args.min_points),
            "--min_confidence", str(min_confidence),
            "--min_support", str(min_support),
            "--temporal_decay", str(temporal_decay),
            "--alias_min_point_support", str(profile_values["alias_min_point_support"]),
            "--alias_min_shared_views", str(profile_values["alias_min_shared_views"]),
            "--alias_min_weight_support", str(profile_values["alias_min_weight_support"]),
            "--alias_min_support_ratio", str(profile_values["alias_min_support_ratio"]),
            "--alias_min_point_balance", str(profile_values["alias_min_point_balance"]),
            "--alias_min_obs_per_label_per_point", str(profile_values["alias_min_obs_per_label_per_point"]),
        ]
        if args.disable_alias_merge:
            voting_cmd.append("--disable_alias_merge")
        run_step("3D Voting", voting_cmd, dry_run=args.dry_run)

    print("\nPipeline finished.")


if __name__ == "__main__":
    main()
