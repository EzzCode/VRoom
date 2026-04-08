"""
Module-1 End-to-End Runner

Runs the full Module-1 pipeline in sequence:
1) COLMAP reconstruction
2) SAM mask generation and post-processing
3) Multi-modal object tracking
4) 3D point-cloud voting

Usage:
    python Module-1/module1_runner.py --data_path data/room_scene
"""

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


def _quote_for_log(parts):
    return " ".join(shlex.quote(str(p)) for p in parts)


def run_step(step_name, cmd, dry_run=False):
    print(f"\n=== {step_name} ===")
    print(_quote_for_log(cmd))

    if dry_run:
        return

    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"Step failed: {step_name} (exit code {result.returncode})")


def build_paths(data_path):
    return {
        "images": data_path / "images",
        "sam_output": data_path / "sam_output",
        "sam_masks": data_path / "sam_output" / "masks",
        "tracked": data_path / "tracked",
        "tracked_id_maps": data_path / "tracked" / "id_maps",
    }


def get_profile_overrides(profile):
    presets = {
        "balanced": {
            "max_area": 0.50,
            "border_touch": 0.35,
            "points_per_side": 40,
            "pred_iou_thresh": 0.88,
            "stability_score_thresh": 0.92,
            "min_mask_area": 300,
            "merge_thresh": 0.78,
            "proximity_gap": 20,
            "proximity_color_thresh": 0.32,
            "min_area_ratio": 0.0035,
            "match_threshold": 0.74,
            "patience": 28,
            "ema": 0.7,
            "flow_reliability_threshold": 0.25,
            "reid_threshold": 0.5,
            "min_confidence": 0.35,
            "min_support": 3,
            "temporal_decay": 0.02,
        },
        "conservative": {
            "max_area": 0.45,
            "border_touch": 0.30,
            "points_per_side": 32,
            "pred_iou_thresh": 0.90,
            "stability_score_thresh": 0.94,
            "min_mask_area": 350,
            "merge_thresh": 0.82,
            "proximity_gap": 16,
            "proximity_color_thresh": 0.28,
            "min_area_ratio": 0.0045,
            "match_threshold": 0.72,
            "patience": 24,
            "ema": 0.68,
            "flow_reliability_threshold": 0.3,
            "reid_threshold": 0.45,
            "min_confidence": 0.45,
            "min_support": 4,
            "temporal_decay": 0.03,
        },
        "recall": {
            "max_area": 0.55,
            "border_touch": 0.40,
            "points_per_side": 48,
            "pred_iou_thresh": 0.85,
            "stability_score_thresh": 0.90,
            "min_mask_area": 220,
            "merge_thresh": 0.74,
            "proximity_gap": 24,
            "proximity_color_thresh": 0.36,
            "min_area_ratio": 0.003,
            "match_threshold": 0.76,
            "patience": 32,
            "ema": 0.72,
            "flow_reliability_threshold": 0.2,
            "reid_threshold": 0.55,
            "min_confidence": 0.3,
            "min_support": 2,
            "temporal_decay": 0.015,
        },
    }
    return presets[profile]


def resolve_param(args, name, profile_values):
    value = getattr(args, name)
    return profile_values[name] if value is None else value


def main():
    parser = argparse.ArgumentParser(description="Run the full Module-1 pipeline")
    parser.add_argument("--data_path", required=True, help="Scene folder containing images/")

    # Global control
    parser.add_argument("--dry_run", action="store_true", help="Print commands without executing")
    parser.add_argument("--force_colmap", action="store_true", help="Pass --force to colmap_runner.py")
    parser.add_argument("--skip_colmap", action="store_true", help="Skip COLMAP step")
    parser.add_argument("--skip_masks", action="store_true", help="Skip mask_processor step")
    parser.add_argument("--skip_tracking", action="store_true", help="Skip object_tracker step")
    parser.add_argument("--skip_voting", action="store_true", help="Skip vote step")
    parser.add_argument("--profile", default="balanced", choices=["balanced", "conservative", "recall"],
                        help="Preset profile for robust one-size behavior")

    # COLMAP options
    parser.add_argument("--camera_model", default="OPENCV", choices=["PINHOLE", "OPENCV", "SIMPLE_RADIAL", "RADIAL"])
    parser.add_argument("--matcher_type", default="sequential", choices=["exhaustive", "sequential", "spatial"])

    # SAM options
    parser.add_argument("--model_cfg", default="sam2.1_hiera_l")
    parser.add_argument("--sam_ckpt", default=str(Path("Module-1") / "models" / "sam2.1_hiera_large.pt"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_area", type=float, default=None)
    parser.add_argument("--border_touch", type=float, default=None)
    parser.add_argument("--points_per_side", type=int, default=None)
    parser.add_argument("--pred_iou_thresh", type=float, default=None)
    parser.add_argument("--stability_score_thresh", type=float, default=None)
    parser.add_argument("--min_mask_area", type=int, default=None)
    parser.add_argument("--merge_thresh", type=float, default=None)
    parser.add_argument("--proximity_gap", type=int, default=None)
    parser.add_argument("--proximity_color_thresh", type=float, default=None)
    parser.add_argument("--min_area_ratio", type=float, default=None)

    # Tracker options
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--beta", type=float, default=0.25)
    parser.add_argument("--gamma", type=float, default=0.15)
    parser.add_argument("--delta", type=float, default=0.2)
    parser.add_argument("--match_threshold", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--ema", type=float, default=None)
    parser.add_argument("--flow_reliability_threshold", type=float, default=None)
    parser.add_argument("--reid_threshold", type=float, default=None)

    # Voting options
    parser.add_argument("--algorithm", default="majority", choices=["majority", "prob", "corr"])
    parser.add_argument("--output_dir", default="labeled_output", help="Output dir name inside data_path for vote.py")
    parser.add_argument("--min_points", type=int, default=10)
    parser.add_argument("--min_confidence", type=float, default=None)
    parser.add_argument("--min_support", type=int, default=None)
    parser.add_argument("--temporal_decay", type=float, default=None)
    parser.add_argument("--disable_alias_merge", action="store_true", help="Disable correspondence-based alias merging in vote.py")
    parser.add_argument("--alias_min_point_support", type=int, default=12, help="Minimum shared COLMAP points to accept an alias edge")
    parser.add_argument("--alias_min_shared_views", type=int, default=6, help="Minimum distinct views to accept an alias edge")
    parser.add_argument("--alias_min_weight_support", type=float, default=0.0, help="Minimum weighted co-support to accept an alias edge")

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
        max_area = resolve_param(args, "max_area", profile_values)
        border_touch = resolve_param(args, "border_touch", profile_values)
        points_per_side = resolve_param(args, "points_per_side", profile_values)
        pred_iou_thresh = resolve_param(args, "pred_iou_thresh", profile_values)
        stability_score_thresh = resolve_param(args, "stability_score_thresh", profile_values)
        min_mask_area = resolve_param(args, "min_mask_area", profile_values)
        merge_thresh = resolve_param(args, "merge_thresh", profile_values)
        proximity_gap = resolve_param(args, "proximity_gap", profile_values)
        proximity_color_thresh = resolve_param(args, "proximity_color_thresh", profile_values)
        min_area_ratio = resolve_param(args, "min_area_ratio", profile_values)

        masks_cmd = [
            py,
            str(script_dir / "mask_processor.py"),
            "--input_dir", str(paths["images"]),
            "--output_dir", str(paths["sam_output"]),
            "--model_cfg", args.model_cfg,
            "--sam_ckpt", args.sam_ckpt,
            "--device", args.device,
            "--max_area", str(max_area),
            "--border_touch", str(border_touch),
            "--points_per_side", str(points_per_side),
            "--pred_iou_thresh", str(pred_iou_thresh),
            "--stability_score_thresh", str(stability_score_thresh),
            "--min_mask_area", str(min_mask_area),
            "--merge_thresh", str(merge_thresh),
            "--proximity_gap", str(proximity_gap),
            "--proximity_color_thresh", str(proximity_color_thresh),
            "--min_area_ratio", str(min_area_ratio),
        ]
        run_step("Mask Generation", masks_cmd, dry_run=args.dry_run)

    if not args.skip_tracking:
        match_threshold = resolve_param(args, "match_threshold", profile_values)
        patience = resolve_param(args, "patience", profile_values)
        ema = resolve_param(args, "ema", profile_values)
        flow_reliability_threshold = resolve_param(args, "flow_reliability_threshold", profile_values)
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
            "--flow_reliability_threshold", str(flow_reliability_threshold),
            "--reid_threshold", str(reid_threshold),
        ]
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
            "--alias_min_point_support", str(args.alias_min_point_support),
            "--alias_min_shared_views", str(args.alias_min_shared_views),
            "--alias_min_weight_support", str(args.alias_min_weight_support),
        ]
        if args.disable_alias_merge:
            voting_cmd.append("--disable_alias_merge")
        run_step("3D Voting", voting_cmd, dry_run=args.dry_run)

    print("\nPipeline finished.")


if __name__ == "__main__":
    main()
