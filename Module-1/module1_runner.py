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

    # COLMAP options
    parser.add_argument("--camera_model", default="OPENCV", choices=["PINHOLE", "OPENCV", "SIMPLE_RADIAL", "RADIAL"])
    parser.add_argument("--matcher_type", default="sequential", choices=["exhaustive", "sequential", "spatial"])

    # SAM options
    parser.add_argument("--model_cfg", default="sam2.1_hiera_l")
    parser.add_argument("--sam_ckpt", default=str(Path("Module-1") / "models" / "sam2.1_hiera_large.pt"))
    parser.add_argument("--device", default="cuda")

    # Voting options
    parser.add_argument("--algorithm", default="majority", choices=["majority", "prob", "corr"])
    parser.add_argument("--output_dir", default="labeled_output", help="Output dir name inside data_path for vote.py")
    parser.add_argument("--min_points", type=int, default=10)

    args = parser.parse_args()

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
        masks_cmd = [
            py,
            str(script_dir / "mask_processor.py"),
            "--input_dir", str(paths["images"]),
            "--output_dir", str(paths["sam_output"]),
            "--model_cfg", args.model_cfg,
            "--sam_ckpt", args.sam_ckpt,
            "--device", args.device,
        ]
        run_step("Mask Generation", masks_cmd, dry_run=args.dry_run)

    if not args.skip_tracking:
        tracking_cmd = [
            py,
            str(script_dir / "object_tracker.py"),
            "--input_dir", str(paths["images"]),
            "--mask_dir", str(paths["sam_masks"]),
            "--output_dir", str(paths["tracked"]),
        ]
        run_step("Object Tracking", tracking_cmd, dry_run=args.dry_run)

    if not args.skip_voting:
        voting_cmd = [
            py,
            str(script_dir / "vote.py"),
            "--data_path", str(data_path),
            "--sparse_dir", "sparse/0",
            "--mask_dir", "tracked/id_maps",
            "--output_dir", args.output_dir,
            "--algorithm", args.algorithm,
            "--min_points", str(args.min_points),
        ]
        run_step("3D Voting", voting_cmd, dry_run=args.dry_run)

    print("\nPipeline finished.")


if __name__ == "__main__":
    main()
