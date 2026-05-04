"""Phase 6 — Object isolation training pipeline.

Assumes Phases 0–5 have already run (``obj_<id>/phase5/hallucination_index.json`` must exist).

Per object:
    Phase 6 — build aligned supervision views from Phase-3 real outputs + Phase-5 SV3D outputs.
    Phase 7 — train an ObjectGS object model from COLMAP seed points + aligned views.
    Phase 8 — render before/after comparison orbit and save the scene package under obj_<id>/.

Usage::

    python -m object_isolation.run_phase6 \
        --model_path temp_deps/ObjectGS/outputs/3dovs/.../2026-03-19_04-01-38 \
        --output_root object_isolation/outputs \
        --object_ids 8 \
        --iterations 1200

    # Run then immediately build debug visuals:
    python -m object_isolation.run_phase6 ... --debug
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from object_isolation.core.scope import discover_object_scope
from object_isolation.core.training import run_training
from object_isolation.core.reintegration import (
    build_orbit_cameras, render_with_orbit, save_compare_grid,
    save_final_model, save_scene_package, build_reintegration_metadata,
    label_anchor_counts, render_composited_with_orbit,
)


logger = logging.getLogger(__name__)


def _setup_logging(level: int = logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def _resolve_halluc_index(output_root: Path, object_id: int) -> Path:
    p = output_root / f"obj_{object_id}" / "phase5" / "hallucination_index.json"
    if not p.exists():
        raise FileNotFoundError(
            f"Phase-5 manifest missing for obj {object_id}: {p}\n"
            f"Run Phase 5 first."
        )
    return p


def run(
    *,
    model_path: str,
    output_root: str | Path,
    object_ids: List[int],
    iterations: int = 1200,
    hallucination_weight: float = 1.0,
    real_weight: float = 1.0,
    novel_rgb_weight: float = 1.0,
    hallucination_rgb_scale: float = 1.0,
    depth_weight: float = 0.1,
    depth_start_iter: int = 100,
    depth_front_weight: float = 1.0,
    depth_back_weight: float = 0.15,
    fov_y_deg: float = 50.0,
    colmap_init_target_points: int = 8000,
    enable_densification: bool = False,
    max_anchor_count: int = 20000,
    densify_grad_threshold: float = 0.00005,
    densify_extra_ratio: float = 0.08,
    max_splat_world_size: float = 0.15,
    densify_real_views_only: bool = True,
    densify_warmup_frac: float = 0.15,
    densify_stop_frac: float = 0.60,
    halluc_decay_start_frac: float = 0.5,
    halluc_weight_floor: float = 0.25,
    min_halluc_iou: float = 0.55,
    min_halluc_area_ratio: float = 0.65,
    max_halluc_area_ratio: float = 1.45,
    n_compare_views: int = 8,
    skip_compare: bool = False,
    debug: bool = False,
) -> dict:
    """Run Phases 6/7/8 for every requested object_id."""
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if not object_ids:
        raise ValueError("object_ids must not be empty.")

    first_id = int(object_ids[0])
    logger.info("Discovering scope for first object %d at %s", first_id, model_path)
    scope0, _wl0, local_sv3d_first, gaussians, pipe_config = discover_object_scope(
        model_path, first_id,
    )

    parent_label_counts_pre = label_anchor_counts(gaussians)
    logger.info("Parent model anchor counts (pre): %s",
                {k: v for k, v in sorted(parent_label_counts_pre.items())})

    per_object_summaries: List[dict] = []

    for idx, obj_id in enumerate(object_ids):
        obj_id = int(obj_id)
        logger.info("\n%s\n=== Object %d (%d/%d) ===\n%s",
                    "=" * 60, obj_id, idx + 1, len(object_ids), "=" * 60)

        if obj_id == first_id and idx == 0:
            scope = scope0
            local_sv3d = local_sv3d_first
        else:
            scope, _wl, local_sv3d, _g, _p = discover_object_scope(model_path, obj_id)

        halluc_index = _resolve_halluc_index(output_root, obj_id)
        obj_out = output_root / f"obj_{obj_id}"
        renders_dir = obj_out / "renders"
        renders_dir.mkdir(parents=True, exist_ok=True)

        # ── Build comparison cameras BEFORE training ───────────────────────
        compare_cams = None
        before_obj_frames = before_full_frames = None
        if not skip_compare:
            ref_cam_pos = scope.cam_centers_visible_W.mean(axis=0).astype(np.float32)
            object_half_size = float(np.linalg.norm(
                np.asarray(scope.aabb_max_W, dtype=np.float32)
                - np.asarray(scope.aabb_min_W, dtype=np.float32)
            ) / 2.0)
            compare_cams = build_orbit_cameras(
                center=np.asarray(scope.centroid_W, dtype=np.float32),
                radius=object_half_size,
                orbit_radius=float(scope.radius),
                up=np.asarray(scope.up_W, dtype=np.float32),
                ref_cam_position=ref_cam_pos,
                n_views=int(n_compare_views),
            )
            before_obj_frames = render_with_orbit(
                gaussians, pipe_config, compare_cams, object_label_id=obj_id,
            )
            before_full_frames = render_with_orbit(
                gaussians, pipe_config, compare_cams, object_label_id=None,
            )

        # ── Phase 6/7: align hallucinations + train object model ───────────
        obj_gaussians = None
        try:
            summary = run_training(
                model_path=model_path,
                object_label_id=obj_id,
                halluc_index_path=str(halluc_index),
                output_dir=str(output_root),
                gaussians=gaussians,
                pipe_config=pipe_config,
                scope=scope,
                local_sv3d=local_sv3d,
                iterations=int(iterations),
                hallucination_weight=float(hallucination_weight),
                real_weight=float(real_weight),
                novel_rgb_weight=float(novel_rgb_weight),
                hallucination_rgb_scale=float(hallucination_rgb_scale),
                depth_weight=float(depth_weight),
                depth_start_iter=int(depth_start_iter),
                depth_front_weight=float(depth_front_weight),
                depth_back_weight=float(depth_back_weight),
                fov_y_deg=float(fov_y_deg),
                colmap_init_target_points=int(colmap_init_target_points),
                enable_densification=bool(enable_densification),
                max_anchor_count=int(max_anchor_count),
                densify_grad_threshold=float(densify_grad_threshold),
                densify_extra_ratio=float(densify_extra_ratio),
                max_splat_world_size=float(max_splat_world_size),
                densify_real_views_only=bool(densify_real_views_only),
                densify_warmup_frac=float(densify_warmup_frac),
                densify_stop_frac=float(densify_stop_frac),
                halluc_decay_start_frac=float(halluc_decay_start_frac),
                halluc_weight_floor=float(halluc_weight_floor),
                min_halluc_iou=float(min_halluc_iou),
                min_halluc_area_ratio=float(min_halluc_area_ratio),
                max_halluc_area_ratio=float(max_halluc_area_ratio),
            )
            obj_gaussians = summary.pop("_gaussians", None)
        except Exception as e:
            logger.exception("Training failed for obj %d: %s", obj_id, e)
            summary = {"object_id": obj_id, "skipped_reason": f"training_error: {e}"}

        per_object_summaries.append(summary)

        # ── Phase 8: before/after renders ─────────────────────────────────
        if compare_cams is not None and not skip_compare and obj_gaussians is not None:
            try:
                after_obj_frames = render_with_orbit(
                    obj_gaussians, pipe_config, compare_cams, object_label_id=None,
                )
                after_full_frames = render_composited_with_orbit(
                    gaussians, obj_gaussians, pipe_config, compare_cams, object_label_id=obj_id,
                )
                save_compare_grid(
                    before_obj_frames, after_obj_frames,
                    renders_dir / "object_only",
                    prefix="obj",
                )
                save_compare_grid(
                    before_full_frames, after_full_frames,
                    renders_dir / "full_scene",
                    prefix="scene",
                )
                logger.info("Phase 8: saved compare grids to %s", renders_dir)
            except Exception as e:
                logger.exception("Phase 8 compare-render failed for obj %d: %s", obj_id, e)

    # ── Final model export ────────────────────────────────────────────────
    parent_label_counts_post = label_anchor_counts(gaussians)
    metadata = build_reintegration_metadata(
        parent_label_counts_pre=parent_label_counts_pre,
        parent_label_counts_post=parent_label_counts_post,
        per_object_summaries=per_object_summaries,
        reference_model_path=str(model_path),
    )

    all_object_training = all(s.get("mode") == "object_training" for s in per_object_summaries)
    if all_object_training:
        # For a single object the scene package lives next to its model;
        # for multiple objects it aggregates at output_root/scene/.
        scene_out = (
            output_root / f"obj_{int(object_ids[0])}"
            if len(object_ids) == 1
            else output_root
        )
        final_output = save_scene_package(
            output_dir=scene_out,
            reference_model_path=model_path,
            per_object_summaries=per_object_summaries,
            extra_metadata=metadata,
        )
    else:
        final_output = save_final_model(
            gaussians,
            output_dir=output_root,
            reference_model_path=model_path,
            extra_metadata=metadata,
        )

    with open(output_root / "pipeline_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "model_path": str(model_path),
            "object_ids": [int(x) for x in object_ids],
            "metadata": metadata,
        }, f, indent=2)

    logger.info("Phase 6 pipeline complete. Output -> %s", final_output)

    if debug:
        from object_isolation.debug.visualize_phase6 import run_debug
        for obj_id in object_ids:
            run_debug(
                model_path=model_path,
                object_id=int(obj_id),
                output_root=str(output_root),
                iterations=int(iterations),
                hallucination_weight=float(hallucination_weight),
                real_weight=float(real_weight),
                novel_rgb_weight=float(novel_rgb_weight),
                hallucination_rgb_scale=float(hallucination_rgb_scale),
                depth_weight=float(depth_weight),
                depth_start_iter=int(depth_start_iter),
                depth_front_weight=float(depth_front_weight),
                depth_back_weight=float(depth_back_weight),
                fov_y_deg=float(fov_y_deg),
                colmap_init_target_points=int(colmap_init_target_points),
                enable_densification=bool(enable_densification),
                max_anchor_count=int(max_anchor_count),
                densify_grad_threshold=float(densify_grad_threshold),
                densify_extra_ratio=float(densify_extra_ratio),
                n_compare_views=int(n_compare_views),
                no_run=True,  # pipeline already ran above
            )

    return metadata


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Object-isolation Phase 6 pipeline orchestrator")
    p.add_argument("--model_path", required=True,
                   help="Path to trained ObjectGS output dir (containing point_cloud/, cameras.json).")
    p.add_argument("--output_root", default="object_isolation/outputs",
                   help="Root holding obj_<id>/phase5 inputs and where outputs are written.")
    p.add_argument("--object_ids", type=int, nargs="+", required=True,
                   help="Object label IDs to train (Phase 5 must already exist for each).")
    p.add_argument("--iterations", type=int, default=1200,
                   help="Number of ObjectGS training iterations.")
    p.add_argument("--hallucination_weight", type=float, default=1.0)
    p.add_argument("--real_weight", type=float, default=1.0)
    p.add_argument("--novel_rgb_weight", type=float, default=1.0)
    p.add_argument("--hallucination_rgb_scale", type=float, default=1.0,
                   help="Extra RGB-loss scale for hallucinated views; masks/alpha still train shape.")
    p.add_argument("--depth_weight", type=float, default=0.1,
                   help="Asymmetric depth regularization weight for reliable real views only.")
    p.add_argument("--depth_start_iter", type=int, default=100,
                   help="First iteration where real-view depth regularization is active.")
    p.add_argument("--depth_front_weight", type=float, default=1.0,
                   help="Penalty scale when the object renders in front of reliable real depth.")
    p.add_argument("--depth_back_weight", type=float, default=0.15,
                   help="Penalty scale when the object renders behind reliable real depth.")
    p.add_argument("--fov_y_deg", type=float, default=50.0,
                   help="Vertical FOV used for SV3D outputs (must match Phase-5 setting).")
    p.add_argument("--colmap_init_target_points", type=int, default=8000,
                   help="Fresh seed points via COLMAP-neighbor interpolation (no parent ObjectGS points used).")
    p.add_argument("--enable_densification", action="store_true",
                   help="Enable ObjectGS densification from COLMAP seed points.")
    p.add_argument("--max_anchor_count", type=int, default=20000,
                   help="Stop densification once this anchor count is reached.")
    p.add_argument("--densify_grad_threshold", type=float, default=0.00005)
    p.add_argument("--densify_extra_ratio", type=float, default=0.08)
    p.add_argument("--max_splat_world_size", type=float, default=0.15,
                   help="Max splat world size as fraction of spatial_extent (log-space cap).")
    p.add_argument("--no_densify_real_views_only", action="store_true",
                   help="Accumulate densification stats from all views (default: real views only).")
    p.add_argument("--densify_warmup_frac", type=float, default=0.15,
                   help="Fraction of iterations before densification starts.")
    p.add_argument("--densify_stop_frac", type=float, default=0.60,
                   help="Fraction of iterations after which densification is locked off.")
    p.add_argument("--halluc_decay_start_frac", type=float, default=0.5,
                   help="Fraction of iterations after which hallucination RGB weight decays.")
    p.add_argument("--halluc_weight_floor", type=float, default=0.25,
                   help="Minimum hallucination weight multiplier at end of decay.")
    p.add_argument("--min_halluc_iou", type=float, default=0.55,
                   help="Min mask-IoU for hallucinated-view alignment audit. "
                        "Lower values accept more back/side views (e.g. 0.10).")
    p.add_argument("--min_halluc_area_ratio", type=float, default=0.65,
                   help="Min SV3D/reference mask area ratio (lower = more lenient).")
    p.add_argument("--max_halluc_area_ratio", type=float, default=1.45,
                   help="Max SV3D/reference mask area ratio (higher = more lenient).")
    p.add_argument("--n_compare_views", type=int, default=8)
    p.add_argument("--skip_compare", action="store_true",
                   help="Skip before/after orbit rendering.")
    p.add_argument("--debug", action="store_true",
                   help="Build debug visuals under obj_<id>/debug/ immediately after training.")
    p.add_argument("--log_level", default="INFO")
    return p.parse_args()


def main():
    args = _parse_args()
    _setup_logging(getattr(logging, str(args.log_level).upper(), logging.INFO))
    torch.backends.cudnn.benchmark = True

    run(
        model_path=args.model_path,
        output_root=args.output_root,
        object_ids=args.object_ids,
        iterations=args.iterations,
        hallucination_weight=args.hallucination_weight,
        real_weight=args.real_weight,
        novel_rgb_weight=args.novel_rgb_weight,
        hallucination_rgb_scale=args.hallucination_rgb_scale,
        depth_weight=args.depth_weight,
        depth_start_iter=args.depth_start_iter,
        depth_front_weight=args.depth_front_weight,
        depth_back_weight=args.depth_back_weight,
        fov_y_deg=args.fov_y_deg,
        colmap_init_target_points=args.colmap_init_target_points,
        enable_densification=args.enable_densification,
        max_anchor_count=args.max_anchor_count,
        densify_grad_threshold=args.densify_grad_threshold,
        densify_extra_ratio=args.densify_extra_ratio,
        max_splat_world_size=args.max_splat_world_size,
        densify_real_views_only=not args.no_densify_real_views_only,
        densify_warmup_frac=args.densify_warmup_frac,
        densify_stop_frac=args.densify_stop_frac,
        halluc_decay_start_frac=args.halluc_decay_start_frac,
        halluc_weight_floor=args.halluc_weight_floor,
        min_halluc_iou=args.min_halluc_iou,
        min_halluc_area_ratio=args.min_halluc_area_ratio,
        max_halluc_area_ratio=args.max_halluc_area_ratio,
        n_compare_views=args.n_compare_views,
        skip_compare=args.skip_compare,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
