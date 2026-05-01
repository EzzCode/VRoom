"""End-to-end orchestrator for Phases 6 → 7 → 8 of the object-isolation pipeline.

Assumes Phase 0–5 have already been run (i.e. ``object_isolation/outputs/
obj_<id>/phase5/hallucination_index.json`` exists).

Per object:
    Phase 6 — build aligned supervision_views from Phase-3 real outputs + Phase-5 SV3D outputs.
    Phase 7 — scratch ObjectGS object training from aligned real + hallucinated views.
    Phase 8 — render scratch composite comparisons + save a scratch scene package.

Usage::

    python -m object_isolation.run_phase678 \
        --model_path temp_deps/ObjectGS/outputs/replica/.../office_0/.../ \
        --output_root object_isolation/outputs \
        --object_ids 8 9 \
        --scratch_iterations 1200 \
        --hallucination_weight 1.0
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
from object_isolation.core.training import run_phase7
from object_isolation.core.reintegration import (
    build_orbit_cameras, render_with_orbit, save_compare_grid,
    save_final_model, save_scratch_scene_package, build_reintegration_metadata,
    label_anchor_counts, render_composited_scratch_with_orbit,
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
    scratch_iterations: int = 1200,
    hallucination_weight: float = 1.0,
    real_weight: float = 1.0,
    novel_rgb_weight: float = 1.0,
    fov_y_deg: float = 50.0,
    grid_resolution: int = 25,
    visual_hull_min_views: int = 10,
    n_compare_views: int = 8,
    skip_compare: bool = False,
) -> dict:
    """Run Phases 6/7/8 for every requested object_id."""
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # ── Load parent model once via Phase-1 discovery for the first object ─
    # All subsequent objects reuse the same parent model for scope discovery
    # and parent-with-object-hidden composite renders.
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

        # Re-discover scope for this object (cheap — gaussians stays mutated).
        if obj_id == first_id and idx == 0:
            scope = scope0
            local_sv3d = local_sv3d_first
        else:
            scope, _wl, local_sv3d, _g, _p = discover_object_scope(model_path, obj_id)

        halluc_index = _resolve_halluc_index(output_root, obj_id)
        obj_out = output_root / f"obj_{obj_id}"
        (obj_out / "phase78").mkdir(parents=True, exist_ok=True)

        # ── Build comparison cameras BEFORE Phase 7 ───────────────────────
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

        # ── Phase 7: scratch-train object model ───────────────────────────
        scratch_gaussians = None
        try:
            summary = run_phase7(
                model_path=model_path,
                object_label_id=obj_id,
                halluc_index_path=str(halluc_index),
                output_dir=str(output_root),
                gaussians=gaussians,
                pipe_config=pipe_config,
                scope=scope,
                local_sv3d=local_sv3d,
                scratch_iterations=int(scratch_iterations),
                hallucination_weight=float(hallucination_weight),
                real_weight=float(real_weight),
                novel_rgb_weight=float(novel_rgb_weight),
                fov_y_deg=float(fov_y_deg),
                grid_resolution=int(grid_resolution),
                visual_hull_min_views=int(visual_hull_min_views),
            )
            scratch_gaussians = summary.pop("_scratch_gaussians", None)
        except Exception as e:
            logger.exception("Phase 7 failed for obj %d: %s", obj_id, e)
            summary = {"object_id": obj_id, "skipped_reason": f"phase7_error: {e}"}

        per_object_summaries.append(summary)

        # ── Phase 8: after renders + compare ──────────────────────────────
        if compare_cams is not None and not skip_compare and scratch_gaussians is not None:
            try:
                after_obj_frames = render_with_orbit(
                    scratch_gaussians, pipe_config, compare_cams, object_label_id=None,
                )
                after_full_frames = render_composited_scratch_with_orbit(
                    gaussians, scratch_gaussians, pipe_config, compare_cams, object_label_id=obj_id,
                )
                save_compare_grid(
                    before_obj_frames, after_obj_frames,
                    obj_out / "phase78" / "compare_object_only",
                    prefix="obj",
                )
                save_compare_grid(
                    before_full_frames, after_full_frames,
                    obj_out / "phase78" / "compare_full_scene",
                    prefix="scene",
                )
                logger.info("Phase 8: saved compare grids to %s", obj_out / "phase78")
            except Exception as e:
                logger.exception("Phase 8 compare-render failed for obj %d: %s", obj_id, e)

    # ── Final model export ────────────────────────────────────────────────
    parent_label_counts_post = label_anchor_counts(gaussians)
    logger.info("Parent model anchor counts (post): %s",
                {k: v for k, v in sorted(parent_label_counts_post.items())})

    metadata = build_reintegration_metadata(
        parent_label_counts_pre=parent_label_counts_pre,
        parent_label_counts_post=parent_label_counts_post,
        per_object_summaries=per_object_summaries,
        reference_model_path=str(model_path),
    )
    all_scratch = all(s.get("mode") == "scratch_object_training" for s in per_object_summaries)
    if all_scratch:
        final_output = save_scratch_scene_package(
            output_dir=output_root,
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

    with open(output_root / "phase678_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "model_path": str(model_path),
            "object_ids": [int(x) for x in object_ids],
            "metadata": metadata,
        }, f, indent=2)

    logger.info("\nPhase 6/7/8 complete. Output -> %s", final_output)
    return metadata


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Object-isolation Phase 6/7/8 orchestrator")
    p.add_argument("--model_path", required=True,
                   help="Path to trained ObjectGS output dir (containing point_cloud/, cameras.json).")
    p.add_argument("--output_root", default="object_isolation/outputs",
                   help="Root holding obj_<id>/phase5 inputs and where Phase 7/8 writes.")
    p.add_argument("--object_ids", type=int, nargs="+", required=True,
                   help="One or more object label IDs to train from scratch (Phase 5 must already exist).")
    p.add_argument("--scratch_iterations", type=int, default=1200,
                   help="Number of scratch ObjectGS training iterations.")
    p.add_argument("--hallucination_weight", type=float, default=1.0)
    p.add_argument("--real_weight", type=float, default=1.0)
    p.add_argument("--novel_rgb_weight", type=float, default=1.0)
    p.add_argument("--fov_y_deg", type=float, default=50.0,
                   help="Vertical FOV used for SV3D outputs (must match Phase-5 setting).")
    p.add_argument("--grid_resolution", type=int, default=25,
                   help="Visual-hull initialization grid resolution per axis.")
    p.add_argument("--visual_hull_min_views", type=int, default=10,
                   help="Minimum number of supervision masks a point must project inside to initialize a scratch anchor.")
    p.add_argument("--n_compare_views", type=int, default=8)
    p.add_argument("--skip_compare", action="store_true",
                   help="Skip before/after orbit rendering (faster for batch runs).")
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
        scratch_iterations=args.scratch_iterations,
        hallucination_weight=args.hallucination_weight,
        real_weight=args.real_weight,
        novel_rgb_weight=args.novel_rgb_weight,
        fov_y_deg=args.fov_y_deg,
        grid_resolution=args.grid_resolution,
        visual_hull_min_views=args.visual_hull_min_views,
        n_compare_views=args.n_compare_views,
        skip_compare=args.skip_compare,
    )


if __name__ == "__main__":
    main()
