"""End-to-end orchestrator for Phases 6 → 7 → 8 of the object-isolation pipeline.

Assumes Phase 0–5 have already been run (i.e. ``object_isolation/outputs/
obj_<id>/phase5/hallucination_index.json`` exists).

Per object:
    Phase 6 — build supervision_views from Phase-5 SV3D outputs.
    Phase 7 — anchor seeding + optimizer fine-tune (mutates parent gaussians).
    Phase 8 — render before/after comparison + save merged final model.

Usage::

    python -m object_isolation.run_phase678 \
        --model_path temp_deps/ObjectGS/outputs/replica/.../office_0/.../ \
        --output_root object_isolation/outputs \
        --object_ids 8 9 \
        --finetune_iterations 1200 \
        --hallucination_weight 0.10
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
    save_final_model, build_reintegration_metadata, label_anchor_counts,
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
    finetune_iterations: int = 1200,
    hallucination_weight: float = 0.10,
    novel_rgb_weight: float = 1.0,
    fov_y_deg: float = 50.0,
    grid_resolution: int = 25,
    n_compare_views: int = 8,
    skip_compare: bool = False,
    freeze_originals: bool = True,
) -> dict:
    """Run Phases 6/7/8 for every requested object_id and save final model."""
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # ── Load parent model once via Phase-1 discovery for the first object ─
    # All subsequent objects reuse the same gaussians/pipe_config so seeding
    # accumulates into the SAME tensor (cross-object reintegration is implicit).
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

        # ── Build comparison cameras BEFORE Phase 7 mutates the gaussians ─
        compare_cams = None
        before_obj_frames = before_full_frames = None
        if not skip_compare:
            ref_cam_pos = scope.cam_centers_visible_W.mean(axis=0).astype(np.float32)
            compare_cams = build_orbit_cameras(
                center=np.asarray(scope.centroid_W, dtype=np.float32),
                radius=float(scope.radius),
                orbit_radius=float(scope.radius * 2.5),
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

        # ── Phase 7: seed + optimize (mutates parent gaussians in-place) ──
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
                finetune_iterations=int(finetune_iterations),
                hallucination_weight=float(hallucination_weight),
                novel_rgb_weight=float(novel_rgb_weight),
                fov_y_deg=float(fov_y_deg),
                grid_resolution=int(grid_resolution),
                freeze_originals=bool(freeze_originals),
            )
        except Exception as e:
            logger.exception("Phase 7 failed for obj %d: %s", obj_id, e)
            summary = {"object_id": obj_id, "skipped_reason": f"phase7_error: {e}"}

        per_object_summaries.append(summary)

        # ── Phase 8: after renders + compare ──────────────────────────────
        if compare_cams is not None and not skip_compare:
            try:
                after_obj_frames = render_with_orbit(
                    gaussians, pipe_config, compare_cams, object_label_id=obj_id,
                )
                after_full_frames = render_with_orbit(
                    gaussians, pipe_config, compare_cams, object_label_id=None,
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
    save_final_model(
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

    logger.info("\nPhase 6/7/8 complete. Final model -> %s", output_root / "final_model")
    return metadata


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Object-isolation Phase 6/7/8 orchestrator")
    p.add_argument("--model_path", required=True,
                   help="Path to trained ObjectGS output dir (containing point_cloud/, cameras.json).")
    p.add_argument("--output_root", default="object_isolation/outputs",
                   help="Root holding obj_<id>/phase5 inputs and where Phase 7/8 writes.")
    p.add_argument("--object_ids", type=int, nargs="+", required=True,
                   help="One or more object label IDs to replenish (Phase 5 must already exist).")
    p.add_argument("--finetune_iterations", type=int, default=1200)
    p.add_argument("--hallucination_weight", type=float, default=0.10)
    p.add_argument("--novel_rgb_weight", type=float, default=1.0)
    p.add_argument("--fov_y_deg", type=float, default=50.0,
                   help="Vertical FOV used for SV3D outputs (must match Phase-5 setting).")
    p.add_argument("--grid_resolution", type=int, default=25,
                   help="Backside seeding grid resolution per axis.")
    p.add_argument("--n_compare_views", type=int, default=8)
    p.add_argument("--skip_compare", action="store_true",
                   help="Skip before/after orbit rendering (faster for batch runs).")
    p.add_argument("--no_freeze_originals", action="store_true",
                   help="Allow optimizer to update the original-object anchors too.")
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
        finetune_iterations=args.finetune_iterations,
        hallucination_weight=args.hallucination_weight,
        novel_rgb_weight=args.novel_rgb_weight,
        fov_y_deg=args.fov_y_deg,
        grid_resolution=args.grid_resolution,
        n_compare_views=args.n_compare_views,
        skip_compare=args.skip_compare,
        freeze_originals=not args.no_freeze_originals,
    )


if __name__ == "__main__":
    main()
