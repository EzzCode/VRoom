"""ModuleTBD end-to-end pipeline CLI.

Drop-in replacement for ``object_isolation.run_pipeline`` using the
rewritten ModuleTBD modules.

Usage::

    python -m ModuleTBD \\
        --model_path temp_deps/ObjectGS/outputs/3dovs/2d_crossentropy_loss_01/2026-03-19_04-01-38 \\
        --scene_dir  data/3dovs/bed \\
        --output_root object_isolation/outputs \\
        --object_id  8 \\
        --iterations 12000 \\
        --debug \\
        --enable_densification \\
        --generated_weight 0.6

By default ``--ply_path`` is resolved to ``<model_path>/point_cloud/point_cloud.ply``
(the standard ObjectGS checkpoint layout). Pass it explicitly if your layout differs.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ── Logging ───────────────────────────────────────────────────────────────────

def _setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run(
    model_path,
    scene_dir,
    output_root,
    object_id,
    ply_path=None,
    iterations=1200,
    # Extraction
    tau_alpha=0.4,
    tracked_id_map_dir="auto",
    tracked_object_id=None,
    # Frame scoring
    top_k=5,
    # Novel views
    reuse_sv3d=False,
    # Training
    generated_weight=1.0,
    real_weight=1.0,
    rgb_weight=1.0,
    generated_rgb_scale=1.0,
    depth_weight=0.1,
    depth_start_iter=100,
    depth_front_weight=1.0,
    depth_back_weight=0.15,
    colmap_init_target_points=8000,
    enable_densification=False,
    max_anchor_count=20000,
    densify_grad_threshold=0.00005,
    debug=False,
):
    """Run extraction → frame scoring → hallucination → training for one object."""
    import torch
    from ModuleTBD.config import ObjectTrainingConfig
    from ModuleTBD.utils.scene_analysis import compute_object_scope, load_gaussians
    from ModuleTBD.utils.transforms import ObjectFrame
    from ModuleTBD.view_selection import run_extraction, run_scoring
    from ModuleTBD.view_generation import run_generation
    from ModuleTBD.pipeline import run_pipeline

    torch.backends.cudnn.benchmark = True

    model_path = Path(model_path)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    obj_id = int(object_id)
    obj_dir = output_root / f"obj_{obj_id}"

    config = ObjectTrainingConfig(
        iterations=iterations,
        generated_weight=generated_weight,
        real_weight=real_weight,
        rgb_weight=rgb_weight,
        generated_rgb_scale=generated_rgb_scale,
        depth_weight=depth_weight,
        depth_start_iter=depth_start_iter,
        depth_front_weight=depth_front_weight,
        depth_back_weight=depth_back_weight,
        colmap_init_target_points=colmap_init_target_points,
        enable_densification=enable_densification,
        max_anchor_count=max_anchor_count,
        densify_grad_threshold=densify_grad_threshold,
    )

    # Resolve PLY path — prefer latest iteration_* checkpoint (has label_ids);
    # fall back to flat point_cloud/point_cloud.ply if no iterations exist.
    if ply_path:
        resolved_ply = Path(ply_path)
    else:
        pc_base = model_path / "point_cloud"
        iter_dirs = sorted(
            [d for d in pc_base.iterdir() if d.is_dir() and d.name.startswith("iteration_")],
            key=lambda d: int(d.name.split("_")[-1]),
        ) if pc_base.exists() else []
        resolved_ply = (iter_dirs[-1] / "point_cloud.ply") if iter_dirs else (pc_base / "point_cloud.ply")

    logger.info("\n" + "=" * 80)
    logger.info("STARTING COMPLETE PIPELINE FOR OBJECT %d", obj_id)
    logger.info("=" * 80 + "\n")

    summary: dict[str, Any] = {
        "object_id": obj_id,
        "model_path": str(model_path),
        "ply_path": str(resolved_ply),
        "scene_dir": str(scene_dir),
        "output_root": str(output_root),
        "phases": {},
    }

    # ── Scope & model ─────────────────────────────────────────────────────────
    logger.info("Computing scope and loading model …")
    scope, frame, pipe_config = compute_object_scope(
        str(model_path), obj_id, ply_path=str(resolved_ply)
    )
    gaussians, _ = load_gaussians(str(model_path), ply_path=str(resolved_ply))

    # ── Extraction ───────────────────────────────────────────────────────────
    logger.info("\n" + "─" * 80)
    logger.info("EXTRACTION: Object Extraction")
    logger.info("─" * 80)

    try:
        scene_p = Path(scene_dir)
        images_dir = scene_p / "images"
        resolved_tracked_id_map_dir = None
        if str(tracked_id_map_dir).lower() == "auto":
            for candidate in (
                scene_p / "tracked" / "id_maps",
                scene_p / "semantic_instance",
                scene_p / "object_mask",
                scene_p / "object_mask_deva",
            ):
                if candidate.exists() and any(candidate.iterdir()):
                    resolved_tracked_id_map_dir = candidate
                    break
        elif str(tracked_id_map_dir).lower() not in ("none", "null", ""):
            resolved_tracked_id_map_dir = Path(tracked_id_map_dir)

        if resolved_tracked_id_map_dir is None:
            raise RuntimeError(
                "No tracked id-map directory found. Pass --tracked_id_map_dir explicitly "
                "or ensure one of the auto-discovery paths (tracked/id_maps, "
                "semantic_instance, object_mask, object_mask_deva) exists and is non-empty "
                "under the scene directory."
            )

        # TODO(label-alignment): replace this entire block with `tracked_object_id = object_label_id`
        #   once Module1 and ObjectGS share the same label namespace; delete vote_tracked_object_id() too.
        if tracked_object_id is None:
            from ModuleTBD.utils.helpers import vote_tracked_object_id
            tracked_object_id = vote_tracked_object_id(
                scope, gaussians, pipe_config, resolved_tracked_id_map_dir, tau_alpha=tau_alpha
            )
            if tracked_object_id is None:
                raise RuntimeError(
                    "vote_tracked_object_id could not match any tracked id-map label to the GS "
                    "model silhouette. Pass --tracked_object_id explicitly."
                )

        extraction_manifest = run_extraction(
            scope=scope,
            images_dir=images_dir,
            output_dir=obj_dir / "01_extraction",
            tracked_id_map_dir=resolved_tracked_id_map_dir,
            tracked_object_id=tracked_object_id,
        )
        summary["phases"]["extraction"] = extraction_manifest
        logger.info("✓ Extraction: %d frames extracted", len(extraction_manifest.get("frames", [])))
    except Exception as exc:
        logger.exception("✗ Extraction failed: %s", exc)
        summary["phases"]["extraction"] = {"error": str(exc)}
        return summary

    # ── Frame scoring ─────────────────────────────────────────────────────────
    logger.info("\n" + "─" * 80)
    logger.info("FRAME SCORING: Pick SV3D conditioning view")
    logger.info("─" * 80)

    scores_manifest = None

    try:
        scores_manifest = run_scoring(
            extraction_index_path=obj_dir / "01_extraction" / "extraction_index.json",
            top_k=top_k,
        )
        summary["phases"]["frame_scoring"] = scores_manifest
        top1 = scores_manifest.get("top_k", [{}])[0] if scores_manifest.get("top_k") else {}
        logger.info("✓ Frame scoring: best score = %.3f  cam=%s  az=%.1f°",
                    top1.get("score", 0.0), top1.get("cam_index", "?"),
                    top1.get("azimuth_deg", 0.0))
    except Exception as exc:
        logger.exception("✗ Frame scoring failed: %s", exc)
        summary["phases"]["frame_scoring"] = {"error": str(exc)}
        return summary

    # ── Novel-view hallucination ──────────────────────────────────────────────
    logger.info("\n" + "─" * 80)
    logger.info("NOVEL VIEWS: SV3D Hallucination")
    logger.info("─" * 80)

    try:
        halluc_manifest = run_generation(
            scope=scope,
            frame=frame,
            gaussians=gaussians,
            pipeline_config=pipe_config,
            scores=scores_manifest,
            output_dir=obj_dir / "03_novel_views",
            reuse_sv3d=reuse_sv3d,
        )
        summary["phases"]["novel_views"] = halluc_manifest
        logger.info("✓ Novel views: %d/%d kept",
                    halluc_manifest.get("n_kept", 0), halluc_manifest.get("n_views", 0))
    except Exception as exc:
        logger.exception("✗ Novel views failed: %s", exc)
        summary["phases"]["novel_views"] = {"error": str(exc)}
        return summary

    # ── Training ──────────────────────────────────────────────────────────────
    logger.info("\n" + "─" * 80)
    logger.info("TRAINING: Supervision + ObjectGS scratch")
    logger.info("─" * 80)

    try:
        training_result = run_pipeline(
            model_path=str(model_path),
            object_id=obj_id,
            generation_path=obj_dir / "03_novel_views" / "generation.json",
            output_dir=output_root,
            halluc_manifest=halluc_manifest,
            gaussians=gaussians,
            pipe_config=pipe_config,
            scope=scope,
            frame=frame,
            config=config,
        )
        training_result_clean = {k: v for k, v in training_result.items() if k != "_gaussians"}
        summary["phases"]["training"] = training_result_clean
        logger.info("✓ Training complete")
    except Exception as exc:
        logger.exception("✗ Training failed: %s", exc)
        summary["phases"]["training"] = {"error": str(exc)}
        return summary

    if debug:
        try:
            from ModuleTBD.debug import generate_all_debug_artifacts
            generate_all_debug_artifacts(
                obj_dir=obj_dir,
                scope=scope,
                frame=frame,
                gaussians=gaussians,
                trained_gaussians=training_result.get("_gaussians"),
                pipe_config=pipe_config,
                images_dir=images_dir,
                extraction_manifest=extraction_manifest,
                scores_manifest=scores_manifest,
                halluc_manifest=halluc_manifest,
                model_path=str(model_path),
                object_id=obj_id,
            )
            logger.info("✓ Debug artifacts written under %s", obj_dir)
        except Exception as exc:
            logger.warning("Debug artifacts failed (non-fatal): %s", exc)

    # ── Save summary ──────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 80)
    logger.info("PIPELINE COMPLETE — object %d", obj_id)
    logger.info("  extraction : %d frames", len(summary["phases"].get("extraction", {}).get("frames", [])))
    logger.info("  novel views: %d kept", summary["phases"].get("novel_views", {}).get("n_kept", 0))
    logger.info("  output dir : %s", obj_dir)
    logger.info("=" * 80 + "\n")

    summary_path = obj_dir / "99_pipeline_summary.json"
    if debug:
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("Summary saved to %s", summary_path)
    return summary


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="ModuleTBD — end-to-end object isolation pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Required
    p.add_argument("--model_path", required=True,
                   help="Trained ObjectGS run dir (contains cameras.json, config.yaml)")
    p.add_argument("--scene_dir", required=True,
                   help="Scene dir with images/ subdir")
    p.add_argument("--object_id", required=True, type=int)

    # Paths
    p.add_argument("--output_root", default="object_isolation/outputs")
    p.add_argument("--ply_path", default=None,
                   help="Explicit path to point_cloud.ply "
                        "(default: <model_path>/point_cloud/point_cloud.ply)")

    # Training length
    p.add_argument("--iterations", type=int, default=1200)

    # Extraction
    p.add_argument("--tau_alpha", type=float, default=0.4)
    p.add_argument("--tracked_id_map_dir", default="auto",
                   help="'auto', 'none', or path to the per-frame tracked id-map directory "
                        "(Module1 object_tracker output, e.g. <scene>/tracked/id_maps)")
    p.add_argument("--tracked_object_id", type=int, default=None,
                   help="Instance label from the tracked id-maps (same integer label that "
                        "vote.py writes into the labeled COLMAP PLY); auto-voted if omitted")

    # Frame scoring
    p.add_argument("--top_k", type=int, default=5)

    # Novel views
    p.add_argument("--reuse_sv3d", action="store_true")

    # Training
    p.add_argument("--generated_weight", type=float, default=1.0)
    p.add_argument("--real_weight", type=float, default=1.0)
    p.add_argument("--rgb_weight", type=float, default=1.0)
    p.add_argument("--generated_rgb_scale", type=float, default=1.0)
    p.add_argument("--depth_weight", type=float, default=0.1)
    p.add_argument("--depth_start_iter", type=int, default=100)
    p.add_argument("--depth_front_weight", type=float, default=1.0)
    p.add_argument("--depth_back_weight", type=float, default=0.15)
    p.add_argument("--colmap_init_target_points", type=int, default=8000)
    p.add_argument("--enable_densification", action="store_true")
    p.add_argument("--max_anchor_count", type=int, default=20000)
    p.add_argument("--densify_grad_threshold", type=float, default=0.00005)

    # General
    p.add_argument("--debug", action="store_true")
    p.add_argument("--log_level", default="INFO")

    return p.parse_args()


def main():
    args = _parse_args()
    _setup_logging(getattr(logging, args.log_level.upper(), logging.INFO))
    run(
        model_path=args.model_path,
        scene_dir=args.scene_dir,
        output_root=args.output_root,
        object_id=args.object_id,
        ply_path=args.ply_path,
        tracked_id_map_dir=args.tracked_id_map_dir,
        tracked_object_id=args.tracked_object_id,
        iterations=args.iterations,
        tau_alpha=args.tau_alpha,
        top_k=args.top_k,
        reuse_sv3d=args.reuse_sv3d,
        generated_weight=args.generated_weight,
        real_weight=args.real_weight,
        rgb_weight=args.rgb_weight,
        generated_rgb_scale=args.generated_rgb_scale,
        depth_weight=args.depth_weight,
        depth_start_iter=args.depth_start_iter,
        depth_front_weight=args.depth_front_weight,
        depth_back_weight=args.depth_back_weight,
        colmap_init_target_points=args.colmap_init_target_points,
        enable_densification=args.enable_densification,
        max_anchor_count=args.max_anchor_count,
        densify_grad_threshold=args.densify_grad_threshold,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
