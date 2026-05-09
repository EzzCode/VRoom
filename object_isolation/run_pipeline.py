"""
Complete pipeline orchestrator — end-to-end object isolation.

Runs all object isolation stages in sequence:
    Scope discovery  — Object scope discovery (automatic, embedded in extraction)
    Extraction       — Hybrid object extraction from real masks + ObjectGS renders
    Frame scoring    — Pick best conditioning view for SV3D novel-view synthesis
    Novel views      — SV3D novel-view hallucination
    Supervision      — Supervision dataset alignment
    Training         — Object model training (ObjectGS scratch)
    Comparison       — Before/after render comparison

Output layout per object::

    <output_root>/obj_<id>/
        01_extraction/
            extraction_index.json
            extracted/<seq>__<cam_id>__<img_name>.png
            masks/<seq>__<cam_id>__<img_name>_mask.png
        01_extraction_debug/
            triptych/...
            contact_sheet.png
            summary.json
        02_frame_scoring/
            scores.json
        02_frame_scoring_debug/
            bar_chart.png  scatter.png  top1.png  top_k_strip.png  summary.json
        03_novel_views/
            conditioning.png
            hallucinated/<seq>__az<DEG>.png
            objgs_refs/<seq>__az<DEG>.png
            sv3d_raw/<seq>__az<DEG>.png
            hallucination_index.json
        03_novel_views_debug/
            conditioning_panel.png  sv3d_grid.png  iou_strip.png
            coverage_overlay.png    summary.json
        04_supervision_manifest.json
        04_supervision_audit/
        04_supervision_debug/
        05_training_summary.json
        05_renders/
        06_model/
            point_cloud.ply  color_mlp.pt  cov_mlp.pt  opacity_mlp.pt
        07_scene/
        99_pipeline_summary.json

Run command::

    python -m object_isolation.run_pipeline \\
        --model_path temp_deps/ObjectGS/outputs/3dovs/2d_crossentropy_loss_01/2026-03-19_04-01-38 \\
        --scene_dir data/3dovs/bed \\
        --output_root object_isolation/outputs \\
        --object_id 8 \\
        --iterations 1200

Key optional flags::

    --debug              enable debug visualizations for all stages
    --skip_compare       skip before/after rendering
    --reuse_sv3d         skip SV3D diffusion if 03_novel_views/ outputs already exist
    --id_map_dir auto    'auto' | 'none' | explicit path to Module-1 id_maps
    --iou_threshold 0.20 minimum SV3D↔reference IoU to accept a hallucinated view
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from object_isolation.paths import OBJECT_SUMMARY_FILE

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger(__name__)


def _setup_logging(level: int = logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def run_pipeline(
    *,
    model_path: str,
    scene_dir: str,
    output_root: str,
    object_id: int,
    iterations: int = 1200,
    # Extraction params
    id_map_dir: str = "auto",
    module1_obj_id: int | None = None,
    tau_alpha: float = 0.4,
    min_pixels: int = 64,
    # Frame scoring params
    top_k: int = 5,
    # Novel views params
    iou_threshold: float = 0.20,
    fov_y_deg: float = 50.0,
    num_inference_steps: int = 25,
    safe_mode: bool = False,
    seed: int = 0,
    reuse_sv3d: bool = False,
    # Training params
    hallucination_weight: float = 1.0,
    real_weight: float = 1.0,
    novel_rgb_weight: float = 1.0,
    hallucination_rgb_scale: float = 1.0,
    depth_weight: float = 0.1,
    depth_start_iter: int = 100,
    depth_front_weight: float = 1.0,
    depth_back_weight: float = 0.15,
    colmap_init_target_points: int = 8000,
    enable_densification: bool = False,
    max_anchor_count: int = 20000,
    densify_grad_threshold: float = 0.00005,
    densify_extra_ratio: float = 0.08,
    n_compare_views: int = 8,
    skip_compare: bool = False,
    debug: bool = False,
) -> dict:
    """
    Run the complete object isolation pipeline for one object.
    
    Returns a summary dict with outputs from all stages.
    """
    import torch
    from object_isolation.core.object_scope import discover_object_scope
    from object_isolation.core.extraction import run_extraction
    from object_isolation.core.frame_scoring import run_scoring
    from object_isolation.core.hallucination import run_hallucination
    from object_isolation.core.diffusion_priors.sv3d import SV3DBackend
    from object_isolation.run_training import run as run_training

    torch.backends.cudnn.benchmark = True
    
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    
    obj_id = int(object_id)
    obj_dir = output_root / f"obj_{obj_id}"
    
    logger.info("\n" + "=" * 80)
    logger.info(f"STARTING COMPLETE PIPELINE FOR OBJECT {obj_id}")
    logger.info("=" * 80 + "\n")
    
    summary = {
        "object_id": obj_id,
        "model_path": str(model_path),
        "scene_dir": str(scene_dir),
        "output_root": str(output_root),
        "phases": {}
    }

    # ═══════════════════════════════════════════════════════════════════════════
    # Model Loading & Scope Discovery
    # ═══════════════════════════════════════════════════════════════════════════
    logger.info("Discovering scope and loading model...")
    scope, world_local, local_sv3d, gaussians, pipe_config = discover_object_scope(
        str(model_path), obj_id
    )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # Extraction
    # ═══════════════════════════════════════════════════════════════════════════
    logger.info("\n" + "─" * 80)
    logger.info("EXTRACTION: Hybrid Object Extraction")
    logger.info("─" * 80)
    
    try:
        scene_p = Path(scene_dir)
        images_dir = scene_p / "images"
        
        resolved_id_map_dir = None
        if id_map_dir == "auto":
            from object_isolation.core.extraction import auto_resolve_module1_id
            candidates = [
                scene_p / "tracked" / "id_maps",
                scene_p / "semantic_instance",
                scene_p / "object_mask",
            ]
            for c in candidates:
                if c.exists() and any(c.iterdir()):
                    resolved_id_map_dir = c
                    break
        elif id_map_dir.lower() not in ("none", "null", ""):
            resolved_id_map_dir = Path(id_map_dir)

        extraction_summary = run_extraction(
            scope=scope,
            gaussians=gaussians,
            pipe_config=pipe_config,
            images_dir=images_dir,
            id_map_dir=resolved_id_map_dir,
            module1_obj_id=module1_obj_id,
            output_dir=obj_dir / "01_extraction",
            tau_alpha=tau_alpha,
            min_pixels=min_pixels,
            auto_resolve=True,
        )
        summary["phases"]["extraction"] = extraction_summary
        
        if debug:
            from object_isolation.debug.debug_extraction import generate_debug_artifacts
            generate_debug_artifacts(
                manifest=extraction_summary,
                scope=scope,
                gaussians=gaussians,
                pipe_config=pipe_config,
                images_dir=images_dir,
                id_map_dir=resolved_id_map_dir,
                debug_dir=obj_dir / "01_extraction_debug",
                tau_alpha=tau_alpha,
            )

        logger.info(f"✓ Extraction complete: {extraction_summary['n_extracted']} frames extracted")
    except Exception as e:
        logger.exception(f"✗ Extraction failed: {e}")
        summary["phases"]["extraction"] = {"error": str(e)}
        return summary
    
    # ═══════════════════════════════════════════════════════════════════════════
    # Frame Scoring
    # ═══════════════════════════════════════════════════════════════════════════
    logger.info("\n" + "─" * 80)
    logger.info("FRAME SCORING: Pick SV3D conditioning view")
    logger.info("─" * 80)
    
    try:
        frame_scoring_summary = run_scoring(
            extraction_index_path=obj_dir / "01_extraction" / "extraction_index.json",
            scope_cameras=scope.cameras,
            output_dir=obj_dir / "02_frame_scoring",
            top_k=top_k,
        )
        if not frame_scoring_summary.get("top1") and frame_scoring_summary.get("top_k"):
            frame_scoring_summary["top1"] = frame_scoring_summary["top_k"][0]
        frame_scoring_summary["scores_json"] = str(obj_dir / "02_frame_scoring" / "scores.json")
        summary["phases"]["frame_scoring"] = frame_scoring_summary
        
        if debug:
            from object_isolation.debug.debug_frame_scoring import generate_debug_artifacts
            generate_debug_artifacts(frame_scoring_summary, obj_dir / "02_frame_scoring_debug", top_k)

        if frame_scoring_summary.get("top1"):
            logger.info(f"✓ Frame scoring complete: Best frame score = {frame_scoring_summary['top1'].get('score', 0.0):.3f}")
            logger.info(f"  Conditioning view: cam={frame_scoring_summary['top1'].get('cam_index', 'N/A')}, "
                       f"az={frame_scoring_summary['top1'].get('azimuth_V_deg', 0.0):.1f}°")
    except Exception as e:
        logger.exception(f"✗ Frame scoring failed: {e}")
        summary["phases"]["frame_scoring"] = {"error": str(e)}
        return summary
    
    # ═══════════════════════════════════════════════════════════════════════════
    # Novel Views
    # ═══════════════════════════════════════════════════════════════════════════
    logger.info("\n" + "─" * 80)
    logger.info("NOVEL VIEWS: SV3D Hallucination")
    logger.info("─" * 80)
    
    try:
        backend = SV3DBackend(num_inference_steps=num_inference_steps, safe_mode=safe_mode)
        novel_views_summary = run_hallucination(
            scope=scope,
            local_sv3d=local_sv3d,
            gaussians=gaussians,
            pipe_config=pipe_config,
            scores_json_path=obj_dir / "02_frame_scoring" / "scores.json",
            output_dir=obj_dir / "03_novel_views",
            object_label_id=obj_id,
            backend=backend,
            iou_threshold=iou_threshold,
            fov_y_deg=fov_y_deg,
            seed=seed,
            reuse_sv3d=reuse_sv3d,
        )
        summary["phases"]["novel_views"] = novel_views_summary
        
        if debug:
            from object_isolation.debug.debug_novel_views import generate_debug_artifacts
            generate_debug_artifacts(novel_views_summary, scope.cameras, obj_dir / "03_novel_views_debug")

        logger.info(f"✓ Novel views complete: {novel_views_summary['n_kept']}/{novel_views_summary['n_views']} views kept")
    except Exception as e:
        logger.exception(f"✗ Novel views failed: {e}")
        summary["phases"]["novel_views"] = {"error": str(e)}
        return summary
    
    # ═══════════════════════════════════════════════════════════════════════════
    # Training
    # ═══════════════════════════════════════════════════════════════════════════
    logger.info("\n" + "─" * 80)
    logger.info("TRAINING: Supervision + ObjectGS + Comparison")
    logger.info("─" * 80)
    
    try:
        training_result = run_training(
            model_path=str(model_path),
            output_root=output_root,
            object_ids=[obj_id],
            iterations=iterations,
            hallucination_weight=hallucination_weight,
            real_weight=real_weight,
            novel_rgb_weight=novel_rgb_weight,
            hallucination_rgb_scale=hallucination_rgb_scale,
            depth_weight=depth_weight,
            depth_start_iter=depth_start_iter,
            depth_front_weight=depth_front_weight,
            depth_back_weight=depth_back_weight,
            fov_y_deg=fov_y_deg,
            colmap_init_target_points=colmap_init_target_points,
            enable_densification=enable_densification,
            max_anchor_count=max_anchor_count,
            densify_grad_threshold=densify_grad_threshold,
            densify_extra_ratio=densify_extra_ratio,
            n_compare_views=n_compare_views,
            skip_compare=skip_compare,
            debug=debug,
            preloaded_data=(scope, world_local, local_sv3d, gaussians, pipe_config),
        )
        summary["phases"]["training"] = training_result
        logger.info("✓ Training complete")
        logger.info(f"  Batch summary: {output_root / '99_batch_summary.json'}")
    except Exception as e:
        logger.exception(f"✗ Training failed: {e}")
        summary["phases"]["training"] = {"error": str(e)}
        return summary
    
    # ═══════════════════════════════════════════════════════════════════════════
    # Final Summary
    # ═══════════════════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 80)
    logger.info("COMPLETE PIPELINE FINISHED")
    logger.info("=" * 80)
    logger.info(f"Object {obj_id}:")
    logger.info(f"  Extraction: {summary['phases']['extraction'].get('n_extracted', 0)} frames extracted")
    logger.info(f"  Frame scoring: Best conditioning score = {summary['phases']['frame_scoring'].get('top1', {}).get('score', 0.0):.3f}")
    logger.info(f"  Novel views: {summary['phases']['novel_views'].get('n_kept', 0)} hallucinated views")
    logger.info(f"  Training: complete with {iterations} iterations")
    logger.info(f"\nAll outputs saved to: {obj_dir}")
    logger.info("=" * 80 + "\n")
    
    # Save summary
    summary_path = obj_dir / OBJECT_SUMMARY_FILE
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Complete summary saved to: {summary_path}")
    
    return summary


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Complete object isolation pipeline — end-to-end",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Required arguments
    p.add_argument("--model_path", required=True,
                   help="Path to trained ObjectGS output dir (containing point_cloud/, cameras.json)")
    p.add_argument("--scene_dir", required=True,
                   help="Scene directory containing images/ and optionally tracked/id_maps/ or semantic_instance/")
    p.add_argument("--object_id", required=True, type=int,
                   help="Object label ID to process")
    
    # Optional arguments
    p.add_argument("--output_root", default="object_isolation/outputs",
                   help="Root directory for all outputs")
    p.add_argument("--iterations", type=int, default=1200,
                   help="Number of training iterations")
    
    # Extraction params
    p.add_argument("--id_map_dir", default="auto",
                   help="'auto' (default) | 'none' | explicit path to Module-1 id_maps")
    p.add_argument("--module1_obj_id", type=int, default=None,
                   help="Override Module-1 instance ID; auto-resolved if omitted")
    p.add_argument("--tau_alpha", type=float, default=0.4,
                   help="ObjectGS alpha threshold for mask extraction")
    p.add_argument("--min_pixels", type=int, default=64,
                   help="Minimum object pixels to accept a frame")
    
    # Frame scoring params
    p.add_argument("--top_k", type=int, default=5,
                   help="Number of top frames to save during scoring")
    
    # Novel views params
    p.add_argument("--iou_threshold", type=float, default=0.20,
                   help="Minimum IoU between SV3D and reference to keep a hallucinated view")
    p.add_argument("--fov_y_deg", type=float, default=50.0,
                   help="Vertical FOV for SV3D outputs")
    p.add_argument("--num_inference_steps", type=int, default=25,
                   help="SV3D diffusion steps")
    p.add_argument("--safe_mode", action="store_true",
                   help="Reduce SV3D resolution/frames for low VRAM")
    p.add_argument("--seed", type=int, default=0,
                   help="Random seed for SV3D")
    p.add_argument("--reuse_sv3d", action="store_true",
                   help="Skip SV3D diffusion if 03_novel_views/ already exists")
    
    # Training params
    p.add_argument("--hallucination_weight", type=float, default=1.0,
                   help="Loss weight for hallucinated views")
    p.add_argument("--real_weight", type=float, default=1.0,
                   help="Loss weight for real extracted views")
    p.add_argument("--novel_rgb_weight", type=float, default=1.0,
                   help="RGB loss weight for novel views")
    p.add_argument("--hallucination_rgb_scale", type=float, default=1.0,
                   help="Extra RGB loss scale for hallucinated views")
    p.add_argument("--depth_weight", type=float, default=0.1,
                   help="Depth regularization weight for real views")
    p.add_argument("--depth_start_iter", type=int, default=100,
                   help="Iteration to start depth regularization")
    p.add_argument("--depth_front_weight", type=float, default=1.0,
                   help="Penalty when object renders in front of depth")
    p.add_argument("--depth_back_weight", type=float, default=0.15,
                   help="Penalty when object renders behind depth")
    p.add_argument("--colmap_init_target_points", type=int, default=8000,
                   help="Target number of COLMAP seed points")
    p.add_argument("--enable_densification", action="store_true",
                   help="Enable ObjectGS densification")
    p.add_argument("--max_anchor_count", type=int, default=20000,
                   help="Maximum anchor count for densification")
    p.add_argument("--densify_grad_threshold", type=float, default=0.00005)
    p.add_argument("--densify_extra_ratio", type=float, default=0.08)
    p.add_argument("--n_compare_views", type=int, default=8,
                   help="Number of orbit views for comparison")
    p.add_argument("--skip_compare", action="store_true",
                   help="Skip before/after rendering")
    
    # General flags
    p.add_argument("--debug", action="store_true",
                   help="Enable debug visualizations for all stages")
    p.add_argument("--log_level", default="INFO",
                   help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    
    return p.parse_args()


def main():
    args = _parse_args()
    _setup_logging(getattr(logging, str(args.log_level).upper(), logging.INFO))
    
    run_pipeline(
        model_path=args.model_path,
        scene_dir=args.scene_dir,
        output_root=args.output_root,
        object_id=args.object_id,
        iterations=args.iterations,
        # Extraction
        id_map_dir=args.id_map_dir,
        module1_obj_id=args.module1_obj_id,
        tau_alpha=args.tau_alpha,
        min_pixels=args.min_pixels,
        # Frame scoring
        top_k=args.top_k,
        # Novel views
        iou_threshold=args.iou_threshold,
        fov_y_deg=args.fov_y_deg,
        num_inference_steps=args.num_inference_steps,
        safe_mode=args.safe_mode,
        seed=args.seed,
        reuse_sv3d=args.reuse_sv3d,
        # Training
        hallucination_weight=args.hallucination_weight,
        real_weight=args.real_weight,
        novel_rgb_weight=args.novel_rgb_weight,
        hallucination_rgb_scale=args.hallucination_rgb_scale,
        depth_weight=args.depth_weight,
        depth_start_iter=args.depth_start_iter,
        depth_front_weight=args.depth_front_weight,
        depth_back_weight=args.depth_back_weight,
        colmap_init_target_points=args.colmap_init_target_points,
        enable_densification=args.enable_densification,
        max_anchor_count=args.max_anchor_count,
        densify_grad_threshold=args.densify_grad_threshold,
        densify_extra_ratio=args.densify_extra_ratio,
        n_compare_views=args.n_compare_views,
        skip_compare=args.skip_compare,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
