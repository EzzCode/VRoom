"""
Complete pipeline orchestrator — Phases 1-8 end-to-end.

Runs all object isolation phases in sequence:
    Phase 1-2 — Object scope discovery (automatic)
    Phase 3   — Hybrid object extraction
    Phase 4   — Frame scoring
    Phase 5   — SV3D hallucination
    Phase 6   — Supervision alignment
    Phase 7   — Object training
    Phase 8   — Before/after comparison

Usage::

    python -m object_isolation.run_all_phases \
        --model_path temp_deps/ObjectGS/outputs/3dovs/.../2026-03-19_04-01-38 \
        --scene_dir data/office_0 \
        --output_root object_isolation/outputs \
        --object_id 8 \
        --iterations 1200

Optional flags:
    --debug              — Enable debug visualizations for all phases
    --skip_compare       — Skip Phase 8 before/after rendering
    --reuse_sv3d         — Skip SV3D diffusion if Phase 5 already exists
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

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


def run_all_phases(
    *,
    model_path: str,
    scene_dir: str,
    output_root: str,
    object_id: int,
    iterations: int = 1200,
    # Phase 3 params
    id_map_dir: str = "auto",
    module1_obj_id: int | None = None,
    tau_alpha: float = 0.4,
    min_pixels: int = 64,
    # Phase 4 params
    top_k: int = 5,
    # Phase 5 params
    iou_threshold: float = 0.20,
    fov_y_deg: float = 50.0,
    num_inference_steps: int = 25,
    safe_mode: bool = False,
    seed: int = 0,
    reuse_sv3d: bool = False,
    # Phase 6-8 params
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
    
    Returns a summary dict with outputs from all phases.
    """
    import torch
    from object_isolation.debug.visualize_phase03 import run_debug as run_phase3
    from object_isolation.debug.visualize_phase04 import run_debug as run_phase4
    from object_isolation.debug.visualize_phase05 import run_debug as run_phase5
    from object_isolation.run_phases import run as run_phase678

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
    # Phase 3: Extraction
    # ═══════════════════════════════════════════════════════════════════════════
    logger.info("\n" + "─" * 80)
    logger.info("PHASE 3: Hybrid Object Extraction")
    logger.info("─" * 80)
    
    try:
        phase3_summary = run_phase3(
            model_path=str(model_path),
            object_id=obj_id,
            scene_dir=str(scene_dir),
            output_root=str(output_root),
            id_map_dir=id_map_dir,
            module1_obj_id=module1_obj_id,
            tau_alpha=tau_alpha,
            min_pixels=min_pixels,
        )
        summary["phases"]["phase3"] = phase3_summary
        logger.info(f"✓ Phase 3 complete: {phase3_summary['n_extracted']} frames extracted")
        logger.info(f"  Output: {phase3_summary['manifest_path']}")
    except Exception as e:
        logger.exception(f"✗ Phase 3 failed: {e}")
        summary["phases"]["phase3"] = {"error": str(e)}
        return summary
    
    # ═══════════════════════════════════════════════════════════════════════════
    # Phase 4: Frame Scoring
    # ═══════════════════════════════════════════════════════════════════════════
    logger.info("\n" + "─" * 80)
    logger.info("PHASE 4: Frame Scoring")
    logger.info("─" * 80)
    
    try:
        phase4_summary = run_phase4(
            model_path=str(model_path),
            object_id=obj_id,
            output_root=str(output_root),
            top_k=top_k,
        )
        summary["phases"]["phase4"] = phase4_summary
        if phase4_summary.get("top1"):
            logger.info(f"✓ Phase 4 complete: Best frame score = {phase4_summary['top1'].get('score', 0.0):.3f}")
            logger.info(f"  Conditioning view: cam={phase4_summary['top1'].get('cam_index', 'N/A')}, "
                       f"az={phase4_summary['top1'].get('azimuth_V_deg', 0.0):.1f}°")
        logger.info(f"  Output: {phase4_summary.get('scores_json', 'N/A')}")
    except Exception as e:
        logger.exception(f"✗ Phase 4 failed: {e}")
        summary["phases"]["phase4"] = {"error": str(e)}
        return summary
    
    # ═══════════════════════════════════════════════════════════════════════════
    # Phase 5: Hallucination (SV3D)
    # ═══════════════════════════════════════════════════════════════════════════
    logger.info("\n" + "─" * 80)
    logger.info("PHASE 5: SV3D Hallucination")
    logger.info("─" * 80)
    
    try:
        phase5_summary = run_phase5(
            model_path=str(model_path),
            object_id=obj_id,
            output_root=str(output_root),
            iou_threshold=iou_threshold,
            fov_y_deg=fov_y_deg,
            num_inference_steps=num_inference_steps,
            safe_mode=safe_mode,
            seed=seed,
            reuse_sv3d=reuse_sv3d,
        )
        summary["phases"]["phase5"] = phase5_summary
        logger.info(f"✓ Phase 5 complete: {phase5_summary['n_kept']}/{phase5_summary['n_views']} views kept")
        logger.info(f"  Output: {phase5_summary['manifest_path']}")
    except Exception as e:
        logger.exception(f"✗ Phase 5 failed: {e}")
        summary["phases"]["phase5"] = {"error": str(e)}
        return summary
    
    # ═══════════════════════════════════════════════════════════════════════════
    # Phase 6-8: Training and Comparison
    # ═══════════════════════════════════════════════════════════════════════════
    logger.info("\n" + "─" * 80)
    logger.info("PHASE 6-8: Training and Comparison")
    logger.info("─" * 80)
    
    try:
        phase678_result = run_phase678(
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
        )
        summary["phases"]["phase678"] = phase678_result
        logger.info("✓ Phase 6-8 complete")
        logger.info(f"  Pipeline summary: {output_root / 'pipeline_summary.json'}")
    except Exception as e:
        logger.exception(f"✗ Phase 6-8 failed: {e}")
        summary["phases"]["phase678"] = {"error": str(e)}
        return summary
    
    # ═══════════════════════════════════════════════════════════════════════════
    # Final Summary
    # ═══════════════════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 80)
    logger.info("COMPLETE PIPELINE FINISHED")
    logger.info("=" * 80)
    logger.info(f"Object {obj_id}:")
    logger.info(f"  Phase 3: {summary['phases']['phase3'].get('n_extracted', 0)} frames extracted")
    logger.info(f"  Phase 4: Best conditioning score = {summary['phases']['phase4'].get('top1', {}).get('score', 0.0):.3f}")
    logger.info(f"  Phase 5: {summary['phases']['phase5'].get('n_kept', 0)} hallucinated views")
    logger.info(f"  Phase 7: Training complete with {iterations} iterations")
    logger.info(f"\nAll outputs saved to: {obj_dir}")
    logger.info("=" * 80 + "\n")
    
    # Save summary
    summary_path = obj_dir / "complete_pipeline_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Complete summary saved to: {summary_path}")
    
    return summary


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Complete object isolation pipeline — Phases 1-8",
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
                   help="Number of training iterations (Phase 7)")
    
    # Phase 3 params
    p.add_argument("--id_map_dir", default="auto",
                   help="'auto' (default) | 'none' | explicit path to Module-1 id_maps")
    p.add_argument("--module1_obj_id", type=int, default=None,
                   help="Override Module-1 instance ID; auto-resolved if omitted")
    p.add_argument("--tau_alpha", type=float, default=0.4,
                   help="ObjectGS alpha threshold for mask extraction")
    p.add_argument("--min_pixels", type=int, default=64,
                   help="Minimum object pixels to accept a frame")
    
    # Phase 4 params
    p.add_argument("--top_k", type=int, default=5,
                   help="Number of top frames to save in Phase 4")
    
    # Phase 5 params
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
                   help="Skip SV3D diffusion if Phase 5 already exists")
    
    # Phase 6-8 params
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
                   help="Number of orbit views for Phase 8 comparison")
    p.add_argument("--skip_compare", action="store_true",
                   help="Skip Phase 8 before/after rendering")
    
    # General flags
    p.add_argument("--debug", action="store_true",
                   help="Enable debug visualizations for all phases")
    p.add_argument("--log_level", default="INFO",
                   help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    
    return p.parse_args()


def main():
    args = _parse_args()
    _setup_logging(getattr(logging, str(args.log_level).upper(), logging.INFO))
    
    run_all_phases(
        model_path=args.model_path,
        scene_dir=args.scene_dir,
        output_root=args.output_root,
        object_id=args.object_id,
        iterations=args.iterations,
        # Phase 3
        id_map_dir=args.id_map_dir,
        module1_obj_id=args.module1_obj_id,
        tau_alpha=args.tau_alpha,
        min_pixels=args.min_pixels,
        # Phase 4
        top_k=args.top_k,
        # Phase 5
        iou_threshold=args.iou_threshold,
        fov_y_deg=args.fov_y_deg,
        num_inference_steps=args.num_inference_steps,
        safe_mode=args.safe_mode,
        seed=args.seed,
        reuse_sv3d=args.reuse_sv3d,
        # Phase 6-8
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
