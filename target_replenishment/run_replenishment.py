"""
VRoom Target Replenishment — Orchestrator

Iterative PAInpainter loop for object enhancement:
  1. Detect defects (anchor_renderer)
  2. Inpaint anchor view (inpainter)
  3. Propagate to neighbors (content_propagation)
  4. Inpaint neighbor views with priors (inpainter)
  5. Verify multi-view consistency (consistency_verifier)
  6. Fine-tune ObjectGS model (optimizer)
  7. Repeat until convergence

Usage:
    python target_replenishment/run_replenishment.py \\
        --model_path outputs/scene_01 \\
        --output_dir replenished_output \\
        --object_ids 1 3 \\
        --max_iterations 3
"""

import sys
import logging
import argparse
import numpy as np
from pathlib import Path

# Add project root to sys.path so we can import target_replenishment modules
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger(__name__)


def run_replenishment(
    model_path: str,
    output_dir: str = "replenished_output",
    iteration: int = -1,
    max_iterations: int = 3,
    quality_threshold: float = 0.75,
    target_object_ids: list = None,
    scoring_views: int = 4,
    finetune_iterations: int = 500,
    finetune_lr_scale: float = 0.1,
    sd_model_id: str = None,
    prompt: str = "",
    seed: int = 42,
    strength: float = 0.99,
    guidance_scale: float = 7.5,
    mask_dilation_px: int = 10,
):
    """Run the full target replenishment pipeline.

    Args:
        model_path: Path to trained ObjectGS output directory.
        output_dir: Where to save results.
        iteration: Training iteration to load (-1 = latest).
        max_iterations: Number of PAInpainter outer loop iterations.
        quality_threshold: Below this quality score = healthy. Above = degraded.
        target_object_ids: Specific object IDs to enhance (None = all).
        scoring_views: Number of training cameras for multi-view quality scoring.
        finetune_iterations: Per-round fine-tuning iterations.
        finetune_lr_scale: Learning rate scale for fine-tuning.
        sd_model_id: Override Stable Diffusion model ID.
        prompt: Text prompt for inpainting.
        seed: Random seed for reproducibility.
        strength: Inpainting denoising strength.
        guidance_scale: CFG scale for prompt guidance.
        mask_dilation_px: Expand repair mask by this many pixels.

    Returns:
        dict with 'iteration_results', 'final_metrics'.
    """
    import torch
    from target_replenishment.core.anchor_renderer import run_anchor_detection
    from target_replenishment.core.content_propagation import propagate_to_neighbors
    from target_replenishment.core.inpainter import load_inpainter, inpaint_view
    from target_replenishment.core.consistency_verifier import verify_consistency
    from target_replenishment.core.optimizer import optimize_with_inpainted_views
    from target_replenishment.core.objectgs_bridge import load_gaussians
    from target_replenishment.core.metrics import compute_psnr, compute_ssim

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading model from {model_path} (iteration {iteration})")
    gaussians, pipe_config = load_gaussians(model_path, iteration)

    logger.info("Loading inpainting model...")
    inpaint_pipeline = load_inpainter(device="cuda", model_id=sd_model_id)

    iteration_results = []

    for outer_iter in range(1, max_iterations + 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"ITERATION {outer_iter}/{max_iterations}")
        logger.info(f"{'='*60}")

        iter_dir = out / f"iter_{outer_iter:02d}"
        iter_dir.mkdir(parents=True, exist_ok=True)

        # ── Step 1: Detect defects ──
        logger.info("Step 1: Detecting defects...")
        detection = run_anchor_detection(
            model_path,
            output_dir=str(iter_dir / "detection"),
            quality_threshold=quality_threshold,
            target_object_ids=target_object_ids,
            scoring_views=scoring_views,
            mask_dilation_px=mask_dilation_px,
        )

        objects_with_defects = {
            oid: r for oid, r in detection.items()
            if r['defect_regions']
        }

        if not objects_with_defects:
            logger.info("No defects detected in any object. Pipeline converged.")
            break

        logger.info(f"Found defects in {len(objects_with_defects)} object(s)")

        round_results = {}

        for obj_id, result in objects_with_defects.items():
            logger.info(f"\n--- Object {obj_id} ---")
            renders = result['renders']
            cam_params = result['camera_params']

            # ── Step 2: Inpaint anchor view ──
            logger.info("Step 2: Inpainting anchor view...")
            inpainted_anchor = inpaint_view(
                inpaint_pipeline,
                renders['rgb'],
                renders['repair_mask'],
                prompt=prompt,
                seed=seed + outer_iter,
                strength=strength,
                guidance_scale=guidance_scale,
            )

            _save_image(inpainted_anchor, iter_dir / f"obj_{obj_id}_inpainted_anchor.png")

            # ── Step 3: Propagate to neighbors ──
            neighbor_cams = cam_params.get('neighbors', [])
            propagated = []
            if neighbor_cams and renders['depth'] is not None:
                logger.info(f"Step 3: Propagating to {len(neighbor_cams)} neighbor(s)...")
                propagated = propagate_to_neighbors(
                    inpainted_anchor,
                    renders['depth'],
                    cam_params,
                    neighbor_cams,
                    renders['repair_mask'],
                )
            else:
                logger.info("Step 3: No neighbor cameras or depth — skipping propagation.")

            # ── Step 4: Inpaint neighbor views ──
            m_candidates = 4
            neighbor_candidates = []
            for ni, prop in enumerate(propagated):
                logger.info(f"Step 4: Inpainting neighbor {ni} ({m_candidates} candidates)...")
                from target_replenishment.core.objectgs_bridge import (
                    create_virtual_camera, render_view,
                )
                nc = prop['camera_params']
                cam = create_virtual_camera(nc['R'], nc['T'], nc['K'], nc['width'], nc['height'])
                bg = torch.zeros(3, dtype=torch.float32, device="cuda")
                nr = render_view(gaussians, cam, pipe_config, bg)
                nr_rgb = (nr['rgb'].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

                # Build neighbor mask from warped mask
                n_mask = prop['mask_warped']

                cands = []
                for m_idx in range(m_candidates):
                    n_inpainted = inpaint_view(
                        inpaint_pipeline,
                        nr_rgb,
                        n_mask,
                        propagated_prior=prop['rgb_warped'],
                        prompt=prompt,
                        seed=seed + outer_iter * 100 + ni * 10 + m_idx,
                        strength=strength,
                        guidance_scale=guidance_scale,
                    )
                    cands.append({
                        'rgb_inpainted': n_inpainted,
                        'rgb_warped': prop['rgb_warped'],
                        'mask_warped': n_mask,
                        'mask_inpainted': n_mask,
                        'camera_params': nc,
                    })
                    
                    _save_image(n_inpainted, iter_dir / f"obj_{obj_id}_neighbor_{ni}_cand_{m_idx}.png")
                    
                neighbor_candidates.append(cands)

            # ── Step 5: Verify consistency ──
            logger.info("Step 5: Verifying consistency...")
            anchor_for_verify = {
                'rgb_inpainted': inpainted_anchor,
                'mask': renders['repair_mask'],
                'depth': renders['depth'],
                'camera_params': cam_params,
            }
            verified = verify_consistency(
                anchor_for_verify,
                neighbor_candidates,
                gaussians=gaussians,
                pipe_config=pipe_config,
            )
            logger.info(f"Accepted: {len(verified['accepted_views'])}, "
                        f"Rejected: {len(verified['rejected_views'])}")

            # ── Step 6: Optimize ──
            if verified['accepted_views']:
                logger.info(f"Step 6: Fine-tuning model ({finetune_iterations} iters)...")
                opt_result = optimize_with_inpainted_views(
                    gaussians, pipe_config,
                    verified['accepted_views'],
                    n_iterations=finetune_iterations,
                    lr_scale=finetune_lr_scale,
                    save_path=str(iter_dir / f"obj_{obj_id}_model"),
                )
                round_results[obj_id] = {
                    'defect_regions': len(result['defect_regions']),
                    'accepted_views': len(verified['accepted_views']),
                    'final_loss': opt_result['final_loss'],
                }
            else:
                logger.warning("No views accepted — skipping optimization for this object.")
                round_results[obj_id] = {
                    'defect_regions': len(result['defect_regions']),
                    'accepted_views': 0,
                    'final_loss': None,
                }

        iteration_results.append(round_results)

    # Save final model
    final_path = out / "final_model"
    final_path.mkdir(parents=True, exist_ok=True)
    try:
        gaussians.save_ply(str(final_path / "point_cloud.ply"))
        gaussians.save_mlp_checkpoints(str(final_path))
        logger.info(f"Saved final model to {final_path}")
    except Exception as e:
        logger.error(f"Failed to save final model: {e}")

    logger.info(f"\n{'='*60}")
    logger.info("REPLENISHMENT COMPLETE")
    logger.info(f"{'='*60}")

    return {
        'iteration_results': iteration_results,
        'output_dir': str(out),
    }


def _save_image(img: np.ndarray, path: Path):
    """Save RGB numpy image to disk."""
    import cv2
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%H:%M:%S',
    )

    parser = argparse.ArgumentParser(
        description="VRoom Target Replenishment — Object Enhancement Pipeline"
    )
    parser.add_argument("--model_path", required=True,
                        help="ObjectGS training output directory")
    parser.add_argument("--iteration", type=int, default=-1,
                        help="Training iteration to load (-1 = latest)")
    parser.add_argument("--output_dir", default="replenished_output",
                        help="Output directory for results")
    parser.add_argument("--max_iterations", type=int, default=3,
                        help="Number of PAInpainter outer loop iterations")
    parser.add_argument("--quality_threshold", type=float, default=0.75,
                        help="Geometric degradation score threshold [0, 1]. Anchors above this score are masked for repair. Default 0.75.")
    parser.add_argument("--object_ids", type=int, nargs='+', default=None,
                        help="Specific object IDs to enhance")
    parser.add_argument("--scoring_views", type=int, default=4,
                        help="Training cameras for multi-view scoring")
    parser.add_argument("--finetune_iters", type=int, default=500,
                        help="Fine-tuning iterations per round")
    parser.add_argument("--lr_scale", type=float, default=0.1,
                        help="Learning rate scale for fine-tuning")
    parser.add_argument("--prompt", type=str, default="",
                        help="Text prompt for SD Inpainting")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--strength", type=float, default=0.99,
                        help="Inpainting denoising strength (0-1.0). Use 0.99 for structural completion, 0.5 for texture enhancement.")
    parser.add_argument("--mask_dilation_px", type=int, default=10,
                        help="Expand repair mask by this many pixels. 10 is recommended for general defect covering.")
    parser.add_argument("--guidance_scale", type=float, default=7.5,
                        help="CFG scale for prompt guidance")
    args = parser.parse_args()

    run_replenishment(
        model_path=args.model_path,
        output_dir=args.output_dir,
        iteration=args.iteration,
        max_iterations=args.max_iterations,
        quality_threshold=args.quality_threshold,
        target_object_ids=args.object_ids,
        scoring_views=args.scoring_views,
        finetune_iterations=args.finetune_iters,
        finetune_lr_scale=args.lr_scale,
        prompt=args.prompt,
        seed=args.seed,
        strength=args.strength,
        guidance_scale=args.guidance_scale,
    )


if __name__ == "__main__":
    main()
