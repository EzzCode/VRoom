"""
VRoom Target Replenishment — Era3D Novel View Pipeline

Replaces the PAInpainter multi-candidate inpainting pipeline with a direct
novel-view generation approach using Era3D:

  1. Analyze coverage gaps to find unseen object hemispheres
  2. Render best visible view of isolated object
  3. Generate 6 novel views via Era3D (~10 sec)
  4. Align views to Scaffold-GS world coordinate frame
  5. Fine-tune 2DGS model with frozen MLPs (~2 min)

Usage:
    python target_replenishment/run_replenishment.py \\
        --model_path outputs/scene_01 \\
        --object_ids 8 \\
        --up_axis z
"""

import sys
import json
import shutil
import logging
import argparse
import numpy as np
import torch
import cv2
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger(__name__)


def run_replenishment(
    model_path: str,
    output_dir: str = "replenished_output",
    iteration: int = -1,
    target_object_ids: list = None,
    up_axis: str = 'auto',
    finetune_iterations: int = 1200,
    finetune_lr_scale: float = 1.0,
    hallucination_weight: float = 0.08,
    novel_rgb_weight: float = 1.0,
    target_mask_erode_px: int = 0,
    freeze_feat_when_rgb_off: bool = True,
    conservative_seed_render: bool = True,
    visual_hull_seed_filter: bool = True,
    visual_hull_min_views: int = 2,
    surface_shell_seed_filter: bool = True,
    surface_shell_min_norm: float = 0.65,
    seed_opacity_gate_init: float = 0.02,
    seed_opacity_gate_lr_scale: float = 50.0,
    seed_opacity_gate_reg_weight: float = 0.005,
    seed_opacity_lift_init: float = 0.0,
    seed_opacity_lift_lr_scale: float = 10.0,
    seed_opacity_lift_reg_weight: float = 0.02,
    seed_opacity_accept_threshold: float = 0.10,
    diffusion_steps: int = 75,
    seed: int = 42,
    auto_compare: bool = True,
    comparison_views: int = 8,
    offset_scale_frac: float = 0.5,
    seeded_scale_max_frac: float = 0.06,
    bounds_expand_frac: float = 0.05,
    originals_lr_scale: float = 0.05,
    originals_max_scale_delta: float = 0.05,
    originals_max_offset_delta: float = 0.05,
    originals_reg_weight: float = 0.5,
    seeded_scale_reg_weight: float = 0.20,
    seeded_offset_reg_weight: float = 0.20,
    seeded_max_scale_delta: float = 0.20,
    seeded_max_offset_delta: float = 0.20,
    feat_lr_scale: float = 0.25,
    feat_reg_weight: float = 0.05,
    silhouette_iou_thresh: float = 0.35,
    cage_padding_frac: float = 0.02,
    hole_weight_max: float = 2.5,
    seeded_anisotropy_max: float = 3.0,
    train_mlp_opacity: bool = False,
    mlp_opacity_lr_scale: float = 0.001,
    mlp_opacity_reg_weight: float = 1.0,
    freeze_originals: bool = True,
    repair_diagnostics: bool = False,
    repair_diagnostics_only: bool = False,
    repair_diag_alpha_threshold: float = 0.03,
    repair_filter_supervision: bool = True,
    repair_filter_min_trust: float = 0.45,
    repair_filter_max_outside_alpha: float = 0.20,
    repair_filter_max_missing_ratio: float = 0.55,
    repair_filter_min_target_iou: float = 0.30,
    repair_filter_min_views: int = 2,
    repair_filter_allow_inspect_prior: bool = True,
    aligned_repair_candidates: bool = False,
    aligned_repair_candidates_only: bool = False,
    aligned_repair_support_dilate_px: int = 12,
    aligned_repair_min_component_px: int = 32,
    aligned_repair_max_components: int = 6,
    aligned_repair_max_area_ratio: float = 0.40,
    aligned_repair_min_target_render_iou: float = 0.20,
    azimuth_sign: int = -1,
    elevation_sign: int = 1,
):
    """Run the Zero123++ novel-view target replenishment pipeline.

    Args:
        model_path: Path to trained ObjectGS output directory.
        output_dir: Where to save results.
        iteration: Training iteration to load (-1 = latest).
        target_object_ids: Specific object IDs to enhance (None = all).
        up_axis: World up axis ('x', 'y', 'z', or 'auto').
        finetune_iterations: Per-object fine-tuning iterations.
        finetune_lr_scale: Learning rate scale for fine-tuning.
        hallucination_weight: Weight for Era3D views in loss (0.0-1.0).
        era3d_steps: Era3D diffusion inference steps.
        seed: Random seed.
        auto_compare: If True, save fixed-pose before/after renders per object.
        comparison_views: Number of orbit views for auto comparison.
    """
    from target_replenishment.core.objectgs_bridge import (
        load_gaussians, get_anchor_positions,
    )
    from target_replenishment.core.perspective_graph import build_perspective_graph, get_top_k_views_for_object
    from target_replenishment.core.coverage_analyzer import analyze_coverage
    from target_replenishment.core.novel_view_generator import (
        load_zero123pp, generate_novel_views, render_object_for_input,
    )
    from target_replenishment.core.view_alignment import compute_novel_cameras
    from target_replenishment.core.optimizer import optimize_with_novel_views
    from target_replenishment.core.anchor_seeding import (
        seed_backside_anchors, build_visual_hull_seed_constraints,
    )
    from target_replenishment.core.repair_diagnostics import (
        analyze_repair_candidates, filter_supervision_views,
    )
    from target_replenishment.core.repair_candidate_stage import analyze_aligned_repair_candidates
    from target_replenishment.core.io_utils import (
        save_image as _save_image,
        save_coverage_plot as _save_coverage_plot,
        build_comparison_cameras as _build_comparison_cameras,
        render_object_with_cameras as _render_object_with_cameras,
        save_camera_metadata as _save_camera_metadata,
        save_auto_comparison as _save_auto_comparison,
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load model ──
    logger.info(f"Loading model from {model_path} (iteration {iteration})")
    gaussians, pipe_config = load_gaussians(model_path, iteration)

    # ── Step 2: Load cameras + build perspective graph ──
    cameras_json = Path(model_path) / "cameras.json"
    anchor_xyz_global = get_anchor_positions(gaussians)
    graph = build_perspective_graph(str(cameras_json), anchor_xyz_global, overlap_method='visibility')

    # Resolve training images directory from cfg_args (for real-image preservation).
    train_images_dir = None
    try:
        cfg_args_path = Path(model_path) / "cfg_args"
        if cfg_args_path.exists():
            cfg_text = cfg_args_path.read_text()
            import re as _re
            m_src = _re.search(r"source_path=['\"]([^'\"]+)['\"]", cfg_text)
            m_img = _re.search(r"images=['\"]([^'\"]+)['\"]", cfg_text)
            if m_src:
                src = Path(m_src.group(1))
                images_subdir = m_img.group(1) if m_img else "images"
                cand = src / images_subdir
                if cand.exists():
                    train_images_dir = cand
                    logger.info("Real-image preservation: using %s", train_images_dir)
                else:
                    logger.warning("Real-image preservation: %s does not exist; "
                                   "falling back to self-rendered snapshots.", cand)
    except Exception as e:
        logger.warning("Real-image preservation: cfg_args parse failed (%s); "
                       "falling back to self-rendered snapshots.", e)

    # ── Step 3: Load Zero123++ ──
    logger.info("Loading Zero123++ model...")
    zero123_pipeline = load_zero123pp(device="cuda")

    # ── Step 4: Process each object ──
    labels = gaussians.label_ids.squeeze(-1).cpu().numpy()
    process_ids = target_object_ids if target_object_ids is not None else np.unique(labels).tolist()

    results = {}
    final_rep_payload = None
    model_was_mutated = False
    for obj_id in process_ids:
        logger.info(f"\n{'='*60}")
        logger.info(f"OBJECT {obj_id}")
        logger.info(f"{'='*60}")

        obj_mask = (labels == obj_id)
        object_anchors = anchor_xyz_global[obj_mask]

        if len(object_anchors) < 10:
            logger.warning(f"Object {obj_id} has too few anchors ({len(object_anchors)}). Skipping.")
            continue

        obj_dir = out / f"obj_{obj_id}"
        obj_dir.mkdir(parents=True, exist_ok=True)

        # ── Step 4a: Coverage analysis ──
        logger.info("Analyzing coverage gaps...")
        coverage = analyze_coverage(
            object_anchors, graph.cameras,
            up_axis=up_axis,
        )

        # Save coverage visualization
        _save_coverage_plot(coverage, obj_dir / "coverage.png")

        # ── Step 4b: Render centered object view for Zero123++ input ──
        logger.info("Rendering centered object view (virtual camera)...")
        best_cam = coverage.best_input_cam
        input_render = render_object_for_input(
            gaussians, pipe_config,
            object_center=coverage.object_center,
            object_radius=coverage.object_radius,
            input_cam_position=best_cam['position'],
            up_vector=coverage.up_vector,
            object_id=obj_id,
            render_size=512,
            reference_K=best_cam.get('K'),
            reference_width=best_cam.get('width'),
            reference_height=best_cam.get('height'),
        )

        rgb_np = input_render['rgb']
        alpha_np = input_render['alpha']

        _save_image(rgb_np, obj_dir / "input_view.png")
        _save_image((alpha_np * 255).astype(np.uint8), obj_dir / "input_alpha.png")

        comparison_cameras = None
        before_frames = []
        if auto_compare:
            comparison_cameras = _build_comparison_cameras(
                center=coverage.object_center,
                radius=coverage.object_radius,
                orbit_radius=coverage.orbit_radius,
                up_vector=coverage.up_vector,
                input_cam_position=best_cam['position'],
                width=512,
                height=512,
                n_views=comparison_views,
            )
            _save_camera_metadata(
                obj_dir / "camera_metadata.json",
                obj_id,
                coverage.object_center,
                coverage.object_radius,
                comparison_cameras,
            )
            before_frames = _render_object_with_cameras(
                gaussians,
                pipe_config,
                comparison_cameras,
                obj_id,
            )

        # ── Step 4c: Generate novel views with Zero123++ ──
        logger.info("Generating novel views with Zero123++...")
        novel_views = generate_novel_views(
            zero123_pipeline,
            rgb_np,
            alpha_mask=alpha_np,
            num_inference_steps=diffusion_steps,
            seed=seed,
        )

        # Save generated views
        for i, view in enumerate(novel_views):
            az = view.get('azimuth_offset_deg', '?')
            el = view.get('elevation_offset_deg', '?')
            _save_image(view['rgb'], obj_dir / f"novel_view_{i}_az{az}_el{el}.png")

        logger.info(f"Generated {len(novel_views)} novel views")

        # Anchor orbit radius to the ACTUAL input camera distance to object
        # center (not the median over all training cameras). This guarantees
        # that an azimuth-offset of 0 with elevation-offset of 0 lands the
        # supervision camera at the input camera's location, which is the
        # only setting in which Zero123++'s 6 views are guaranteed self-
        # consistent with the input we showed it.
        input_orbit_radius = float(np.linalg.norm(
            np.asarray(best_cam['position'], dtype=np.float32)
            - np.asarray(coverage.object_center, dtype=np.float32)
        ))
        if not np.isfinite(input_orbit_radius) or input_orbit_radius < 1e-4:
            input_orbit_radius = float(coverage.orbit_radius)
            logger.warning(
                "Input camera distance to object center is degenerate; "
                "falling back to median orbit_radius=%.3f.",
                input_orbit_radius,
            )
        else:
            logger.info(
                "Using input-anchored orbit_radius=%.3f (median over training cams was %.3f)",
                input_orbit_radius, float(coverage.orbit_radius),
            )

        # ── Step 4d: Align to world frame ──
        logger.info("Aligning novel views to world coordinate frame...")
        aligned_cameras_all = compute_novel_cameras(
            object_center=coverage.object_center,
            input_azimuth=coverage.input_azimuth,
            orbit_radius=input_orbit_radius,
            up_vector=coverage.up_vector,
            reference_K=input_render['camera_K'],
            reference_width=512,
            reference_height=512,
            gap_azimuths=None,
            azimuth_sign=azimuth_sign,
            elevation_sign=elevation_sign,
        )

        aligned_cameras = compute_novel_cameras(
            object_center=coverage.object_center,
            input_azimuth=coverage.input_azimuth,
            orbit_radius=input_orbit_radius,
            up_vector=coverage.up_vector,
            reference_K=input_render['camera_K'],
            reference_width=512,
            reference_height=512,
            gap_azimuths=coverage.gap_azimuths,
            azimuth_sign=azimuth_sign,
            elevation_sign=elevation_sign,
        )

        # If gap-filtering leaves too few views, use full generated set.
        if len(aligned_cameras) < min(4, len(novel_views)):
            logger.warning(
                "Too few aligned views after gap filtering (%d). Falling back to all generated views.",
                len(aligned_cameras),
            )
            aligned_cameras = list(aligned_cameras_all)

        # Pair novel views with aligned cameras by (azimuth, elevation) to avoid
        # convention mismatches.
        supervision_views = []

        def _view_key(d: dict):
            az = int(round(float(d.get('azimuth_offset_deg', 0))))
            el = int(round(float(d.get('elevation_offset_deg', 0))))
            return (az, el)

        novel_by_key = {_view_key(view): view for view in novel_views}

        missing_keys_gap = []
        for cam_dict in aligned_cameras:
            key = _view_key(cam_dict)
            view = novel_by_key.get(key)
            if view is None:
                missing_keys_gap.append(key)
                continue
            supervision_views.append({
                'rgb': view['rgb'],
                'camera': cam_dict,
                'weight': hallucination_weight,
            })

        if missing_keys_gap:
            logger.warning(
                "Missing generated novel view(s) for gap-filtered keys: %s",
                sorted(set(missing_keys_gap)),
            )

        logger.info(
            "Prepared %d supervision views in coverage gap",
            len(supervision_views),
        )

        if not supervision_views:
            logger.warning(f"No views to use for object {obj_id}. Skipping.")
            continue

        raw_supervision_count = len(supervision_views)
        repair_diag_result = {}
        repair_filter_result = {}
        aligned_repair_result = {}
        if repair_diagnostics or repair_diagnostics_only:
            logger.info("Running read-only repair diagnostics before seeding/fine-tune...")
            repair_diag_result = analyze_repair_candidates(
                gaussians=gaussians,
                pipe_config=pipe_config,
                supervision_views=supervision_views,
                object_id=int(obj_id),
                object_anchors=object_anchors,
                object_radius=float(coverage.object_radius),
                output_dir=obj_dir / "repair_diagnostics",
                target_mask_erode_px=int(max(0, target_mask_erode_px)),
                alpha_threshold=float(repair_diag_alpha_threshold),
                save_debug_images=True,
            )
            with open(obj_dir / "repair_diagnostics.json", "w", encoding="utf-8") as f:
                json.dump(repair_diag_result, f, indent=2)

        if aligned_repair_candidates or aligned_repair_candidates_only:
            logger.info("Running read-only aligned repair-candidate stage (no model mutation)...")
            aligned_repair_result = analyze_aligned_repair_candidates(
                gaussians=gaussians,
                pipe_config=pipe_config,
                supervision_views=supervision_views,
                object_id=int(obj_id),
                object_anchors=object_anchors,
                object_radius=float(coverage.object_radius),
                output_dir=obj_dir / "repair_candidates",
                target_mask_erode_px=int(max(0, target_mask_erode_px)),
                alpha_threshold=float(repair_diag_alpha_threshold),
                support_dilate_px=int(aligned_repair_support_dilate_px),
                min_repair_component_px=int(aligned_repair_min_component_px),
                max_repair_components=int(aligned_repair_max_components),
                max_repair_area_ratio=float(aligned_repair_max_area_ratio),
                min_target_render_iou=float(aligned_repair_min_target_render_iou),
                save_debug_images=True,
            )
            with open(obj_dir / "repair_candidates.json", "w", encoding="utf-8") as f:
                json.dump(aligned_repair_result, f, indent=2)
            logger.info(
                "Aligned repair-candidate stage: accept=%d inspect=%d reject=%d (n_views=%d)",
                aligned_repair_result.get('summary', {}).get('n_accept_repair_views', 0),
                aligned_repair_result.get('summary', {}).get('n_inspect_repair_views', 0),
                aligned_repair_result.get('summary', {}).get('n_reject_repair_views', 0),
                aligned_repair_result.get('n_views', 0),
            )

        if repair_diagnostics or repair_diagnostics_only or aligned_repair_candidates_only:
            if repair_diagnostics_only or aligned_repair_candidates_only:
                results[obj_id] = {
                    'diagnostic_only': True,
                    'n_gap_bins': len(coverage.gap_azimuths),
                    'n_generated_views': len(novel_views),
                    'n_aligned_views': len(aligned_cameras),
                    'n_supervision_views': len(supervision_views),
                    'repair_diagnostics': repair_diag_result,
                    'aligned_repair_candidates': aligned_repair_result,
                }
                with open(obj_dir / "replenishment_summary.json", "w", encoding="utf-8") as f:
                    json.dump(results[obj_id], f, indent=2)
                logger.info("Object %d diagnostic-only complete; skipping seeding and optimization.", int(obj_id))
                continue

            if repair_filter_supervision:
                supervision_views, repair_filter_result = filter_supervision_views(
                    supervision_views,
                    repair_diag_result,
                    min_trust=float(repair_filter_min_trust),
                    max_outside_alpha_ratio=float(repair_filter_max_outside_alpha),
                    max_missing_ratio=float(repair_filter_max_missing_ratio),
                    min_target_render_iou=float(repair_filter_min_target_iou),
                    min_kept_views=int(repair_filter_min_views),
                    allow_inspect_prior=bool(repair_filter_allow_inspect_prior),
                )
                with open(obj_dir / "repair_filter.json", "w", encoding="utf-8") as f:
                    json.dump(repair_filter_result, f, indent=2)
                logger.info(
                    "Repair diagnostic filter kept %d/%d supervision views: %s",
                    len(supervision_views),
                    raw_supervision_count,
                    repair_filter_result.get('kept_view_indices', []),
                )
                if not supervision_views:
                    results[obj_id] = {
                        'diagnostic_only': False,
                        'skipped_reason': 'repair_filter_rejected_all_supervision_views',
                        'n_gap_bins': len(coverage.gap_azimuths),
                        'n_generated_views': len(novel_views),
                        'n_aligned_views': len(aligned_cameras),
                        'n_supervision_views_raw': int(raw_supervision_count),
                        'n_supervision_views': 0,
                        'repair_diagnostics': repair_diag_result,
                        'aligned_repair_candidates': aligned_repair_result,
                        'repair_filter': repair_filter_result,
                    }
                    with open(obj_dir / "replenishment_summary.json", "w", encoding="utf-8") as f:
                        json.dump(results[obj_id], f, indent=2)
                    logger.warning(
                        "Object %d skipped because repair diagnostics rejected every supervision view.",
                        int(obj_id),
                    )
                    continue

        # Real-view preservation cameras from training data to avoid degrading seen sides.
        preservation_cameras = []
        for c in get_top_k_views_for_object(graph, object_anchors, k=4):
            entry = {
                'R': c['R'],
                'T': c['T'],
                'K': c['K'],
                'width': c['width'],
                'height': c['height'],
            }
            if train_images_dir is not None:
                img_name = c.get('img_name', '')
                # img_name may be a stem (no ext); try common extensions.
                resolved = None
                cand_path = train_images_dir / img_name
                if cand_path.exists() and cand_path.is_file():
                    resolved = cand_path
                else:
                    for ext in ('.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG'):
                        cp = train_images_dir / (img_name + ext)
                        if cp.exists():
                            resolved = cp
                            break
                if resolved is not None:
                    entry['image_path'] = str(resolved)
                else:
                    logger.warning("Preservation: could not resolve image for img_name=%r in %s",
                                   img_name, train_images_dir)
            preservation_cameras.append(entry)

        # Seed missing backside geometry before optimization.
        view_dir = best_cam['position'] - coverage.object_center
        view_dir = view_dir / (np.linalg.norm(view_dir) + 1e-8)

        n_original_anchors = int(gaussians._anchor.shape[0])
        gaussians.n_original_anchors = n_original_anchors
        gaussians.override_view_dir = torch.tensor(
            view_dir,
            dtype=torch.float32,
            device="cuda",
        )

        dynamic_seed_cap = int(np.clip(0.35 * len(object_anchors), 300, 900))
        dynamic_grid = int(np.clip(14 + int(np.log2(max(len(object_anchors), 16))), 14, 18))

        seed_cfg = {
            'grid_resolution': dynamic_grid,
            'k_neighbors': 5,
            'max_new_anchors': dynamic_seed_cap,
            'bounds_quantile_low': 0.02,
            'bounds_quantile_high': 0.98,
            'hemisphere_margin': 0.03,
            'bounds_expand_frac': float(bounds_expand_frac),
            'offset_scale_frac': float(offset_scale_frac),
            'scale_max_frac': float(seeded_scale_max_frac),
            'conservative_seed_render': bool(conservative_seed_render),
            'visual_hull_seed_filter': bool(visual_hull_seed_filter),
            'visual_hull_min_views': int(visual_hull_min_views),
            'surface_shell_seed_filter': bool(surface_shell_seed_filter),
            'surface_shell_min_norm': float(surface_shell_min_norm),
        }
        visual_hull_constraints = []
        if visual_hull_seed_filter:
            visual_hull_constraints = build_visual_hull_seed_constraints(
                supervision_views,
                object_anchors,
                erode_px=max(0, int(target_mask_erode_px)),
            )
            logger.info(
                "Prepared %d visual-hull seed constraints for object %d.",
                len(visual_hull_constraints),
                int(obj_id),
            )
        seed_call_cfg = dict(seed_cfg)
        seed_call_cfg.pop('visual_hull_seed_filter', None)
        seed_call_cfg.pop('surface_shell_seed_filter', None)
        seed_call_cfg['visual_hull_constraints'] = visual_hull_constraints
        seed_call_cfg['visual_hull_min_views'] = int(visual_hull_min_views)
        seed_call_cfg['surface_shell_filter'] = bool(surface_shell_seed_filter)
        seed_call_cfg['surface_shell_min_norm'] = float(surface_shell_min_norm)
        n_seeded = seed_backside_anchors(
            gaussians=gaussians,
            object_center=coverage.object_center,
            view_direction=view_dir,
            object_id=int(obj_id),
            **seed_call_cfg,
        )
        seed_visual_hull_stats = getattr(gaussians, '_replenishment_seed_filter_stats', {})

        # Re-read anchor arrays after in-place tensor expansion.
        labels_post = gaussians.label_ids.squeeze(-1).cpu().numpy()
        anchor_xyz_post = get_anchor_positions(gaussians)
        obj_mask_post = (labels_post == obj_id)
        object_anchors_post = anchor_xyz_post[obj_mask_post]

        logger.info(
            "Seeded %d anchors for object %d (original=%d, total=%d)",
            n_seeded,
            int(obj_id),
            n_original_anchors,
            int(anchor_xyz_post.shape[0]),
        )

        # Compute trainable masks. Seeded backside anchors get full update budget;
        # original-object anchors get a tight low-LR/clamp/reg budget so they can
        # adjust slightly without being cannibalized.
        seeded_anchor_mask_np = np.zeros(anchor_xyz_post.shape[0], dtype=bool)
        if n_seeded > 0:
            seeded_anchor_mask_np[n_original_anchors:] = True
        seeded_anchor_mask_np &= obj_mask_post

        if freeze_originals:
            # Default safe path: only the freshly-seeded backside anchors are
            # trainable. The original-object anchors fit observed views and
            # MUST NOT be moved by hallucinated Zero123++ supervision (which
            # is unaligned in elevation, lighting and identity). Photometric
            # gradients from imperfect novel views otherwise corrupt the
            # already-correct frontside geometry.
            originals_mask_np = np.zeros_like(obj_mask_post, dtype=bool)
            update_mask_np = seeded_anchor_mask_np.copy()
            if not update_mask_np.any():
                # No seeds were added (e.g. coverage already complete);
                # nothing to fine-tune. Skip optimization for this object.
                logger.warning(
                    "Object %d: freeze_originals=True and no seeded anchors -> skipping fine-tune.",
                    int(obj_id),
                )
                continue
        else:
            # Opt-in: trainable set = seeded anchors ∪ originals (whole object,
            # bounded by tight clamps + low LR + reg). NEVER expand to other
            # objects. If backside detection failed, we already have the seeds.
            originals_mask_np = obj_mask_post & (~seeded_anchor_mask_np)
            update_mask_np = seeded_anchor_mask_np | originals_mask_np

        anchor_update_mask = torch.from_numpy(update_mask_np).to(device="cuda", dtype=torch.bool)
        seeded_anchor_mask = torch.from_numpy(seeded_anchor_mask_np).to(device="cuda", dtype=torch.bool)
        originals_anchor_mask = torch.from_numpy(originals_mask_np).to(device="cuda", dtype=torch.bool)

        logger.info(
            "Anchor masks: %d seeded, %d originals, %d total trainable / %d in object",
            int(seeded_anchor_mask.sum().item()),
            int(originals_anchor_mask.sum().item()),
            int(anchor_update_mask.sum().item()),
            int(obj_mask_post.sum()),
        )

        if n_seeded > 0:
            gate_init = float(np.clip(seed_opacity_gate_init, 1e-5, 0.95))
            if finetune_iterations <= 0 or seed_opacity_gate_init <= 0.0:
                gaussians.replenishment_seed_opacity_gate = torch.zeros(
                    n_seeded, 1, dtype=torch.float32, device="cuda"
                )
                if hasattr(gaussians, 'replenishment_seed_opacity_logit'):
                    delattr(gaussians, 'replenishment_seed_opacity_logit')
                logger.info("Seed opacity gate: fixed at 0.0000 for seed-only render")
            else:
                gate_logit = float(np.log(gate_init / (1.0 - gate_init)))
                gaussians.replenishment_seed_opacity_logit = torch.nn.Parameter(
                    torch.full((n_seeded, 1), gate_logit, dtype=torch.float32, device="cuda")
                )
                if hasattr(gaussians, 'replenishment_seed_opacity_gate'):
                    delattr(gaussians, 'replenishment_seed_opacity_gate')
                logger.info(
                    "Seed opacity gate: trainable init=%.4f, lr_scale=%.1f, reg=%.4f",
                    gate_init,
                    float(seed_opacity_gate_lr_scale),
                    float(seed_opacity_gate_reg_weight),
                )

            if finetune_iterations <= 0 or seed_opacity_lift_init <= 0.0:
                if hasattr(gaussians, 'replenishment_seed_opacity_lift'):
                    delattr(gaussians, 'replenishment_seed_opacity_lift')
                logger.info("Seed opacity lift: disabled")
            else:
                lift_init = float(max(0.0, seed_opacity_lift_init))
                gaussians.replenishment_seed_opacity_lift = torch.nn.Parameter(
                    torch.full(
                        (n_seeded, int(gaussians.n_offsets)),
                        lift_init,
                        dtype=torch.float32,
                        device="cuda",
                    )
                )
                logger.info(
                    "Seed opacity lift: trainable init=%.4f, lr_scale=%.1f, reg=%.4f",
                    lift_init,
                    float(seed_opacity_lift_lr_scale),
                    float(seed_opacity_lift_reg_weight),
                )

        # Pull AABB + scale ceiling from the seeding step (None if seeding skipped).
        aabb_min = aabb_max = None
        scale_ceiling_log = None
        if hasattr(gaussians, '_replenishment_aabb') and int(obj_id) in gaussians._replenishment_aabb:
            aabb_meta = gaussians._replenishment_aabb[int(obj_id)]
            aabb_min = aabb_meta['min'].detach().cpu().numpy()
            aabb_max = aabb_meta['max'].detach().cpu().numpy()
            # Use median axis (extent_med) so elongated/thin objects get a
            # ceiling sized to their characteristic dimension, not the long axis.
            extent_med_val = float(aabb_meta.get('extent_med', aabb_meta['extent_max']))
            scale_ceiling_log = float(np.log(max(extent_med_val * float(seeded_scale_max_frac), 1e-6)))

        # ── Step 4e: Fine-tune ──
        logger.info(f"Fine-tuning model ({finetune_iterations} iters)...")
        opt_result = optimize_with_novel_views(
            gaussians, pipe_config,
            supervision_views,
            n_iterations=finetune_iterations,
            lr_scale=finetune_lr_scale,
            hallucination_weight=hallucination_weight,
            novel_rgb_weight=float(novel_rgb_weight),
            target_mask_erode_px=int(max(0, target_mask_erode_px)),
            freeze_feat_when_rgb_off=bool(freeze_feat_when_rgb_off),
            object_id=obj_id,
            object_anchors=object_anchors,
            object_radius=float(coverage.object_radius),
            object_center=coverage.object_center,
            silhouette_weight=max(2.5, hallucination_weight * 4.0),
            anchor_update_mask=anchor_update_mask,
            seeded_anchor_mask=seeded_anchor_mask,
            originals_anchor_mask=originals_anchor_mask,
            seeded_scale_reg_weight=float(seeded_scale_reg_weight),
            outside_alpha_weight=8.0,
            seeded_offset_reg_weight=float(seeded_offset_reg_weight),
            seeded_max_scale_delta=float(seeded_max_scale_delta),
            seeded_max_offset_delta=float(seeded_max_offset_delta),
            originals_lr_scale=float(originals_lr_scale),
            originals_max_scale_delta=float(originals_max_scale_delta),
            originals_max_offset_delta=float(originals_max_offset_delta),
            originals_reg_weight=float(originals_reg_weight),
            feat_lr_scale=float(feat_lr_scale),
            feat_reg_weight=float(feat_reg_weight),
            seed_opacity_gate_lr_scale=float(seed_opacity_gate_lr_scale),
            seed_opacity_gate_reg_weight=float(seed_opacity_gate_reg_weight),
            seed_opacity_lift_lr_scale=float(seed_opacity_lift_lr_scale),
            seed_opacity_lift_reg_weight=float(seed_opacity_lift_reg_weight),
            aabb_min=aabb_min,
            aabb_max=aabb_max,
            cage_padding_frac=float(cage_padding_frac),
            scale_ceiling_log=scale_ceiling_log,
            silhouette_iou_thresh=float(silhouette_iou_thresh),
            hole_weight_max=float(hole_weight_max),
            seeded_anisotropy_max=float(seeded_anisotropy_max),
            train_mlp_opacity=bool(train_mlp_opacity),
            mlp_opacity_lr_scale=float(mlp_opacity_lr_scale),
            mlp_opacity_reg_weight=float(mlp_opacity_reg_weight),
            train_mlp_cov=True,
            mlp_cov_lr_scale=0.005,
            preservation_cameras=preservation_cameras,
            preservation_weight=4.0,
            save_path=str(obj_dir / "model"),
            reference_model_path=model_path,
        )

        if not opt_result.get('loss_history'):
            results[obj_id] = {
                'diagnostic_only': False,
                'skipped_reason': 'optimizer_produced_no_training_steps',
                'n_gap_bins': len(coverage.gap_azimuths),
                'n_generated_views': len(novel_views),
                'n_aligned_views': len(aligned_cameras),
                'n_supervision_views_raw': int(raw_supervision_count),
                'n_supervision_views': len(supervision_views),
                'n_seeded_anchors': int(n_seeded),
                'repair_diagnostics': repair_diag_result,
                'aligned_repair_candidates': aligned_repair_result,
                'repair_filter': repair_filter_result,
                'optimizer_result': opt_result,
            }
            with open(obj_dir / "replenishment_summary.json", "w", encoding="utf-8") as f:
                json.dump(results[obj_id], f, indent=2)
            logger.warning(
                "Object %d skipped final export because optimizer produced no training steps.",
                int(obj_id),
            )
            continue

        model_was_mutated = True

        final_seed_gate_stats = {}
        final_seed_lift_stats = {}
        final_seed_gates_tensor = None
        final_seed_lifts_tensor = None
        if n_seeded > 0:
            with torch.no_grad():
                if hasattr(gaussians, 'replenishment_seed_opacity_logit'):
                    raw_gates = torch.sigmoid(gaussians.replenishment_seed_opacity_logit.detach())
                elif hasattr(gaussians, 'replenishment_seed_opacity_gate'):
                    raw_gates = gaussians.replenishment_seed_opacity_gate.detach()
                else:
                    raw_gates = torch.ones(n_seeded, 1, dtype=torch.float32, device="cuda")

                raw_gates = raw_gates.reshape(n_seeded, 1).clamp(0.0, 1.0)
                accept_thresh = float(seed_opacity_accept_threshold)
                if accept_thresh > 0.0:
                    final_seed_gates_tensor = torch.where(
                        raw_gates >= accept_thresh,
                        raw_gates,
                        torch.zeros_like(raw_gates),
                    )
                else:
                    final_seed_gates_tensor = raw_gates.clone()

                # Use fixed accepted gates for final renders and saved model metadata.
                gaussians.replenishment_seed_opacity_gate = final_seed_gates_tensor
                if hasattr(gaussians, 'replenishment_seed_opacity_logit'):
                    delattr(gaussians, 'replenishment_seed_opacity_logit')

                if hasattr(gaussians, 'replenishment_seed_opacity_lift'):
                    raw_lifts = gaussians.replenishment_seed_opacity_lift.detach().clamp(min=0.0, max=2.0)
                    raw_lifts = raw_lifts.reshape(n_seeded, -1)
                    final_seed_lifts_tensor = torch.where(
                        final_seed_gates_tensor.reshape(n_seeded, 1) > 0.0,
                        raw_lifts,
                        torch.zeros_like(raw_lifts),
                    )
                    gaussians.replenishment_seed_opacity_lift = final_seed_lifts_tensor
                    final_lift_flat = final_seed_lifts_tensor.reshape(-1)
                    final_seed_lift_stats = {
                        'min': float(final_lift_flat.min().item()),
                        'median': float(final_lift_flat.median().item()),
                        'mean': float(final_lift_flat.mean().item()),
                        'max': float(final_lift_flat.max().item()),
                    }

                accepted_count = int((final_seed_gates_tensor > 0.0).sum().item())
                final_seed_gate_stats = {
                    'accept_threshold': float(accept_thresh),
                    'accepted_count': accepted_count,
                    'total_count': int(n_seeded),
                    'min': float(final_seed_gates_tensor.min().item()),
                    'median': float(final_seed_gates_tensor.median().item()),
                    'mean': float(final_seed_gates_tensor.mean().item()),
                    'max': float(final_seed_gates_tensor.max().item()),
                }
                logger.info(
                    "Seed opacity final acceptance: %d/%d gates kept at threshold %.3f (max=%.4f)",
                    accepted_count,
                    int(n_seeded),
                    float(accept_thresh),
                    final_seed_gate_stats['max'],
                )

        seeded_opacity_gates = []
        seeded_opacity_lifts = []
        if n_seeded > 0:
            with torch.no_grad():
                if final_seed_gates_tensor is not None:
                    gate_tensor = final_seed_gates_tensor
                elif hasattr(gaussians, 'replenishment_seed_opacity_logit'):
                    gate_tensor = torch.sigmoid(gaussians.replenishment_seed_opacity_logit.detach())
                elif hasattr(gaussians, 'replenishment_seed_opacity_gate'):
                    gate_tensor = gaussians.replenishment_seed_opacity_gate.detach()
                else:
                    gate_tensor = torch.ones(n_seeded, 1, dtype=torch.float32, device="cuda")
                seeded_opacity_gates = [float(v) for v in gate_tensor.reshape(-1).cpu().tolist()]
                if final_seed_lifts_tensor is not None:
                    lift_tensor = final_seed_lifts_tensor
                elif hasattr(gaussians, 'replenishment_seed_opacity_lift'):
                    lift_tensor = gaussians.replenishment_seed_opacity_lift.detach()
                else:
                    lift_tensor = torch.zeros(n_seeded, int(gaussians.n_offsets), dtype=torch.float32, device="cuda")
                seeded_opacity_lifts = [
                    [float(v) for v in row]
                    for row in lift_tensor.reshape(n_seeded, -1).cpu().tolist()
                ]

        rep_payload = {
            'object_id': int(obj_id),
            'n_original_anchors': int(n_original_anchors),
            'override_view_dir': [float(v) for v in view_dir.tolist()],
            'seeded_anchors': int(n_seeded),
            'seed_settings': seed_cfg,
            'seed_opacity_gate_init': float(seed_opacity_gate_init),
            'seed_opacity_gate_lr_scale': float(seed_opacity_gate_lr_scale),
            'seed_opacity_gate_reg_weight': float(seed_opacity_gate_reg_weight),
            'seed_opacity_lift_init': float(seed_opacity_lift_init),
            'seed_opacity_lift_lr_scale': float(seed_opacity_lift_lr_scale),
            'seed_opacity_lift_reg_weight': float(seed_opacity_lift_reg_weight),
            'seed_opacity_accept_threshold': float(seed_opacity_accept_threshold),
            'seeded_opacity_gates': seeded_opacity_gates,
            'seeded_opacity_lifts': seeded_opacity_lifts,
            'seed_visual_hull_stats': seed_visual_hull_stats,
        }
        model_dir = obj_dir / "model"
        model_dir.mkdir(parents=True, exist_ok=True)
        with open(model_dir / "replenishment.json", "w", encoding="utf-8") as f:
            json.dump(rep_payload, f, indent=2)
        final_rep_payload = rep_payload

        compare_summary = {}
        if auto_compare and comparison_cameras is not None:
            after_frames = _render_object_with_cameras(
                gaussians,
                pipe_config,
                comparison_cameras,
                obj_id,
            )
            compare_summary = _save_auto_comparison(
                before_frames,
                after_frames,
                obj_dir / "auto_compare",
            )

        results[obj_id] = {
            'n_gap_bins': len(coverage.gap_azimuths),
            'n_generated_views': len(novel_views),
            'n_aligned_views': len(aligned_cameras),
            'n_supervision_views_raw': int(raw_supervision_count),
            'n_supervision_views': len(supervision_views),
            'n_seeded_anchors': int(n_seeded),
            'final_loss': opt_result['final_loss'],
            'view_usage_counts': opt_result.get('view_usage_counts', []),
            'param_delta_norms': opt_result.get('param_delta_norms', {}),
            'seeded_delta_norms': opt_result.get('seeded_delta_norms', {}),
            'seeded_scaling_stats': opt_result.get('seeded_scaling_stats', {}),
            'seeded_offset_stats': opt_result.get('seeded_offset_stats', {}),
            'seed_opacity_gate_stats': opt_result.get('seed_opacity_gate_stats', {}),
            'seed_opacity_lift_stats': opt_result.get('seed_opacity_lift_stats', {}),
            'final_seed_opacity_gate_stats': final_seed_gate_stats,
            'final_seed_opacity_lift_stats': final_seed_lift_stats,
            'supervision_diagnostics': opt_result.get('supervision_diagnostics', []),
            'seeded_scale_reg_last': opt_result.get('seeded_scale_reg_last', 0.0),
            'seeded_offset_reg_last': opt_result.get('seeded_offset_reg_last', 0.0),
            'outside_alpha_loss_last': opt_result.get('outside_alpha_loss_last', 0.0),
            'seed_opacity_gate_reg_last': opt_result.get('seed_opacity_gate_reg_last', 0.0),
            'seed_opacity_lift_reg_last': opt_result.get('seed_opacity_lift_reg_last', 0.0),
            'mlp_opacity_reg_last': opt_result.get('mlp_opacity_reg_last', 0.0),
            'mlp_opacity_delta_norm': opt_result.get('mlp_opacity_delta_norm'),
            'train_mlp_opacity': opt_result.get('train_mlp_opacity', False),
            'mlp_opacity_lr_scale': opt_result.get('mlp_opacity_lr_scale'),
            'mlp_opacity_reg_weight': opt_result.get('mlp_opacity_reg_weight'),
            'aabb_escape_total': opt_result.get('aabb_escape_total', 0),
            'n_anchors_caged_last': opt_result.get('n_anchors_caged_last', 0),
            'n_scale_clipped_last': opt_result.get('n_scale_clipped_last', 0),
            'n_aniso_capped_last': opt_result.get('n_aniso_capped_last', 0),
            'max_scale_log': opt_result.get('max_scale_log'),
            'mean_scale_log': opt_result.get('mean_scale_log'),
            'n_views_dropped': opt_result.get('n_views_dropped', 0),
            'scale_ceiling_log': opt_result.get('scale_ceiling_log'),
            'offset_abs_cap': opt_result.get('offset_abs_cap'),
            'extent_max': opt_result.get('extent_max'),
            'extent_med': opt_result.get('extent_med'),
            'extent_min': opt_result.get('extent_min'),
            'novel_rgb_weight': opt_result.get('novel_rgb_weight'),
            'target_mask_erode_px': opt_result.get('target_mask_erode_px'),
            'freeze_feat_when_rgb_off': opt_result.get('freeze_feat_when_rgb_off'),
            'conservative_seed_render': bool(conservative_seed_render),
            'seed_opacity_gate_init': float(seed_opacity_gate_init),
            'seed_opacity_lift_init': float(seed_opacity_lift_init),
            'seed_opacity_accept_threshold': float(seed_opacity_accept_threshold),
            'seed_visual_hull_stats': seed_visual_hull_stats,
            'repair_diagnostics': repair_diag_result,
            'aligned_repair_candidates': aligned_repair_result,
            'repair_filter': repair_filter_result,
            'comparison': compare_summary,
        }

        summary_path = obj_dir / "replenishment_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(results[obj_id], f, indent=2)

        logger.info(
            f"Object {obj_id} complete: "
            f"gap={len(coverage.gap_azimuths)} bins, "
            f"views={len(supervision_views)}, "
            f"loss={opt_result['final_loss']:.5f}"
        )

    # ── Save final model ──
    if model_was_mutated:
        final_path = out / "final_model"
        final_path.mkdir(parents=True, exist_ok=True)
        try:
            # Legacy/simple dump
            gaussians.save_ply(str(final_path / "point_cloud.ply"))
            gaussians.save_mlp_checkpoints(str(final_path))

            # ObjectGS-compatible checkpoint layout for load_gaussians()
            iter_dir = final_path / "point_cloud" / "iteration_1"
            iter_dir.mkdir(parents=True, exist_ok=True)
            gaussians.save_ply(str(iter_dir / "point_cloud.ply"))
            gaussians.save_mlp_checkpoints(str(iter_dir))

            ref = Path(model_path)
            for name in ("config.yaml", "cameras.json"):
                src = ref / name
                dst = final_path / name
                if src.exists() and not dst.exists():
                    shutil.copy2(src, dst)

            if final_rep_payload is not None:
                with open(final_path / "replenishment.json", "w", encoding="utf-8") as f:
                    json.dump(final_rep_payload, f, indent=2)

            logger.info(f"Saved final model to {final_path}")
        except Exception as e:
            logger.error(f"Failed to save final model: {e}")
    else:
        logger.info("No model mutation was performed; skipping final model export.")

    logger.info(f"\n{'='*60}")
    logger.info("REPLENISHMENT COMPLETE")
    logger.info(f"{'='*60}")

    with open(out / "pipeline_summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%H:%M:%S',
    )

    parser = argparse.ArgumentParser(
        description="VRoom Target Replenishment — Era3D Novel View Pipeline"
    )
    parser.add_argument("--model_path", required=True,
                        help="ObjectGS training output directory")
    parser.add_argument("--iteration", type=int, default=-1,
                        help="Training iteration to load (-1 = latest)")
    parser.add_argument("--output_dir", default="replenished_output",
                        help="Output directory for results")
    parser.add_argument("--object_ids", type=int, nargs='+', default=None,
                        help="Specific object IDs to enhance")
    parser.add_argument("--up_axis", default="auto", choices=['x', 'y', 'z', 'auto', 'spread'],
                        help="World up axis (auto-detect from cameras)")
    parser.add_argument("--finetune_iters", type=int, default=1200,
                        help="Fine-tuning iterations per object")
    parser.add_argument("--lr_scale", type=float, default=1.0,
                        help="Learning rate scale for fine-tuning")
    parser.add_argument("--hallucination_weight", type=float, default=0.15,
                        help="Loss weight for hallucinated views (0.0–1.0)")
    parser.add_argument("--novel_rgb_weight", type=float, default=1.0,
                        help="Extra multiplier on direct Zero123++ RGB L1/SSIM. "
                            "Use 0.0 for geometry/mask-only ablations.")
    parser.add_argument("--target_mask_erode_px", type=int, default=0,
                        help="Erode generated target masks by this many pixels before "
                            "loss/filtering to ignore uncertain edges and drips.")
    parser.add_argument("--no_freeze_feat_when_rgb_off", action="store_true",
                        help="Allow anchor features to update even when --novel_rgb_weight is 0.0")
    parser.add_argument("--diffusion_steps", type=int, default=75,
                        help="Zero123++ diffusion inference steps")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--no_auto_compare", action="store_true",
                        help="Disable automatic fixed-pose before/after comparison outputs")
    parser.add_argument("--comparison_views", type=int, default=8,
                        help="Number of orbit views for auto comparison")
    # Seeding controls
    parser.add_argument("--offset_scale_frac", type=float, default=0.5,
                        help="Seed offset magnitude as a fraction of grid spacing")
    parser.add_argument("--seeded_scale_max_frac", type=float, default=0.10,
                        help="Max Gaussian scale as fraction of object's largest extent")
    parser.add_argument("--bounds_expand_frac", type=float, default=0.05,
                        help="Fractional expansion of robust object AABB for seeding")
    parser.add_argument("--legacy_seed_render", action="store_true",
                        help="Use old random child offsets and KNN visual scales for seeded anchors")
    parser.add_argument("--no_visual_hull_seed_filter", action="store_true",
                        help="Disable pre-seed visual-hull carving from generated novel-view masks")
    parser.add_argument("--visual_hull_min_views", type=int, default=2,
                        help="Minimum generated-view masks a candidate seed must project into")
    parser.add_argument("--no_surface_shell_seed_filter", action="store_true",
                        help="Disable outer-shell filtering of candidate seed positions")
    parser.add_argument("--surface_shell_min_norm", type=float, default=0.65,
                        help="Keep candidate seeds whose normalized AABB shell score is at least this value")
    parser.add_argument("--seed_opacity_gate_init", type=float, default=0.02,
                        help="Initial opacity multiplier for newly seeded anchors; 0 disables seed-only visibility")
    parser.add_argument("--seed_opacity_gate_lr_scale", type=float, default=50.0,
                        help="Learning-rate multiplier for trainable seeded opacity gates")
    parser.add_argument("--seed_opacity_gate_reg_weight", type=float, default=0.005,
                        help="Sparsity regularization weight for seeded opacity gates")
    parser.add_argument("--seed_opacity_lift_init", type=float, default=0.0,
                        help="Initial additive pre-mask opacity lift for seeded anchors; 0 disables")
    parser.add_argument("--seed_opacity_lift_lr_scale", type=float, default=10.0,
                        help="Learning-rate multiplier for seeded pre-mask opacity lift")
    parser.add_argument("--seed_opacity_lift_reg_weight", type=float, default=0.02,
                        help="Magnitude regularization weight for seeded opacity lift")
    parser.add_argument("--seed_opacity_accept_threshold", type=float, default=0.10,
                        help="Final gate threshold for accepting seeded anchors into saved/rendered output; <=0 disables")
    # Originals (pre-existing object Gaussians) trainability budget
    parser.add_argument("--originals_lr_scale", type=float, default=0.05,
                        help="Effective LR factor applied via tight clamps/reg on originals")
    parser.add_argument("--originals_max_scale_delta", type=float, default=0.05,
                        help="Hard delta-from-init clamp on originals' _scaling")
    parser.add_argument("--originals_max_offset_delta", type=float, default=0.05,
                        help="Hard delta-from-init clamp on originals' _offset")
    parser.add_argument("--originals_reg_weight", type=float, default=0.5,
                        help="MSE-to-init regularization weight for originals")
    # Seeded regularization / clamps
    parser.add_argument("--seeded_scale_reg_weight", type=float, default=0.20)
    parser.add_argument("--seeded_offset_reg_weight", type=float, default=0.20)
    parser.add_argument("--seeded_max_scale_delta", type=float, default=0.20)
    parser.add_argument("--seeded_max_offset_delta", type=float, default=0.20)
    # Anchor-feature LR / regularization
    parser.add_argument("--feat_lr_scale", type=float, default=0.25,
                        help="Multiplier on _anchor_feat LR (1.0 = legacy)")
    parser.add_argument("--feat_reg_weight", type=float, default=0.05,
                        help="MSE-to-init regularization weight on seeded _anchor_feat")
    # Novel-view confidence + spatial cage
    parser.add_argument("--silhouette_iou_thresh", type=float, default=0.2,
                        help="Drop novel views with mask-vs-AABB IoU below this")
    parser.add_argument("--cage_padding_frac", type=float, default=0.05,
                        help="Absolute |offset| cap as fraction of object's largest extent")
    parser.add_argument("--hole_weight_max", type=float, default=2.5,
                        help="Cap on hole-fill loss weight multiplier")
    parser.add_argument("--seeded_anisotropy_max", type=float, default=3.0,
                        help="Max ratio of largest/smallest gaussian scale axis "
                             "for seeded anchors. Caps radial spike artifacts "
                             "from single-view depth ambiguity. Set <=1 to disable.")
    parser.add_argument("--train_mlp_opacity", action="store_true",
                        help="Adapt ObjectGS mlp_opacity at a tiny LR for seeded backside OOD feature/view pairs")
    parser.add_argument("--mlp_opacity_lr_scale", type=float, default=0.001,
                        help="LR multiplier for mlp_opacity when --train_mlp_opacity is enabled")
    parser.add_argument("--mlp_opacity_reg_weight", type=float, default=1.0,
                        help="MSE-to-initial-weights regularization for mlp_opacity adaptation")
    parser.add_argument("--no_freeze_originals", action="store_true",
                        help="Allow original-object anchors to also be fine-tuned by Zero123++ supervision (default: frozen)")
    parser.add_argument("--repair_diagnostics", action="store_true",
                        help="Run read-only candidate-view verification diagnostics before seeding/fine-tune")
    parser.add_argument("--repair_diagnostics_only", action="store_true",
                        help="Run repair diagnostics and skip seeding/fine-tune/model mutation")
    parser.add_argument("--repair_diag_alpha_threshold", type=float, default=0.03,
                        help="Rendered alpha threshold used by repair diagnostics")
    parser.add_argument("--no_repair_filter_supervision", action="store_true",
                        help="Do not use repair diagnostics to filter Zero123++ supervision views")
    parser.add_argument("--repair_filter_min_trust", type=float, default=0.45,
                        help="Minimum diagnostic trust score for inspect_prior views to drive optimization")
    parser.add_argument("--repair_filter_max_outside_alpha", type=float, default=0.20,
                        help="Maximum outside-alpha ratio for inspect_prior views to drive optimization")
    parser.add_argument("--repair_filter_max_missing_ratio", type=float, default=0.55,
                        help="Maximum target-mask missing ratio allowed for views to drive optimization")
    parser.add_argument("--repair_filter_min_target_iou", type=float, default=0.30,
                        help="Minimum target/render IoU allowed for views to drive optimization")
    parser.add_argument("--repair_filter_min_views", type=int, default=2,
                        help="Minimum number of accepted diagnostic views required before mutating the model")
    parser.add_argument("--no_repair_filter_inspect_prior", action="store_true",
                        help="Only allow usable_prior views through the repair diagnostic filter")
    parser.add_argument("--aligned_repair_candidates", action="store_true",
                        help="Run the read-only aligned repair-candidate stage (current-render-aligned conservative repair masks)")
    parser.add_argument("--aligned_repair_candidates_only", action="store_true",
                        help="Run aligned repair-candidate stage and skip seeding/fine-tune/model mutation")
    parser.add_argument("--aligned_repair_support_dilate_px", type=int, default=12,
                        help="Pixels of dilation around the current render mask defining the support zone for repair components")
    parser.add_argument("--aligned_repair_min_component_px", type=int, default=32,
                        help="Minimum connected-component size for a missing region to be considered a repair candidate")
    parser.add_argument("--aligned_repair_max_components", type=int, default=6,
                        help="Maximum number of repair components kept per view (largest by area)")
    parser.add_argument("--aligned_repair_max_area_ratio", type=float, default=0.40,
                        help="Reject views whose total repair-mask area exceeds this fraction of the target-mask area")
    parser.add_argument("--aligned_repair_min_target_render_iou", type=float, default=0.20,
                        help="Minimum target/render IoU required for a view to be considered for aligned repair")
    parser.add_argument("--azimuth_sign", type=int, choices=[-1, 1], default=-1,
                        help="Azimuth rotation sign for novel cameras. Default -1 matches Zero123++ v1.2 convention.")
    parser.add_argument("--elevation_sign", type=int, choices=[-1, 1], default=1,
                        help="Elevation sign for novel cameras. Default +1 with camera-local-up consensus matches Zero123++ v1.2.")
    args = parser.parse_args()

    run_replenishment(
        model_path=args.model_path,
        output_dir=args.output_dir,
        iteration=args.iteration,
        target_object_ids=args.object_ids,
        up_axis=args.up_axis,
        finetune_iterations=args.finetune_iters,
        finetune_lr_scale=args.lr_scale,
        hallucination_weight=args.hallucination_weight,
        novel_rgb_weight=args.novel_rgb_weight,
        target_mask_erode_px=args.target_mask_erode_px,
        freeze_feat_when_rgb_off=not args.no_freeze_feat_when_rgb_off,
        conservative_seed_render=not args.legacy_seed_render,
        visual_hull_seed_filter=not args.no_visual_hull_seed_filter,
        visual_hull_min_views=args.visual_hull_min_views,
        surface_shell_seed_filter=not args.no_surface_shell_seed_filter,
        surface_shell_min_norm=args.surface_shell_min_norm,
        seed_opacity_gate_init=args.seed_opacity_gate_init,
        seed_opacity_gate_lr_scale=args.seed_opacity_gate_lr_scale,
        seed_opacity_gate_reg_weight=args.seed_opacity_gate_reg_weight,
        seed_opacity_lift_init=args.seed_opacity_lift_init,
        seed_opacity_lift_lr_scale=args.seed_opacity_lift_lr_scale,
        seed_opacity_lift_reg_weight=args.seed_opacity_lift_reg_weight,
        seed_opacity_accept_threshold=args.seed_opacity_accept_threshold,
        diffusion_steps=args.diffusion_steps,
        seed=args.seed,
        auto_compare=not args.no_auto_compare,
        comparison_views=args.comparison_views,
        offset_scale_frac=args.offset_scale_frac,
        seeded_scale_max_frac=args.seeded_scale_max_frac,
        bounds_expand_frac=args.bounds_expand_frac,
        originals_lr_scale=args.originals_lr_scale,
        originals_max_scale_delta=args.originals_max_scale_delta,
        originals_max_offset_delta=args.originals_max_offset_delta,
        originals_reg_weight=args.originals_reg_weight,
        seeded_scale_reg_weight=args.seeded_scale_reg_weight,
        seeded_offset_reg_weight=args.seeded_offset_reg_weight,
        seeded_max_scale_delta=args.seeded_max_scale_delta,
        seeded_max_offset_delta=args.seeded_max_offset_delta,
        feat_lr_scale=args.feat_lr_scale,
        feat_reg_weight=args.feat_reg_weight,
        silhouette_iou_thresh=args.silhouette_iou_thresh,
        cage_padding_frac=args.cage_padding_frac,
        hole_weight_max=args.hole_weight_max,
        seeded_anisotropy_max=args.seeded_anisotropy_max,
        train_mlp_opacity=args.train_mlp_opacity,
        mlp_opacity_lr_scale=args.mlp_opacity_lr_scale,
        mlp_opacity_reg_weight=args.mlp_opacity_reg_weight,
        freeze_originals=not args.no_freeze_originals,
        repair_diagnostics=args.repair_diagnostics,
        repair_diagnostics_only=args.repair_diagnostics_only,
        repair_diag_alpha_threshold=args.repair_diag_alpha_threshold,
        repair_filter_supervision=not args.no_repair_filter_supervision,
        repair_filter_min_trust=args.repair_filter_min_trust,
        repair_filter_max_outside_alpha=args.repair_filter_max_outside_alpha,
        repair_filter_max_missing_ratio=args.repair_filter_max_missing_ratio,
        repair_filter_min_target_iou=args.repair_filter_min_target_iou,
        repair_filter_min_views=args.repair_filter_min_views,
        repair_filter_allow_inspect_prior=not args.no_repair_filter_inspect_prior,
        aligned_repair_candidates=args.aligned_repair_candidates,
        aligned_repair_candidates_only=args.aligned_repair_candidates_only,
        aligned_repair_support_dilate_px=args.aligned_repair_support_dilate_px,
        aligned_repair_min_component_px=args.aligned_repair_min_component_px,
        aligned_repair_max_components=args.aligned_repair_max_components,
        aligned_repair_max_area_ratio=args.aligned_repair_max_area_ratio,
        aligned_repair_min_target_render_iou=args.aligned_repair_min_target_render_iou,
        azimuth_sign=args.azimuth_sign,
        elevation_sign=args.elevation_sign,
    )


if __name__ == "__main__":
    main()
