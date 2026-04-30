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
    from target_replenishment.core.anchor_seeding import seed_backside_anchors
    from target_replenishment.core.repair_diagnostics import analyze_repair_candidates
    from target_replenishment.core.repair_candidate_stage import analyze_aligned_repair_candidates

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
                supervision_views, repair_filter_result = _filter_supervision_by_repair_diagnostics(
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
            visual_hull_constraints = _build_visual_hull_seed_constraints(
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


def _filter_supervision_by_repair_diagnostics(
    supervision_views: list,
    repair_diag_result: dict,
    min_trust: float = 0.45,
    max_outside_alpha_ratio: float = 0.20,
    max_missing_ratio: float = 0.55,
    min_target_render_iou: float = 0.30,
    min_kept_views: int = 2,
    allow_inspect_prior: bool = True,
):
    """Keep only candidate views that passed read-only repair diagnostics.

    The filter is intentionally conservative: views flagged as floaters or
    rejected priors are not allowed to drive seeding/optimization. Kept views
    are reweighted by their trust score so marginal priors contribute less.
    """
    view_scores = repair_diag_result.get('view_scores', []) if repair_diag_result else []
    scores_by_index = {
        int(score.get('view_index')): score
        for score in view_scores
        if score.get('view_index') is not None
    }

    kept_views = []
    kept_entries = []
    rejected_entries = []

    for idx, view in enumerate(supervision_views):
        score = scores_by_index.get(idx)
        if score is None:
            rejected_entries.append({
                'view_index': int(idx),
                'reason': 'missing_repair_diagnostic_score',
            })
            continue

        recommendation = str(score.get('recommendation', ''))
        trust_score = float(score.get('trust_score', 0.0))
        outside_alpha_ratio = float(score.get('outside_alpha_ratio', 1.0))
        missing_ratio = float(score.get('missing_ratio', 1.0))
        target_render_iou = float(score.get('target_render_iou', 0.0))

        metric_ok = (
            trust_score >= float(min_trust)
            and outside_alpha_ratio <= float(max_outside_alpha_ratio)
            and missing_ratio <= float(max_missing_ratio)
            and target_render_iou >= float(min_target_render_iou)
        )
        keep = recommendation == 'usable_prior' and metric_ok
        if allow_inspect_prior and recommendation == 'inspect_prior':
            keep = metric_ok

        entry = {
            'view_index': int(idx),
            'azimuth_offset_deg': score.get('azimuth_offset_deg'),
            'elevation_offset_deg': score.get('elevation_offset_deg'),
            'recommendation': recommendation,
            'trust_score': trust_score,
            'outside_alpha_ratio': outside_alpha_ratio,
            'missing_ratio': missing_ratio,
            'target_render_iou': target_render_iou,
        }

        if keep:
            filtered_view = dict(view)
            original_weight = float(filtered_view.get('weight', 1.0))
            filtered_view['weight'] = original_weight * max(trust_score, 1e-3)
            filtered_view['repair_diagnostic'] = entry
            entry['original_weight'] = original_weight
            entry['filtered_weight'] = float(filtered_view['weight'])
            kept_views.append(filtered_view)
            kept_entries.append(entry)
        else:
            if recommendation in {'inspect_floaters', 'reject_prior'}:
                reason = recommendation
            elif trust_score < float(min_trust):
                reason = 'below_min_trust'
            elif outside_alpha_ratio > float(max_outside_alpha_ratio):
                reason = 'above_max_outside_alpha'
            elif missing_ratio > float(max_missing_ratio):
                reason = 'above_max_missing_ratio'
            elif target_render_iou < float(min_target_render_iou):
                reason = 'below_min_target_render_iou'
            else:
                reason = 'not_allowed_by_filter'
            entry['reason'] = reason
            rejected_entries.append(entry)

    min_kept_views = max(0, int(min_kept_views))
    if len(kept_views) < min_kept_views:
        for entry in kept_entries:
            rejected = dict(entry)
            rejected['reason'] = 'below_min_filtered_view_count'
            rejected_entries.append(rejected)
        kept_views = []
        kept_entries = []

    filter_result = {
        'enabled': True,
        'raw_view_count': int(len(supervision_views)),
        'kept_view_count': int(len(kept_views)),
        'rejected_view_count': int(len(rejected_entries)),
        'min_trust': float(min_trust),
        'max_outside_alpha_ratio': float(max_outside_alpha_ratio),
        'max_missing_ratio': float(max_missing_ratio),
        'min_target_render_iou': float(min_target_render_iou),
        'min_kept_views': int(min_kept_views),
        'allow_inspect_prior': bool(allow_inspect_prior),
        'kept_view_indices': [int(entry['view_index']) for entry in kept_entries],
        'kept_views': kept_entries,
        'rejected_views': rejected_entries,
    }
    return kept_views, filter_result


def _save_image(img: np.ndarray, path: Path):
    """Save an image to disk."""
    if img.ndim == 3 and img.shape[2] == 3:
        cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    else:
        cv2.imwrite(str(path), img)


def _save_coverage_plot(coverage, path: Path):
    """Save a simple coverage histogram visualization."""
    n_bins = len(coverage.coverage_map)
    H, W = 200, 400
    img = np.ones((H, W, 3), dtype=np.uint8) * 255

    bar_w = W // n_bins
    for i, val in enumerate(coverage.coverage_map):
        bar_h = int(val * (H - 20))
        x0 = i * bar_w
        x1 = x0 + bar_w - 1
        y0 = H - 10 - bar_h
        y1 = H - 10

        # Green = covered, red = gap
        color = (0, 180, 0) if val >= 0.1 else (0, 0, 200)
        cv2.rectangle(img, (x0, y0), (x1, y1), color, -1)

    # Mark input camera azimuth
    input_bin = int((coverage.input_azimuth + np.pi) / (2 * np.pi) * n_bins)
    input_bin = np.clip(input_bin, 0, n_bins - 1)
    cx = input_bin * bar_w + bar_w // 2
    cv2.circle(img, (cx, 5), 5, (255, 0, 0), -1)

    cv2.imwrite(str(path), img)


def _build_comparison_cameras(center, radius, orbit_radius, up_vector, input_cam_position, width, height, n_views):
    from target_replenishment.render_360 import look_at

    up = up_vector.astype(np.float32)

    # Keep comparison cameras outside the object and near training-view distance.
    dist = max(float(orbit_radius) * 0.9, float(radius) * 2.5, 0.5)

    # Recompute a wider focal length for comparison rendering so object framing is stable.
    angular = 2.0 * np.arctan(float(radius) / max(dist, 1e-6))
    fov = np.clip(angular / 0.55, np.radians(30.0), np.radians(100.0))
    fx = (width / 2.0) / np.tan(fov / 2.0)
    fy = (height / 2.0) / np.tan(fov / 2.0)
    k = np.array([[fx, 0.0, width / 2.0], [0.0, fy, height / 2.0], [0.0, 0.0, 1.0]], dtype=np.float32)

    # Use input camera direction as the orbit start to avoid random extreme angles.
    ref_vec = (input_cam_position.astype(np.float32) - center.astype(np.float32))
    vertical = float(np.dot(ref_vec, up))
    horizontal = ref_vec - vertical * up
    if np.linalg.norm(horizontal) < 1e-6:
        horizontal = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if abs(np.dot(horizontal, up)) > 0.9:
            horizontal = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    basis_h = horizontal / (np.linalg.norm(horizontal) + 1e-8)
    basis_v = np.cross(up, basis_h)
    basis_v = basis_v / (np.linalg.norm(basis_v) + 1e-8)

    # Preserve some elevation from the source camera while clamping extremes.
    z_offset = float(np.clip(vertical, -0.3 * dist, 0.3 * dist))

    cams = []
    for i in range(n_views):
        angle = 2.0 * np.pi * i / n_views
        cam_pos = (
            center
            + dist * np.cos(angle) * basis_h
            + dist * np.sin(angle) * basis_v
            + z_offset * up
        ).astype(np.float32)
        r, t = look_at(cam_pos.astype(np.float32), center.astype(np.float32), up)
        cams.append(
            {
                'index': i,
                'azimuth_deg': float(np.degrees(angle)),
                'cam_pos': cam_pos,
                'R': r,
                'T': t,
                'K': k.copy(),
                'width': width,
                'height': height,
            }
        )
    return cams


def _render_object_with_cameras(gaussians, pipe_config, cameras, object_id):
    from target_replenishment.core.objectgs_bridge import create_virtual_camera, render_view

    bg = torch.ones(3, dtype=torch.float32, device='cuda')
    frames = []
    for cam_data in cameras:
        cam = create_virtual_camera(
            cam_data['R'],
            cam_data['T'],
            cam_data['K'],
            cam_data['width'],
            cam_data['height'],
        )
        res = render_view(gaussians, cam, pipe_config, bg, object_label_id=object_id)
        rgb = (res['rgb'].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        frames.append(rgb)
    return frames


def _build_visual_hull_seed_constraints(supervision_views, object_anchors, erode_px: int = 0):
    """Build coarse foreground masks for pre-seed visual-hull carving.

    Zero123++ outputs are object-centered, while our novel cameras use scene
    intrinsics. Aligning the generated foreground bbox to the projected object
    AABB gives the seeder a rough but useful multi-view support test before it
    appends any renderable anchors.
    """
    from target_replenishment.core.image_alignment import align_image_to_render_bbox

    object_anchors_np = np.asarray(object_anchors, dtype=np.float32)
    if object_anchors_np.size == 0:
        return []

    q_low = np.quantile(object_anchors_np, 0.02, axis=0)
    q_high = np.quantile(object_anchors_np, 0.98, axis=0)
    corners = np.array([
        [q_low[0], q_low[1], q_low[2]],
        [q_high[0], q_low[1], q_low[2]],
        [q_low[0], q_high[1], q_low[2]],
        [q_high[0], q_high[1], q_low[2]],
        [q_low[0], q_low[1], q_high[2]],
        [q_high[0], q_low[1], q_high[2]],
        [q_low[0], q_high[1], q_high[2]],
        [q_high[0], q_high[1], q_high[2]],
    ], dtype=np.float32)

    constraints = []
    for view in supervision_views:
        cam = view.get('camera')
        rgb = np.asarray(view.get('rgb'))
        if cam is None or rgb.ndim != 3:
            continue

        height = int(cam.get('height', rgb.shape[0]))
        width = int(cam.get('width', rgb.shape[1]))
        mask = (rgb.astype(np.float32).mean(axis=2) < 250.0)
        mask = _largest_component_mask_np(mask, min_pixels=64)
        if int(erode_px) > 0 and mask.any():
            kernel_size = 2 * int(erode_px) + 1
            kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
            eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1) > 0
            if eroded.sum() >= 64:
                mask = eroded
        if not mask.any():
            continue

        ref = _project_aabb_bbox_image(cam, corners, height, width)
        if ref is None:
            continue

        target_img = np.ones((rgb.shape[0], rgb.shape[1], 3), dtype=np.uint8) * 255
        target_img[mask] = 0
        aligned = align_image_to_render_bbox(target_img, ref, bg_color=(255, 255, 255))
        aligned_mask = aligned.mean(axis=2) < 250.0
        aligned_mask = _largest_component_mask_np(aligned_mask, min_pixels=64)
        if aligned_mask.shape[:2] != (height, width):
            aligned_mask = cv2.resize(
                aligned_mask.astype(np.uint8),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            ) > 0
        if aligned_mask.sum() >= 64:
            constraints.append({
                'camera': cam,
                'mask': aligned_mask.astype(bool),
                'azimuth_offset_deg': cam.get('azimuth_offset_deg'),
                'elevation_offset_deg': cam.get('elevation_offset_deg'),
            })
    return constraints


def _project_aabb_bbox_image(cam, corners: np.ndarray, height: int, width: int):
    R = np.asarray(cam['R'], dtype=np.float32)
    T = np.asarray(cam['T'], dtype=np.float32).reshape(1, 3)
    K = np.asarray(cam['K'], dtype=np.float32)
    cam_pts = (R @ corners.T).T + T
    z = cam_pts[:, 2]
    valid = z > 1e-4
    if not np.any(valid):
        return None
    u = K[0, 0] * cam_pts[:, 0] / (z + 1e-8) + K[0, 2]
    v = K[1, 1] * cam_pts[:, 1] / (z + 1e-8) + K[1, 2]
    valid &= np.isfinite(u) & np.isfinite(v)
    if not np.any(valid):
        return None
    x0 = int(np.floor(np.clip(np.min(u[valid]), 0, width - 1)))
    x1 = int(np.ceil(np.clip(np.max(u[valid]), 0, width - 1)))
    y0 = int(np.floor(np.clip(np.min(v[valid]), 0, height - 1)))
    y1 = int(np.ceil(np.clip(np.max(v[valid]), 0, height - 1)))
    if x1 <= x0 or y1 <= y0:
        return None
    ref = np.ones((height, width, 3), dtype=np.uint8) * 255
    cv2.rectangle(ref, (x0, y0), (x1, y1), (0, 0, 0), thickness=-1)
    return ref


def _largest_component_mask_np(mask: np.ndarray, min_pixels: int = 16) -> np.ndarray:
    mask_u8 = (np.asarray(mask).astype(np.uint8) > 0).astype(np.uint8)
    if int(mask_u8.sum()) < min_pixels:
        return mask_u8.astype(bool)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if n_labels <= 1:
        return mask_u8.astype(bool)
    areas = stats[1:, cv2.CC_STAT_AREA]
    if areas.size == 0 or int(areas.max()) < min_pixels:
        return mask_u8.astype(bool)
    keep_label = 1 + int(np.argmax(areas))
    return labels == keep_label


def _save_camera_metadata(path, object_id, center, radius, cameras):
    payload = {
        'object_id': int(object_id),
        'object_center': np.asarray(center, dtype=np.float32).tolist(),
        'object_radius': float(radius),
        'n_views': len(cameras),
        'cameras': [
            {
                'index': c['index'],
                'azimuth_deg': c['azimuth_deg'],
                'cam_pos': np.asarray(c['cam_pos'], dtype=np.float32).tolist(),
                'R': np.asarray(c['R'], dtype=np.float32).tolist(),
                'T': np.asarray(c['T'], dtype=np.float32).tolist(),
                'K': np.asarray(c['K'], dtype=np.float32).tolist(),
                'width': int(c['width']),
                'height': int(c['height']),
            }
            for c in cameras
        ],
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)


def _save_auto_comparison(before_frames, after_frames, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    diffs = []
    for i, (before_rgb, after_rgb) in enumerate(zip(before_frames, after_frames)):
        _save_image(before_rgb, out_dir / f"before_view_{i}.png")
        _save_image(after_rgb, out_dir / f"after_view_{i}.png")

        compare = np.hstack([before_rgb.copy(), after_rgb.copy()])
        cv2.putText(compare, 'BEFORE', (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
        cv2.putText(compare, 'AFTER', (before_rgb.shape[1] + 12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        _save_image(compare, out_dir / f"compare_view_{i}.png")

        abs_diff = np.abs(after_rgb.astype(np.int16) - before_rgb.astype(np.int16)).astype(np.uint8)
        diff_gray = np.mean(abs_diff, axis=2).astype(np.uint8)
        boosted = np.clip(diff_gray.astype(np.float32) * 4.0, 0, 255).astype(np.uint8)
        diff_heat = cv2.applyColorMap(boosted, cv2.COLORMAP_JET)
        cv2.imwrite(str(out_dir / f"diff_view_{i}.png"), diff_heat)

        diffs.append(float(diff_gray.mean()))

    return {
        'n_views': len(diffs),
        'mean_abs_diff': float(np.mean(diffs)) if diffs else 0.0,
        'max_abs_diff': float(np.max(diffs)) if diffs else 0.0,
    }


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