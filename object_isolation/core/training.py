"""Phase 7 — Replenishment training using Phase-5 SV3D supervision.

Wraps ``target_replenishment.core.{anchor_seeding,optimizer}`` so the same
seed-and-fine-tune machinery used by the Zero123++ pipeline is driven by
Phase-5 SV3D outputs instead.

Key differences vs ``target_replenishment.run_replenishment``:
- Skips ``coverage_analyzer.analyze_coverage`` (we already know the gap
  hemisphere from Phase 1's ``base_dir_W``).
- Skips Zero123++ generation, ``compute_novel_cameras``, and the
  Zero123++-specific bbox alignment workaround.
- Uses ``object_isolation.core.dataset_builder.build_supervision_views``
  for the supervision list.
- Mutates the shared parent ObjectGS model in-place (same as the existing
  pipeline) and saves the merged result for Phase 8 reintegration.

Output layout (per object_id):
    <output_dir>/obj_<id>/
        supervision_manifest.json   # Phase-6 dump (camera metadata only)
        replenishment_summary.json  # subset of optimizer return dict
        model/
            point_cloud.ply
            color_mlp.pt cov_mlp.pt opacity_mlp.pt
            replenishment.json

The orchestrator (``run_phase678.py``) is responsible for saving the final
merged scene-level model after all objects have been processed.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


def _ensure_paths():
    """Make sure target_replenishment is importable."""
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


def run_phase7(
    *,
    model_path: str,
    object_label_id: int,
    halluc_index_path: str | Path,
    output_dir: str | Path,
    gaussians=None,
    pipe_config=None,
    scope=None,
    local_sv3d=None,
    finetune_iterations: int = 1200,
    finetune_lr_scale: float = 1.0,
    hallucination_weight: float = 0.10,
    novel_rgb_weight: float = 1.0,
    target_mask_erode_px: int = 0,
    freeze_originals: bool = True,
    grid_resolution: int = 25,
    seed_opacity_gate_init: float = 0.02,
    seed_opacity_gate_lr_scale: float = 50.0,
    seed_opacity_gate_reg_weight: float = 0.005,
    seed_opacity_lift_init: float = 0.0,
    seed_opacity_lift_lr_scale: float = 10.0,
    seed_opacity_lift_reg_weight: float = 0.02,
    seed_opacity_accept_threshold: float = 0.10,
    seeded_scale_max_frac: float = 0.06,
    seeded_scale_reg_weight: float = 0.20,
    seeded_offset_reg_weight: float = 0.20,
    seeded_max_scale_delta: float = 0.20,
    seeded_max_offset_delta: float = 0.20,
    originals_lr_scale: float = 0.05,
    originals_max_scale_delta: float = 0.05,
    originals_max_offset_delta: float = 0.05,
    originals_reg_weight: float = 0.5,
    feat_lr_scale: float = 0.25,
    feat_reg_weight: float = 0.05,
    cage_padding_frac: float = 0.02,
    silhouette_iou_thresh: float = 0.20,
    hole_weight_max: float = 2.5,
    seeded_anisotropy_max: float = 3.0,
    visual_hull_min_views: int = 2,
    surface_shell_min_norm: float = 0.65,
    use_cond_cam_up: bool = True,
    fov_y_deg: float = 50.0,
) -> dict:
    """Run anchor seeding + optimizer fine-tune for one object.

    Either pass pre-loaded ``gaussians/pipe_config/scope/local_sv3d`` or
    omit them — they will be (re-)discovered from ``model_path``.

    Returns a summary dict (also written to disk).
    """
    _ensure_paths()

    from target_replenishment.core.objectgs_bridge import (
        load_gaussians, get_anchor_positions,
    )
    from target_replenishment.core.anchor_seeding import seed_backside_anchors
    from target_replenishment.core.optimizer import optimize_with_novel_views

    from .scope import discover_object_scope
    from .dataset_builder import build_supervision_views, save_supervision_manifest

    out_dir = Path(output_dir)
    obj_dir = out_dir / f"obj_{int(object_label_id)}"
    (obj_dir / "model").mkdir(parents=True, exist_ok=True)

    # ── Load (or reuse) parent model + scope ──────────────────────────────
    if gaussians is None or pipe_config is None or scope is None or local_sv3d is None:
        logger.info("Phase 7: rediscovering scope for obj %d at %s",
                    object_label_id, model_path)
        scope, _world_local, local_sv3d, gaussians, pipe_config = discover_object_scope(
            model_path, int(object_label_id),
        )

    # ── Pull cond cam up (matches Phase 5 reference renders) ──────────────
    cond_cam_up_W: Optional[np.ndarray] = None
    cond_cam_idx: Optional[int] = None
    try:
        with open(halluc_index_path) as f:
            manifest = json.load(f)
        cond_cam_idx = int(manifest.get("conditioning", {}).get("cam_index", -1))
        if use_cond_cam_up and cond_cam_idx >= 0 and cond_cam_idx < len(scope.cameras):
            R_cond = np.asarray(scope.cameras[cond_cam_idx]["R"], dtype=np.float64)
            cond_cam_up_W = -R_cond[1]  # camera up in world = -row1 of R_w2c
            ang = float(np.degrees(np.arccos(np.clip(
                cond_cam_up_W @ scope.up_W /
                (np.linalg.norm(cond_cam_up_W) * max(np.linalg.norm(scope.up_W), 1e-9)),
                -1.0, 1.0,
            ))))
            logger.info("Phase 7: using cond cam %d up (%.2f deg from scope.up_W).",
                        cond_cam_idx, ang)
    except Exception as e:
        logger.warning("Could not read cond cam up from halluc_index (%s); "
                       "falling back to scope.up_W.", e)
        cond_cam_up_W = None

    # ── Phase 6: build supervision views ──────────────────────────────────
    supervision_views = build_supervision_views(
        halluc_index_path=halluc_index_path,
        local_sv3d=local_sv3d,
        weight=hallucination_weight,
        fov_y_deg=fov_y_deg,
        target_resolution=576,
        up_W_override=cond_cam_up_W,
    )
    if not supervision_views:
        raise RuntimeError(f"Phase 6 produced no supervision views for obj {object_label_id}.")

    save_supervision_manifest(supervision_views, obj_dir / "supervision_manifest.json")
    logger.info("Phase 7: %d supervision views queued for obj %d.",
                len(supervision_views), object_label_id)

    # ── Phase 7a: seeding ─────────────────────────────────────────────────
    n_original_anchors = int(gaussians._anchor.shape[0])
    anchor_xyz_global = get_anchor_positions(gaussians)
    labels = gaussians.label_ids.squeeze(-1).cpu().numpy()
    obj_mask = (labels == int(object_label_id))
    n_obj_anchors = int(obj_mask.sum())
    object_anchors = anchor_xyz_global[obj_mask]

    # view_dir = unit vector FROM object centre TO best (cond) camera.
    if cond_cam_idx is not None and cond_cam_idx >= 0 and cond_cam_idx < len(scope.cameras):
        cam_pos = np.asarray(scope.cameras[cond_cam_idx]["position"], dtype=np.float64)
    else:
        cam_pos = scope.cam_centers_visible_W.mean(axis=0)
    view_dir = cam_pos - np.asarray(scope.centroid_W, dtype=np.float64)
    view_dir = view_dir / max(np.linalg.norm(view_dir), 1e-9)

    object_radius = float(scope.radius)
    object_center = np.asarray(scope.centroid_W, dtype=np.float32)

    logger.info("Phase 7: seeding obj %d (n_orig_anchors=%d, n_obj_anchors=%d, radius=%.3f)",
                object_label_id, n_original_anchors, n_obj_anchors, object_radius)

    n_seeded = seed_backside_anchors(
        gaussians=gaussians,
        object_center=object_center,
        view_direction=view_dir.astype(np.float32),
        object_id=int(object_label_id),
        grid_resolution=int(grid_resolution),
        scale_max_frac=float(seeded_scale_max_frac),
        visual_hull_min_views=int(visual_hull_min_views),
        surface_shell_min_norm=float(surface_shell_min_norm),
    )
    logger.info("Phase 7: seeded %d new anchors for obj %d.", n_seeded, object_label_id)

    # ── Phase 7b: trainable masks ─────────────────────────────────────────
    n_total = int(gaussians._anchor.shape[0])
    seeded_mask_np = np.zeros(n_total, dtype=bool)
    if n_seeded > 0:
        seeded_mask_np[n_original_anchors:n_original_anchors + n_seeded] = True

    labels_post = gaussians.label_ids.squeeze(-1).cpu().numpy()
    obj_mask_post = (labels_post == int(object_label_id))
    seeded_mask_np &= obj_mask_post

    if freeze_originals:
        originals_mask_np = np.zeros_like(obj_mask_post, dtype=bool)
        update_mask_np = seeded_mask_np.copy()
        if not update_mask_np.any():
            logger.warning("Phase 7: no seeds for obj %d and freeze_originals=True; "
                           "skipping fine-tune.", object_label_id)
            summary = {
                "object_id": int(object_label_id),
                "n_seeded_anchors": 0,
                "skipped_reason": "no_seeds_with_freeze_originals",
            }
            with open(obj_dir / "replenishment_summary.json", "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            return summary
    else:
        originals_mask_np = obj_mask_post & (~seeded_mask_np)
        update_mask_np = seeded_mask_np | originals_mask_np

    anchor_update_mask = torch.from_numpy(update_mask_np).to(device="cuda", dtype=torch.bool)
    seeded_anchor_mask = torch.from_numpy(seeded_mask_np).to(device="cuda", dtype=torch.bool)
    originals_anchor_mask = torch.from_numpy(originals_mask_np).to(device="cuda", dtype=torch.bool)

    # ── Seed opacity gate / lift parameters (mirror run_replenishment) ────
    if n_seeded > 0:
        if finetune_iterations <= 0 or seed_opacity_gate_init <= 0.0:
            gaussians.replenishment_seed_opacity_gate = torch.zeros(
                n_seeded, 1, dtype=torch.float32, device="cuda"
            )
        else:
            gate_init = float(np.clip(seed_opacity_gate_init, 1e-5, 0.95))
            gate_logit = float(np.log(gate_init / (1.0 - gate_init)))
            gaussians.replenishment_seed_opacity_logit = torch.nn.Parameter(
                torch.full((n_seeded, 1), gate_logit, dtype=torch.float32, device="cuda")
            )

        if finetune_iterations > 0 and seed_opacity_lift_init > 0.0:
            gaussians.replenishment_seed_opacity_lift = torch.nn.Parameter(
                torch.full(
                    (n_seeded, int(gaussians.n_offsets)),
                    float(seed_opacity_lift_init),
                    dtype=torch.float32, device="cuda",
                )
            )

    # ── AABB / scale ceiling from seeding ─────────────────────────────────
    aabb_min = aabb_max = None
    scale_ceiling_log = None
    if hasattr(gaussians, "_replenishment_aabb") and int(object_label_id) in gaussians._replenishment_aabb:
        ameta = gaussians._replenishment_aabb[int(object_label_id)]
        aabb_min = ameta["min"].detach().cpu().numpy()
        aabb_max = ameta["max"].detach().cpu().numpy()
        extent_med = float(ameta.get("extent_med", ameta["extent_max"]))
        scale_ceiling_log = float(np.log(max(extent_med * float(seeded_scale_max_frac), 1e-6)))

    # ── Phase 7c: optimizer ───────────────────────────────────────────────
    logger.info("Phase 7: fine-tuning obj %d for %d iters.",
                object_label_id, finetune_iterations)
    opt_result = optimize_with_novel_views(
        gaussians, pipe_config,
        supervision_views,
        n_iterations=int(finetune_iterations),
        lr_scale=float(finetune_lr_scale),
        hallucination_weight=float(hallucination_weight),
        novel_rgb_weight=float(novel_rgb_weight),
        target_mask_erode_px=int(max(0, target_mask_erode_px)),
        object_id=int(object_label_id),
        object_anchors=object_anchors,
        object_radius=object_radius,
        object_center=object_center,
        silhouette_weight=max(2.5, float(hallucination_weight) * 4.0),
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
        save_path=str(obj_dir / "model"),
        reference_model_path=model_path,
    )

    # ── Apply opacity gate acceptance threshold (drop low-conf seeds) ─────
    accepted_seed_count = int(n_seeded)
    if n_seeded > 0:
        with torch.no_grad():
            if hasattr(gaussians, "replenishment_seed_opacity_logit"):
                raw_gates = torch.sigmoid(gaussians.replenishment_seed_opacity_logit.detach())
            elif hasattr(gaussians, "replenishment_seed_opacity_gate"):
                raw_gates = gaussians.replenishment_seed_opacity_gate.detach()
            else:
                raw_gates = torch.ones(n_seeded, 1, dtype=torch.float32, device="cuda")
            raw_gates = raw_gates.reshape(n_seeded, 1).clamp(0.0, 1.0)
            thr = float(seed_opacity_accept_threshold)
            if thr > 0.0:
                final_gates = torch.where(raw_gates >= thr, raw_gates,
                                          torch.zeros_like(raw_gates))
            else:
                final_gates = raw_gates.clone()
            gaussians.replenishment_seed_opacity_gate = final_gates
            if hasattr(gaussians, "replenishment_seed_opacity_logit"):
                delattr(gaussians, "replenishment_seed_opacity_logit")
            accepted_seed_count = int((final_gates > 0.0).sum().item())

    # ── Save per-object snapshot (mirrors target_replenishment layout) ────
    try:
        gaussians.save_ply(str(obj_dir / "model" / "point_cloud.ply"))
        gaussians.save_mlp_checkpoints(str(obj_dir / "model"))
    except Exception as e:
        logger.error("Phase 7: failed to save per-object model: %s", e)

    summary = {
        "object_id": int(object_label_id),
        "n_supervision_views": len(supervision_views),
        "n_original_anchors": int(n_original_anchors),
        "n_obj_anchors_pre": int(n_obj_anchors),
        "n_seeded_anchors": int(n_seeded),
        "n_seeded_accepted": int(accepted_seed_count),
        "seed_opacity_accept_threshold": float(seed_opacity_accept_threshold),
        "final_loss": float(opt_result.get("final_loss", 0.0)),
        "loss_history": opt_result.get("loss_history", [])[:50],  # truncate
        "supervision_diagnostics": opt_result.get("supervision_diagnostics", []),
        "param_delta_norms": opt_result.get("param_delta_norms", {}),
        "halluc_index_path": str(halluc_index_path),
        "model_path": str(model_path),
    }

    # Write replenishment.json (per-object metadata) for Phase 8 to inspect.
    rep_payload = {
        "object_id": int(object_label_id),
        "n_original_anchors": int(n_original_anchors),
        "n_seeded": int(n_seeded),
        "n_seeded_accepted": int(accepted_seed_count),
        "seed_opacity_accept_threshold": float(seed_opacity_accept_threshold),
        "view_direction": view_dir.tolist(),
        "object_center": object_center.tolist(),
        "object_radius": object_radius,
    }
    with open(obj_dir / "model" / "replenishment.json", "w", encoding="utf-8") as f:
        json.dump(rep_payload, f, indent=2)

    with open(obj_dir / "replenishment_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(
        "Phase 7 complete for obj %d: seeded=%d accepted=%d final_loss=%.5f",
        object_label_id, n_seeded, accepted_seed_count, summary["final_loss"],
    )
    return summary
