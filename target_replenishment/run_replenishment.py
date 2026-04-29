"""
VRoom Target Replenishment — Deterministic Shrinkwrap Runner.

Per-object pipeline:
    A. Surface extraction + render-space + AABB floater pruning.
    B. Directional 6-sided projected height-field cap seeding.
    C. KNN normal-aware direct-drive seed init (originals frozen).
    D. Post-seed render-space + 3D component floater elimination.
    E. Save before/after diagnostics + per-object summary.

CLI:
    python target_replenishment/run_replenishment.py \
        --model_path <ObjectGS run dir> \
        --object_id 9 \
        --output_dir replenished_output
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

# Path bootstrap
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
_OBJECTGS_DIR = _PROJECT_ROOT / "temp_deps" / "ObjectGS"
if str(_OBJECTGS_DIR) not in sys.path:
    sys.path.insert(0, str(_OBJECTGS_DIR))

from target_replenishment.core.objectgs_bridge import (   # noqa: E402
    load_gaussians, get_anchor_positions, get_label_ids, create_virtual_camera,
)
from target_replenishment.core.surface_extraction import extract_dense_surface   # noqa: E402
from target_replenishment.core.directional_shrinkwrap import (                   # noqa: E402
    build_directional_seeds, save_side_scan_pngs,
)
from target_replenishment.core.knn_initializer import (                          # noqa: E402
    knn_direct_drive_init, prune_target_object_floaters, prune_anchor_indices,
    find_3d_floater_anchors,
)
from target_replenishment.core.render_floater_pruning import find_render_space_floaters  # noqa: E402
from target_replenishment.core import diagnostics as diag                        # noqa: E402

logger = logging.getLogger("target_replenishment")


# ───────────────────────────────────────────────────────────────────────────
def _save_object_model(gaussians, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    pc_dir = out_dir / "point_cloud" / "iteration_final"
    pc_dir.mkdir(parents=True, exist_ok=True)
    gaussians.save_ply(str(pc_dir / "point_cloud.ply"))
    if hasattr(gaussians, "save_mlp_checkpoints"):
        gaussians.save_mlp_checkpoints(str(pc_dir))


def _persist_replenishment_metadata(out_dir: Path, payload: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "replenishment.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _copy_side_files(model_path: Path, out_dir: Path):
    """Copy cameras.json + config.yaml so downstream tools can re-load."""
    import shutil
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("cameras.json", "config.yaml"):
        src = model_path / name
        if src.exists():
            shutil.copy2(src, out_dir / name)


def _render_auto_compare_set(
    gaussians, pipe_config, object_id: int, cams: list,
    out_dir: Path, prefix: str,
):
    paths = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, cam in enumerate(cams):
        path = out_dir / f"{prefix}_{idx:03d}.png"
        diag.render_and_save(
            gaussians, pipe_config, cam, path,
            bg_white=True, object_label_id=object_id,
        )
        paths.append(path)
    return paths


def _finish_auto_compare_set(auto_dir: Path, before_paths: list[Path], after_paths: list[Path]):
    compare_paths = []
    diff_means = []
    for idx, (before_path, after_path) in enumerate(zip(before_paths, after_paths)):
        before = cv2.cvtColor(cv2.imread(str(before_path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        after = cv2.cvtColor(cv2.imread(str(after_path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        compare_path = auto_dir / f"compare_{idx:03d}.png"
        diff_path = auto_dir / f"diff_{idx:03d}.png"
        diag.make_compare(before, after, compare_path)
        diff_means.append(diag.make_diff(before, after, diff_path))
        compare_paths.append(compare_path)
    diag.make_contact_sheet(compare_paths, auto_dir / "compare_contact_sheet.png", columns=2)
    return diff_means


def _make_stage_compare_sheet(auto_dir: Path, stage_names: list[str], stage_paths: dict[str, list[Path]]):
    """Write per-view horizontal stage strips and a contact sheet.

    Each strip is: before | after_prune | after_seed | after, using identical
    camera poses. This makes pruning and seeding effects visually separable.
    """
    strip_paths = []
    n_views = min((len(stage_paths.get(name, [])) for name in stage_names), default=0)
    for idx in range(n_views):
        imgs = []
        for name in stage_names:
            img = cv2.imread(str(stage_paths[name][idx]), cv2.IMREAD_COLOR)
            if img is None:
                break
            cv2.putText(
                img,
                name.upper(),
                (12, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )
            imgs.append(img)
        if len(imgs) != len(stage_names):
            continue
        h = max(im.shape[0] for im in imgs)
        padded = []
        for im in imgs:
            if im.shape[0] == h:
                padded.append(im)
                continue
            pad = np.full((h, im.shape[1], 3), 255, np.uint8)
            pad[:im.shape[0], :im.shape[1]] = im
            padded.append(pad)
        strip = np.hstack(padded)
        strip_path = auto_dir / f"stages_{idx:03d}.png"
        cv2.imwrite(str(strip_path), strip)
        strip_paths.append(strip_path)
    diag.make_contact_sheet(strip_paths, auto_dir / "stage_contact_sheet.png", columns=1)


def _estimate_render_median_color(image_paths: list[Path]) -> np.ndarray | None:
    pixels = []
    for path in image_paths:
        img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            continue
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        # Isolated object renders use white background. Keep colored/gray couch
        # pixels and ignore the pure-white canvas plus near-black annotation/noise.
        non_bg = np.linalg.norm(img - 1.0, axis=2) > 0.08
        non_black = img.mean(axis=2) > 0.08
        keep = non_bg & non_black
        if keep.any():
            pixels.append(img[keep])
    if not pixels:
        return None
    all_pixels = np.concatenate(pixels, axis=0)
    if all_pixels.shape[0] > 100_000:
        step = max(1, all_pixels.shape[0] // 100_000)
        all_pixels = all_pixels[::step]
    return np.median(all_pixels, axis=0).astype(np.float32)


# ───────────────────────────────────────────────────────────────────────────
def run_object(
    gaussians,
    pipe_config,
    object_id: int,
    args,
    obj_out_dir: Path,
    diag_cam_template=None,
    auto_compare_template=None,
    cam_centers: np.ndarray | None = None,
    scene_up: np.ndarray | None = None,
):
    obj_out_dir.mkdir(parents=True, exist_ok=True)

    label_ids = get_label_ids(gaussians)
    obj_mask = (label_ids == object_id)
    obj_global_idx = np.where(obj_mask)[0].astype(np.int64)
    if obj_global_idx.size == 0:
        logger.warning("Object %s has no anchors; skipping.", object_id)
        return {"object_id": int(object_id), "skipped": True}

    object_xyz = get_anchor_positions(gaussians)[obj_global_idx]
    logger.info("[obj %s] %d original anchors", object_id, obj_global_idx.size)

    # ── Stage A ────────────────────────────────────────────────────────────
    surf = extract_dense_surface(
        object_xyz,
        knn_k=args.knn_k_surface,
        iso_factor=args.floater_iso_factor,
        edge_factor=args.floater_edge_factor,
        min_component_frac=args.min_component_frac,
        min_component_size=args.min_component_size,
    )
    survivor_local_idx = surf.survivor_indices
    survivor_global_idx = obj_global_idx[survivor_local_idx]
    logger.info(
        "[obj %s] Stage A: %d -> %d survivors (extent %.3f -> %.3f)",
        object_id, surf.n_in, surf.n_out,
        float(np.linalg.norm(surf.extent_before)),
        float(np.linalg.norm(surf.extent_after)),
    )
    if surf.n_out < 8:
        logger.warning("[obj %s] too few survivors after Stage A; skipping.", object_id)
        return {"object_id": int(object_id), "stage_a": surf.to_dict(), "skipped": True}

    survivor_xyz = object_xyz[survivor_local_idx]

    auto_compare_cams = []
    if (
        auto_compare_template is not None
        and not args.no_auto_compare
        and args.auto_compare_views > 0
    ):
        auto_compare_cams = diag.build_orbit_cameras(
            survivor_xyz,
            auto_compare_template,
            n_views=args.auto_compare_views,
            start_azimuth_deg=args.diag_azimuth_deg,
            elevation_deg=args.auto_compare_elevation_deg,
            dist_factor=args.auto_compare_dist_factor,
            cam_centers=cam_centers,
            zoom_scale=args.auto_compare_zoom_scale,
            up_vector=scene_up,
        )

    # Raw before diagnostics must happen before any pruning or seeding.
    pre_render = None
    if diag_cam_template is not None:
        pre_render = diag.render_and_save(
            gaussians, pipe_config, diag_cam_template,
            obj_out_dir / "before.png", bg_white=True, object_label_id=object_id,
        )

    auto_compare_dir = obj_out_dir / "auto_compare"
    auto_before_paths = []
    auto_stage_paths = {}
    if auto_compare_cams:
        auto_before_paths = _render_auto_compare_set(
            gaussians, pipe_config, object_id, auto_compare_cams,
            auto_compare_dir, "before",
        )
        auto_stage_paths["before"] = auto_before_paths

    prune = prune_target_object_floaters(
        gaussians=gaussians,
        object_id=int(object_id),
        survivor_global_indices=survivor_global_idx,
        bounds_min=survivor_xyz.min(axis=0),
        bounds_max=survivor_xyz.max(axis=0),
        padding=float(args.prune_aabb_padding_factor * surf.r_med),
    )
    survivor_global_idx = prune.survivor_indices_mapped
    logger.info(
        "[obj %s] Stage A prune (aabb): %d target anchors -> %d kept (deleted %d)",
        object_id, prune.n_target_before, prune.n_target_after, prune.n_pruned,
    )

    render_prune_summary = {"n_pruned": 0}
    if args.render_prune and auto_compare_cams:
        render_prune = find_render_space_floaters(
            gaussians=gaussians,
            pipe_config=pipe_config,
            object_id=int(object_id),
            cameras=auto_compare_cams,
            alpha_threshold=args.render_prune_alpha_threshold,
            close_kernel=args.render_prune_close_kernel,
            min_blob_area_px=args.render_prune_min_blob_area_px,
            vote_threshold=args.render_prune_vote_threshold,
        )
        render_prune_apply = prune_anchor_indices(gaussians, render_prune.prune_indices)
        render_prune_summary = render_prune.to_dict()
        render_prune_summary.update(render_prune_apply)
        logger.info(
            "[obj %s] render-space prune: %d candidates, deleted %d anchors",
            object_id, render_prune.n_candidate_anchors, render_prune_apply["n_pruned"],
        )

    if diag_cam_template is not None:
        diag.render_and_save(
            gaussians, pipe_config, diag_cam_template,
            obj_out_dir / "after_prune.png", bg_white=True, object_label_id=object_id,
        )
    if auto_compare_cams:
        auto_stage_paths["prune"] = _render_auto_compare_set(
            gaussians, pipe_config, object_id, auto_compare_cams,
            auto_compare_dir, "after_prune",
        )

    # ── Diagnostic: silhouette before/after extraction (text only) ─────────
    diag_dir = obj_out_dir / "cleanup_diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    label_ids_after_prune = get_label_ids(gaussians)
    current_target_indices = np.where(label_ids_after_prune == object_id)[0].astype(np.int64)
    current_target_xyz = get_anchor_positions(gaussians)[current_target_indices]
    if current_target_indices.size == 0:
        logger.warning("[obj %s] no target anchors remain after pruning; skipping.", object_id)
        return {
            "object_id": int(object_id),
            "stage_a_surface_extraction": surf.to_dict(),
            "stage_a_prune": prune.to_dict(),
            "render_space_prune": render_prune_summary,
            "skipped": True,
        }

    # ── Stage B ────────────────────────────────────────────────────────────
    ds = build_directional_seeds(
        survivor_xyz=survivor_xyz,
        existing_object_xyz=current_target_xyz,
        r_med=surf.r_med,
        cam_centers=cam_centers,
        cell_size_factor=args.directional_cell_factor,
        depth_percentile=args.directional_depth_percentile,
        cap_offset_factor=args.directional_cap_offset_factor,
        morphological_close_iters=args.directional_close_iters,
        depth_smooth_iters=args.directional_depth_smooth_iters,
        samples_per_cell=args.directional_samples_per_cell,
        min_camera_support=args.directional_min_camera_support,
        min_projected_area_frac=args.directional_min_area_frac,
        min_uncovered_frac=args.directional_min_uncovered_frac,
        existing_coverage_alpha=args.existing_coverage_alpha,
    )
    save_side_scan_pngs(ds, obj_out_dir / "side_scans")
    seed_xyz_for_init = ds.seed_xyz
    seed_sheet_tangent_u = ds.seed_tangent_u
    seed_sheet_tangent_v = ds.seed_tangent_v
    # Per-seed wall normal = u x v (right-handed wall frame). Used for
    # 2DGS-aligned donor selection in Stage C.
    if seed_sheet_tangent_u.shape[0] > 0:
        seed_normal_arr = np.cross(seed_sheet_tangent_u, seed_sheet_tangent_v).astype(np.float32)
        nrm = np.linalg.norm(seed_normal_arr, axis=1, keepdims=True)
        seed_normal_arr = seed_normal_arr / np.maximum(nrm, 1e-8)
    else:
        seed_normal_arr = np.zeros((0, 3), dtype=np.float32)
    bounds_min_for_init = survivor_xyz.min(axis=0).astype(np.float32)
    bounds_max_for_init = survivor_xyz.max(axis=0).astype(np.float32)
    spacing_for_init = ds.cell_size if ds.cell_size > 0 else max(surf.r_med, 1e-4)
    sw_dict = ds.to_dict()
    sw_dict["mode"] = "directional"
    sw_dict["n_seeded"] = int(seed_xyz_for_init.shape[0])
    logger.info(
        "[obj %s] Stage B (directional): cell=%.4f sides_selected=%s seeds=%d",
        object_id, ds.cell_size, ds.selected_side_ids,
        seed_xyz_for_init.shape[0],
    )

    # ── Stage C ────────────────────────────────────────────────────────────
    knn = knn_direct_drive_init(
        gaussians=gaussians,
        object_id=int(object_id),
        seed_xyz=seed_xyz_for_init,
        survivor_global_indices=current_target_indices,
        bounds_min=bounds_min_for_init,
        bounds_max=bounds_max_for_init,
        grid_spacing=spacing_for_init,
        knn_k=args.knn_k_init,
        scale_log_floor_offset=args.scale_log_floor_offset,
        scale_log_ceil_offset=args.scale_log_ceil_offset,
        seed_opacity_lift=args.seed_opacity_lift,
        seed_opacity_gate=args.seed_opacity_gate,
        seed_fixed_opacity=args.seed_fixed_opacity,
        seed_scaling_boost=args.seed_scaling_boost,
        seed_sheet_tangent_u=seed_sheet_tangent_u,
        seed_sheet_tangent_v=seed_sheet_tangent_v,
        seed_sheet_radius_factor=args.seed_sheet_radius_factor,
        seed_normal=seed_normal_arr,
        normal_align_min_cos=args.normal_align_min_cos,
        normal_donor_pool_k=args.normal_donor_pool_k,
    )
    logger.info("[obj %s] Stage C: %d -> %d total anchors (+%d seeds)",
                object_id, knn.n_originals, knn.n_total, knn.n_seeded)

    seed_color_rgb = None
    if knn.n_seeded > 0:
        color_source_paths = auto_stage_paths.get("prune", []) or auto_before_paths
        seed_color_rgb = _estimate_render_median_color(color_source_paths)
        if seed_color_rgb is not None:
            gaussians.replenishment_seed_color_rgb = torch.tensor(
                seed_color_rgb.reshape(1, 3),
                dtype=gaussians._anchor.dtype,
                device=gaussians._anchor.device,
            ).expand(knn.n_seeded, -1).clone()
            logger.info(
                "[obj %s] seed render color override RGB=%s",
                object_id,
                np.round(seed_color_rgb, 3).tolist(),
            )

    if diag_cam_template is not None:
        diag.render_and_save(
            gaussians, pipe_config, diag_cam_template,
            obj_out_dir / "after_seed.png", bg_white=True, object_label_id=object_id,
        )
    if auto_compare_cams:
        auto_stage_paths["seed"] = _render_auto_compare_set(
            gaussians, pipe_config, object_id, auto_compare_cams,
            auto_compare_dir, "after_seed",
        )

    # ── Stage D (post-seed floater pass) ───────────────────────────────────
    # Task 3: aggressive floater elimination after the new shell is in place.
    post_seed_render_prune = {"n_pruned": 0}
    post_seed_3d_prune = {"n_pruned": 0}
    if knn.n_seeded > 0 and auto_compare_cams and args.post_seed_render_prune:
        rfp = find_render_space_floaters(
            gaussians=gaussians,
            pipe_config=pipe_config,
            object_id=int(object_id),
            cameras=auto_compare_cams,
            alpha_threshold=args.render_prune_alpha_threshold,
            close_kernel=args.render_prune_close_kernel,
            min_blob_area_px=args.render_prune_min_blob_area_px,
            vote_threshold=args.post_seed_render_vote_threshold,
        )
        rfa = prune_anchor_indices(gaussians, rfp.prune_indices)
        post_seed_render_prune = {**rfp.to_dict(), **rfa}
        logger.info(
            "[obj %s] post-seed render prune: %d candidates, deleted %d anchors",
            object_id, rfp.n_candidate_anchors, rfa["n_pruned"],
        )
    if knn.n_seeded > 0 and args.post_seed_3d_prune:
        floaters_3d = find_3d_floater_anchors(
            gaussians=gaussians,
            object_id=int(object_id),
            edge_factor=args.post_seed_3d_edge_factor,
            min_component_size=args.post_seed_3d_min_component_size,
            knn_k=args.post_seed_3d_knn_k,
        )
        if floaters_3d.size > 0:
            rfa3 = prune_anchor_indices(gaussians, floaters_3d)
            post_seed_3d_prune = {
                "n_candidates": int(floaters_3d.size),
                **rfa3,
            }
            logger.info(
                "[obj %s] post-seed 3D component prune: %d candidates, deleted %d anchors",
                object_id, int(floaters_3d.size), rfa3["n_pruned"],
            )

    # ── Stage E ────────────────────────────────────────────────────────────
    post_render = None
    diff_mean = None
    auto_diff_means = []
    if diag_cam_template is not None:
        post_render = diag.render_and_save(
            gaussians, pipe_config, diag_cam_template,
            obj_out_dir / "after.png", bg_white=True, object_label_id=object_id,
        )
        if pre_render is not None and post_render is not None:
            diag.make_compare(pre_render, post_render, obj_out_dir / "compare.png")
            diff_mean = diag.make_diff(pre_render, post_render, obj_out_dir / "diff.png")
            # AABB overlay on after image
            aabb_overlay = diag.overlay_aabb(
                post_render, diag_cam_template, bounds_min_for_init, bounds_max_for_init,
            )
            cv2.imwrite(
                str(obj_out_dir / "after_with_aabb.png"),
                cv2.cvtColor(aabb_overlay, cv2.COLOR_RGB2BGR),
            )

    if auto_compare_cams and auto_before_paths:
        auto_after_paths = _render_auto_compare_set(
            gaussians, pipe_config, object_id, auto_compare_cams,
            auto_compare_dir, "after",
        )
        auto_stage_paths["after"] = auto_after_paths
        auto_diff_means = _finish_auto_compare_set(
            auto_compare_dir, auto_before_paths, auto_after_paths,
        )
        _make_stage_compare_sheet(
            auto_compare_dir,
            ["before", "prune", "seed", "after"],
            auto_stage_paths,
        )

    summary = {
        "object_id": int(object_id),
        "n_anchors_input": int(obj_global_idx.size),
        "stage_a_surface_extraction": surf.to_dict(),
        "stage_a_prune": prune.to_dict(),
        "render_space_prune": render_prune_summary,
        "stage_b_shrinkwrap": sw_dict,
        "stage_c_knn_init": knn.to_dict(),
        "stage_d_post_seed_render_prune": post_seed_render_prune,
        "stage_d_post_seed_3d_prune": post_seed_3d_prune,
        "seed_render_color_rgb": seed_color_rgb.tolist() if seed_color_rgb is not None else None,
        "diff_mean_before_after": diff_mean,
        "auto_compare": {
            "n_views": len(auto_diff_means),
            "mean_diff_per_view": auto_diff_means,
            "mean_diff_all_views": float(np.mean(auto_diff_means)) if auto_diff_means else None,
        },
    }
    with open(obj_out_dir / "replenishment_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


# ───────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True, help="ObjectGS run dir.")
    ap.add_argument("--output_dir", default="replenished_output")
    ap.add_argument("--iteration", type=int, default=-1)
    ap.add_argument("--object_id", type=int, default=None,
                    help="Single object id; if omitted, runs all label ids > 0.")
    ap.add_argument("--all_objects", action="store_true")
    ap.add_argument("--seed", type=int, default=42)

    # Stage A
    ap.add_argument("--knn_k_surface", type=int, default=16)
    ap.add_argument("--floater_iso_factor", type=float, default=3.0)
    ap.add_argument("--floater_edge_factor", type=float, default=2.0)
    ap.add_argument("--min_component_frac", type=float, default=0.005)
    ap.add_argument("--min_component_size", type=int, default=8)
    ap.add_argument("--prune_aabb_padding_factor", type=float, default=8.0,
                    help="Padding around survivor AABB in multiples of Stage A r_med.")
    ap.add_argument("--render_prune", action="store_true", default=True,
                    help="Prune target anchors that render as disconnected alpha blobs.")
    ap.add_argument("--no_render_prune", dest="render_prune", action="store_false")
    ap.add_argument("--render_prune_alpha_threshold", type=float, default=0.03)
    ap.add_argument("--render_prune_close_kernel", type=int, default=9)
    ap.add_argument("--render_prune_min_blob_area_px", type=int, default=24)
    ap.add_argument("--render_prune_vote_threshold", type=int, default=2)

    # Stage B (directional shrinkwrap — 6-sided projected height-field cap seeding)
    ap.add_argument("--existing_coverage_alpha", type=float, default=0.6)
    ap.add_argument("--directional_cell_factor", type=float, default=1.0,
                    help="Tangent-plane cell edge = factor * r_med.")
    ap.add_argument("--directional_depth_percentile", type=float, default=95.0,
                    help="Percentile (0-100) of survivor depth used as cap surface.")
    ap.add_argument("--directional_cap_offset_factor", type=float, default=0.35,
                    help="Cap is placed (factor * r_med) outside outer_depth.")
    ap.add_argument("--directional_close_iters", type=int, default=2,
                    help="Morphological close iterations on per-side support mask.")
    ap.add_argument("--directional_depth_smooth_iters", type=int, default=5,
                    help="Smooth per-side outer-depth sheets before seeding.")
    ap.add_argument("--directional_samples_per_cell", type=int, default=3,
                    help="Sub-cell samples per tangent cell axis. 3 means 9 seeds/cell.")
    ap.add_argument("--directional_min_camera_support", type=float, default=0.15,
                    help="Side flagged as missing if camera_support BELOW this.")
    ap.add_argument("--directional_min_area_frac", type=float, default=0.10,
                    help="Side requires >= this fraction of largest projected area.")
    ap.add_argument("--directional_min_uncovered_frac", type=float, default=0.12,
                    help="Also seed camera-supported sides with this much uncovered shrinkwrap sheet.")

    # Stage C
    ap.add_argument("--knn_k_init", type=int, default=4)
    ap.add_argument("--scale_log_floor_offset", type=float, default=0.0)
    ap.add_argument("--scale_log_ceil_offset", type=float, default=0.75)
    ap.add_argument("--seed_sheet_radius_factor", type=float, default=0.45,
                    help="Child patch radius as a factor of wall cell size.")
    ap.add_argument("--normal_align_min_cos", type=float, default=0.2,
                    help="Donors must have cos(donor_normal, seed_normal) >= this. "
                         "Aligns the new shell with the original 2DGS surface frame.")
    ap.add_argument("--normal_donor_pool_k", type=int, default=32,
                    help="Donor candidate pool per seed before normal filtering.")
    ap.add_argument("--seed_opacity_lift", type=float, default=0.40,
                    help="Seed-only additive raw-opacity lift before renderer masking.")
    ap.add_argument("--seed_opacity_gate", type=float, default=0.70,
                    help="Seed-only multiplicative opacity gate after opacity lift.")
    ap.add_argument("--seed_fixed_opacity", type=float, default=0.18,
                    help="If >0, force seed child opacity to this raw value for continuous shell rendering.")
    ap.add_argument("--seed_scaling_boost", type=float, default=1.35,
                    help="Seed-only covariance multiplier in the ObjectGS renderer.")

    # Stage D (post-seed floater elimination)
    ap.add_argument("--post_seed_render_prune", action="store_true", default=True,
                    help="After seeding, re-run render-space prune to delete pop-out floaters.")
    ap.add_argument("--no_post_seed_render_prune",
                    dest="post_seed_render_prune", action="store_false")
    ap.add_argument("--post_seed_render_vote_threshold", type=int, default=1,
                    help="Lower threshold than Stage A: prune anchors flagged in any single view.")
    ap.add_argument("--post_seed_3d_prune", action="store_true", default=True,
                    help="After seeding, prune small disconnected 3D anchor components.")
    ap.add_argument("--no_post_seed_3d_prune",
                    dest="post_seed_3d_prune", action="store_false")
    ap.add_argument("--post_seed_3d_edge_factor", type=float, default=2.5,
                    help="Connectivity edge length = factor * median target-anchor knn distance.")
    ap.add_argument("--post_seed_3d_min_component_size", type=int, default=12,
                    help="Components smaller than this are pruned as floaters.")
    ap.add_argument("--post_seed_3d_knn_k", type=int, default=8,
                    help="kNN used to estimate connectivity radius.")

    # Diagnostics
    ap.add_argument("--diag_azimuth_deg", type=float, default=180.0,
                    help="Orbit azimuth for before/after fixed-pose snapshot.")
    ap.add_argument("--diag_elevation_deg", type=float, default=15.0)
    ap.add_argument("--diag_dist_factor", type=float, default=0.5,
                    help="Orbit radius = median(cam-to-object dist) * dist_factor.")
    ap.add_argument("--diag_zoom_scale", type=float, default=0.85,
                    help="Scale diagnostic focal length; <1 zooms out.")
    ap.add_argument("--no_diag_render", action="store_true")
    ap.add_argument("--diag_use_training_cam", action="store_true", default=True,
                    help="Use the best-framing training camera for before/after "
                         "snapshots. Disables synthetic orbit pose.")
    ap.add_argument("--no_diag_use_training_cam", dest="diag_use_training_cam",
                    action="store_false")
    ap.add_argument("--auto_compare_views", type=int, default=8,
                    help="Number of orbit before/after comparison views to save.")
    ap.add_argument("--auto_compare_dist_factor", type=float, default=0.9,
                    help="Orbit radius = median(cam-to-object dist) * this factor.")
    ap.add_argument("--auto_compare_zoom_scale", type=float, default=0.8,
                    help="Scale orbit comparison focal length; <1 zooms out.")
    ap.add_argument("--auto_compare_elevation_deg", type=float, default=15.0)
    ap.add_argument("--no_auto_compare", action="store_true")

    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_root = Path(args.output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    logger.info("Loading ObjectGS model from %s", args.model_path)
    gaussians, pipe_config = load_gaussians(args.model_path, args.iteration)

    # Determine object id list
    label_ids = get_label_ids(gaussians)
    if args.all_objects or args.object_id is None:
        unique_ids = sorted(int(x) for x in np.unique(label_ids).tolist() if int(x) > 0)
        target_ids = unique_ids
        if args.object_id is not None and args.object_id in unique_ids:
            target_ids = [args.object_id]
    else:
        target_ids = [int(args.object_id)]
    logger.info("Target object ids: %s", target_ids)

    # Build diagnostic camera once (shared) using first training cam intrinsics.
    diag_cam = None
    cam_centers = None
    cam_data = None
    scene_up = None
    if not args.no_diag_render:
        cams_json = Path(args.model_path) / "cameras.json"
        if cams_json.exists():
            with open(cams_json, "r", encoding="utf-8") as f:
                cams = json.load(f)
            cam_data = cams
            c0 = cams[0]
            w = int(c0["width"]); h = int(c0["height"])
            fx = float(c0["fx"]); fy = float(c0["fy"])
            R0 = np.array(c0["rotation"], np.float32)  # ObjectGS/Colmap world-to-camera
            T0 = np.array(c0["position"], np.float32)
            K = np.array([[fx, 0, w / 2.0], [0, fy, h / 2.0], [0, 0, 1.0]], np.float32)
            diag_cam_template = create_virtual_camera(R0, T0, K, w, h)
            cam_centers = diag.camera_centers_from_cameras_json(cams)
            scene_up = diag.estimate_scene_up_from_cameras(cams)
            logger.info("Estimated scene up from cameras.json: %s", np.round(scene_up, 4).tolist())
            diag_cam = diag_cam_template

    pipeline_summary = {
        "model_path": str(Path(args.model_path).resolve()),
        "output_dir": str(out_root),
        "object_results": [],
    }

    for obj_id in target_ids:
        obj_xyz = get_anchor_positions(gaussians)[label_ids == obj_id]
        if obj_xyz.size == 0:
            continue
        # Per-object orbit camera (centered on the object centroid).
        obj_diag_cam = None
        if diag_cam is not None:
            if args.diag_use_training_cam and cam_data is not None:
                best = diag.pick_best_training_camera(cam_data, obj_xyz)
                if best is not None:
                    obj_diag_cam = diag.build_camera_from_entry(
                        best, zoom_scale=args.diag_zoom_scale,
                    )
            if obj_diag_cam is None:
                obj_diag_cam = diag.build_orbit_camera(
                    obj_xyz, diag_cam,
                    azimuth_deg=args.diag_azimuth_deg,
                    elevation_deg=args.diag_elevation_deg,
                    dist_factor=args.diag_dist_factor,
                    cam_centers=cam_centers,
                    zoom_scale=args.diag_zoom_scale,
                    up_vector=scene_up,
                )

        obj_out_dir = out_root / f"obj_{obj_id}"
        try:
            summary = run_object(
                gaussians, pipe_config, int(obj_id), args, obj_out_dir,
                diag_cam_template=obj_diag_cam,
                auto_compare_template=diag_cam,
                cam_centers=cam_centers,
                scene_up=scene_up,
            )
        except Exception as exc:
            logger.exception("Object %s failed: %s", obj_id, exc)
            summary = {"object_id": int(obj_id), "error": str(exc)}
        pipeline_summary["object_results"].append(summary)

    # Persist final model + replenishment metadata
    final_dir = out_root / "final_model"
    _save_object_model(gaussians, final_dir)
    _copy_side_files(Path(args.model_path), final_dir)

    seeded_mask = getattr(gaussians, "_replenishment_seeded_mask", None)
    rep_meta = {
        "n_original_anchors": int(getattr(gaussians, "n_original_anchors", 0)),
        "n_total_anchors": int(gaussians._anchor.shape[0]),
        "n_seeded": int(seeded_mask.sum().item()) if seeded_mask is not None else 0,
        "object_ids_processed": [int(s.get("object_id", -1)) for s in pipeline_summary["object_results"]],
    }
    seed_lifts = getattr(gaussians, "replenishment_seed_opacity_lift", None)
    if seed_lifts is not None:
        rep_meta["seeded_opacity_lifts"] = seed_lifts.detach().cpu().numpy().tolist()
    seed_gates = getattr(gaussians, "replenishment_seed_opacity_gate", None)
    if seed_gates is not None:
        rep_meta["seeded_opacity_gates"] = seed_gates.detach().cpu().numpy().tolist()
    seed_fixed = getattr(gaussians, "replenishment_seed_fixed_opacity", None)
    if seed_fixed is not None:
        rep_meta["seeded_fixed_opacities"] = seed_fixed.detach().cpu().numpy().tolist()
    seed_color = getattr(gaussians, "replenishment_seed_color_rgb", None)
    if seed_color is not None:
        rep_meta["seeded_color_rgb"] = seed_color.detach().cpu().numpy().tolist()
    seed_scaling_boost = getattr(gaussians, "replenishment_seed_scaling_boost", None)
    if seed_scaling_boost is not None:
        rep_meta["seeded_scaling_boost"] = float(seed_scaling_boost)
    _persist_replenishment_metadata(final_dir, rep_meta)
    _persist_replenishment_metadata(final_dir / "point_cloud" / "iteration_final", rep_meta)

    with open(out_root / "pipeline_summary.json", "w", encoding="utf-8") as f:
        json.dump(pipeline_summary, f, indent=2)
    logger.info("Done. Output: %s", out_root)


if __name__ == "__main__":
    main()
