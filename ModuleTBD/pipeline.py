import json
import logging
import math
import shutil
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .utils.gstrain_wrapper import make_camera, render_rgba
from .utils.transforms import look_at, ObjectFrame
from .utils.scene_analysis import compute_object_scope, load_gaussians
from .trainer import train_object
from .utils.colmap_init import load_colmap_object_point_cloud

logger = logging.getLogger(__name__)


# ── Comparison orbit cameras ──────────────────────────────────────────────────

def build_orbit_cameras(*, center, radius, orbit_radius, up, ref_cam_position, n_views=8, width=512, height=512):
    """Build n_views cameras orbiting center, anchored to the ref camera's azimuth."""
    center = np.asarray(center, np.float32)
    up = np.asarray(up, np.float32)
    up = up / max(float(np.linalg.norm(up)), 1e-9)

    dist = max(float(orbit_radius) * 0.9, float(radius) * 2.5, 0.5)
    fov = float(np.clip(
        2.0 * np.arctan(float(radius) / max(dist, 1e-6)) / 0.55,
        np.radians(30.0), np.radians(100.0),
    ))
    fx = (width / 2.0) / np.tan(fov / 2.0)
    K = np.array([[fx, 0.0, width / 2.0],
                  [0.0, fx, height / 2.0],
                  [0.0, 0.0, 1.0]], np.float32)

    ref = np.asarray(ref_cam_position, np.float32) - center
    vert = float(np.dot(ref, up))
    horiz = ref - vert * up
    if np.linalg.norm(horiz) < 1e-6:
        horiz = np.array([1.0, 0.0, 0.0], np.float32)
        if abs(np.dot(horiz, up)) > 0.9:
            horiz = np.array([0.0, 1.0, 0.0], np.float32)
    basis_h = horiz / np.linalg.norm(horiz)
    basis_v = np.cross(up, basis_h)
    basis_v = basis_v / np.linalg.norm(basis_v)
    z_off = float(np.clip(vert, -0.3 * dist, 0.3 * dist))

    cams = []
    for i in range(int(n_views)):
        a = 2.0 * np.pi * i / int(n_views)
        cam_pos = center + dist * np.cos(a) * basis_h + dist * np.sin(a) * basis_v + z_off * up
        R, T = look_at(cam_pos, center, up)
        cams.append({"R": R, "T": T, "K": K.copy(), "width": int(width), "height": int(height)})
    return cams


def render_orbit(gaussians, pipe_config, orbit_cams, object_label_id=None, exclude_label_id=None, bg_white=True):
    """Render orbit cameras. Returns list of HxWx3 uint8 RGB frames."""
    frames = []
    for c in orbit_cams:
        cam = make_camera(c["R"], c["T"], c["K"], c["width"], c["height"])
        pkg = render_rgba(gaussians, cam, pipe_config, bg_white=bool(bg_white),
                          object_label_id=object_label_id, exclude_label_id=exclude_label_id)
        rgb = (pkg["rgb"].detach().permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
        frames.append(rgb)
    return frames


def save_compare_grid(before, after, out_dir, prefix="view"):
    """Save before/after PNGs and side-by-side composites."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, (b, a) in enumerate(zip(before, after)):
        cv2.imwrite(str(out_dir / f"before_{prefix}_{i:02d}.png"), cv2.cvtColor(b, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out_dir / f"after_{prefix}_{i:02d}.png"), cv2.cvtColor(a, cv2.COLOR_RGB2BGR))
        if b.shape == a.shape:
            sep = np.full((b.shape[0], 6, 3), 255, dtype=np.uint8)
            cv2.imwrite(str(out_dir / f"compare_{prefix}_{i:02d}.png"),
                        cv2.cvtColor(np.concatenate([b, sep, a], axis=1), cv2.COLOR_RGB2BGR))


# ── Model export ──────────────────────────────────────────────────────────────

def save_model(gaussians, *, output_dir, reference_model_path, metadata=None):
    """Save gaussians in ObjectGS-compatible layout under output_dir."""
    out = Path(output_dir)
    iter_dir = out / "point_cloud" / "iteration_1"
    iter_dir.mkdir(parents=True, exist_ok=True)

    gaussians.save_ply(str(out / "point_cloud.ply"))
    gaussians.save_mlp_checkpoints(str(out))
    gaussians.save_ply(str(iter_dir / "point_cloud.ply"))
    gaussians.save_mlp_checkpoints(str(iter_dir))

    ref = Path(reference_model_path)
    for name in ("config.yaml", "cameras.json"):
        src, dst = ref / name, out / name
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)

    if metadata is not None:
        with open(out / "reintegration_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    logger.info("Saved model to %s", out)
    return out


def label_anchor_counts(gaussians):
    """Return {label_id: n_anchors} from current gaussians state."""
    labels = gaussians.label_ids.squeeze(-1).cpu().numpy().astype(np.int64)
    uniq, counts = np.unique(labels, return_counts=True)
    return {int(k): int(v) for k, v in zip(uniq.tolist(), counts.tolist())}


def build_metadata(*, counts_pre, counts_post, per_object_summaries, model_path):
    """Aggregate per-object training summaries into a scene-wide metadata dict."""
    return {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "model_path": str(model_path),
        "n_anchors_pre": int(sum(counts_pre.values())),
        "n_anchors_post": int(sum(counts_post.values())),
        "anchors_added": int(sum(counts_post.values()) - sum(counts_pre.values())),
        "label_counts_pre": {str(k): int(v) for k, v in counts_pre.items()},
        "label_counts_post": {str(k): int(v) for k, v in counts_post.items()},
        "objects": per_object_summaries,
    }


# ── Object training pipeline ──────────────────────────────────────────────────

def run_pipeline(
    *,
    model_path,
    object_label_id,
    halluc_index_path,
    output_dir,
    halluc_manifest=None,
    gaussians=None,
    pipe_config=None,
    scope=None,
    frame=None,
    iterations=1200,
    lr_scale=1.0,
    hallucination_weight=1.0,
    real_weight=1.0,
    rgb_weight=1.0,
    hallucination_rgb_scale=1.0,
    depth_weight=0.1,
    depth_start_iter=100,
    depth_front_weight=1.0,
    depth_back_weight=0.15,
    colmap_init_target_points=8000,
    enable_densification=False,
    max_anchor_count=20000,
    densify_grad_threshold=0.00005,
    densify_extra_ratio=0.08,
    use_cond_cam_up=True,
    fov_y_deg=50.0,
    debug=False,
):
    """Train a fresh ObjectGS model for one object.

    Pass pre-loaded gaussians/pipe_config/scope/frame to skip reloading
    (useful when processing multiple objects from the same scene).
    Returns a summary dict; the trained GaussianModel is under key '_gaussians'.
    """
    from .dataset_builder import build_supervision_views, save_supervision_manifest

    out_dir = Path(output_dir)
    obj_id = int(object_label_id)
    obj_dir = out_dir / f"obj_{obj_id}"
    obj_dir.mkdir(parents=True, exist_ok=True)

    # ── Load scope and model ──────────────────────────────────────────────
    if scope is None or pipe_config is None:
        logger.info("Computing scope for obj %d from %s", obj_id, model_path)
        scope, pipe_config = compute_object_scope(model_path, obj_id)
    if gaussians is None:
        gaussians, _ = load_gaussians(model_path)
    if frame is None:
        frame = ObjectFrame(centroid=scope.centroid, up=scope.up,
                            base_dir=scope.base_dir, radius=scope.radius)

    # ── Validate hallucination manifest ──────────────────────────────────
    halluc_path = Path(halluc_index_path)
    if halluc_manifest is not None:
        halluc = halluc_manifest
    else:
        with open(halluc_path) as f:
            halluc = json.load(f)

    cam_idx = int(halluc.get("conditioning", {}).get("cam_index", -1))
    if not (0 <= cam_idx < len(scope.cameras)):
        raise RuntimeError(
            f"hallucination_index.json conditioning.cam_index={cam_idx} out of range "
            f"(scope has {len(scope.cameras)} cameras). Re-run hallucination."
        )

    manifest_az = float(halluc.get("conditioning", {}).get("azimuth_deg", float("nan")))
    manifest_el = float(halluc.get("conditioning", {}).get("elevation_deg", float("nan")))
    if math.isfinite(manifest_az) and math.isfinite(manifest_el):
        current_az, current_el = frame.world_to_virtual(
            np.asarray(scope.cameras[cam_idx]["position"], np.float32)
        )
        current_az = ((float(current_az) + 180.0) % 360.0) - 180.0
        delta_az = abs(((float(manifest_az) - current_az + 180.0) % 360.0) - 180.0)
        if delta_az > 0.5 or abs(float(manifest_el) - float(current_el)) > 0.5:
            raise RuntimeError(
                f"Hallucination manifest frame mismatch for obj {obj_id}: "
                f"manifest az/el=({manifest_az:.2f}, {manifest_el:.2f}) vs "
                f"current ({current_az:.2f}, {float(current_el):.2f}). Re-run hallucination."
            )

    # ── Up vector for supervision alignment ──────────────────────────────
    if use_cond_cam_up:
        up_override = -np.asarray(scope.cameras[cam_idx]["R"], np.float32)[1]  # -row1 of R_w2c
    else:
        up_override = np.asarray(scope.up, np.float32)

    # ── Build supervision views ───────────────────────────────────────────
    extraction_index_path = obj_dir / "01_extraction" / "extraction_index.json"

    pcd, _ = load_colmap_object_point_cloud(
        model_path=model_path, object_id=obj_id, scope=scope,
        extraction_index_path=extraction_index_path,
        max_points=20000, target_points=int(colmap_init_target_points),
    )
    seed_points = np.asarray(pcd.points, np.float32)

    supervision_views = build_supervision_views(
        halluc_index_path=halluc_path,
        extraction_index_path=extraction_index_path,
        scope=scope,
        frame=frame,
        seed_points_W=seed_points,
        real_weight=float(real_weight),
        hallucination_weight=float(hallucination_weight),
        fov_y_deg=float(fov_y_deg),
        resolution=576,
        real_target_long_edge=576,
        up_override=up_override,
    )
    if not supervision_views:
        raise RuntimeError(f"No supervision views produced for obj {obj_id}.")

    if debug:
        (obj_dir / "04_supervision").mkdir(parents=True, exist_ok=True)
        save_supervision_manifest(supervision_views, obj_dir / "04_supervision" / "supervision_manifest.json")

    n_real = sum(1 for v in supervision_views if v.get("source") == "real")
    n_hall = len(supervision_views) - n_real
    logger.info("%d supervision views for obj %d (real=%d hall=%d)", len(supervision_views), obj_id, n_real, n_hall)

    # ── Train ─────────────────────────────────────────────────────────────
    n_parent_anchors = int(gaussians._anchor.shape[0]) if gaussians is not None else 0
    n_parent_obj_anchors = 0
    if gaussians is not None:
        labels = gaussians.label_ids.squeeze(-1).cpu().numpy()
        n_parent_obj_anchors = int((labels == obj_id).sum())

    result = train_object(
        supervision_views=supervision_views,
        scope=scope,
        object_id=obj_id,
        model_path=model_path,
        output_dir=obj_dir,
        n_iterations=int(iterations),
        extraction_index_path=extraction_index_path,
        parent_gaussians=gaussians,
        pipe_config=pipe_config,
        lr_scale=float(lr_scale),
        colmap_init_target_points=int(colmap_init_target_points),
        rgb_weight=float(rgb_weight),
        hallucination_rgb_scale=float(hallucination_rgb_scale),
        depth_weight=float(depth_weight),
        depth_start_iter=int(depth_start_iter),
        depth_front_weight=float(depth_front_weight),
        depth_back_weight=float(depth_back_weight),
        enable_densification=bool(enable_densification),
        max_anchor_count=int(max_anchor_count),
        densify_grad_threshold=float(densify_grad_threshold),
        densify_extra_ratio=float(densify_extra_ratio),
        debug=bool(debug),
    )

    summary = dict(result["summary"])
    summary.update({
        "n_real_supervision_views": n_real,
        "n_hallucinated_supervision_views": n_hall,
        "n_parent_anchors": n_parent_anchors,
        "n_parent_obj_anchors": n_parent_obj_anchors,
        "halluc_index_path": str(halluc_path),
        "extraction_index_path": str(extraction_index_path),
        "model_path": str(model_path),
    })
    summary["_gaussians"] = result["gaussians"]

    logger.info("obj %d done: anchors=%d final_loss=%.5f",
                obj_id, summary.get("n_final_anchors", 0), summary.get("final_loss", 0.0))
    return summary
