"""
Visual debug for Phases 0-2.

Outputs (under <output_root>/<scene>/<obj>/debug_phase01/):
    summary.json           — numeric snapshot of the scope
    aabb_overlays/         — AABB drawn on a subset of training images
    object_isolation/      — current ObjectGS object render at the same cams
    topdown.png            — birds-eye view of cameras + object axes + V-azimuths
    coord_roundtrip.txt    — Phase 2 unit-test results
    sv3d_pose_preview.png  — example SV3D virtual cams projected to topdown
"""
from __future__ import annotations

from pathlib import Path
import json
import logging
import sys

import cv2
import numpy as np
import torch

_VROOM_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_VROOM_ROOT) not in sys.path:
    sys.path.insert(0, str(_VROOM_ROOT))

from target_replenishment.core.objectgs_bridge import (
    create_virtual_camera, render_view,
)
from target_replenishment.core.diagnostics import overlay_aabb

from object_isolation.core.scope import (
    discover_object_scope, find_uncovered_azimuth_sectors,
)
from object_isolation.core.coordinate_frames import WorldLocal, LocalSV3D, R_LV


logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 round-trip unit test
# ─────────────────────────────────────────────────────────────────────────────

def coord_roundtrip_test(world_local: WorldLocal, local_sv3d: LocalSV3D,
                         out_path: Path, n_pts: int = 100, seed: int = 0) -> dict:
    """Round-trip 100 random points and 20 SV3D views; assert tiny error."""
    rng = np.random.default_rng(seed)

    # Points: W → L → W
    pts_W = rng.normal(size=(n_pts, 3)).astype(np.float64) * world_local.radius + world_local.centroid_W
    pts_L = world_local.world_to_local_pts(pts_W)
    pts_W2 = world_local.local_to_world_pts(pts_L)
    pts_err = float(np.max(np.linalg.norm(pts_W - pts_W2, axis=1)))

    # SV3D views: V→W→V (use the inverse mapping)
    az_in = rng.uniform(0, 360, size=20)
    el_in = rng.uniform(-30, 30, size=20)
    view_errs = []
    for az, el in zip(az_in, el_in):
        R_w2c, T_w2c, C_W = local_sv3d.sv3d_view_to_world_camera(az, el)
        az_back, el_back = local_sv3d.world_camera_to_sv3d_view(C_W)
        # Compare on the unit-radius sphere position to avoid az wrap issues
        from object_isolation.core.coordinate_frames import sv3d_view_position_V
        p_in = sv3d_view_position_V(az, el)
        p_back = sv3d_view_position_V(az_back, el_back)
        view_errs.append(float(np.linalg.norm(p_in - p_back)))

    # Orthonormality of R_LV
    ortho_err = float(np.linalg.norm(R_LV @ R_LV.T - np.eye(3)))

    # WL rotation orthonormality
    R_WL = world_local.R_WL
    wl_ortho_err = float(np.linalg.norm(R_WL @ R_WL.T - np.eye(3)))

    result = {
        "points_max_err_world_units": pts_err,
        "sv3d_view_max_err_unit_sphere": float(max(view_errs)),
        "sv3d_view_mean_err": float(np.mean(view_errs)),
        "R_LV_orthogonality_err": ortho_err,
        "R_WL_orthogonality_err": wl_ortho_err,
        "n_points": n_pts,
        "n_views": len(view_errs),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("Round-trip: pts_max=%.2e | view_max=%.2e | orthog=%.2e/%.2e",
                result["points_max_err_world_units"],
                result["sv3d_view_max_err_unit_sphere"],
                result["R_LV_orthogonality_err"],
                result["R_WL_orthogonality_err"])
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Top-down plot
# ─────────────────────────────────────────────────────────────────────────────

def make_topdown_plot(scope, world_local: WorldLocal, local_sv3d: LocalSV3D,
                      out_path: Path, canvas_px: int = 1024,
                      n_sv3d_preview: int = 21):
    """Birds-eye view (looking down +Z_W if up_W ≈ +Z_W; otherwise looking
    along -up_W) showing all training cameras, the object AABB footprint,
    the L-frame axes, the orbit circle, and a preview of SV3D azimuths.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cam_centers_W = np.array([c['position'] for c in scope.cameras], dtype=np.float32)
    visible_centers_W = scope.cam_centers_visible_W

    # Project onto plane ⊥ up. Use base_dir as +U axis, (up × base_dir) as +V axis.
    up = world_local.up_W
    u_axis = world_local.base_dir_W                       # +X_L
    v_axis = np.cross(up, u_axis); v_axis /= np.linalg.norm(v_axis)  # +Y_L

    def to_uv(pts_W: np.ndarray) -> np.ndarray:
        d = pts_W.astype(np.float64) - world_local.centroid_W
        u = d @ u_axis
        v = d @ v_axis
        return np.stack([u, v], axis=-1)

    cam_uv = to_uv(cam_centers_W)
    aabb_corners_W = np.array([
        [scope.aabb_min_W[0], scope.aabb_min_W[1], scope.aabb_min_W[2]],
        [scope.aabb_max_W[0], scope.aabb_min_W[1], scope.aabb_min_W[2]],
        [scope.aabb_max_W[0], scope.aabb_max_W[1], scope.aabb_min_W[2]],
        [scope.aabb_min_W[0], scope.aabb_max_W[1], scope.aabb_min_W[2]],
        [scope.aabb_min_W[0], scope.aabb_min_W[1], scope.aabb_max_W[2]],
        [scope.aabb_max_W[0], scope.aabb_min_W[1], scope.aabb_max_W[2]],
        [scope.aabb_max_W[0], scope.aabb_max_W[1], scope.aabb_max_W[2]],
        [scope.aabb_min_W[0], scope.aabb_max_W[1], scope.aabb_max_W[2]],
    ], dtype=np.float32)
    aabb_uv = to_uv(aabb_corners_W)

    # SV3D preview: 21 azimuths at elev=0 (SV3D_p default-ish trajectory).
    sv3d_uv = []
    for k in range(n_sv3d_preview):
        az = 360.0 * k / n_sv3d_preview
        C_W = local_sv3d.sv3d_camera_in_W(az, 0.0)
        sv3d_uv.append(C_W)
    sv3d_uv = to_uv(np.array(sv3d_uv, dtype=np.float32))

    # Compute extents and pixel scale.
    all_uv = np.concatenate([cam_uv, aabb_uv, sv3d_uv, np.zeros((1, 2))], axis=0)
    pad = 0.10
    u_min, v_min = all_uv.min(axis=0)
    u_max, v_max = all_uv.max(axis=0)
    span = max(u_max - u_min, v_max - v_min)
    span = max(span, 1e-3)
    u_min -= span * pad; v_min -= span * pad
    u_max += span * pad; v_max += span * pad
    span = max(u_max - u_min, v_max - v_min)

    img = np.full((canvas_px, canvas_px, 3), 245, dtype=np.uint8)

    def to_px(uv: np.ndarray) -> np.ndarray:
        u = (uv[..., 0] - u_min) / span
        v = (uv[..., 1] - v_min) / span
        x = (u * (canvas_px - 1)).astype(int)
        y = ((1.0 - v) * (canvas_px - 1)).astype(int)
        return np.stack([x, y], axis=-1)

    # Grid + origin
    cv2.line(img, (0, canvas_px // 2), (canvas_px, canvas_px // 2), (220, 220, 220), 1)
    cv2.line(img, (canvas_px // 2, 0), (canvas_px // 2, canvas_px), (220, 220, 220), 1)

    # AABB footprint (project all 8 corners to plane and take convex hull-ish bbox)
    aabb_px = to_px(aabb_uv)
    bb_x0, bb_y0 = aabb_px.min(axis=0)
    bb_x1, bb_y1 = aabb_px.max(axis=0)
    cv2.rectangle(img, (int(bb_x0), int(bb_y0)), (int(bb_x1), int(bb_y1)),
                  (60, 60, 200), 2)

    # Centroid
    centroid_px = to_px(np.zeros((1, 2), dtype=np.float32))[0]
    cv2.circle(img, tuple(centroid_px.astype(int)), 5, (60, 60, 200), -1)

    # L-frame axes from centroid: +X_L (red), +Y_L (green), +Z_L is into page
    axis_len_uv = scope.radius * 0.6
    x_end = to_px(np.array([[axis_len_uv, 0.0]]))[0]
    y_end = to_px(np.array([[0.0, axis_len_uv]]))[0]
    cv2.arrowedLine(img, tuple(centroid_px.astype(int)), tuple(x_end.astype(int)),
                    (40, 40, 200), 2, tipLength=0.05)
    cv2.arrowedLine(img, tuple(centroid_px.astype(int)), tuple(y_end.astype(int)),
                    (40, 200, 40), 2, tipLength=0.05)
    cv2.putText(img, "+X_L (front)", tuple((x_end + np.array([5, 0])).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (40, 40, 200), 1, cv2.LINE_AA)
    cv2.putText(img, "+Y_L", tuple((y_end + np.array([5, 0])).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (40, 200, 40), 1, cv2.LINE_AA)

    # Orbit circle (radius)
    r_px = int(round(scope.radius / span * (canvas_px - 1)))
    cv2.circle(img, tuple(centroid_px.astype(int)), r_px, (180, 180, 180), 1)

    # Training cameras (gray = unseen by obj, blue = visible)
    cam_px = to_px(cam_uv)
    visible_set = set(scope.visible_cam_indices)
    for ci in range(len(cam_px)):
        x, y = int(cam_px[ci, 0]), int(cam_px[ci, 1])
        if ci in visible_set:
            cv2.circle(img, (x, y), 4, (200, 60, 60), -1)
        else:
            cv2.circle(img, (x, y), 3, (160, 160, 160), 1)

    # SV3D azimuth preview (small green crosses)
    sv3d_px = to_px(sv3d_uv)
    for k in range(len(sv3d_px)):
        x, y = int(sv3d_px[k, 0]), int(sv3d_px[k, 1])
        cv2.drawMarker(img, (x, y), (60, 160, 60), cv2.MARKER_CROSS, 8, 1)
        if k % 3 == 0:
            # OpenCV's default Hershey fonts don't include U+00B0 (°); use 'deg'.
            cv2.putText(img, f"{int(360.0 * k / len(sv3d_px))}deg",
                        (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                        (60, 120, 60), 1, cv2.LINE_AA)

    # Legend
    legend = [
        ("BLUE dot   training cam (sees object)", (200, 60, 60)),
        ("GRAY dot   training cam (no view)", (140, 140, 140)),
        ("GREEN +    SV3D azimuth preview", (60, 160, 60)),
        ("RED rect   object AABB footprint", (60, 60, 200)),
        ("RED arrow  +X_L (orbit zero-azimuth)", (40, 40, 200)),
        ("GREEN arrow +Y_L", (40, 200, 40)),
    ]
    for i, (txt, col) in enumerate(legend):
        cv2.putText(img, txt, (12, 22 + 18 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)

    # Title
    title = (f"obj={scope.object_label_id} | radius={scope.radius:.2f} | "
             f"vis={len(scope.visible_cam_indices)}/{len(scope.cameras)}")
    cv2.putText(img, title, (12, canvas_px - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1, cv2.LINE_AA)

    cv2.imwrite(str(out_path), img)
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# AABB overlays on training images
# ─────────────────────────────────────────────────────────────────────────────

def render_aabb_overlays(scope, gaussians, pipe_config, out_dir: Path,
                         max_views: int = 6) -> list[Path]:
    """Render the current ObjectGS isolation of the object on a few visible
    training cameras, and overlay the AABB. This is the "what the model
    currently thinks the object looks like" reference — useful baseline for
    Phase 1 hybrid extraction.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    # Sample evenly-spaced visible cams.
    indices = scope.visible_cam_indices
    if len(indices) > max_views:
        step = max(1, len(indices) // max_views)
        indices = indices[::step][:max_views]

    bg = torch.ones(3, dtype=torch.float32, device="cuda")
    for ci in indices:
        cam_p = scope.cameras[ci]
        cam = create_virtual_camera(cam_p['R'], cam_p['T'], cam_p['K'],
                                    cam_p['width'], cam_p['height'])
        # Object-only render
        res = render_view(gaussians, cam, pipe_config, bg,
                          object_label_id=scope.object_label_id)
        rgb = res['rgb'].detach().clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
        rgb_u8 = (rgb * 255 + 0.5).astype(np.uint8)
        # Overlay AABB
        rgb_u8 = overlay_aabb(rgb_u8, cam, scope.aabb_min_W, scope.aabb_max_W,
                              color=(0, 200, 0), thickness=2)

        # Annotate cam id and visible-anchor count
        from target_replenishment.core.perspective_graph import _count_visible_anchors
        n_vis = int(_count_visible_anchors(cam_p, scope.anchor_xyz_W).sum())
        cv2.putText(rgb_u8, f"cam={ci} | vis_anchors={n_vis} | img={cam_p['img_name']}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(rgb_u8, f"cam={ci} | vis_anchors={n_vis} | img={cam_p['img_name']}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        out_path = out_dir / f"cam_{ci:03d}_{cam_p['img_name']}.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR))
        saved.append(out_path)

    return saved


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration
# ─────────────────────────────────────────────────────────────────────────────

def run_debug(model_path: str, object_id: int, output_root: str,
              max_aabb_views: int = 6) -> dict:
    """Run Phases 1 & 2 + emit all visual debug artefacts."""
    out_dir = Path(output_root) / f"obj_{object_id}" / "debug_phase01"
    out_dir.mkdir(parents=True, exist_ok=True)

    scope, world_local, local_sv3d, gaussians, pipe = discover_object_scope(
        model_path=model_path, object_label_id=object_id,
    )

    # Round-trip math test (no GPU).
    rt = coord_roundtrip_test(world_local, local_sv3d, out_dir / "coord_roundtrip.json")

    # Top-down plot.
    make_topdown_plot(scope, world_local, local_sv3d, out_dir / "topdown.png")

    # AABB overlays (GPU).
    overlays = render_aabb_overlays(scope, gaussians, pipe,
                                    out_dir / "aabb_overlays",
                                    max_views=max_aabb_views)

    # Uncovered sectors.
    uncovered = find_uncovered_azimuth_sectors(scope, bin_deg=10.0, min_gap_deg=30.0)

    # Summary.
    summary = {
        "model_path": str(model_path),
        "object_id": int(object_id),
        "n_anchors": int(scope.n_anchors),
        "centroid_W": np.asarray(scope.centroid_W).tolist(),
        "aabb_min_W": np.asarray(scope.aabb_min_W).tolist(),
        "aabb_max_W": np.asarray(scope.aabb_max_W).tolist(),
        "principal_extents": np.asarray(scope.principal_extents).tolist(),
        "up_W": np.asarray(scope.up_W).tolist(),
        "base_dir_W": np.asarray(scope.base_dir_W).tolist(),
        "radius": float(scope.radius),
        "n_cameras_total": len(scope.cameras),
        "n_cameras_visible": len(scope.visible_cam_indices),
        "azimuth_histogram_V": {int(k): int(v) for k, v in scope.azimuth_histogram_V.items()},
        "uncovered_sectors_V_deg": uncovered,
        "round_trip_test": rt,
        "aabb_overlay_files": [str(p) for p in overlays],
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Phase 0-2 debug saved to: %s", out_dir)
    return summary


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Phase 0-2 visual debug.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--object_id", required=True, type=int)
    parser.add_argument("--output_root", default="object_isolation/outputs")
    parser.add_argument("--max_aabb_views", type=int, default=6)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
    run_debug(args.model_path, args.object_id, args.output_root, args.max_aabb_views)


if __name__ == "__main__":
    main()
