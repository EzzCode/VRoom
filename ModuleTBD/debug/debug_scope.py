"""Visual Debug for Scope Discovery + Coordinate Frames (ModuleTBD).

Outputs under ``<obj_dir>/00_scope/debug/``::

    summary.json             numeric snapshot of the scope
    aabb_overlays/           AABB drawn on a subset of training images
    topdown.png              birds-eye view of cameras + object axes + V-azimuths
    coord_roundtrip.json     coordinate-frame round-trip test results

Run standalone::

    python -m ModuleTBD.debug.debug_scope \\
        --model_path temp_deps/ObjectGS/outputs/3dovs/.../2026-03-19_04-01-38 \\
        --object_id 8 \\
        --output_root ModuleTBD/outputs
"""
from __future__ import annotations

from pathlib import Path
import json
import logging
import sys

import cv2
import numpy as np

_VROOM_ROOT = Path(__file__).resolve().parents[2]
if str(_VROOM_ROOT) not in sys.path:
    sys.path.insert(0, str(_VROOM_ROOT))

from ModuleTBD.utils.scene_analysis import compute_object_scope, load_gaussians
from ModuleTBD.utils.gstrain_wrapper import make_camera, render_rgba

logger = logging.getLogger(__name__)


# ── helpers ────────────────────────────────────────────────────────────────────

def _project_aabb_corners(cam_p, aabb_min, aabb_max):
    corners = np.array([
        [aabb_min[0], aabb_min[1], aabb_min[2]],
        [aabb_max[0], aabb_min[1], aabb_min[2]],
        [aabb_max[0], aabb_max[1], aabb_min[2]],
        [aabb_min[0], aabb_max[1], aabb_min[2]],
        [aabb_min[0], aabb_min[1], aabb_max[2]],
        [aabb_max[0], aabb_min[1], aabb_max[2]],
        [aabb_max[0], aabb_max[1], aabb_max[2]],
        [aabb_min[0], aabb_max[1], aabb_max[2]],
    ], dtype=np.float64)
    R = np.asarray(cam_p["R"], np.float64)
    T = np.asarray(cam_p["T"], np.float64).reshape(3)
    K = np.asarray(cam_p["K"], np.float64)
    cam_pts = (R @ corners.T).T + T
    px = K @ cam_pts.T
    px[:2] /= np.clip(px[2:3], 1e-6, None)
    return px[:2].T.astype(int)


def _overlay_aabb(rgb_u8, cam_p, aabb_min, aabb_max,
                  color=(0, 200, 0), thickness=2):
    img = rgb_u8.copy()
    px = _project_aabb_corners(cam_p, aabb_min, aabb_max)
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),
             (0,4),(1,5),(2,6),(3,7)]
    for a, b in edges:
        p1 = tuple(np.clip(px[a], -10000, 10000).astype(int))
        p2 = tuple(np.clip(px[b], -10000, 10000).astype(int))
        cv2.line(img, p1, p2, color[::-1], thickness, cv2.LINE_AA)
    return img


# ── round-trip test ───────────────────────────────────────────────────────────

def coord_roundtrip_test(frame, out_path, n_pts=100, n_views=20, seed=0):
    """Round-trip random points (W→L→W) and SV3D views (V→W→V)."""
    rng = np.random.default_rng(seed)
    pts_W = rng.normal(size=(n_pts, 3)).astype(np.float64) * frame.radius + frame.centroid
    pts_L = frame.world_to_local(pts_W)
    pts_W2 = frame.local_to_world(pts_L)
    pts_err = float(np.max(np.linalg.norm(pts_W - pts_W2, axis=1)))

    az_in = rng.uniform(0, 360, size=n_views)
    el_in = rng.uniform(-30, 30, size=n_views)
    view_errs = []
    for az, el in zip(az_in, el_in):
        _R, _T, C_W = frame.virtual_to_world_camera(float(az), float(el))
        az_back, el_back = frame.world_to_virtual(C_W)
        # Compare on the unit-sphere position to avoid azimuth wrap issues.
        from ModuleTBD.utils.transforms import orbit_position
        p_in = orbit_position(az, el)
        p_back = orbit_position(az_back, el_back)
        view_errs.append(float(np.linalg.norm(p_in - p_back)))

    R = frame.R
    ortho_err = float(np.linalg.norm(R @ R.T - np.eye(3)))

    result = {
        "points_max_err_world_units": pts_err,
        "sv3d_view_max_err_unit_sphere": float(max(view_errs)) if view_errs else 0.0,
        "sv3d_view_mean_err": float(np.mean(view_errs)) if view_errs else 0.0,
        "R_orthogonality_err": ortho_err,
        "n_points": n_pts,
        "n_views": len(view_errs),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("Round-trip: pts_max=%.2e | view_max=%.2e | ortho=%.2e",
                result["points_max_err_world_units"],
                result["sv3d_view_max_err_unit_sphere"],
                result["R_orthogonality_err"])
    return result


# ── top-down plot ─────────────────────────────────────────────────────────────

def make_topdown_plot(scope, frame, out_path,
                      canvas_px=1024, n_sv3d_preview=21):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cam_centers = np.array([c["position"] for c in scope.cameras], np.float32)
    up = frame.up
    u_axis = frame.base_dir
    v_axis = np.cross(up, u_axis)
    v_axis = v_axis / max(float(np.linalg.norm(v_axis)), 1e-9)

    def to_uv(pts_W):
        d = np.asarray(pts_W, np.float64) - scope.centroid
        return np.stack([d @ u_axis, d @ v_axis], axis=-1)

    cam_uv = to_uv(cam_centers)
    aabb_corners = np.array([
        [scope.aabb_min[0], scope.aabb_min[1], scope.aabb_min[2]],
        [scope.aabb_max[0], scope.aabb_min[1], scope.aabb_min[2]],
        [scope.aabb_max[0], scope.aabb_max[1], scope.aabb_min[2]],
        [scope.aabb_min[0], scope.aabb_max[1], scope.aabb_min[2]],
        [scope.aabb_min[0], scope.aabb_min[1], scope.aabb_max[2]],
        [scope.aabb_max[0], scope.aabb_min[1], scope.aabb_max[2]],
        [scope.aabb_max[0], scope.aabb_max[1], scope.aabb_max[2]],
        [scope.aabb_min[0], scope.aabb_max[1], scope.aabb_max[2]],
    ], np.float32)
    aabb_uv = to_uv(aabb_corners)

    sv3d_C = []
    for k in range(n_sv3d_preview):
        az = 360.0 * k / n_sv3d_preview
        _R, _T, C_W = frame.virtual_to_world_camera(float(az), 0.0)
        sv3d_C.append(C_W)
    sv3d_uv = to_uv(np.asarray(sv3d_C, np.float32))

    all_uv = np.concatenate([cam_uv, aabb_uv, sv3d_uv, np.zeros((1, 2))], axis=0)
    pad = 0.10
    u_min, v_min = all_uv.min(axis=0)
    u_max, v_max = all_uv.max(axis=0)
    span = max(u_max - u_min, v_max - v_min, 1e-3)
    u_min -= span * pad; v_min -= span * pad
    span = max(u_max - u_min, v_max - v_min, 1e-3) + 2 * span * pad

    img = np.full((canvas_px, canvas_px, 3), 245, np.uint8)

    def to_px(uv):
        u = (uv[..., 0] - u_min) / span
        v = (uv[..., 1] - v_min) / span
        x = (u * (canvas_px - 1)).astype(int)
        y = ((1.0 - v) * (canvas_px - 1)).astype(int)
        return np.stack([x, y], axis=-1)

    cv2.line(img, (0, canvas_px // 2), (canvas_px, canvas_px // 2), (220, 220, 220), 1)
    cv2.line(img, (canvas_px // 2, 0), (canvas_px // 2, canvas_px), (220, 220, 220), 1)

    aabb_px = to_px(aabb_uv)
    bb0 = aabb_px.min(axis=0); bb1 = aabb_px.max(axis=0)
    cv2.rectangle(img, tuple(bb0.astype(int)), tuple(bb1.astype(int)),
                  (60, 60, 200), 2)
    centroid_px = to_px(np.zeros((1, 2), np.float32))[0]
    cv2.circle(img, tuple(centroid_px.astype(int)), 5, (60, 60, 200), -1)

    axis_len = scope.radius * 0.6
    x_end = to_px(np.array([[axis_len, 0.0]]))[0]
    y_end = to_px(np.array([[0.0, axis_len]]))[0]
    cv2.arrowedLine(img, tuple(centroid_px.astype(int)), tuple(x_end.astype(int)),
                    (40, 40, 200), 2, tipLength=0.05)
    cv2.arrowedLine(img, tuple(centroid_px.astype(int)), tuple(y_end.astype(int)),
                    (40, 200, 40), 2, tipLength=0.05)
    cv2.putText(img, "+base (front)", tuple((x_end + np.array([5, 0])).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (40, 40, 200), 1, cv2.LINE_AA)
    cv2.putText(img, "+right", tuple((y_end + np.array([5, 0])).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (40, 200, 40), 1, cv2.LINE_AA)

    r_px = int(round(scope.radius / span * (canvas_px - 1)))
    cv2.circle(img, tuple(centroid_px.astype(int)), r_px, (180, 180, 180), 1)

    cam_px = to_px(cam_uv)
    visible = set(scope.visible_cam_indices)
    for ci in range(len(cam_px)):
        x, y = int(cam_px[ci, 0]), int(cam_px[ci, 1])
        if ci in visible:
            cv2.circle(img, (x, y), 4, (200, 60, 60), -1)
        else:
            cv2.circle(img, (x, y), 3, (160, 160, 160), 1)

    sv3d_px = to_px(sv3d_uv)
    for k in range(len(sv3d_px)):
        x, y = int(sv3d_px[k, 0]), int(sv3d_px[k, 1])
        cv2.drawMarker(img, (x, y), (60, 160, 60), cv2.MARKER_CROSS, 8, 1)
        if k % 3 == 0:
            cv2.putText(img, f"{int(360.0 * k / max(len(sv3d_px), 1))}deg",
                        (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                        (60, 120, 60), 1, cv2.LINE_AA)

    legend = [
        ("BLUE dot   training cam (sees object)", (200, 60, 60)),
        ("GRAY dot   training cam (no view)", (140, 140, 140)),
        ("GREEN +    SV3D azimuth preview", (60, 160, 60)),
        ("RED rect   object AABB footprint", (60, 60, 200)),
        ("RED arrow  +base_dir (orbit az=0)", (40, 40, 200)),
        ("GREEN arrow +right (orbit az=90)", (40, 200, 40)),
    ]
    for i, (txt, col) in enumerate(legend):
        cv2.putText(img, txt, (12, 22 + 18 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)
    title = (f"obj={scope.object_label_id} | radius={scope.radius:.2f} | "
             f"vis={len(scope.visible_cam_indices)}/{len(scope.cameras)}")
    cv2.putText(img, title, (12, canvas_px - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1, cv2.LINE_AA)

    cv2.imwrite(str(out_path), img)
    return out_path


# ── AABB overlays ──────────────────────────────────────────────────────────────

def render_aabb_overlays(scope, gaussians, pipe_config, out_dir,
                         max_views=6):
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    if gaussians is None or pipe_config is None:
        return saved
    indices = list(scope.visible_cam_indices)
    if len(indices) > max_views:
        step = max(1, len(indices) // max_views)
        indices = indices[::step][:max_views]
    for ci in indices:
        cam_p = scope.cameras[ci]
        cam = make_camera(cam_p["R"], cam_p["T"], cam_p["K"],
                          cam_p["width"], cam_p["height"])
        try:
            res = render_rgba(gaussians, cam, pipe_config, bg_white=True,
                              object_label_id=scope.object_label_id)
        except Exception:
            continue
        rgb = res["rgb"].detach().clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
        rgb_u8 = (rgb * 255 + 0.5).astype(np.uint8)
        rgb_u8 = _overlay_aabb(rgb_u8, cam_p, scope.aabb_min, scope.aabb_max)
        img_name = cam_p.get("image_name", f"cam_{ci}")
        cv2.putText(rgb_u8, f"cam={ci} | img={img_name}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(rgb_u8, f"cam={ci} | img={img_name}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        out_path = out_dir / f"cam_{ci:03d}_{img_name}.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR))
        saved.append(out_path)
    return saved


# ── orchestrator ──────────────────────────────────────────────────────────────

def generate_debug_artifacts(*, scope, frame, debug_dir,
                             gaussians=None, pipe_config=None,
                             max_aabb_views=6):
    debug_dir = Path(debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)

    rt = coord_roundtrip_test(frame, debug_dir / "coord_roundtrip.json")
    make_topdown_plot(scope, frame, debug_dir / "topdown.png")
    overlays = render_aabb_overlays(scope, gaussians, pipe_config,
                                    debug_dir / "aabb_overlays",
                                    max_views=max_aabb_views)

    summary = {
        "object_id": int(scope.object_label_id),
        "n_anchors": int(scope.n_anchors),
        "centroid": np.asarray(scope.centroid).tolist(),
        "aabb_min": np.asarray(scope.aabb_min).tolist(),
        "aabb_max": np.asarray(scope.aabb_max).tolist(),
        "obb_extents": np.asarray(scope.obb_extents).tolist(),
        "up": np.asarray(scope.up).tolist(),
        "base_dir": np.asarray(scope.base_dir).tolist(),
        "radius": float(scope.radius),
        "n_cameras_total": len(scope.cameras),
        "n_cameras_visible": len(scope.visible_cam_indices),
        "round_trip_test": rt,
        "aabb_overlay_files": [str(p) for p in overlays],
    }
    with open(debug_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Scope debug saved to: %s", debug_dir)
    return summary


def main():
    import argparse
    parser = argparse.ArgumentParser(description="ModuleTBD scope/frame visual debug.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--object_id", required=True, type=int)
    parser.add_argument("--output_root", default="ModuleTBD/outputs")
    parser.add_argument("--ply_path", default=None)
    parser.add_argument("--max_aabb_views", type=int, default=6)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
    scope, frame, pipe_config = compute_object_scope(
        args.model_path, args.object_id, ply_path=args.ply_path,
    )
    gaussians, _ = load_gaussians(args.model_path, ply_path=args.ply_path)
    out_dir = Path(args.output_root) / f"obj_{args.object_id}" / "00_scope" / "debug"
    generate_debug_artifacts(
        scope=scope, frame=frame, debug_dir=out_dir,
        gaussians=gaussians, pipe_config=pipe_config,
        max_aabb_views=args.max_aabb_views,
    )


if __name__ == "__main__":
    main()
