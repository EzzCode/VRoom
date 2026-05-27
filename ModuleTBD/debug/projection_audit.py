"""Projection Audit — Two Sanity Tests Before Training (ModuleTBD).

Adapted from ``object_isolation/debug/projection_audit.py`` to use ModuleTBD's
single-stage ObjectFrame and its dataset_builder API.

Outputs under ``<obj_dir>/06_projection_audit/``::

    projection_overlay/         per-view PNGs with seed-point dots
    audit_report.json           per-view numerical report

Usage::

    python -m ModuleTBD.debug.projection_audit \\
        --model_path "temp_deps/ObjectGS/outputs/3dovs/.../2026-03-19_04-01-38" \\
        --output_root ModuleTBD/outputs \\
        --object_id 8
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

_VROOM_ROOT = Path(__file__).resolve().parents[2]
if str(_VROOM_ROOT) not in sys.path:
    sys.path.insert(0, str(_VROOM_ROOT))

logger = logging.getLogger(__name__)


def _signed_angle_delta_deg(a, b):
    return float(((float(a) - float(b) + 180.0) % 360.0) - 180.0)


# ── Test 2: geometry ──────────────────────────────────────────────────────────

def test_point_cloud_geometry(xyz_W, scope, supervision_views):
    centroid = xyz_W.mean(axis=0)
    pmin = xyz_W.min(axis=0)
    pmax = xyz_W.max(axis=0)
    extent = pmax - pmin
    radius = float(np.linalg.norm(extent) / 2.0)

    print("\n" + "=" * 65)
    print("TEST 2 — POINT CLOUD GEOMETRY")
    print("=" * 65)
    print(f"  N points        : {xyz_W.shape[0]}")
    print(f"  Centroid (pcd)  : {centroid}")
    print(f"  scope.centroid  : {np.asarray(scope.centroid)}")
    print(f"  scope.radius    : {scope.radius:.4f}")
    print(f"  scope.aabb_min  : {np.asarray(scope.aabb_min)}")
    print(f"  scope.aabb_max  : {np.asarray(scope.aabb_max)}")

    centroid_offset = float(np.linalg.norm(centroid - np.asarray(scope.centroid, np.float32)))
    print(f"  centroid offset : {centroid_offset:.4f}  (warn > {scope.radius * 0.5:.4f})")
    if centroid_offset > scope.radius * 0.5:
        print("  *** WARNING: seed centroid far from scope centroid ***")

    aabb_min = np.asarray(scope.aabb_min, np.float32)
    aabb_max = np.asarray(scope.aabb_max, np.float32)
    outside = np.any((xyz_W < aabb_min - 0.01) | (xyz_W > aabb_max + 0.01), axis=1)
    print(f"  Points outside scope AABB: {int(outside.sum())} / {xyz_W.shape[0]}")

    print(f"\n  {'Source':<12} {'Az':>8} {'El':>10}  cam->centroid")
    print("  " + "-" * 50)
    cam_dists = []
    for v in supervision_views:
        cp = v["camera"].get("position")
        if cp is None:
            continue
        c = np.asarray(cp, np.float32)
        d = float(np.linalg.norm(c - centroid))
        cam_dists.append(d)
        az = v["camera"].get("azimuth_offset_deg", 0.0)
        el = v["camera"].get("elevation_offset_deg", 0.0)
        src = v.get("source", "?")
        print(f"  {src:<12} {az:>8.1f} {el:>10.1f}  {d:.4f}")

    return {
        "n_points": int(xyz_W.shape[0]),
        "centroid": centroid.tolist(),
        "aabb_min": pmin.tolist(),
        "aabb_max": pmax.tolist(),
        "radius": radius,
        "n_outside_aabb": int(outside.sum()),
        "centroid_offset_from_scope": centroid_offset,
    }


# ── Test 1: projection overlay ────────────────────────────────────────────────

def test_projection_overlay(xyz_W, supervision_views, output_dir):
    from ModuleTBD.dataset_builder import write_projection_overlays, _SEED_DEPTH_MIN

    overlay_dir = Path(output_dir) / "projection_overlay"
    write_projection_overlays(xyz_W, supervision_views, overlay_dir)

    print("\n" + "=" * 65)
    print("TEST 1 — PROJECTION OVERLAY")
    print("=" * 65)
    print(f"  Overlays saved to: {overlay_dir}")
    print(f"  {'#':>3}  {'Source':<12} {'Az':>6} {'El':>6}  {'In-frame':>10}  {'Behind':>7}  Result")
    print("  " + "-" * 62)

    results = []
    for i, view in enumerate(supervision_views):
        cam = view["camera"]
        R = np.asarray(cam["R"], np.float64)
        T = np.asarray(cam["T"], np.float64).flatten()
        K = np.asarray(cam["K"], np.float64)
        W = int(cam["width"]); H = int(cam["height"])
        source = view.get("source", "?")
        az = cam.get("azimuth_offset_deg", 0.0)
        el = cam.get("elevation_offset_deg", 0.0)

        pts_c = (R @ xyz_W.T).T + T.reshape(1, 3)
        in_front = pts_c[:, 2] > _SEED_DEPTH_MIN
        n_behind = int((~in_front).sum())
        pts_f = pts_c[in_front]

        if pts_f.shape[0] == 0:
            print(f"  {i:>3}  {source:<12} {az:>6.1f} {el:>6.1f}  {'0':>10}  {n_behind:>7}  *** ALL BEHIND ***")
            results.append({"view_idx": i, "source": source, "n_in_frame": 0, "n_behind": n_behind})
            continue

        u = K[0, 0] * (pts_f[:, 0] / pts_f[:, 2]) + K[0, 2]
        v = K[1, 1] * (pts_f[:, 1] / pts_f[:, 2]) + K[1, 2]
        valid = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        n_in_frame = int(valid.sum())
        depths = pts_f[valid, 2]
        flag = "OK" if n_in_frame > 10 else "*** FEW IN FRAME ***"
        print(f"  {i:>3}  {source:<12} {az:>6.1f} {el:>6.1f}  {n_in_frame:>10}  {n_behind:>7}  {flag}")
        results.append({
            "view_idx": i, "source": source, "azimuth": az, "elevation": el,
            "n_in_frame": n_in_frame, "n_behind": n_behind,
            "mean_depth": float(depths.mean()) if depths.size else 0.0,
            "depth_min": float(depths.min()) if depths.size else 0.0,
            "depth_max": float(depths.max()) if depths.size else 0.0,
        })
    return results


# ── Test 3: manifest acceptance ────────────────────────────────────────────────

def test_halluc_manifest(halluc_index_path):
    print("\n" + "=" * 65)
    print("TEST 3 — HALLUCINATION MANIFEST ACCEPTANCE")
    print("=" * 65)
    if not Path(halluc_index_path).exists():
        print(f"  Not found: {halluc_index_path}")
        return
    with open(halluc_index_path) as f:
        manifest = json.load(f)
    frames = manifest.get("frames", [])
    n_total = len(frames)
    n_acc = sum(1 for fr in frames if fr.get("accepted"))
    print(f"  Total frames: {n_total}  accepted: {n_acc}  rejected: {n_total - n_acc}")
    if n_acc == 0:
        print("  *** CRITICAL: 0 accepted ***")
    print(f"\n  {'#':>3}  {'Az':>7} {'El':>7}  {'Accept':>7}  {'IoU':>6}  Path")
    print("  " + "-" * 60)
    for fr in frames:
        idx = fr.get("index", "?")
        az = fr.get("azimuth_deg", 0.0)
        el = fr.get("elevation_deg", 0.0)
        acc = "YES" if fr.get("accepted") else "NO"
        iou = fr.get("iou_with_objgs", 0.0) or 0.0
        path = Path(fr.get("out_rgba_path", "")).name
        print(f"  {idx:>3}  {az:>7.1f} {el:>7.1f}  {acc:>7}  {iou:>6.3f}  {path}")


# ── main ──────────────────────────────────────────────────────────────────────

def run(*, model_path, output_root, object_id):
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")

    from ModuleTBD.utils.scene_analysis import compute_object_scope
    from ModuleTBD.colmap_init import load_colmap_object_point_cloud
    from ModuleTBD.dataset_builder import build_supervision_views

    _SV3D_RESOLUTION = 576
    output_root_p = Path(output_root)
    obj_dir = output_root_p / f"obj_{object_id}"
    halluc_index = obj_dir / "03_novel_views" / "hallucination_index.json"
    extraction_index = obj_dir / "01_extraction" / "extraction_index.json"
    debug_out = obj_dir / "06_projection_audit"
    debug_out.mkdir(parents=True, exist_ok=True)

    test_halluc_manifest(halluc_index)

    print(f"\nLoading scope for object {object_id} ...")
    scope, frame, _pipe_config = compute_object_scope(model_path, int(object_id))

    print("Loading COLMAP seed points ...")
    pcd, metadata = load_colmap_object_point_cloud(
        model_path=model_path, object_id=int(object_id), scope=scope,
        extraction_index_path=extraction_index,
        max_points=20000, target_points=8000,
    )
    xyz_W = np.asarray(pcd.points, np.float32)
    print(f"  Loaded {xyz_W.shape[0]} seed points  (source: {metadata.get('init_source')})")

    with open(halluc_index) as f:
        manifest = json.load(f)
    cam_idx = int(manifest.get("conditioning", {}).get("cam_index", -1))
    if not (0 <= cam_idx < len(scope.cameras)):
        raise RuntimeError(f"conditioning.cam_index={cam_idx} out of range.")

    R_cond = np.asarray(scope.cameras[cam_idx]["R"], np.float64)
    cond_cam_up = -R_cond[1]
    cond_cam_up = cond_cam_up / np.linalg.norm(cond_cam_up)

    cur_az, cur_el = frame.world_to_virtual(
        np.asarray(scope.cameras[cam_idx]["position"], np.float32)
    )
    cur_az = ((float(cur_az) + 180.0) % 360.0) - 180.0
    man_az = float(manifest.get("conditioning", {}).get("azimuth_deg", cur_az))
    man_el = float(manifest.get("conditioning", {}).get("elevation_deg", cur_el))
    if abs(_signed_angle_delta_deg(man_az, cur_az)) > 0.5 or abs(man_el - float(cur_el)) > 0.5:
        print(f"  *** STALE manifest: ({man_az:.2f},{man_el:.2f}) vs ({cur_az:.2f},{cur_el:.2f}) ***")

    print("Building supervision views ...")
    supervision_views = build_supervision_views(
        halluc_index_path=halluc_index,
        extraction_index_path=extraction_index,
        scope=scope, frame=frame,
        seed_points_W=xyz_W,
        real_weight=1.0, hallucination_weight=1.0,
        fov_y_deg=50.0, resolution=_SV3D_RESOLUTION,
        real_target_long_edge=_SV3D_RESOLUTION,
        up_override=cond_cam_up,
    )
    print(f"  {len(supervision_views)} supervision views built.")

    geo = test_point_cloud_geometry(xyz_W, scope, supervision_views)
    proj = test_projection_overlay(xyz_W, supervision_views, debug_out)

    report = {
        "object_id": int(object_id),
        "n_seed_points": int(xyz_W.shape[0]),
        "init_source": metadata.get("init_source"),
        "geometry": geo,
        "projection": proj,
    }
    report_path = debug_out / "audit_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Full report: {report_path}")


def _parse_args():
    p = argparse.ArgumentParser(description="ModuleTBD projection audit.")
    p.add_argument("--model_path", required=True)
    p.add_argument("--output_root", default="ModuleTBD/outputs")
    p.add_argument("--object_id", type=int, required=True)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(model_path=args.model_path, output_root=args.output_root,
        object_id=args.object_id)
