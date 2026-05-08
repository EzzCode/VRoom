"""
Projection Audit — Two Sanity Tests Before Training

Test 1 — Projection Overlay
    Load the 8k COLMAP seed points for the object.
    Load every supervision view (real + hallucinated) and its camera (R, T, K).
    Project the 3D points onto each 2D image using raw matrix math (no ObjectGS).
    Overlay the projections as red dots on the supervision image and save to disk.
    If the dots DON'T trace the banana silhouette → coordinate-frame is broken.

Test 2 — Point Cloud Geometry
    Print min/max/centroid/radius of the seed points.
    Print distance from centroid to each camera.
    Flag if anything looks pathological.

Usage (from VRoom root):
    conda activate objectgs
    python -m object_isolation.debug.projection_audit \
        --model_path "temp_deps/ObjectGS/outputs/3dovs/2d_crossentropy_loss_01/2026-03-19_04-01-38" \
        --output_root object_isolation/outputs \
        --object_id 8
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

# ── path setup ─────────────────────────────────────────────────────────────
_VROOM_ROOT = Path(__file__).resolve().parents[2]
_OBJECTGS_DIR = _VROOM_ROOT / "temp_deps" / "ObjectGS"
if str(_OBJECTGS_DIR) not in sys.path:
    sys.path.insert(0, str(_OBJECTGS_DIR))
if str(_VROOM_ROOT) not in sys.path:
    sys.path.insert(0, str(_VROOM_ROOT))

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Pure-numpy projection (no ObjectGS, no torch, no magic)
# ─────────────────────────────────────────────────────────────────────────────

def project_points(
    xyz_W: np.ndarray,   # (N, 3)  world-space points
    R_w2c: np.ndarray,   # (3, 3)  world-to-camera rotation
    T_w2c: np.ndarray,   # (3,)    world-to-camera translation
    K: np.ndarray,       # (3, 3)  intrinsics
    width: int,
    height: int,
    min_depth: float = 0.01,
) -> np.ndarray:
    """Return (M, 2) pixel coords of points in front of the camera."""
    # Transform to camera space
    pts_c = (R_w2c @ xyz_W.T).T + T_w2c.reshape(1, 3)   # (N, 3)
    # Keep only points in front
    in_front = pts_c[:, 2] > min_depth
    pts_c = pts_c[in_front]
    if pts_c.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    # Perspective divide + apply K
    x = pts_c[:, 0] / pts_c[:, 2]
    y = pts_c[:, 1] / pts_c[:, 2]
    u = K[0, 0] * x + K[0, 2]
    v = K[1, 1] * y + K[1, 2]
    # Clip to image
    valid = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    return np.stack([u[valid], v[valid]], axis=1).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: Point Cloud Geometry
# ─────────────────────────────────────────────────────────────────────────────

def test_point_cloud_geometry(
    xyz_W: np.ndarray,
    scope,
    supervision_views: list[dict],
) -> dict:
    centroid = xyz_W.mean(axis=0)
    pmin = xyz_W.min(axis=0)
    pmax = xyz_W.max(axis=0)
    extent = pmax - pmin
    radius = float(np.linalg.norm(extent) / 2.0)
    dists_to_centroid = np.linalg.norm(xyz_W - centroid, axis=1)

    print("\n" + "=" * 65)
    print("TEST 2 — POINT CLOUD GEOMETRY")
    print("=" * 65)
    print(f"  N points        : {xyz_W.shape[0]}")
    print(f"  Min XYZ         : [{pmin[0]:.4f}, {pmin[1]:.4f}, {pmin[2]:.4f}]")
    print(f"  Max XYZ         : [{pmax[0]:.4f}, {pmax[1]:.4f}, {pmax[2]:.4f}]")
    print(f"  Centroid        : [{centroid[0]:.4f}, {centroid[1]:.4f}, {centroid[2]:.4f}]")
    print(f"  Diagonal extent : {radius * 2:.4f}  (radius ≈ {radius:.4f})")
    print(f"  scope.centroid  : {np.asarray(scope.centroid_W)}")
    print(f"  scope.radius    : {scope.radius:.4f}")
    print(f"  scope.aabb_min  : {np.asarray(scope.aabb_min_W)}")
    print(f"  scope.aabb_max  : {np.asarray(scope.aabb_max_W)}")

    centroid_offset = np.linalg.norm(centroid - np.asarray(scope.centroid_W, dtype=np.float32))
    print(f"\n  COLMAP centroid vs scope.centroid: {centroid_offset:.4f} world units")
    if centroid_offset > radius * 0.2:
        print("  *** WARNING: COLMAP seed centroid deviates from scope centroid by "
              f"{centroid_offset:.4f} (>{radius * 0.2:.4f}). "
              "Seed points may not be centred on the object! ***")

    # Spread check: are the points inside the scope AABB?
    aabb_min = np.asarray(scope.aabb_min_W, dtype=np.float32)
    aabb_max = np.asarray(scope.aabb_max_W, dtype=np.float32)
    outside_aabb = np.any((xyz_W < aabb_min - 0.01) | (xyz_W > aabb_max + 0.01), axis=1)
    print(f"\n  Points outside scope AABB: {int(outside_aabb.sum())} / {xyz_W.shape[0]}")
    if outside_aabb.sum() > xyz_W.shape[0] * 0.3:
        print("  *** WARNING: >30% of seed points fall outside the scope AABB! "
              "The COLMAP init may be pulling in unrelated scene geometry. ***")

    # Camera-to-centroid distance for each supervision view
    print(f"\n  {'Source':<12} {'Azimuth':>8} {'Elevation':>10}  Camera-to-centroid dist")
    print("  " + "-" * 58)
    cam_dists = []
    for v in supervision_views:
        cp = v["camera"].get("position")
        if cp is None:
            continue
        c = np.asarray(cp, dtype=np.float32)
        d = float(np.linalg.norm(c - centroid))
        cam_dists.append(d)
        az = v["camera"].get("azimuth_offset_deg", 0.0)
        el = v["camera"].get("elevation_offset_deg", 0.0)
        src = v.get("source", "?")
        print(f"  {src:<12} {az:>8.1f} {el:>10.1f}  {d:.4f}")

    if cam_dists:
        print(f"\n  Camera dist range: [{min(cam_dists):.4f}, {max(cam_dists):.4f}]")
        print(f"  scope.radius     : {scope.radius:.4f}")
        ratio_range = [d / max(scope.radius, 1e-6) for d in cam_dists]
        print(f"  dist/radius range: [{min(ratio_range):.2f}, {max(ratio_range):.2f}]  (expect ≈ 1–3)")
        if max(ratio_range) > 10 or min(ratio_range) < 0.1:
            print("  *** WARNING: Camera-to-centroid distances are pathological. "
                  "Check scope.radius and camera placement. ***")

    return {
        "n_points": xyz_W.shape[0],
        "centroid": centroid.tolist(),
        "aabb_min": pmin.tolist(),
        "aabb_max": pmax.tolist(),
        "radius": float(radius),
        "n_outside_aabb": int(outside_aabb.sum()),
        "centroid_offset_from_scope": float(centroid_offset),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: Projection Overlay
# ─────────────────────────────────────────────────────────────────────────────

def test_projection_overlay(
    xyz_W: np.ndarray,
    supervision_views: list[dict],
    output_dir: Path,
) -> list[dict]:
    overlay_dir = output_dir / "projection_overlay"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 65)
    print("TEST 1 — PROJECTION OVERLAY")
    print("=" * 65)
    print(f"  Saving overlays to: {overlay_dir}")
    print(f"  {'#':>3}  {'Source':<12} {'Az':>6} {'El':>6}  {'In-frame':>10}  {'Behind':>7}  Result")
    print("  " + "-" * 62)

    results = []
    for i, view in enumerate(supervision_views):
        cam = view["camera"]
        R = np.asarray(cam["R"], dtype=np.float64)
        T = np.asarray(cam["T"], dtype=np.float64).flatten()
        K = np.asarray(cam["K"], dtype=np.float64)
        W = int(cam["width"])
        H = int(cam["height"])
        source = view.get("source", "?")

        # ── Project ──────────────────────────────────────────────────────
        pts_c = (R @ xyz_W.T).T + T.reshape(1, 3)
        in_front = pts_c[:, 2] > 0.01
        n_behind = int((~in_front).sum())
        pts_c_front = pts_c[in_front]

        if pts_c_front.shape[0] == 0:
            print(f"  {i:>3}  {source:<12} "
                  f"{cam.get('azimuth_offset_deg',0):>6.1f} {cam.get('elevation_offset_deg',0):>6.1f}"
                  f"  {'0':>10}  {n_behind:>7}  *** ALL POINTS BEHIND CAMERA ***")
            results.append({"view_idx": i, "source": source, "n_in_frame": 0, "n_behind": n_behind,
                             "mean_depth": None, "depth_min": None, "depth_max": None})
            continue

        x = pts_c_front[:, 0] / pts_c_front[:, 2]
        y = pts_c_front[:, 1] / pts_c_front[:, 2]
        u = K[0, 0] * x + K[0, 2]
        v = K[1, 1] * y + K[1, 2]
        valid_mask = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        n_in_frame = int(valid_mask.sum())
        depths = pts_c_front[valid_mask, 2]
        mean_depth = float(depths.mean()) if depths.size else 0.0
        depth_min = float(depths.min()) if depths.size else 0.0
        depth_max = float(depths.max()) if depths.size else 0.0

        az = cam.get("azimuth_offset_deg", 0.0)
        el = cam.get("elevation_offset_deg", 0.0)
        flag = "OK" if n_in_frame > 10 else "*** FEW POINTS IN FRAME ***"
        print(f"  {i:>3}  {source:<12} {az:>6.1f} {el:>6.1f}"
              f"  {n_in_frame:>10}  {n_behind:>7}  {flag}")

        # ── Build overlay image ───────────────────────────────────────────
        rgb = np.asarray(view["rgb"], dtype=np.uint8)
        if rgb.shape[0] != H or rgb.shape[1] != W:
            rgb = cv2.resize(rgb, (W, H))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        # Draw mask outline in green
        mask_np = np.asarray(view["mask"], dtype=np.uint8) * 255
        if mask_np.shape[0] != H or mask_np.shape[1] != W:
            mask_np = cv2.resize(mask_np, (W, H), interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(bgr, contours, -1, (0, 255, 0), 2)

        # Draw projected COLMAP points in red
        u_valid = u[valid_mask].astype(np.int32)
        v_valid = v[valid_mask].astype(np.int32)
        for pu, pv in zip(u_valid, v_valid):
            cv2.circle(bgr, (int(pu), int(pv)), 3, (0, 0, 255), -1)

        # Depth-colorize to help diagnose flipped Z
        # Draw a legend bar with depth range
        if depths.size > 0:
            d_norm = ((depths - depth_min) / max(depth_max - depth_min, 1e-6)).clip(0, 1)
            cmap = cv2.applyColorMap((d_norm * 255).astype(np.uint8).reshape(-1, 1), cv2.COLORMAP_JET)
            for j, (pu, pv) in enumerate(zip(u_valid, v_valid)):
                col = tuple(int(c) for c in cmap[j, 0])
                cv2.circle(bgr, (int(pu), int(pv)), 2, col, -1)

        # Annotate
        cv2.putText(bgr, f"{source} az={az:.0f} el={el:.0f}",
                    (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(bgr, f"pts in-frame={n_in_frame} behind={n_behind}",
                    (6, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(bgr, f"depth [{depth_min:.3f}, {depth_max:.3f}]  mean={mean_depth:.3f}",
                    (6, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        fname = f"{i:03d}_{source}_az{az:.0f}_el{el:.0f}.jpg"
        cv2.imwrite(str(overlay_dir / fname), bgr)

        results.append({
            "view_idx": i,
            "source": source,
            "azimuth": az,
            "elevation": el,
            "n_in_frame": n_in_frame,
            "n_behind": n_behind,
            "mean_depth": mean_depth,
            "depth_min": depth_min,
            "depth_max": depth_max,
        })

    # ── Summarise ─────────────────────────────────────────────────────────
    n_ok = sum(1 for r in results if r["n_in_frame"] > 10)
    n_bad = len(results) - n_ok
    print(f"\n  Total views: {len(results)}   in-frame>10pts: {n_ok}   problematic: {n_bad}")
    if n_bad > 0:
        print("  *** DIAGNOSIS: Some views have few/no COLMAP points in frame.")
        print("      Possible causes:")
        print("      A) Camera T is expressed in the wrong frame (L vs W).")
        print("      B) K uses different units / resolution than the rendered image.")
        print("      C) R and T signs are flipped (OpenCV vs OpenGL convention).")
        print("      D) Points are in local frame but cameras are in world frame or vice-versa.")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Bonus Test 3: Inspect hallucination manifest frames
# ─────────────────────────────────────────────────────────────────────────────

def test_halluc_manifest(halluc_index_path: Path) -> None:
    print("\n" + "=" * 65)
    print("TEST 3 — HALLUCINATION MANIFEST ACCEPTANCE")
    print("=" * 65)
    if not halluc_index_path.exists():
        print(f"  Not found: {halluc_index_path}")
        return
    with open(halluc_index_path) as f:
        manifest = json.load(f)
    frames = manifest.get("frames", [])
    n_total = len(frames)
    n_accepted = sum(1 for f in frames if f.get("accepted", False))
    print(f"  Total frames in manifest : {n_total}")
    print(f"  Accepted by Phase 5      : {n_accepted}")
    print(f"  Rejected by Phase 5      : {n_total - n_accepted}")
    if n_accepted == 0:
        print("  *** CRITICAL: Phase 5 accepted 0 frames — no hallucinated views at all! ***")
    print()
    print(f"  {'#':>3}  {'Az':>7} {'El':>7}  {'Accepted':>9}  {'IoU':>6}  Path")
    print("  " + "-" * 70)
    for fr in frames:
        idx = fr.get("index", "?")
        az = fr.get("azimuth_V_deg", 0.0)
        el = fr.get("elevation_V_deg", 0.0)
        acc = "YES" if fr.get("accepted") else "NO"
        iou = fr.get("iou_with_objgs", 0.0) or 0.0
        path = Path(fr.get("out_rgba_path", "")).name
        exists_mark = "" if not fr.get("out_rgba_path") else (
            "✓" if Path(_VROOM_ROOT / fr["out_rgba_path"]).exists() else "MISSING"
        )
        print(f"  {idx:>3}  {az:>7.1f} {el:>7.1f}  {acc:>9}  {iou:>6.3f}  {path} {exists_mark}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(
    *,
    model_path: str,
    output_root: str,
    object_id: int,
) -> None:
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    from object_isolation.core.object_scope import discover_object_scope
    from object_isolation.core.colmap_init import load_colmap_object_point_cloud
    from object_isolation.core.pipeline import run_pipeline   # noqa: just to warm imports
    from object_isolation.core.dataset_builder import build_joint_supervision_views

    output_root_p = Path(output_root)
    obj_dir = output_root_p / f"obj_{object_id}"
    halluc_index = obj_dir / "phase5" / "hallucination_index.json"
    extraction_index = obj_dir / "phase3" / "extraction_index.json"
    debug_out = obj_dir / "debug_projection_audit"
    debug_out.mkdir(parents=True, exist_ok=True)

    # ── Test 3: manifest ──────────────────────────────────────────────────
    test_halluc_manifest(halluc_index)

    # ── Load scope ────────────────────────────────────────────────────────
    print(f"\nLoading scope for object {object_id} from {model_path} ...")
    scope, world_local, local_sv3d, gaussians, pipe_config = discover_object_scope(
        model_path, int(object_id)
    )

    # ── Load COLMAP seed points ───────────────────────────────────────────
    print(f"Loading COLMAP seed points ...")
    pcd, metadata = load_colmap_object_point_cloud(
        model_path=model_path,
        object_id=int(object_id),
        scope=scope,
        extraction_index_path=extraction_index,
        max_points=20000,
        target_points=8000,
    )
    xyz_W = np.asarray(pcd.points, dtype=np.float32)
    print(f"  Loaded {xyz_W.shape[0]} seed points  (source: {metadata.get('init_source')})")

    # ── Read conditioning-camera up vector (matches training exactly) ─────
    cond_cam_up_W: np.ndarray | None = None
    try:
        with open(halluc_index) as _f:
            _manifest = json.load(_f)
        _cam_idx = int(_manifest.get("conditioning", {}).get("cam_index", -1))
        if 0 <= _cam_idx < len(scope.cameras):
            _R_cond = np.asarray(scope.cameras[_cam_idx]["R"], dtype=np.float64)
            cond_cam_up_W = -_R_cond[1]
            print(f"  Using cond cam {_cam_idx} up vector for up_W_override.")
    except Exception as _e:
        print(f"  Could not read cond cam up ({_e}); using scope.up_W.")

    # ── Load supervision views (mirrors training call exactly) ────────────
    print(f"Building supervision views ...")
    supervision_views = build_joint_supervision_views(
        halluc_index_path=halluc_index,
        extraction_index_path=extraction_index,
        scope=scope,
        local_sv3d=local_sv3d,
        real_weight=1.0,
        hallucination_weight=1.0,
        fov_y_deg=50.0,
        hallucination_resolution=576,
        real_target_long_edge=576,
        up_W_override=cond_cam_up_W,
        seed_points_W=xyz_W,
    )
    print(f"  {len(supervision_views)} supervision views built.")

    # ── Run tests ─────────────────────────────────────────────────────────
    geo_report = test_point_cloud_geometry(xyz_W, scope, supervision_views)
    proj_results = test_projection_overlay(xyz_W, supervision_views, debug_out)

    # ── Save JSON report ──────────────────────────────────────────────────
    report = {
        "object_id": int(object_id),
        "n_seed_points": xyz_W.shape[0],
        "init_source": metadata.get("init_source"),
        "geometry": geo_report,
        "projection": proj_results,
    }
    report_path = debug_out / "audit_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Full report: {report_path}")
    print(f"  Overlay images: {debug_out / 'projection_overlay'}\n")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Projection audit — coordinate-space sanity tests")
    p.add_argument("--model_path", required=True)
    p.add_argument("--output_root", default="object_isolation/outputs")
    p.add_argument("--object_id", type=int, required=True)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        model_path=args.model_path,
        output_root=args.output_root,
        object_id=args.object_id,
    )
