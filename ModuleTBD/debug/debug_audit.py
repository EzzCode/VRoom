"""Supervision & Projection Audit — Verify training-data quality (ModuleTBD).

Combines two audit stages into one module:

**Stage 05 — Supervision manifest audit**
Reads ``04_supervision_manifest.json`` and runs:

1. File-existence check — every ``image_path`` must resolve on disk.
2. Weight distribution — real vs hallucinated weight ratio + bar chart.
3. Coverage polar plot — azimuth/elevation scatter coloured by source.
4. K/R/T sanity — rotation orthogonality, finite T, plausible K diagonal.
5. Projection check — if COLMAP seed points are loadable, projects them onto
   each supervision camera and reports how many land inside the image plane.
   Saves colour-coded overlay JPGs per view (red=far, blue=near).

Output under ``<obj_dir>/05_supervision_audit/debug/``::

    audit_report.json
    coverage_polar.png
    weight_distribution.png
    projection_overlays/view_NNN_<source>.jpg   (if model_path+object_id given)

**Stage 06 — Projection audit**
Rebuilds supervision views from scratch, then runs:

1. Projection overlay — seed points projected onto each rebuilt view.
2. Point-cloud geometry — centroid/AABB/camera-distance checks.
3. Hallucination manifest acceptance — per-frame IoU + accept/reject.

Output under ``<obj_dir>/06_projection_audit/debug/``::

    projection_overlay/     per-view PNGs
    audit_report.json       per-view numerical report

Usage (standalone)::

    # supervision audit only:
    python -m ModuleTBD.debug.debug_audit supervision \\
        --obj_dir ModuleTBD/outputs/obj_8 \\
        --model_path "temp_deps/ObjectGS/outputs/..." \\
        --object_id 8

    # projection audit only:
    python -m ModuleTBD.debug.debug_audit projection \\
        --model_path "temp_deps/ObjectGS/outputs/..." \\
        --output_root ModuleTBD/outputs \\
        --object_id 8

    # both (default):
    python -m ModuleTBD.debug.debug_audit both \\
        --obj_dir ModuleTBD/outputs/obj_8 \\
        --model_path "temp_deps/ObjectGS/outputs/..." \\
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

from ModuleTBD.constants import SEED_DEPTH_MIN


# ══════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ══════════════════════════════════════════════════════════════════════════════

def _rot_orthogonality_err(R):
    """Max element-wise deviation of R @ R.T from I."""
    R = np.asarray(R, np.float64)
    return float(np.abs(R @ R.T - np.eye(3)).max())


def _signed_angle_delta_deg(a, b):
    return float(((float(a) - float(b) + 180.0) % 360.0) - 180.0)


def _load_manifest(obj_dir):
    path = Path(obj_dir) / "04_supervision_manifest.json"
    if not path.exists():
        raise FileNotFoundError(
            f"04_supervision_manifest.json not found under {obj_dir}. "
            "Run pipeline with --debug to generate it."
        )
    with open(path) as f:
        return json.load(f)


# ══════════════════════════════════════════════════════════════════════════════
# Stage 05 — Supervision manifest checks
# ══════════════════════════════════════════════════════════════════════════════

def check_file_existence(views):
    """Return list of per-view {idx, source, exists, path}."""
    results = []
    for i, v in enumerate(views):
        p = v.get("image_path", "")
        exists = bool(p) and Path(p).exists()
        results.append({
            "view_idx": i,
            "source": v.get("source", "?"),
            "exists": exists,
            "path": str(p),
        })
    n_missing = sum(1 for r in results if not r["exists"])
    logger.info("[1] File existence: %d views, %d missing", len(results), n_missing)
    for r in results:
        if not r["exists"]:
            logger.warning("  MISSING view %d (%s): %s", r["view_idx"], r["source"], r["path"])
    return results


def check_weight_distribution(views, debug_dir):
    """Compute weight stats and save a bar chart."""
    real_weights = [v["weight"] for v in views if v.get("source") == "real"]
    halluc_weights = [v["weight"] for v in views if v.get("source") == "hallucinated"]

    real_total = float(sum(real_weights))
    halluc_total = float(sum(halluc_weights))
    grand_total = real_total + halluc_total

    logger.info("[2] Weight distribution:")
    logger.info("  real        : %3d views,  total weight = %.3f", len(real_weights), real_total)
    logger.info("  hallucinated: %3d views,  total weight = %.3f", len(halluc_weights), halluc_total)
    logger.info("  ratio real/(real+halluc) = %.3f", real_total / max(grand_total, 1e-9))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        sources = ["real", "hallucinated"]
        totals = [real_total, halluc_total]
        counts = [len(real_weights), len(halluc_weights)]
        bars = ax.bar(sources, totals, color=["steelblue", "coral"])
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"n={count}", ha="center", va="bottom", fontsize=9)
        ax.set_ylabel("Sum of weights")
        ax.set_title("Supervision view weight distribution")
        out = Path(debug_dir) / "weight_distribution.png"
        fig.savefig(str(out), dpi=100, bbox_inches="tight")
        plt.close(fig)
        logger.info("  Saved: %s", out.name)
    except Exception as exc:  # noqa: BLE001
        logger.debug("weight_distribution plot failed: %s", exc)

    return {
        "n_real": len(real_weights),
        "n_hallucinated": len(halluc_weights),
        "real_weight_total": real_total,
        "halluc_weight_total": halluc_total,
        "real_fraction": real_total / max(grand_total, 1e-9),
    }


def check_coverage(views, debug_dir):
    """Save azimuth/elevation polar scatter coloured by source."""
    real_az = [v["azimuth_deg"] for v in views if v.get("source") == "real"]
    real_el = [v["elevation_deg"] for v in views if v.get("source") == "real"]
    halluc_az = [v["azimuth_deg"] for v in views if v.get("source") == "hallucinated"]
    halluc_el = [v["elevation_deg"] for v in views if v.get("source") == "hallucinated"]
    cond_az = [v["azimuth_deg"] for v in views if v.get("is_conditioning")]
    cond_el = [v["elevation_deg"] for v in views if v.get("is_conditioning")]

    az_all = [v["azimuth_deg"] for v in views]
    el_all = [v["elevation_deg"] for v in views]
    az_norm = [float(a) % 360.0 for a in az_all]
    az_spread = 0.0
    if az_norm:
        az_sorted = sorted(az_norm)
        gaps = [(az_sorted[(i + 1) % len(az_sorted)] - az_sorted[i]) % 360.0 for i in range(len(az_sorted))]
        az_spread = float(360.0 - max(gaps))
    el_spread = float(max(el_all) - min(el_all)) if el_all else 0.0
    logger.info("[3] Coverage: az_spread=%.1f° el_spread=%.1f°", az_spread, el_spread)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="polar")

        def _plot(azs, els, label, marker, color, zorder=2):
            if not azs:
                return
            theta = [np.deg2rad(float(a) % 360.0) for a in azs]
            r = [90.0 - e for e in els]
            ax.scatter(theta, r, label=label, marker=marker,
                       color=color, s=40, zorder=zorder)

        _plot(real_az, real_el, "real", "o", "steelblue")
        _plot(halluc_az, halluc_el, "hallucinated", "^", "coral")
        _plot(cond_az, cond_el, "conditioning", "*", "gold", zorder=3)

        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        r_values = [90.0 - float(e) for e in el_all]
        r_max = max(90.0, max(r_values) + 5.0) if r_values else 90.0
        ax.set_rlim(0, r_max)
        ax.set_rticks([30, 60, 90] + ([round(r_max)] if r_max > 95 else []))
        ax.set_rlabel_position(10)
        ax.set_title("Supervision coverage\n(r = 90−elevation, θ = azimuth)", pad=14)
        ax.legend(loc="lower right", fontsize=8)
        out = Path(debug_dir) / "coverage_polar.png"
        fig.savefig(str(out), dpi=100, bbox_inches="tight")
        plt.close(fig)
        logger.info("  Saved: %s", out.name)
    except Exception as exc:  # noqa: BLE001
        logger.debug("coverage_polar plot failed: %s", exc)

    return {
        "az_spread_deg": az_spread,
        "el_spread_deg": el_spread,
        "az_min": float(min(az_norm)) if az_norm else None,
        "az_max": float(max(az_norm)) if az_norm else None,
        "el_min": float(min(el_all)) if el_all else None,
        "el_max": float(max(el_all)) if el_all else None,
    }


def check_krt_sanity(views):
    """Check rotation orthogonality, finite T, plausible K."""
    results = []
    logger.info("[4] K/R/T sanity (%d views):", len(views))
    logger.info("  %3s  %-12s %8s  %8s  %7s  %7s  OK?", "#", "Source", "R_err", "T_norm", "fx", "fy")
    logger.info("  " + "-" * 58)
    all_ok = True
    for i, v in enumerate(views):
        R = np.asarray(v["R_w2c"], np.float64)
        T = np.asarray(v["T_w2c"], np.float64).flatten()
        K = np.asarray(v["K"], np.float64)
        rot_err = _rot_orthogonality_err(R)
        t_norm = float(np.linalg.norm(T))
        fx = float(K[0, 0])
        fy = float(K[1, 1])
        ok = (
            rot_err < 1e-4
            and np.isfinite(T).all()
            and t_norm < 1e5
            and fx > 1.0 and fy > 1.0
        )
        if not ok:
            all_ok = False
        flag = "OK" if ok else "*** BAD ***"
        src = v.get("source", "?")
        logger.info("  %3d  %-12s %8.2e  %8.3f  %7.1f  %7.1f  %s", i, src, rot_err, t_norm, fx, fy, flag)
        results.append({
            "view_idx": i, "source": src, "rot_err": rot_err,
            "t_norm": t_norm, "fx": fx, "fy": fy, "ok": ok,
        })
    n_bad = sum(1 for r in results if not r["ok"])
    logger.info("  → %d OK, %d bad", len(results) - n_bad, n_bad)
    return results


def check_projections(views, xyz_W, debug_dir):
    """Project COLMAP seed points onto each view; save overlay JPGs."""
    import cv2

    overlay_dir = Path(debug_dir) / "projection_overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    xyz = np.asarray(xyz_W, np.float64)

    logger.info("[5] Projection check (%d views, %d seed pts):", len(views), xyz.shape[0])
    logger.info("  %3s  %-12s  %9s  %7s  %7s  Result", "#", "Source", "In-frame", "Behind", "CtrErr")
    logger.info("  " + "-" * 58)

    results = []
    for i, v in enumerate(views):
        R = np.asarray(v["R_w2c"], np.float64)
        T = np.asarray(v["T_w2c"], np.float64).flatten()
        K = np.asarray(v["K"], np.float64)
        W, H = int(v["width"]), int(v["height"])
        src = v.get("source", "?")
        az = float(v.get("azimuth_deg", 0.0))
        el = float(v.get("elevation_deg", 0.0))

        pts_c = (R @ xyz.T).T + T.reshape(1, 3)
        in_front = pts_c[:, 2] > SEED_DEPTH_MIN
        n_behind = int((~in_front).sum())
        pts_f = pts_c[in_front]

        n_in_frame = 0
        center_err_norm = None
        extent_ratio = None
        if pts_f.shape[0] > 0:
            u = K[0, 0] * (pts_f[:, 0] / pts_f[:, 2]) + K[0, 2]
            v_px = K[1, 1] * (pts_f[:, 1] / pts_f[:, 2]) + K[1, 2]
            valid = (u >= 0) & (u < W) & (v_px >= 0) & (v_px < H)
            n_in_frame = int(valid.sum())

            img_path = v.get("image_path", "")
            rgba = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED) if img_path and Path(img_path).exists() else None
            mask = None
            if rgba is not None:
                if rgba.ndim == 3 and rgba.shape[2] == 4:
                    mask = rgba[..., 3] > 127
                else:
                    rgb_tmp = rgba[..., :3] if rgba.ndim == 3 else cv2.cvtColor(rgba, cv2.COLOR_GRAY2BGR)
                    hsv = cv2.cvtColor(rgb_tmp, cv2.COLOR_BGR2HSV)
                    mask = (hsv[..., 1] > 12) | (hsv[..., 2] < 245)
            if src == "hallucinated" and mask is not None and int(mask.sum()) > 0 and n_in_frame > 10:
                u_v = u[valid]
                v_v = v_px[valid]
                p_x0 = float(np.percentile(u_v, 2)); p_x1 = float(np.percentile(u_v, 98))
                p_y0 = float(np.percentile(v_v, 2)); p_y1 = float(np.percentile(v_v, 98))
                ys, xs = np.where(mask)
                m_x0, m_x1 = float(xs.min()), float(xs.max() + 1)
                m_y0, m_y1 = float(ys.min()), float(ys.max() + 1)
                p_c = np.array([0.5 * (p_x0 + p_x1), 0.5 * (p_y0 + p_y1)], np.float32)
                m_c = np.array([0.5 * (m_x0 + m_x1), 0.5 * (m_y0 + m_y1)], np.float32)
                diag = float(np.hypot(max(m_x1 - m_x0, 1.0), max(m_y1 - m_y0, 1.0)))
                center_err_norm = float(np.linalg.norm(p_c - m_c) / max(diag, 1.0))
                p_extent = max(p_x1 - p_x0, p_y1 - p_y0)
                m_extent = max(m_x1 - m_x0, m_y1 - m_y0)
                extent_ratio = float(p_extent / max(m_extent, 1.0))

            if img_path and Path(img_path).exists():
                bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                if bgr is None:
                    bgr = np.full((H, W, 3), 30, np.uint8)
                else:
                    bgr = cv2.resize(bgr, (W, H))
                depths = pts_f[valid, 2]
                u_v = u[valid].astype(np.int32)
                v_v = v_px[valid].astype(np.int32)
                if depths.size:
                    d_lo, d_hi = depths.min(), depths.max()
                    d_norm = ((depths - d_lo) / max(d_hi - d_lo, 1e-6)).clip(0, 1)
                    cmap = cv2.applyColorMap(
                        (d_norm * 255).astype(np.uint8).reshape(-1, 1), cv2.COLORMAP_JET
                    )
                    for j, (pu, pv) in enumerate(zip(u_v, v_v)):
                        cv2.circle(bgr, (int(pu), int(pv)), 3,
                                   tuple(int(c) for c in cmap[j, 0]), -1)
                label = f"#{i} {src}  az={az:.0f} el={el:.0f}  n={n_in_frame}"
                cv2.putText(bgr, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 255, 255), 1, cv2.LINE_AA)
                fname = overlay_dir / f"view_{i:03d}_{src}.jpg"
                cv2.imwrite(str(fname), bgr)

        align_ok = center_err_norm is None or center_err_norm <= 0.25
        flag = "OK" if n_in_frame > 10 and align_ok else "*** MISALIGN ***" if n_in_frame > 10 else "*** FEW ***"
        ctr = f"{center_err_norm:.2f}" if center_err_norm is not None else "n/a"
        logger.info("  %3d  %-12s  %9d  %7d  %7s  %s", i, src, n_in_frame, n_behind, ctr, flag)
        results.append({
            "view_idx": i, "source": src, "azimuth_deg": az,
            "n_in_frame": n_in_frame, "n_behind": n_behind,
            "projection_mask_center_error_norm": center_err_norm,
            "projection_mask_extent_ratio": extent_ratio,
        })

    n_ok = sum(1 for r in results if r["n_in_frame"] > 10)
    logger.info("  → %d/%d views have >10 seed points in frame", n_ok, len(results))
    if n_ok < len(results) // 2:
        logger.warning(
            "*** majority of views have few in-frame points — possible coordinate-frame mismatch ***"
        )
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Stage 06 — Projection audit tests (rebuilds views from scratch)
# ══════════════════════════════════════════════════════════════════════════════

def test_halluc_manifest(halluc_index_path):
    logger.info("=" * 65)
    logger.info("TEST 3 — HALLUCINATION MANIFEST ACCEPTANCE")
    logger.info("=" * 65)
    if not Path(halluc_index_path).exists():
        logger.warning("  Not found: %s", halluc_index_path)
        return
    with open(halluc_index_path) as f:
        manifest = json.load(f)
    frames = manifest.get("frames", [])
    n_total = len(frames)
    n_acc = sum(1 for fr in frames if fr.get("accepted"))
    logger.info("  Total frames: %d  accepted: %d  rejected: %d", n_total, n_acc, n_total - n_acc)
    if n_acc == 0:
        logger.warning("*** CRITICAL: 0 accepted frames ***")
    logger.info("  %3s  %7s %7s  %7s  %6s  Path", "#", "Az", "El", "Accept", "IoU")
    logger.info("  " + "-" * 58)
    for fr in frames:
        idx = fr.get("index", "?")
        az = fr.get("azimuth_deg", 0.0)
        el = fr.get("elevation_deg", 0.0)
        acc = "YES" if fr.get("accepted") else "NO"
        iou = fr.get("iou_with_objgs", 0.0) or 0.0
        path = Path(fr.get("out_rgba_path", "")).name
        logger.info("  %3s  %7.1f %7.1f  %7s  %6.3f  %s", str(idx), az, el, acc, iou, path)


def test_point_cloud_geometry(xyz_W, scope, supervision_views):
    centroid = xyz_W.mean(axis=0)
    pmin = xyz_W.min(axis=0)
    pmax = xyz_W.max(axis=0)
    extent = pmax - pmin
    radius = float(np.linalg.norm(extent) / 2.0)

    logger.info("=" * 65)
    logger.info("TEST 2 — POINT CLOUD GEOMETRY")
    logger.info("=" * 65)
    logger.info("  N points        : %d", xyz_W.shape[0])
    logger.info("  Centroid (pcd)  : %s", centroid)
    logger.info("  scope.centroid  : %s", np.asarray(scope.centroid))
    logger.info("  scope.radius    : %.4f", scope.radius)
    logger.info("  scope.aabb_min  : %s", np.asarray(scope.aabb_min))
    logger.info("  scope.aabb_max  : %s", np.asarray(scope.aabb_max))

    centroid_offset = float(np.linalg.norm(centroid - np.asarray(scope.centroid, np.float32)))
    logger.info("  centroid offset : %.4f  (warn > %.4f)", centroid_offset, scope.radius * 0.5)
    if centroid_offset > scope.radius * 0.5:
        logger.warning("*** seed centroid far from scope centroid ***")

    aabb_min = np.asarray(scope.aabb_min, np.float32)
    aabb_max = np.asarray(scope.aabb_max, np.float32)
    outside = np.any((xyz_W < aabb_min - 0.01) | (xyz_W > aabb_max + 0.01), axis=1)
    logger.info("  Points outside scope AABB: %d / %d", int(outside.sum()), xyz_W.shape[0])

    logger.info("  %-12s %8s %10s  cam->centroid", "Source", "Az", "El")
    logger.info("  " + "-" * 48)
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
        logger.info("  %-12s %8.1f %10.1f  %.4f", src, az, el, d)

    return {
        "n_points": int(xyz_W.shape[0]),
        "centroid": centroid.tolist(),
        "aabb_min": pmin.tolist(),
        "aabb_max": pmax.tolist(),
        "radius": radius,
        "n_outside_aabb": int(outside.sum()),
        "centroid_offset_from_scope": centroid_offset,
    }


def test_projection_overlay(xyz_W, supervision_views, output_dir):
    from ModuleTBD.dataset_builder import write_projection_overlays

    overlay_dir = Path(output_dir) / "projection_overlay"
    write_projection_overlays(xyz_W, supervision_views, overlay_dir)

    logger.info("=" * 65)
    logger.info("TEST 1 — PROJECTION OVERLAY")
    logger.info("=" * 65)
    logger.info("  Overlays saved to: %s", overlay_dir)
    logger.info("  %3s  %-12s %6s %6s  %10s  %7s  Result", "#", "Source", "Az", "El", "In-frame", "Behind")
    logger.info("  " + "-" * 60)

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
        in_front = pts_c[:, 2] > SEED_DEPTH_MIN
        n_behind = int((~in_front).sum())
        pts_f = pts_c[in_front]

        if pts_f.shape[0] == 0:
            logger.warning(
                "  %3d  %-12s %6.1f %6.1f  %10d  %7d  *** ALL BEHIND ***",
                i, source, az, el, 0, n_behind,
            )
            results.append({"view_idx": i, "source": source, "n_in_frame": 0, "n_behind": n_behind})
            continue

        u = K[0, 0] * (pts_f[:, 0] / pts_f[:, 2]) + K[0, 2]
        v = K[1, 1] * (pts_f[:, 1] / pts_f[:, 2]) + K[1, 2]
        valid = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        n_in_frame = int(valid.sum())
        depths = pts_f[valid, 2]
        flag = "OK" if n_in_frame > 10 else "*** FEW IN FRAME ***"
        logger.info(
            "  %3d  %-12s %6.1f %6.1f  %10d  %7d  %s",
            i, source, az, el, n_in_frame, n_behind, flag,
        )
        results.append({
            "view_idx": i, "source": source, "azimuth": az, "elevation": el,
            "n_in_frame": n_in_frame, "n_behind": n_behind,
            "mean_depth": float(depths.mean()) if depths.size else 0.0,
            "depth_min": float(depths.min()) if depths.size else 0.0,
            "depth_max": float(depths.max()) if depths.size else 0.0,
        })
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Internal runners
# ══════════════════════════════════════════════════════════════════════════════

def _run_supervision_audit(obj_dir, scope, frame, model_path, object_id, debug_dir):
    """Run the stage-05 supervision manifest audit."""
    debug_dir = Path(debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)

    manifest = _load_manifest(obj_dir)
    views = manifest.get("views", [])
    if not views:
        logger.warning("supervision_audit: manifest has no views")
        return {"error": "no views"}

    n_real = sum(1 for v in views if v.get("source") == "real")
    n_hallucinated = sum(1 for v in views if v.get("source") == "hallucinated")
    n_unknown = len(views) - n_real - n_hallucinated

    logger.info("=" * 65)
    logger.info("SUPERVISION AUDIT  (stage 05)")
    logger.info("=" * 65)
    logger.info("  Manifest: %s", Path(obj_dir) / "04_supervision_manifest.json")
    logger.info("  Views: %d  (real=%d, hallucinated=%d)", len(views), n_real, n_hallucinated)
    if n_unknown:
        logger.warning("  WARNING: %d views have unknown source — manifest may be from old pipeline run", n_unknown)

    report: dict = {"n_views": len(views), "n_real": n_real, "n_hallucinated": n_hallucinated}

    report["file_existence"] = check_file_existence(views)
    report["weight_distribution"] = check_weight_distribution(views, debug_dir)
    report["coverage"] = check_coverage(views, debug_dir)
    report["krt_sanity"] = check_krt_sanity(views)

    if model_path is not None and object_id is not None and scope is not None:
        try:
            from ModuleTBD.utils.colmap_init import load_colmap_object_point_cloud
            extraction_index = Path(obj_dir) / "01_extraction" / "extraction_index.json"
            pcd, _meta = load_colmap_object_point_cloud(
                model_path=str(model_path),
                object_id=int(object_id),
                scope=scope,
                extraction_index_path=extraction_index if extraction_index.exists() else None,
                max_points=20000, target_points=8000,
            )
            xyz_W = np.asarray(pcd.points, np.float32)
            report["projection"] = check_projections(views, xyz_W, debug_dir)
        except Exception as exc:  # noqa: BLE001
            logger.warning("supervision_audit projection check failed: %s", exc)
            report["projection"] = {"error": str(exc)}
    else:
        logger.info("[5] Projection check: skipped (model_path/object_id/scope not provided)")

    report_path = debug_dir / "audit_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("  Audit report: %s", report_path)
    return report


def _run_projection_audit(obj_dir, model_path, object_id, debug_dir):
    """Run the stage-06 projection audit (rebuilds views from scratch)."""
    from ModuleTBD.utils.scene_analysis import compute_object_scope
    from ModuleTBD.utils.colmap_init import load_colmap_object_point_cloud
    from ModuleTBD.dataset_builder import build_supervision_views

    _SV3D_RESOLUTION = 576
    obj_dir = Path(obj_dir)
    debug_dir = Path(debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)

    halluc_index = obj_dir / "03_novel_views" / "hallucination_index.json"
    extraction_index = obj_dir / "01_extraction" / "extraction_index.json"

    test_halluc_manifest(halluc_index)

    logger.info("Loading scope for object %s ...", object_id)
    scope, frame, _pipe_config = compute_object_scope(model_path, int(object_id))

    logger.info("Loading COLMAP seed points ...")
    pcd, metadata = load_colmap_object_point_cloud(
        model_path=model_path, object_id=int(object_id), scope=scope,
        extraction_index_path=extraction_index,
        max_points=20000, target_points=8000,
    )
    xyz_W = np.asarray(pcd.points, np.float32)
    logger.info("  Loaded %d seed points  (source: %s)", xyz_W.shape[0], metadata.get("init_source"))

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
        logger.warning(
            "*** STALE manifest: (%.2f,%.2f) vs (%.2f,%.2f) ***",
            man_az, man_el, cur_az, float(cur_el),
        )

    logger.info("Building supervision views ...")
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
    logger.info("  %d supervision views built.", len(supervision_views))

    geo = test_point_cloud_geometry(xyz_W, scope, supervision_views)
    proj = test_projection_overlay(xyz_W, supervision_views, debug_dir)

    report = {
        "object_id": int(object_id),
        "n_seed_points": int(xyz_W.shape[0]),
        "init_source": metadata.get("init_source"),
        "geometry": geo,
        "projection": proj,
    }
    report_path = debug_dir / "audit_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("  Full report: %s", report_path)
    return report


# ══════════════════════════════════════════════════════════════════════════════
# Public entry point
# ══════════════════════════════════════════════════════════════════════════════

def generate_debug_artifacts(
    *,
    obj_dir,
    scope=None,
    frame=None,
    model_path=None,
    object_id=None,
):
    """Run both audit stages and return combined results.

    Parameters
    ----------
    obj_dir      : object pipeline output dir (must contain 04_supervision_manifest.json)
    scope        : ObjectScope (enables COLMAP projection check in stage 05)
    frame        : ObjectFrame (used by stage 06 to rebuild views)
    model_path   : path to trained gstrain model directory (enables both projection checks)
    object_id    : integer object label id (enables both projection checks)
    """
    obj_dir = Path(obj_dir)
    results = {}

    # Stage 05 — supervision manifest audit
    try:
        results["supervision"] = _run_supervision_audit(
            obj_dir=obj_dir,
            scope=scope,
            frame=frame,
            model_path=model_path,
            object_id=object_id,
            debug_dir=obj_dir / "05_supervision_audit" / "debug",
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("supervision_audit failed: %s", exc)
        results["supervision"] = {"error": str(exc)}

    # Stage 06 — projection audit (rebuild views from scratch; needs model_path)
    if model_path is not None and object_id is not None:
        try:
            results["projection"] = _run_projection_audit(
                obj_dir=obj_dir,
                model_path=model_path,
                object_id=object_id,
                debug_dir=obj_dir / "06_projection_audit" / "debug",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("projection_audit failed: %s", exc)
            results["projection"] = {"error": str(exc)}

    return results


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args():
    p = argparse.ArgumentParser(
        description="ModuleTBD supervision + projection audit.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("mode", nargs="?", default="both",
                   choices=["both", "supervision", "projection"],
                   help="Which audit stage to run.")
    p.add_argument("--obj_dir", default=None,
                   help="Object pipeline output dir (required for supervision/both).")
    p.add_argument("--model_path", default=None,
                   help="Trained gstrain model path (required for projection/both).")
    p.add_argument("--output_root", default=None,
                   help="Pipeline output root (used to infer obj_dir when --obj_dir not given).")
    p.add_argument("--object_id", type=int, default=None,
                   help="Integer object label id.")
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    args = _parse_args()

    obj_dir = args.obj_dir
    if obj_dir is None and args.output_root is not None and args.object_id is not None:
        obj_dir = str(Path(args.output_root) / f"obj_{args.object_id}")
    if obj_dir is None:
        raise SystemExit("--obj_dir (or --output_root + --object_id) is required.")

    scope = None
    frame_obj = None
    if args.model_path and args.object_id is not None:
        try:
            from ModuleTBD.utils.scene_analysis import compute_object_scope
            scope, frame_obj, _ = compute_object_scope(args.model_path, args.object_id)
        except Exception as exc:
            logger.warning("Could not load scope: %s — projection check will be skipped", exc)

    if args.mode == "supervision":
        debug_dir = Path(obj_dir) / "05_supervision_audit" / "debug"
        _run_supervision_audit(
            obj_dir=obj_dir, scope=scope, frame=frame_obj,
            model_path=args.model_path, object_id=args.object_id,
            debug_dir=debug_dir,
        )
    elif args.mode == "projection":
        if args.model_path is None or args.object_id is None:
            raise SystemExit("--model_path and --object_id are required for projection mode.")
        debug_dir = Path(obj_dir) / "06_projection_audit" / "debug"
        _run_projection_audit(
            obj_dir=obj_dir, model_path=args.model_path,
            object_id=args.object_id, debug_dir=debug_dir,
        )
    else:  # both
        generate_debug_artifacts(
            obj_dir=obj_dir, scope=scope, frame=frame_obj,
            model_path=args.model_path, object_id=args.object_id,
        )
