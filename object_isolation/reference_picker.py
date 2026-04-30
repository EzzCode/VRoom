"""Phase 2.1 — pick the cleanest, most front-facing, well-lit reference frame.

Score every Phase-1 view and pick exactly one. The score is a weighted sum
of five terms; each term is normalized into ``[0, 1]`` so the weights are
comparable. The chosen frame's ``meta.json`` row plus all per-term scores
are written to ``reference.json`` for auditability.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ScoreWeights:
    """Per-term weights. Defaults sum to 1.0."""

    center: float = 0.10
    area: float = 0.15
    clip: float = 0.10
    front: float = 0.15
    light: float = 0.10
    complete: float = 0.40


@dataclass
class ScoreBreakdown:
    """All raw scores for one candidate view (for debug)."""

    frame_index: int
    img_name: str
    center_score: float
    area_score: float
    clip_score: float  # higher = less clipping
    front_score: float
    light_score: float
    complete_score: float  # fraction of object anchors inside full image
    total: float


# ── Per-term scorers ────────────────────────────────────────────────────────


def _center_score(K: np.ndarray, width: int, height: int, mean_uv: tuple[float, float]) -> float:
    """Centroid of object pixels close to the principal point -> 1.0."""
    cx, cy = K[0, 2], K[1, 2]
    u, v = mean_uv
    diag = float(np.hypot(width, height))
    dist = float(np.hypot(u - cx, v - cy))
    return float(np.clip(1.0 - 2.0 * dist / max(diag, 1.0), 0.0, 1.0))


def _area_score(visible_pixel_count: int, render_size: int, min_area_frac: float) -> float:
    """Object covers at least ``min_area_frac`` -> 1.0; below that -> 0."""
    total = float(render_size * render_size)
    frac = visible_pixel_count / max(total, 1.0)
    if frac < min_area_frac:
        return 0.0
    # Saturating curve: full credit at >= 4x min_area_frac
    return float(np.clip((frac - min_area_frac) / (3.0 * min_area_frac), 0.0, 1.0))


def _clip_score(crop_bbox_in_full: list, full_w: int, full_h: int) -> float:
    """Penalize crops that touch the original image border (the object was
    clipped). Returns the fraction of crop edges that are *inside* the
    original image, i.e. away from the border.
    """
    x0, y0, x1, y1 = crop_bbox_in_full
    edges_off_border = 0
    if x0 > 0:
        edges_off_border += 1
    if y0 > 0:
        edges_off_border += 1
    if x1 < full_w:
        edges_off_border += 1
    if y1 < full_h:
        edges_off_border += 1
    return edges_off_border / 4.0


def _front_facing_score(
    R_w2c: np.ndarray,
    T_w2c: np.ndarray,
    object_center: np.ndarray,
    object_up_world: np.ndarray,
) -> float:
    """High when the camera looks roughly straight at the object along an
    axis perpendicular to ``object_up_world`` (i.e. *not* from directly
    above or below).

    Concretely:
        forward_world = R_w2c[2, :]    (camera +Z axis in world coords)
        center_dir   = normalize(object_center - cam_pos)
        front       = max(0, forward . center_dir)         in [0,1]
        side_axis_dot = |forward . object_up|              in [0,1]
        side_score = 1 - side_axis_dot                     in [0,1]
        score      = 0.5 * front + 0.5 * side_score
    """
    forward = R_w2c[2, :]  # camera +Z in world
    cam_pos = -R_w2c.T @ T_w2c
    to_center = object_center - cam_pos
    n = float(np.linalg.norm(to_center))
    if n < 1e-6:
        return 0.0
    to_center = to_center / n
    front = float(np.clip(forward @ to_center, 0.0, 1.0))
    up = object_up_world / max(float(np.linalg.norm(object_up_world)), 1e-8)
    side_axis_dot = float(abs(forward @ up))
    side_score = float(np.clip(1.0 - side_axis_dot, 0.0, 1.0))
    return 0.5 * front + 0.5 * side_score


def _lighting_score(rgba_path: Path) -> float:
    """Mean luminance of object pixels mapped through a soft window.

    Penalize over- and under-exposure: ideal mean L* ~ 0.5 in [0,1].
    Also penalize the fraction of saturated pixels (L > 0.97 or L < 0.03).
    """
    img = cv2.imread(str(rgba_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return 0.0
    if img.shape[2] == 4:
        bgr = img[..., :3]
        alpha = img[..., 3].astype(np.float32) / 255.0
    else:
        bgr = img
        alpha = np.ones(img.shape[:2], dtype=np.float32)

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    luminance = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]

    mask = alpha > 0.05
    if not mask.any():
        return 0.0
    obj_lum = luminance[mask]

    mean_l = float(obj_lum.mean())
    # Soft window: 1.0 at L=0.5, falls to 0 at L=0 or L=1
    window = 1.0 - 2.0 * abs(mean_l - 0.5)
    window = float(np.clip(window, 0.0, 1.0))

    sat_frac = float(((obj_lum > 0.97) | (obj_lum < 0.03)).mean())
    sat_penalty = float(np.clip(1.0 - 4.0 * sat_frac, 0.0, 1.0))

    return window * sat_penalty


def _completeness_score(
    anchors_world: np.ndarray,
    R_w2c: np.ndarray,
    T_w2c: np.ndarray,
    K_full: np.ndarray,
    full_w: int,
    full_h: int,
) -> float:
    """Fraction of object anchors that lie in front of the camera AND project
    inside the ORIGINAL (uncropped) image. ``1.0`` => the entire object is
    visible in this frame; lower => the object is partially out-of-frame.
    """
    if anchors_world.size == 0:
        return 0.0
    Pc = (R_w2c @ anchors_world.T).T + T_w2c[None, :]
    z = Pc[:, 2]
    in_front = z > 1e-6
    if not in_front.any():
        return 0.0
    uv = (K_full @ Pc.T).T
    pix = uv[:, :2] / np.maximum(np.abs(uv[:, 2:3]), 1e-9)
    inside = (
        in_front
        & (pix[:, 0] >= 0)
        & (pix[:, 0] < float(full_w))
        & (pix[:, 1] >= 0)
        & (pix[:, 1] < float(full_h))
    )
    return float(inside.sum()) / float(anchors_world.shape[0])


# ── Main entry point ────────────────────────────────────────────────────────


def pick_reference(
    obj_dir: str,
    weights: Optional[ScoreWeights] = None,
    min_area_frac: float = 0.06,
    min_complete_frac: float = 0.90,
) -> dict:
    """Score every view in ``<obj_dir>/meta.json`` and write
    ``<obj_dir>/reference.json`` plus copy the chosen image to
    ``<obj_dir>/reference.png``.

    The reference frame must show the WHOLE object: views where less than
    ``min_complete_frac`` of the object's anchors project inside the
    original (uncropped) image are hard-filtered before scoring. If no view
    passes, the threshold is relaxed adaptively to the highest available
    completeness.

    Returns the chosen view's metadata dict (with extra ``scores`` field).
    """
    obj_dir_p = Path(obj_dir)
    meta_path = obj_dir_p / "meta.json"
    summary_path = obj_dir_p / "extraction_summary.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"meta.json missing at {meta_path}; run Phase 1 first")
    if not summary_path.exists():
        raise FileNotFoundError(f"extraction_summary.json missing at {summary_path}")

    with open(meta_path, "r", encoding="utf-8") as f:
        views = json.load(f)
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    object_frame = summary["object_frame"]
    object_center = np.asarray(object_frame["object_center"], dtype=np.float64)
    object_up = np.asarray(object_frame["object_up_world"], dtype=np.float64)
    render_size = int(summary["render_size"])

    weights = weights or ScoreWeights()

    # Load object anchors (used for completeness score). PLY is written by
    # both extraction.extract and extraction_real.extract_from_real_masks.
    anchors_ply = obj_dir_p / "object_anchors.ply"
    anchors_world = _read_ply_xyz(anchors_ply) if anchors_ply.exists() else np.zeros((0, 3), np.float32)
    if anchors_world.shape[0] == 0:
        logger.warning(
            "No anchors at %s — completeness score disabled.", anchors_ply,
        )

    # Cache per-frame full-image dims + intrinsics (cameras.json).
    cam_lookup = _build_cam_lookup(summary["model_path"]) if anchors_world.size else {}

    breakdowns: list[ScoreBreakdown] = []
    candidates_with_scores: list[tuple[float, dict, ScoreBreakdown]] = []

    for v in views:
        K = np.asarray(v["K"], dtype=np.float64)
        R_w2c = np.asarray(v["R_w2c"], dtype=np.float64)
        T_w2c = np.asarray(v["T_w2c"], dtype=np.float64)
        rgba_path = obj_dir_p / v["image_path"]

        # We need the projected centroid for center_score. Project the world
        # object_center into the (cropped, resized) view's K.
        cam_pt = R_w2c @ object_center + T_w2c
        if cam_pt[2] <= 1e-3:
            # Object behind the camera in this view. Should never happen
            # if Phase 1 already filtered, but be defensive.
            mean_uv = (1e6, 1e6)
        else:
            u = K[0, 0] * cam_pt[0] / cam_pt[2] + K[0, 2]
            v_pix = K[1, 1] * cam_pt[1] / cam_pt[2] + K[1, 2]
            mean_uv = (float(u), float(v_pix))

        # Original full-image dims are not stored per-view (we only stored the
        # cropped dims). We can recover them from crop_bbox upper bound + a
        # fudge factor; or simply use crop_bbox values directly to compute the
        # "edge off border" ratio. _clip_score already only needs full_w/h to
        # know whether the crop reached the border.
        # Heuristic: full_w/full_h = max(crop_x1, ...) — we don't actually
        # need exact dims, only the comparison. We'll use the crop_bbox max
        # extent + 1 as a proxy. Better: read it from the original cameras.json.
        # For correctness we re-read cameras.json once.
        full_dims = _full_image_dims_for_view(summary["model_path"], v["frame_index"])
        full_w, full_h = full_dims if full_dims is not None else (v["width"], v["height"])

        center_s = _center_score(K, v["width"], v["height"], mean_uv)
        area_s = _area_score(v["visible_pixel_count"], render_size, min_area_frac)
        clip_s = _clip_score(v["crop_bbox_in_full"], full_w, full_h)
        front_s = _front_facing_score(R_w2c, T_w2c, object_center, object_up)
        light_s = _lighting_score(rgba_path)
        complete_s = 0.0
        if anchors_world.size and int(v["frame_index"]) in cam_lookup:
            cam = cam_lookup[int(v["frame_index"])]
            K_full = np.array(
                [[cam["fx"], 0.0, cam["width"] * 0.5],
                 [0.0, cam["fy"], cam["height"] * 0.5],
                 [0.0, 0.0, 1.0]],
                dtype=np.float64,
            )
            complete_s = _completeness_score(
                anchors_world, R_w2c, T_w2c, K_full,
                int(cam["width"]), int(cam["height"]),
            )

        total = (
            weights.center * center_s
            + weights.area * area_s
            + weights.clip * clip_s
            + weights.front * front_s
            + weights.light * light_s
            + weights.complete * complete_s
        )

        breakdown = ScoreBreakdown(
            frame_index=int(v["frame_index"]),
            img_name=str(v["img_name"]),
            center_score=float(center_s),
            area_score=float(area_s),
            clip_score=float(clip_s),
            front_score=float(front_s),
            light_score=float(light_s),
            complete_score=float(complete_s),
            total=float(total),
        )
        breakdowns.append(breakdown)
        candidates_with_scores.append((total, v, breakdown))

    if not candidates_with_scores:
        raise RuntimeError(
            f"No candidate views in {meta_path}. Check Phase 1 drop_reasons."
        )

    # Hard filter on whole-object visibility BEFORE scoring (when anchors
    # are available). Adapt the threshold downward if too strict.
    applied_complete_thr = float(min_complete_frac)
    if anchors_world.size:
        eligible = [t for t in candidates_with_scores if t[2].complete_score >= min_complete_frac]
        if not eligible:
            best_complete = max(t[2].complete_score for t in candidates_with_scores)
            applied_complete_thr = max(0.5, best_complete - 0.05)
            logger.warning(
                "No view reaches min_complete_frac=%.2f; relaxing to %.2f "
                "(best available = %.2f).",
                min_complete_frac, applied_complete_thr, best_complete,
            )
            eligible = [t for t in candidates_with_scores if t[2].complete_score >= applied_complete_thr]
        candidates_with_scores = eligible

    # Sort and pick the best
    candidates_with_scores.sort(key=lambda x: -x[0])
    best_total, best_view, best_breakdown = candidates_with_scores[0]
    if best_total <= 0.0:
        raise RuntimeError(
            f"Best candidate score is {best_total:.3f}; no view passed the "
            f"area threshold (min_area_frac={min_area_frac})."
        )

    # Copy best image to reference.png
    ref_image_src = obj_dir_p / best_view["image_path"]
    ref_image_dst = obj_dir_p / "reference.png"
    img = cv2.imread(str(ref_image_src), cv2.IMREAD_UNCHANGED)
    cv2.imwrite(str(ref_image_dst), img)

    out = {
        "weights": asdict(weights),
        "min_area_frac": float(min_area_frac),
        "min_complete_frac": float(min_complete_frac),
        "applied_complete_thr": float(applied_complete_thr),
        "selected": {**best_view, "scores": asdict(best_breakdown)},
        "all_scores": [asdict(b) for b in breakdowns],
    }
    with open(obj_dir_p / "reference.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    logger.info(
        "Reference frame: idx=%d (%s) total=%.3f [center=%.2f area=%.2f "
        "clip=%.2f front=%.2f light=%.2f complete=%.2f]",
        best_breakdown.frame_index, best_breakdown.img_name, best_breakdown.total,
        best_breakdown.center_score, best_breakdown.area_score,
        best_breakdown.clip_score, best_breakdown.front_score,
        best_breakdown.light_score, best_breakdown.complete_score,
    )
    return out["selected"]


def _full_image_dims_for_view(model_path: str, frame_index: int) -> Optional[tuple[int, int]]:
    """Read original (full) ``(width, height)`` for a frame from the source
    cameras.json. Returns None on failure (then ``_clip_score`` falls back
    to the cropped dims, which yields a slightly optimistic score).
    """
    try:
        cam_path = Path(model_path) / "cameras.json"
        with open(cam_path, "r", encoding="utf-8") as f:
            cam_data = json.load(f)
        for c in cam_data:
            if int(c.get("id", -1)) == int(frame_index):
                return int(c["width"]), int(c["height"])
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("full_image_dims lookup failed: %s", exc)
    return None


def _build_cam_lookup(model_path: str) -> dict:
    """Read cameras.json once and return ``{frame_index: cam_dict}``."""
    try:
        cam_path = Path(model_path) / "cameras.json"
        with open(cam_path, "r", encoding="utf-8") as f:
            cam_data = json.load(f)
        return {int(c.get("id", -1)): c for c in cam_data}
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("cameras.json lookup failed: %s", exc)
        return {}


def _read_ply_xyz(path: Path) -> np.ndarray:
    """Minimal PLY reader: returns (N,3) float32 vertex positions. Handles
    ascii and binary little-endian; ignores all other properties.
    """
    if not path.exists():
        return np.zeros((0, 3), dtype=np.float32)
    with open(path, "rb") as f:
        header = b""
        while True:
            line = f.readline()
            header += line
            if line.strip() == b"end_header":
                break
        head = header.decode("latin1", errors="replace")
        ascii_mode = "format ascii" in head
        n_vertex = 0
        props: list[str] = []
        for line in head.splitlines():
            if line.startswith("element vertex"):
                n_vertex = int(line.split()[-1])
            elif line.startswith("property"):
                props.append(line.split()[-1])

        ix, iy, iz = props.index("x"), props.index("y"), props.index("z")
        if ascii_mode:
            xs, ys, zs = [], [], []
            for _ in range(n_vertex):
                parts = f.readline().decode("latin1").split()
                xs.append(float(parts[ix]))
                ys.append(float(parts[iy]))
                zs.append(float(parts[iz]))
            return np.stack([np.asarray(xs), np.asarray(ys), np.asarray(zs)],
                             axis=1).astype(np.float32)
        else:
            dt = []
            for p in props:
                if p in ("red", "green", "blue", "alpha", "label"):
                    dt.append((p, "<u1"))
                else:
                    dt.append((p, "<f4"))
            arr = np.frombuffer(f.read(), dtype=np.dtype(dt), count=n_vertex)
            return np.stack([arr["x"], arr["y"], arr["z"]],
                             axis=1).astype(np.float32)
