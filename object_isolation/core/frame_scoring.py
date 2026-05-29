"""SV3D Conditioning-Frame Scoring.

Pick the SV3D conditioning frame (top-1) and rank backups (top-K). Five
factors with locked weights:

    front   0.35   how close to az=0, el=0 in V-frame
    cover   0.20   how big the object is in frame (peak ~30% of image area)
    sharp   0.20   Laplacian variance on the masked crop (rank-normalized)
    expose  0.10   exposure quality (mean luma near 0.5, few saturated px)
    occl    0.15   |M_hybrid| / |M_objgs|  (drops when something occludes)

All factors are mapped into [0, 1] and combined linearly. The chosen
conditioning view will become the SV3D input image for novel-view synthesis.

Inputs:
    extraction_index.json  (from extraction)

Outputs at <out_root>/obj_<id>/02_frame_scoring/::

    scores.json            — per-frame components + final score, ranked

Run via pipeline orchestrator (recommended)::

    python -m object_isolation.run_pipeline \\
        --model_path temp_deps/ObjectGS/outputs/3dovs/.../2026-03-19_04-01-38 \\
        --object_id 8 \\
        --output_root object_isolation/outputs
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
import json
import logging
import math

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ── Score weights (locked by user decision) ───────────────────────────────────────────────────

WEIGHTS = {
    "front":  0.35,
    "cover":  0.20,
    "sharp":  0.20,
    "expose": 0.10,
    "occl":   0.15,
}

# Cover preference: peaks at this fraction of the image, falls off as object
# becomes too small (lacks detail) or too large (cropping risk).
COVER_TARGET = 0.30
COVER_FLOOR = 0.02   # below 2% of image, treat as zero
COVER_CEIL = 0.85    # above 85%, hard cut


# ── Component scoring functions ───────────────────────────────────────────────────────────────

def _front_score(az_deg: float, el_deg: float) -> float:
    """1 at az=el=0; falls smoothly with angle. Elevation weighted as cos(el)."""
    if not math.isfinite(az_deg) or not math.isfinite(el_deg):
        return 0.0
    az = math.radians(az_deg)
    el = math.radians(el_deg)
    s_az = (1.0 + math.cos(az)) * 0.5  # 1 at 0°, 0 at 180°
    s_el = max(0.0, math.cos(el))      # 1 at 0°, drops to 0 at 90°
    return float(s_az * s_el)


def _cover_score(fg_fraction: float) -> float:
    """Piecewise: rises to COVER_TARGET, plateaus, drops past COVER_CEIL."""
    if fg_fraction <= COVER_FLOOR:
        return 0.0
    if fg_fraction >= COVER_CEIL:
        return 0.0
    if fg_fraction <= COVER_TARGET:
        return float((fg_fraction - COVER_FLOOR) / max(COVER_TARGET - COVER_FLOOR, 1e-6))
    # Above target, decay linearly toward COVER_CEIL.
    return float(max(0.0, 1.0 - (fg_fraction - COVER_TARGET) / max(COVER_CEIL - COVER_TARGET, 1e-6)))


def _sharp_metric(rgb: np.ndarray, mask: np.ndarray) -> float:
    """Laplacian variance over the masked region (raw, not normalized)."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if mask.sum() < 32:
        return 0.0
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    vals = lap[mask]
    return float(np.var(vals)) if vals.size > 0 else 0.0


def _expose_metric(rgb: np.ndarray, mask: np.ndarray) -> dict:
    """Mean luma + saturated-pixel fraction over the mask."""
    if mask.sum() < 32:
        return {"mean_luma": 0.0, "frac_saturated": 1.0, "frac_dark": 1.0}
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    pix = gray[mask]
    return {
        "mean_luma": float(np.mean(pix)),
        "frac_saturated": float(np.mean(pix > 0.98)),
        "frac_dark": float(np.mean(pix < 0.02)),
    }


def _expose_score(em: dict) -> float:
    luma_term = max(0.0, 1.0 - 2.0 * abs(em["mean_luma"] - 0.5))
    sat_pen = 1.0 - min(1.0, em["frac_saturated"] * 5.0)
    dark_pen = 1.0 - min(1.0, em["frac_dark"] * 5.0)
    return float(luma_term * sat_pen * dark_pen)


def _occl_score(n_hybrid: int, n_objgs: int) -> float:
    """Higher when the real-image mask covers most of the predicted projection."""
    if n_objgs <= 0:
        return 0.0
    return float(min(1.0, n_hybrid / max(n_objgs, 1)))


# ── Per-frame metric collection ───────────────────────────────────────────────────────────────

@dataclass
class FrameScore:
    """Container for per-frame raw metrics, normalized components, and final score."""
    cam_index: int
    img_name: str
    out_rgba_path: str
    azimuth_V_deg: float
    elevation_V_deg: float
    fg_fraction: float
    n_pixels_hybrid: int
    n_pixels_objgs: int
    sharpness_raw: float
    expose_metrics: dict
    components: dict = field(default_factory=dict)
    score: float = 0.0


def _collect_raw_metrics(frame: dict, scope_cameras: list[dict]) -> FrameScore:
    """Read RGB + mask once, compute raw metrics."""
    bgr = cv2.imread(frame['image_path'], cv2.IMREAD_COLOR)
    mask_u8 = cv2.imread(frame['out_mask_path'], cv2.IMREAD_GRAYSCALE)
    if bgr is None or mask_u8 is None:
        return None  # type: ignore[return-value]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mask = mask_u8 > 127

    sharp = _sharp_metric(rgb, mask)
    expose = _expose_metric(rgb, mask)

    az = float(frame.get('azimuth_V_deg', float('nan')))
    # Normalize az to (-180, 180] for "front" interpretation.
    if math.isfinite(az):
        az = ((az + 180.0) % 360.0) - 180.0

    el = float(scope_cameras[frame['cam_index']].get('elevation_V_deg', 0.0))
    if not math.isfinite(el):
        el = 0.0

    return FrameScore(
        cam_index=int(frame['cam_index']),
        img_name=str(frame['img_name']),
        out_rgba_path=str(frame['out_rgba_path']),
        azimuth_V_deg=az,
        elevation_V_deg=el,
        fg_fraction=float(frame['fg_fraction']),
        n_pixels_hybrid=int(frame['n_pixels_hybrid']),
        n_pixels_objgs=int(frame['n_pixels_objgs']),
        sharpness_raw=sharp,
        expose_metrics=expose,
    )


def _rank_normalize(values: np.ndarray) -> np.ndarray:
    """Map to [0, 1] by rank (ties averaged). Robust to outliers."""
    n = len(values)
    if n == 0:
        return values
    if n == 1:
        return np.array([1.0])
    order = np.argsort(values)
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.linspace(0.0, 1.0, n)
    return ranks


# ── Top-level ────────────────────────────────────────────────────────────────────────────────

def score_frames(extraction_index_path: Path, scope_cameras: list[dict],
                 weights: dict | None = None,
                 top_k: int = 5) -> dict:
    """Score every frame in the extraction manifest and return a ranked dict."""
    weights = weights or WEIGHTS
    with open(extraction_index_path) as f:
        manifest = json.load(f)
    frames = manifest['frames']
    if not frames:
        return {"weights": weights, "frames": [], "ranking": [], "top_k": []}

    # Pass 1: collect raw metrics.
    scored: list[FrameScore] = []
    for fr in frames:
        s = _collect_raw_metrics(fr, scope_cameras)
        if s is not None:
            scored.append(s)

    # Rank-normalize sharpness across this object's frames.
    sharp_raw = np.array([s.sharpness_raw for s in scored], dtype=np.float64)
    sharp_norm = _rank_normalize(sharp_raw)

    # Pass 2: build component scores.
    for s, sh in zip(scored, sharp_norm):
        comp = {
            "front":  _front_score(s.azimuth_V_deg, s.elevation_V_deg),
            "cover":  _cover_score(s.fg_fraction),
            "sharp":  float(sh),
            "expose": _expose_score(s.expose_metrics),
            "occl":   _occl_score(s.n_pixels_hybrid, s.n_pixels_objgs),
        }
        score = sum(weights[k] * comp[k] for k in weights)
        s.components = comp
        s.score = float(score)

    scored.sort(key=lambda x: -x.score)
    ranking = [{"rank": i + 1, "cam_index": s.cam_index, "score": s.score,
                "components": s.components, "img_name": s.img_name,
                "azimuth_V_deg": s.azimuth_V_deg}
               for i, s in enumerate(scored)]
    out = {
        "weights": weights,
        "n_frames": len(scored),
        "ranking": ranking,
        "top_k": ranking[:top_k],
        "frames": [asdict(s) for s in scored],
    }
    return out


def run_scoring(extraction_index_path: Path, scope_cameras: list[dict],
                output_dir: Path, top_k: int = 5) -> dict:
    """Run :func:`score_frames` and persist the result to ``scores.json``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    result = score_frames(extraction_index_path, scope_cameras, top_k=top_k)
    with open(output_dir / "scores.json", "w") as f:
        json.dump(result, f, indent=2)
    if result["top_k"]:
        best = result["top_k"][0]
        logger.info("Top frame: cam=%d az=%.1f score=%.3f comps=%s",
                    best["cam_index"], best["azimuth_V_deg"], best["score"],
                    {k: round(v, 2) for k, v in best["components"].items()})
    return result
