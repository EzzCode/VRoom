"""Visual Debug for Frame Scoring (ModuleTBD).

Outputs under ``<obj_dir>/02_frame_scoring_debug/``::

    bar_top10.png            top-10 frames bar chart of total score
    scatter_az_score.png     azimuth vs total score for all candidates
    components_top10.png     stacked component breakdown (front/cover/sharp/expose/occl)
    topk_strip.png           horizontal strip of the chosen top-K thumbnails
    summary.json             ranking + selected indices

ModuleTBD's scoring mirrors the old object_isolation components.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import json
import logging
import sys

import cv2
import numpy as np

_VROOM_ROOT = Path(__file__).resolve().parents[2]
if str(_VROOM_ROOT) not in sys.path:
    sys.path.insert(0, str(_VROOM_ROOT))

logger = logging.getLogger(__name__)


_COMPONENTS = ("front", "cover", "sharp", "expose", "occl")
_COMP_COLORS = {
    "front":  (120, 200, 255),
    "cover":  (130, 200, 130),
    "sharp":  (180, 140, 230),
    "expose": (210, 180, 90),
    "occl":   (220, 120, 120),
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_bar(values, labels, *, title, out_path, width=900, bar_h=24, pad=12):
    n = len(values)
    if n == 0:
        return
    h = pad * 2 + n * (bar_h + 6) + 40
    img = np.full((h, width, 3), 250, np.uint8)
    cv2.putText(img, title, (pad, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (30, 30, 30), 1, cv2.LINE_AA)
    vmin, vmax = float(min(values)), float(max(values))
    span = max(vmax - vmin, 1e-6)
    label_w = 260
    for i, (v, lab) in enumerate(zip(values, labels)):
        y = 40 + i * (bar_h + 6)
        w_px = int((v - vmin) / span * (width - label_w - pad * 2))
        cv2.rectangle(img, (label_w, y), (label_w + max(w_px, 1), y + bar_h),
                      (80, 140, 220), -1)
        cv2.putText(img, lab, (pad, y + bar_h - 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (30, 30, 30), 1, cv2.LINE_AA)
        cv2.putText(img, f"{v:+.3f}", (label_w + max(w_px, 1) + 6, y + bar_h - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 30), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_path), img)


def _make_scatter(ranking, *, out_path, width=900, height=500, pad=50,
                  highlight_image_names=None):
    if not ranking:
        return
    az = np.array([r.get("azimuth_deg", 0.0) for r in ranking], np.float32)
    sc = np.array([r.get("score", 0.0) for r in ranking], np.float32)
    img = np.full((height, width, 3), 250, np.uint8)
    cv2.rectangle(img, (pad, pad), (width - pad, height - pad), (200, 200, 200), 1)

    sc_min, sc_max = float(sc.min()), float(sc.max())
    sc_span = max(sc_max - sc_min, 1e-6)

    def to_px(a, s):
        x = pad + int((a + 180.0) / 360.0 * (width - 2 * pad))
        y = (height - pad) - int((s - sc_min) / sc_span * (height - 2 * pad))
        return x, y

    highlight = set(highlight_image_names or [])
    for r in ranking:
        x, y = to_px(r.get("azimuth_deg", 0.0), r.get("score", 0.0))
        if r.get("image_name") in highlight:
            cv2.circle(img, (x, y), 6, (0, 80, 220), -1)
        else:
            cv2.circle(img, (x, y), 3, (140, 140, 140), -1)

    cv2.putText(img, "azimuth (deg)", (width // 2 - 60, height - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1, cv2.LINE_AA)
    cv2.putText(img, "score", (10, height // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1, cv2.LINE_AA)
    cv2.putText(img, "azimuth (V) vs total frame score",
                (pad, pad - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (30, 30, 30), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_path), img)


def _make_components(top_entries, *, out_path, width=1000, row_h=36, pad=12):
    n = len(top_entries)
    if n == 0:
        return
    h = pad * 2 + n * row_h + 60
    img = np.full((h, width, 3), 250, np.uint8)
    cv2.putText(img, "Top frames — score component breakdown",
                (pad, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (30, 30, 30), 1, cv2.LINE_AA)

    label_w = 320
    bar_w_total = width - label_w - pad * 2

    # legend
    lx = pad
    for comp in _COMPONENTS:
        color = _COMP_COLORS[comp][::-1]
        cv2.rectangle(img, (lx, 38), (lx + 14, 52), color, -1)
        cv2.putText(img, comp, (lx + 18, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (30, 30, 30), 1, cv2.LINE_AA)
        lx += 100

    for i, e in enumerate(top_entries):
        y = 60 + i * row_h
        comps = e.get("components", {}) or {}
        vals = np.array([comps.get(c, 0.0) for c in _COMPONENTS], np.float32)
        denom = max(float(np.sum(np.abs(vals))), 1e-6)
        x = label_w
        lab = (f"#{i+1} cam={e.get('cam_index', '?')} "
               f"{str(e.get('image_name', ''))[:24]:<24} "
               f"az={e.get('azimuth_deg', 0.0):+6.1f} "
               f"total={e.get('score', 0.0):+.3f}")
        cv2.putText(img, lab, (pad, y + row_h // 2 + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (30, 30, 30), 1, cv2.LINE_AA)
        for c, v in zip(_COMPONENTS, vals):
            seg_w = max(int(abs(v) / denom * bar_w_total), 1)
            color = _COMP_COLORS[c][::-1]
            cv2.rectangle(img, (x, y + 4), (x + seg_w, y + row_h - 4), color, -1)
            x += seg_w
    cv2.imwrite(str(out_path), img)


def _read_thumb(path, max_h=160):
    if not path:
        return None
    rgba = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if rgba is None:
        return None
    if rgba.ndim == 3 and rgba.shape[-1] == 4:
        alpha = rgba[..., -1:].astype(np.float32) / 255.0
        rgb = rgba[..., :3].astype(np.float32)
        rgb = rgb * alpha + 245.0 * (1.0 - alpha)
        thumb = rgb.astype(np.uint8)
    else:
        thumb = rgba
    h, w = thumb.shape[:2]
    s = min(1.0, max_h / max(h, 1))
    if s < 1.0:
        thumb = cv2.resize(thumb, (int(w * s), int(h * s)), cv2.INTER_AREA)
    return thumb


def _make_topk_strip(top_entries, *, out_path, max_h=180):
    thumbs = []
    for e in top_entries:
        t = _read_thumb(e.get("out_rgba_path"), max_h=max_h)
        if t is None:
            continue
        h, w = t.shape[:2]
        head_h = 26
        head = np.full((head_h, w, 3), 240, np.uint8)
        cv2.putText(head, f"cam={e.get('cam_index')} az={e.get('azimuth_deg', 0):+.1f} "
                          f"score={e.get('score', 0):+.3f}",
                    (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (30, 30, 30), 1, cv2.LINE_AA)
        thumbs.append(np.vstack([head, t]))
    if not thumbs:
        return
    max_h_full = max(t.shape[0] for t in thumbs)
    padded = []
    for t in thumbs:
        if t.shape[0] < max_h_full:
            pad = max_h_full - t.shape[0]
            t = np.vstack([t, np.full((pad, t.shape[1], 3), 240, np.uint8)])
        padded.append(t)
        padded.append(np.full((max_h_full, 4, 3), 220, np.uint8))
    strip = np.hstack(padded[:-1])
    cv2.imwrite(str(out_path), strip)


# ── orchestrator ──────────────────────────────────────────────────────────────

def generate_debug_artifacts(*, scores, debug_dir):
    debug_dir = Path(debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)

    ranking = scores.get("ranking", [])
    top_k = scores.get("top_k", [])
    weights = scores.get("weights", {})

    top10 = ranking[:10]
    if top10:
        labels = [f"cam={e.get('cam_index', '?')} {str(e.get('image_name', ''))[:24]}"
                  for e in top10]
        values = [float(e.get("score", 0.0)) for e in top10]
        _make_bar(values, labels, title="Top-10 frames by total score",
                  out_path=debug_dir / "bar_top10.png")
        _make_components(top10, out_path=debug_dir / "components_top10.png")

    highlight = {e.get("image_name") for e in top_k}
    _make_scatter(ranking, out_path=debug_dir / "scatter_az_score.png",
                  highlight_image_names=highlight)
    _make_topk_strip(top_k, out_path=debug_dir / "topk_strip.png")

    summary = {
        "weights": weights,
        "n_frames": int(scores.get("n_frames", len(ranking))),
        "n_top_k": len(top_k),
        "top_k_image_names": [e.get("image_name") for e in top_k],
        "top_k_cam_indices": [e.get("cam_index") for e in top_k],
        "top_k_scores": [float(e.get("score", 0.0)) for e in top_k],
    }
    with open(debug_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Frame-scoring debug saved to: %s", debug_dir)
    return summary


def main():
    parser = argparse.ArgumentParser(description="ModuleTBD frame-scoring visual debug.")
    parser.add_argument("--scores_json", required=True,
                        help="JSON file containing ranking + top_k (e.g. 99_pipeline_summary.json or a saved scores file).")
    parser.add_argument("--debug_dir", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
    with open(args.scores_json, "r") as f:
        data = json.load(f)
    # Tolerate the pipeline-summary shape
    scores = data.get("phases", {}).get("frame_scoring", data)
    generate_debug_artifacts(scores=scores, debug_dir=Path(args.debug_dir))


if __name__ == "__main__":
    main()
