"""
Visual debug for Phase 4 (frame scoring).

Outputs:
    summary.json           — chosen frame and weights
    bar_chart.png          — top-K stacked bar of component contributions
    scatter.png            — az_V vs final score for all frames
    top1.png               — winning frame with score breakdown overlay
    top_k_strip.png        — thumbnails of top-K with scores
"""
from __future__ import annotations

from pathlib import Path
import json
import logging
import math
import sys

import cv2
import numpy as np

_VROOM_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_VROOM_ROOT) not in sys.path:
    sys.path.insert(0, str(_VROOM_ROOT))

from object_isolation.core.object_scope import discover_object_scope
from object_isolation.core.frame_scoring import run_scoring, WEIGHTS

logger = logging.getLogger(__name__)

# Colors per component (BGR space — we work in RGB then write BGR).
COMP_COLORS_RGB = {
    "front":  (220,  60,  60),
    "cover":  ( 60, 180,  80),
    "sharp":  ( 60, 120, 220),
    "expose": (220, 180,  40),
    "occl":   (160,  80, 200),
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _putlbl(img, text, org, fg=(255, 255, 255), bg=(0, 0, 0), scale=0.5, thick=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, bg, thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, fg, thick, cv2.LINE_AA)


def _crop_object(rgba_path: Path, pad_frac: float = 0.10):
    rgba = cv2.imread(str(rgba_path), cv2.IMREAD_UNCHANGED)
    if rgba is None or rgba.shape[2] != 4:
        return None
    a = rgba[..., 3]
    ys, xs = np.where(a > 127)
    if len(xs) == 0:
        return None
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    pad = int(pad_frac * max(x1 - x0, y1 - y0))
    x0 = max(0, x0 - pad); y0 = max(0, y0 - pad)
    x1 = min(rgba.shape[1], x1 + pad); y1 = min(rgba.shape[0], y1 + pad)
    crop = rgba[y0:y1, x0:x1]
    rgb = cv2.cvtColor(crop[..., :3], cv2.COLOR_BGR2RGB)
    a = crop[..., 3:4].astype(np.float32) / 255.0
    bg = np.full_like(rgb, 245)
    comp = (a * rgb + (1 - a) * bg).astype(np.uint8)
    return comp


# ─────────────────────────────────────────────────────────────────────────────
# Bar chart of top-K components
# ─────────────────────────────────────────────────────────────────────────────

def make_bar_chart(scores: dict, out_path: Path, top_k: int = 5,
                   width: int = 1100, height: int = 520):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = np.full((height, width, 3), 250, dtype=np.uint8)

    # Title.
    _putlbl(img, "Phase 4 frame scoring — top-K component contributions",
            (16, 28), fg=(20, 20, 20), bg=(255, 255, 255), scale=0.6, thick=1)

    rows = scores["top_k"][:top_k]
    if not rows:
        cv2.imwrite(str(out_path), img)
        return out_path

    weights = scores["weights"]
    # Plot area.
    pad_l, pad_r, pad_t, pad_b = 70, 280, 60, 60
    px0, py0 = pad_l, pad_t
    px1, py1 = width - pad_r, height - pad_b
    plot_w = px1 - px0
    plot_h = py1 - py0
    n = len(rows)
    bar_h = int(plot_h / n * 0.7)
    gap = int(plot_h / n * 0.3)

    # Axis lines.
    cv2.line(img, (px0, py1), (px1, py1), (60, 60, 60), 1)
    cv2.line(img, (px0, py0), (px0, py1), (60, 60, 60), 1)

    # Tick labels along top (score axis).
    for f in [0.0, 0.25, 0.5, 0.75, 1.0]:
        x = int(px0 + f * plot_w)
        cv2.line(img, (x, py1), (x, py1 + 4), (60, 60, 60), 1)
        _putlbl(img, f"{f:.2f}", (x - 12, py1 + 18),
                fg=(60, 60, 60), bg=(255, 255, 255), scale=0.4)

    comp_order = ["front", "cover", "sharp", "expose", "occl"]
    for i, row in enumerate(rows):
        y = py0 + i * (bar_h + gap)
        # Bar label.
        lbl = f"#{i+1} cam{row['cam_index']:>3} az={row['azimuth_V_deg']:.0f}deg"
        _putlbl(img, lbl, (10, y + bar_h - 4), fg=(20, 20, 20),
                bg=(255, 255, 255), scale=0.45)
        # Stacked bar of weighted components.
        x_cur = px0
        for k in comp_order:
            w = weights[k]
            v = row['components'][k]
            seg = int(round(w * v * plot_w))
            color = COMP_COLORS_RGB[k]
            cv2.rectangle(img, (x_cur, y), (x_cur + seg, y + bar_h),
                          color[::-1], -1)  # to BGR
            x_cur += seg
        # Score number at end.
        _putlbl(img, f"{row['score']:.3f}", (x_cur + 8, y + bar_h - 4),
                fg=(20, 20, 20), bg=(255, 255, 255), scale=0.5)

    # Legend.
    ly = py0
    lx = px1 + 20
    _putlbl(img, "components x weight", (lx, ly - 8),
            fg=(40, 40, 40), bg=(255, 255, 255), scale=0.5)
    for i, k in enumerate(comp_order):
        yy = ly + i * 26 + 8
        cv2.rectangle(img, (lx, yy), (lx + 22, yy + 16),
                      COMP_COLORS_RGB[k][::-1], -1)
        _putlbl(img, f"{k}  (w={weights[k]:.2f})", (lx + 28, yy + 13),
                fg=(40, 40, 40), bg=(255, 255, 255), scale=0.45)

    cv2.imwrite(str(out_path), img)
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Scatter: az_V vs score
# ─────────────────────────────────────────────────────────────────────────────

def make_scatter(scores: dict, out_path: Path,
                 width: int = 900, height: int = 420):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = np.full((height, width, 3), 250, dtype=np.uint8)

    _putlbl(img, "Frame score vs azimuth (V-frame)", (16, 28),
            fg=(20, 20, 20), bg=(255, 255, 255), scale=0.6)

    pad_l, pad_r, pad_t, pad_b = 60, 30, 60, 50
    px0, py0 = pad_l, pad_t
    px1, py1 = width - pad_r, height - pad_b
    plot_w = px1 - px0
    plot_h = py1 - py0

    # Axes.
    cv2.line(img, (px0, py1), (px1, py1), (60, 60, 60), 1)
    cv2.line(img, (px0, py0), (px0, py1), (60, 60, 60), 1)

    # x-axis ticks: -180 ... 180.
    for az in range(-180, 181, 60):
        x = int(px0 + (az + 180) / 360 * plot_w)
        cv2.line(img, (x, py1), (x, py1 + 4), (60, 60, 60), 1)
        _putlbl(img, f"{az}", (x - 10, py1 + 18),
                fg=(60, 60, 60), bg=(255, 255, 255), scale=0.4)
    _putlbl(img, "azimuth_V (deg)", (px0 + plot_w // 2 - 50, py1 + 38),
            fg=(40, 40, 40), bg=(255, 255, 255), scale=0.45)

    # y-axis ticks: 0..1.
    for f in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = int(py1 - f * plot_h)
        cv2.line(img, (px0 - 4, y), (px0, y), (60, 60, 60), 1)
        _putlbl(img, f"{f:.2f}", (10, y + 4),
                fg=(60, 60, 60), bg=(255, 255, 255), scale=0.4)
    _putlbl(img, "score", (10, py0 - 8),
            fg=(40, 40, 40), bg=(255, 255, 255), scale=0.45)

    # Front-az reference line (az=0).
    x0_ref = int(px0 + 180 / 360 * plot_w)
    cv2.line(img, (x0_ref, py0), (x0_ref, py1), (180, 200, 220), 1)
    _putlbl(img, "az=0", (x0_ref + 4, py0 + 14),
            fg=(80, 120, 180), bg=(255, 255, 255), scale=0.4)

    # Plot all frames.
    for fr in scores["frames"]:
        az = fr["azimuth_V_deg"]
        if not math.isfinite(az):
            continue
        # az already in (-180, 180]
        x = int(px0 + (az + 180) / 360 * plot_w)
        y = int(py1 - fr["score"] * plot_h)
        cv2.circle(img, (x, y), 3, (140, 140, 140), -1)

    # Highlight top-K.
    for i, fr in enumerate(scores["top_k"]):
        az = fr["azimuth_V_deg"]
        if not math.isfinite(az):
            continue
        x = int(px0 + (az + 180) / 360 * plot_w)
        y = int(py1 - fr["score"] * plot_h)
        col = (40, 180, 60) if i == 0 else (60, 100, 200)
        cv2.circle(img, (x, y), 6, col, 2)
        _putlbl(img, f"#{i+1}", (x + 8, y - 6),
                fg=col, bg=(255, 255, 255), scale=0.4)

    cv2.imwrite(str(out_path), img)
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Top-1 thumbnail with score breakdown overlay
# ─────────────────────────────────────────────────────────────────────────────

def make_top1_card(scores: dict, out_path: Path, target_w: int = 720):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not scores["top_k"]:
        return None
    top = scores["top_k"][0]
    full = next((f for f in scores["frames"] if f["cam_index"] == top["cam_index"]), None)
    if full is None:
        return None
    crop = _crop_object(Path(full["out_rgba_path"]))
    if crop is None:
        return None
    H, W = crop.shape[:2]
    new_w = target_w
    new_h = int(round(H * new_w / max(W, 1)))
    crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Sidebar with score breakdown.
    side_w = 320
    sidebar = np.full((new_h, side_w, 3), 245, dtype=np.uint8)
    _putlbl(sidebar, "TOP-1 (SV3D conditioning view)", (10, 24),
            fg=(20, 20, 20), bg=(255, 255, 255), scale=0.55)
    _putlbl(sidebar, f"cam {top['cam_index']}  {top['img_name']}", (10, 50),
            fg=(60, 60, 60), bg=(255, 255, 255), scale=0.45)
    _putlbl(sidebar, f"az_V = {top['azimuth_V_deg']:.1f} deg", (10, 74),
            fg=(60, 60, 60), bg=(255, 255, 255), scale=0.45)
    _putlbl(sidebar, f"score = {top['score']:.3f}", (10, 98),
            fg=(20, 100, 40), bg=(255, 255, 255), scale=0.55)

    weights = scores["weights"]
    y = 130
    bar_w = side_w - 130
    for k in ["front", "cover", "sharp", "expose", "occl"]:
        v = top['components'][k]
        col = COMP_COLORS_RGB[k][::-1]  # to BGR
        _putlbl(sidebar, f"{k:>6}", (10, y + 12),
                fg=(40, 40, 40), bg=(255, 255, 255), scale=0.45)
        # bar
        bx0 = 80
        cv2.rectangle(sidebar, (bx0, y), (bx0 + bar_w, y + 16), (220, 220, 220), 1)
        seg = int(round(v * bar_w))
        cv2.rectangle(sidebar, (bx0, y), (bx0 + seg, y + 16), col, -1)
        _putlbl(sidebar,
                f"{v:.2f} x {weights[k]:.2f} = {v * weights[k]:.3f}",
                (bx0 + bar_w + 8, y + 13),
                fg=(40, 40, 40), bg=(255, 255, 255), scale=0.4)
        y += 28

    out = np.concatenate([cv2.cvtColor(crop, cv2.COLOR_RGB2BGR), sidebar], axis=1)
    cv2.imwrite(str(out_path), out)
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Top-K thumbnail strip
# ─────────────────────────────────────────────────────────────────────────────

def make_topk_strip(scores: dict, out_path: Path, tile_w: int = 220, k: int = 5):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = scores["top_k"][:k]
    tiles = []
    for i, top in enumerate(rows):
        full = next((f for f in scores["frames"] if f["cam_index"] == top["cam_index"]), None)
        if full is None:
            continue
        crop = _crop_object(Path(full["out_rgba_path"]))
        if crop is None:
            continue
        ratio = tile_w / max(crop.shape[1], 1)
        new_h = max(1, int(round(crop.shape[0] * ratio)))
        thumb = cv2.resize(crop, (tile_w, new_h), interpolation=cv2.INTER_AREA)
        thumb = cv2.cvtColor(thumb, cv2.COLOR_RGB2BGR)
        # Header strip.
        hdr = np.full((54, tile_w, 3), 240, dtype=np.uint8)
        _putlbl(hdr, f"#{i+1}  cam{top['cam_index']}", (8, 20),
                fg=(20, 20, 20), bg=(255, 255, 255), scale=0.5)
        _putlbl(hdr, f"score {top['score']:.3f}", (8, 40),
                fg=(20, 100, 40), bg=(255, 255, 255), scale=0.45)
        _putlbl(hdr, f"az {top['azimuth_V_deg']:.0f}deg", (tile_w - 80, 40),
                fg=(60, 60, 120), bg=(255, 255, 255), scale=0.45)
        tiles.append(np.concatenate([hdr, thumb], axis=0))

    if not tiles:
        return None
    max_h = max(t.shape[0] for t in tiles)
    tiles = [_pad_h(t, max_h) for t in tiles]
    strip = np.concatenate(tiles, axis=1)
    cv2.imwrite(str(out_path), strip)
    return out_path


def _pad_h(img: np.ndarray, target_h: int) -> np.ndarray:
    if img.shape[0] >= target_h:
        return img
    pad = np.full((target_h - img.shape[0], img.shape[1], 3), 240, dtype=np.uint8)
    return np.concatenate([img, pad], axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration
# ─────────────────────────────────────────────────────────────────────────────

def run_debug(model_path: str, object_id: int, output_root: str,
              top_k: int = 5) -> dict:
    out_dir = Path(output_root) / f"obj_{object_id}"
    phase4_dir = out_dir / "phase4"
    debug_dir = out_dir / "debug_phase04"
    debug_dir.mkdir(parents=True, exist_ok=True)

    extraction_index = out_dir / "phase3" / "extraction_index.json"
    if not extraction_index.exists():
        raise FileNotFoundError(f"Run Phase 3 first: missing {extraction_index}")

    # Re-run scope discovery to recover azimuth/elevation tags on cameras.
    scope, _, _, _, _ = discover_object_scope(
        model_path=model_path, object_label_id=object_id,
    )

    scores = run_scoring(extraction_index, scope.cameras, phase4_dir, top_k=top_k)

    # Visual debug.
    make_bar_chart(scores, debug_dir / "bar_chart.png", top_k=top_k)
    make_scatter(scores, debug_dir / "scatter.png")
    make_top1_card(scores, debug_dir / "top1.png")
    make_topk_strip(scores, debug_dir / "top_k_strip.png", k=top_k)

    summary = {
        "scores_json": str(phase4_dir / "scores.json"),
        "n_frames_scored": scores["n_frames"],
        "top1": scores["top_k"][0] if scores["top_k"] else None,
        "weights": scores["weights"],
    }
    with open(debug_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Phase 4 debug saved to: %s", debug_dir)
    return summary


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Phase 4 frame-scoring visual debug.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--object_id", required=True, type=int)
    parser.add_argument("--output_root", default="object_isolation/outputs")
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
    run_debug(args.model_path, args.object_id, args.output_root, args.top_k)


if __name__ == "__main__":
    main()
