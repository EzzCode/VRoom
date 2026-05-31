"""Visual Debug for Supervision & Training (ModuleTBD).

Outputs under ``<obj_dir>/04_supervision/debug/``::

    supervision_contact_sheet.png   first ~20 supervision views with source label
    loss_plot.png                   loss/depth-loss curves from 05_training_summary.json
    compare_strip.png               horizontal strip combining 07_compare/ pages
    summary.json                    short numeric summary
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


# ── helpers ───────────────────────────────────────────────────────────────────

def _imread(path):
    if not path or not Path(path).exists():
        return None
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    return img


def _to_rgb(img, bg=(245, 245, 245)):
    if img is None:
        return None
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[-1] == 4:
        a = img[..., -1:].astype(np.float32) / 255.0
        rgb = img[..., :3].astype(np.float32)
        return (rgb * a + np.asarray(bg, np.float32) * (1 - a)).astype(np.uint8)
    return img


def _resize_h(img, h):
    if img is None:
        return None
    ih, iw = img.shape[:2]
    s = min(1.0, h / max(ih, 1))
    if s < 1.0:
        return cv2.resize(img, (int(iw * s), int(ih * s)), cv2.INTER_AREA)
    return img


# ── supervision contact sheet ─────────────────────────────────────────────────

def make_supervision_contact_sheet(obj_dir, debug_dir, n_cols=4, thumb_h=180,
                                   max_views=20):
    manifest_path = Path(obj_dir) / "04_supervision" / "supervision_manifest.json"
    if not manifest_path.exists():
        logger.info("Skipping supervision contact sheet: %s missing", manifest_path)
        return False
    with open(manifest_path) as f:
        man = json.load(f)
    views = man.get("views", [])
    if not views:
        return False
    tiles = []
    for v in views[:max_views]:
        img_path = v.get("image_path")
        rgb = _to_rgb(_imread(img_path))
        if rgb is None:
            continue
        rgb = _resize_h(rgb, thumb_h)
        src = v.get("source", "?")
        col = (60, 180, 60) if src == "real" else (210, 150, 60)
        head = np.full((26, rgb.shape[1], 3), col, np.uint8)
        txt = (f"{src} az={v.get('azimuth_deg', 0.0):+.1f} "
               f"w={v.get('weight', 0.0):.2f}")
        cv2.putText(head, txt, (4, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(np.vstack([head, rgb]))

    if not tiles:
        return False
    while len(tiles) % n_cols != 0:
        tiles.append(np.full_like(tiles[0], 230))
    max_w = max(t.shape[1] for t in tiles)
    tiles = [t if t.shape[1] == max_w else
             np.hstack([t, np.full((t.shape[0], max_w - t.shape[1], 3), 230, np.uint8)])
             for t in tiles]
    rows = [np.hstack(tiles[i:i + n_cols]) for i in range(0, len(tiles), n_cols)]
    cv2.imwrite(str(debug_dir / "supervision_contact_sheet.png"), np.vstack(rows))
    return True


# ── loss plot ─────────────────────────────────────────────────────────────────

def make_loss_plot(obj_dir, debug_dir, training_summary=None,
                   width=900, height=420, pad=50):
    summary = training_summary
    if summary is None:
        summary_path = Path(obj_dir) / "05_training_summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                summary = json.load(f)
    if not summary:
        logger.info("Skipping loss plot: no training summary.")
        return False
    loss_hist = summary.get("loss_history") or []
    depth_hist = summary.get("depth_loss_history") or []
    if not loss_hist:
        return False
    img = np.full((height, width, 3), 250, np.uint8)
    cv2.rectangle(img, (pad, pad), (width - pad, height - pad), (200, 200, 200), 1)

    def plot(series, color, label, y_off):
        if not series:
            return
        ys = np.asarray(series, np.float32)
        xs = np.linspace(0, 1, len(ys))
        y_min, y_max = float(ys.min()), float(ys.max())
        y_span = max(y_max - y_min, 1e-6)
        pts = []
        for x, y in zip(xs, ys):
            px = pad + int(x * (width - 2 * pad))
            py = (height - pad) - int((y - y_min) / y_span * (height - 2 * pad))
            pts.append((px, py))
        for a, b in zip(pts[:-1], pts[1:]):
            cv2.line(img, a, b, color[::-1], 1, cv2.LINE_AA)
        cv2.putText(img, f"{label}: min={y_min:.4f} max={y_max:.4f} last={float(ys[-1]):.4f}",
                    (pad, pad - 12 - y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    color[::-1], 1, cv2.LINE_AA)

    plot(loss_hist, (60, 80, 220), "total loss", 0)
    plot(depth_hist, (60, 160, 60), "depth loss", 18)
    cv2.putText(img, "iteration ->", (width // 2 - 50, height - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 30), 1, cv2.LINE_AA)
    cv2.imwrite(str(debug_dir / "loss_plot.png"), img)
    return True


# ── compare sheet ─────────────────────────────────────────────────────────────

def make_compare_strip(obj_dir, debug_dir, max_w=720):
    cmp_dir = Path(obj_dir) / "07_compare"
    if not cmp_dir.is_dir():
        return False
    items = sorted(cmp_dir.glob("compare_view_*.png"))
    if not items:
        items = sorted(p for p in cmp_dir.iterdir() if p.suffix.lower() in (".png", ".jpg"))
    if not items:
        return False
    tiles = []
    for p in items[:12]:
        img = _to_rgb(_imread(p))
        if img is None:
            continue
        h, w = img.shape[:2]
        scale = min(1.0, float(max_w) / max(float(w), 1.0))
        if scale < 1.0:
            img = cv2.resize(img, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))), cv2.INTER_AREA)
        head = np.full((22, img.shape[1], 3), 40, np.uint8)
        cv2.putText(head, p.name, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(np.vstack([head, img]))
    if not tiles:
        return False
    cols = 2
    rows = []
    for start in range(0, len(tiles), cols):
        row_tiles = tiles[start:start + cols]
        h_full = max(t.shape[0] for t in row_tiles)
        padded = []
        for t in row_tiles:
            if t.shape[0] < h_full:
                t = np.vstack([t, np.full((h_full - t.shape[0], t.shape[1], 3), 245, np.uint8)])
            padded.append(t)
        while len(padded) < cols:
            padded.append(np.full((h_full, padded[0].shape[1], 3), 245, np.uint8))
        rows.append(np.hstack([padded[0], np.full((h_full, 8, 3), 220, np.uint8), padded[1]]))
    w_full = max(r.shape[1] for r in rows)
    rows = [np.hstack([r, np.full((r.shape[0], w_full - r.shape[1], 3), 245, np.uint8)]) if r.shape[1] < w_full else r for r in rows]
    cv2.imwrite(str(debug_dir / "compare_strip.png"), np.vstack([v for row in rows for v in (row, np.full((8, w_full, 3), 220, np.uint8))][:-1]))
    return True


# ── orchestrator ──────────────────────────────────────────────────────────────

def generate_debug_artifacts(*, obj_dir, debug_dir,
                             training_summary=None,
                             scope=None, frame=None, gaussians=None,
                             pipe_config=None, n_compare_views=8,
                             do_compare_renders=False, object_id=None):
    obj_dir = Path(obj_dir)
    debug_dir = Path(debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)

    info = {
        "supervision_sheet": bool(make_supervision_contact_sheet(obj_dir, debug_dir)),
        "loss_plot": bool(make_loss_plot(obj_dir, debug_dir, training_summary)),
        "compare_strip": bool(make_compare_strip(obj_dir, debug_dir)),
    }
    with open(debug_dir / "summary.json", "w") as f:
        json.dump(info, f, indent=2)
    logger.info("Supervision debug saved to: %s (%s)", debug_dir, info)
    return info


def main():
    parser = argparse.ArgumentParser(description="ModuleTBD supervision/training visual debug.")
    parser.add_argument("--obj_dir", required=True,
                        help="Path to obj_<id>/ produced by ModuleTBD.")
    parser.add_argument("--debug_dir", default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
    obj_dir = Path(args.obj_dir)
    debug_dir = Path(args.debug_dir) if args.debug_dir else obj_dir / "04_supervision" / "debug"
    generate_debug_artifacts(obj_dir=obj_dir, debug_dir=debug_dir)


if __name__ == "__main__":
    main()
