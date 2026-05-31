"""Visual Debug for Novel-View Synthesis (ModuleTBD).

Outputs under ``<obj_dir>/03_novel_views/debug/``::

    conditioning_panel.png   side-by-side: conditioning, alpha, alpha-on-bg
    sv3d_grid.png            grid of all SV3D outputs with IoU + accept/reject
    iou_strip.png            bar chart of IoU per generated frame
    coverage_polar.png       polar plot showing real cam azimuths vs SV3D az
    summary.json             counts / accepted indices
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


def _imread(path):
    if not path:
        return None
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    return img


def _rgba_on_bg(rgba, bg=(245, 245, 245)):
    if rgba is None:
        return None
    if rgba.ndim == 2:
        rgb = cv2.cvtColor(rgba, cv2.COLOR_GRAY2BGR)
        return rgb
    if rgba.shape[-1] == 4:
        a = rgba[..., -1:].astype(np.float32) / 255.0
        rgb = rgba[..., :3].astype(np.float32)
        bg_arr = np.full_like(rgb, bg, np.float32)
        out = rgb * a + bg_arr * (1 - a)
        return out.astype(np.uint8)
    return rgba


def _resize_h(img, max_h=192):
    if img is None:
        return None
    h, w = img.shape[:2]
    s = min(1.0, max_h / max(h, 1))
    if s < 1.0:
        return cv2.resize(img, (int(w * s), int(h * s)), cv2.INTER_AREA)
    return img


def _label_band(img, text, color=(255, 255, 255), bg=(40, 40, 40)):
    h, w = img.shape[:2]
    band = np.full((28, w, 3), bg, np.uint8)
    cv2.putText(band, text, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                color, 1, cv2.LINE_AA)
    return np.vstack([band, img])


# ── conditioning panel ────────────────────────────────────────────────────────

def make_conditioning_panel(manifest, debug_dir):
    cond = manifest.get("conditioning", {})
    cond_path = cond.get("image_path")
    rgba = _imread(cond_path)
    if rgba is None:
        return
    rgb_on_bg = _rgba_on_bg(rgba)
    # Prefer the extracted RGBA (which carries the hybrid mask alpha).
    rgba_src = _imread(cond.get("rgba_path")) if cond.get("rgba_path") else None
    if rgba_src is not None and rgba_src.ndim == 3 and rgba_src.shape[-1] == 4:
        alpha = rgba_src[..., -1]
        alpha_vis = cv2.cvtColor(alpha, cv2.COLOR_GRAY2BGR)
    elif rgba is not None and rgba.ndim == 3 and rgba.shape[-1] == 4:
        alpha = rgba[..., -1]
        alpha_vis = cv2.cvtColor(alpha, cv2.COLOR_GRAY2BGR)
    else:
        alpha_vis = None
    rgb_on_bg = _resize_h(rgb_on_bg, 320)
    tiles = [_label_band(rgb_on_bg, "conditioning RGB")]
    if alpha_vis is not None:
        alpha_vis = _resize_h(alpha_vis, 320)
        if alpha_vis.shape[:2] != rgb_on_bg.shape[:2]:
            alpha_vis = cv2.resize(alpha_vis, (rgb_on_bg.shape[1], rgb_on_bg.shape[0]),
                                   cv2.INTER_NEAREST)
        tiles.append(np.full((rgb_on_bg.shape[0] + 28, 4, 3), 220, np.uint8))
        tiles.append(_label_band(alpha_vis, "alpha"))
    panel = np.hstack(tiles)

    header = np.full((40, panel.shape[1], 3), 245, np.uint8)
    txt = (f"cond cam={cond.get('cam_index','?')} | "
           f"{cond.get('image_name','?')} | "
           f"az={cond.get('azimuth_deg', 0.0):+.1f} | "
           f"el={cond.get('elevation_deg', 0.0):+.1f} | "
           f"score={cond.get('score', 0.0):+.3f}")
    cv2.putText(header, txt, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (30, 30, 30), 1, cv2.LINE_AA)
    out = np.vstack([header, panel])
    cv2.imwrite(str(debug_dir / "conditioning_panel.png"), out)


# ── sv3d grid ─────────────────────────────────────────────────────────────────

def make_sv3d_grid(manifest, debug_dir, n_cols=5, thumb_h=170):
    frames = manifest.get("frames", [])
    tiles = []
    for fr in frames:
        sv3d_path = fr.get("sv3d_raw_path") or fr.get("out_rgba_path")
        ref_path = fr.get("objgs_ref_path")
        sv3d = _rgba_on_bg(_imread(sv3d_path))
        ref = _rgba_on_bg(_imread(ref_path))
        if sv3d is None and ref is None:
            continue
        sv3d = _resize_h(sv3d, thumb_h) if sv3d is not None else np.full((thumb_h, thumb_h, 3), 230, np.uint8)
        ref = _resize_h(ref, thumb_h) if ref is not None else np.full((thumb_h, thumb_h, 3), 230, np.uint8)
        if sv3d.shape[1] != ref.shape[1]:
            ref = cv2.resize(ref, (sv3d.shape[1], sv3d.shape[0]), cv2.INTER_AREA)
        pair = np.hstack([sv3d, np.full((sv3d.shape[0], 3, 3), 220, np.uint8), ref])
        acc = fr.get("accepted", True)
        col = (60, 180, 60) if acc else (180, 60, 60)
        rr = (" cond" if fr.get("is_conditioning") else "")
        txt = (f"#{fr.get('index')} az={fr.get('azimuth_deg', 0.0):+.1f} "
               f"iou={fr.get('iou_with_objgs', 0.0):.2f}{rr}")
        if not acc and fr.get("reject_reason"):
            txt += f" [{fr.get('reject_reason')}]"
        pair = _label_band(pair, txt, bg=col)
        tiles.append(pair)

    if not tiles:
        return
    while len(tiles) % n_cols != 0:
        tiles.append(np.full_like(tiles[0], 230))
    rows = []
    for i in range(0, len(tiles), n_cols):
        rows.append(np.hstack(tiles[i:i + n_cols]))
    grid = np.vstack(rows)
    cv2.imwrite(str(debug_dir / "sv3d_grid.png"), grid)


# ── IoU strip ─────────────────────────────────────────────────────────────────

def make_iou_strip(manifest, debug_dir):
    frames = manifest.get("frames", [])
    if not frames:
        return
    iou_thresh = manifest.get("params", {}).get("iou_threshold", 0.0)
    width, height = 900, 320
    pad = 50
    img = np.full((height, width, 3), 250, np.uint8)
    n = len(frames)
    bar_w = max(int((width - 2 * pad) / max(n, 1)) - 2, 2)
    for i, fr in enumerate(frames):
        x = pad + i * (bar_w + 2)
        iou = float(fr.get("iou_with_objgs", 0.0))
        h_px = int(iou * (height - 2 * pad))
        col = ((60, 180, 60) if fr.get("accepted") else (180, 60, 60))[::-1]
        if fr.get("is_conditioning"):
            col = (40, 120, 220)[::-1]
        cv2.rectangle(img, (x, height - pad - h_px), (x + bar_w, height - pad), col, -1)
    y_thr = height - pad - int(iou_thresh * (height - 2 * pad))
    cv2.line(img, (pad, y_thr), (width - pad, y_thr), (40, 40, 40), 1, cv2.LINE_AA)
    cv2.putText(img, f"IoU threshold = {iou_thresh:.2f}",
                (pad, y_thr - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (40, 40, 40), 1, cv2.LINE_AA)
    cv2.putText(img, "Per-view IoU(SV3D mask vs ObjectGS mask)",
                (pad, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (30, 30, 30), 1, cv2.LINE_AA)
    cv2.imwrite(str(debug_dir / "iou_strip.png"), img)


# ── coverage polar ────────────────────────────────────────────────────────────

def make_coverage_polar(manifest, scope_cameras, debug_dir):
    canvas = 640
    img = np.full((canvas, canvas, 3), 250, np.uint8)
    cx, cy = canvas // 2, canvas // 2
    r_outer = canvas // 2 - 30
    cv2.circle(img, (cx, cy), r_outer, (200, 200, 200), 1)
    cv2.line(img, (cx, cy - r_outer), (cx, cy + r_outer), (220, 220, 220), 1)
    cv2.line(img, (cx - r_outer, cy), (cx + r_outer, cy), (220, 220, 220), 1)

    def pt(az_deg, r):
        a = np.deg2rad(az_deg)
        return int(cx + np.sin(a) * r), int(cy - np.cos(a) * r)

    # real cameras
    for cam in scope_cameras or []:
        if "azimuth_deg" not in cam:
            continue
        x, y = pt(float(cam["azimuth_deg"]), r_outer * 0.85)
        cv2.circle(img, (x, y), 4, (180, 60, 60), -1)

    # sv3d frames
    for fr in manifest.get("frames", []):
        az = float(fr.get("azimuth_deg", 0.0))
        x, y = pt(az, r_outer * 0.55)
        col = (60, 180, 60) if fr.get("accepted") else (180, 60, 60)
        cv2.circle(img, (x, y), 6, col, -1)
        if fr.get("is_conditioning"):
            cv2.circle(img, (x, y), 9, (40, 120, 220), 2)

    # conditioning marker
    cond = manifest.get("conditioning", {})
    if cond:
        x, y = pt(float(cond.get("azimuth_deg", 0.0)), r_outer * 0.85)
        cv2.drawMarker(img, (x, y), (40, 120, 220), cv2.MARKER_STAR, 14, 2)

    legend = [
        ("RED dot   real training cam azimuth", (180, 60, 60)),
        ("GREEN dot SV3D accepted frame", (60, 180, 60)),
        ("RED dot   SV3D rejected frame", (180, 60, 60)),
        ("BLUE star Conditioning frame", (40, 120, 220)),
    ]
    for i, (txt, col) in enumerate(legend):
        cv2.putText(img, txt, (12, 22 + i * 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, col, 1, cv2.LINE_AA)
    cv2.putText(img, "Azimuth coverage (top-down, az=0 -> up)",
                (12, canvas - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (30, 30, 30), 1, cv2.LINE_AA)
    cv2.imwrite(str(debug_dir / "coverage_polar.png"), img)


# ── orchestrator ──────────────────────────────────────────────────────────────

def generate_debug_artifacts(*, manifest, scope_cameras, debug_dir):
    debug_dir = Path(debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)
    make_conditioning_panel(manifest, debug_dir)
    make_sv3d_grid(manifest, debug_dir)
    make_iou_strip(manifest, debug_dir)
    make_coverage_polar(manifest, scope_cameras, debug_dir)

    frames = manifest.get("frames", [])
    summary = {
        "n_views": int(manifest.get("n_views", len(frames))),
        "n_kept": int(manifest.get("n_kept", sum(1 for f in frames if f.get("accepted")))),
        "iou_threshold": float(manifest.get("params", {}).get("iou_threshold", 0.0)),
        "accepted_indices": [int(f["index"]) for f in frames if f.get("accepted")],
        "reject_reasons": {f.get("reject_reason", ""): 1 for f in frames if not f.get("accepted")},
    }
    with open(debug_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Novel-views debug saved to: %s", debug_dir)
    return summary


def main():
    parser = argparse.ArgumentParser(description="ModuleTBD novel-views visual debug.")
    parser.add_argument("--hallucination_index", required=True)
    parser.add_argument("--debug_dir", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
    with open(args.hallucination_index, "r") as f:
        manifest = json.load(f)
    generate_debug_artifacts(manifest=manifest, scope_cameras=[],
                             debug_dir=Path(args.debug_dir))


if __name__ == "__main__":
    main()
