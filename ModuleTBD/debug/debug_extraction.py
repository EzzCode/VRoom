"""Visual Debug for Object View Extraction (ModuleTBD).

Outputs under ``<obj_dir>/debug/extraction/``::

    triptych/                    per-frame [source | used mask] grids
    contact_sheet.png            12-frame thumbnail grid
    summary.json                 numeric snapshot of the manifest

Run standalone::

    python -m ModuleTBD.debug.debug_extraction \\
        --extraction_index ModuleTBD/outputs/obj_8/01_extraction/extraction_index.json \\
        --images_dir data/3dovs/bed/images_4 \\
        --output_root ModuleTBD/outputs
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

def _imread_rgb(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _image_path(images_dir, image_name):
    value = str(image_name or "")
    path = Path(value)
    candidates = [images_dir / path]
    if path.suffix == "":
        candidates.extend(images_dir / f"{value}{suffix}" for suffix in (".jpg", ".jpeg", ".png"))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _resize_pair(rgb, mask, max_h=320):
    h, w = rgb.shape[:2]
    s = min(1.0, max_h / max(h, 1))
    if s < 1.0:
        rgb = cv2.resize(rgb, (int(w * s), int(h * s)), cv2.INTER_AREA)
        mask = cv2.resize(mask, (int(w * s), int(h * s)), cv2.INTER_NEAREST)
    return rgb, mask


def _overlay_mask(rgb, mask, color=(0, 200, 0), alpha=0.45):
    base = rgb.astype(np.float32)
    overlay = base.copy()
    overlay[mask > 0] = (1 - alpha) * base[mask > 0] + alpha * np.asarray(color, np.float32)
    return overlay.astype(np.uint8)


def _label(img, text, color=(255, 255, 255)):
    out = img.copy()
    cv2.putText(out, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
    return out


def _make_triptych(rgb, mask, frame_label):
    rgb, mask = _resize_pair(rgb, mask)
    panel_src = _label(rgb, "source")
    panel_mask = _label(_overlay_mask(rgb, mask, (0, 200, 0)),
                        "mask used", (220, 255, 220))
    gap = np.full((panel_src.shape[0], 4, 3), 240, np.uint8)
    row = np.hstack([panel_src, gap, panel_mask])

    header_h = 28
    header = np.full((header_h, row.shape[1], 3), 245, np.uint8)
    cv2.putText(header, frame_label, (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (30, 30, 30), 1, cv2.LINE_AA)
    return np.vstack([header, row])


def _make_contact_sheet(items, n_cols=4, thumb_h=160):
    if not items:
        return None
    thumbs = []
    for rgb, mask, label in items:
        rgb, mask = _resize_pair(rgb, mask, max_h=thumb_h)
        thumb = _overlay_mask(rgb, mask, (0, 200, 0))
        thumb = _label(thumb, label)
        h, w = thumb.shape[:2]
        if w < thumb_h * 4 // 3:
            pad = thumb_h * 4 // 3 - w
            thumb = np.hstack([thumb, np.full((h, pad, 3), 240, np.uint8)])
        else:
            scale = thumb_h * 4 // 3 / w
            thumb = cv2.resize(thumb, (thumb_h * 4 // 3, thumb_h), cv2.INTER_AREA)
        thumbs.append(thumb)

    while len(thumbs) % n_cols != 0:
        thumbs.append(np.full_like(thumbs[0], 240))
    rows = []
    for i in range(0, len(thumbs), n_cols):
        rows.append(np.hstack(thumbs[i:i + n_cols]))
    return np.vstack(rows)


# ── orchestrator ──────────────────────────────────────────────────────────────

def generate_debug_artifacts(*, manifest, images_dir, debug_dir,
                             scope=None, gaussians=None, pipe_config=None,
                             max_triptychs=20,
                             contact_sheet_size=12):
    images_dir = Path(images_dir)
    debug_dir = Path(debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)

    frames = manifest.get("frames", [])
    n_total = len(frames)

    sheet_items = []
    triptych_files = []
    frames_for_triptych = frames[:max_triptychs]
    for f in frames_for_triptych:
        source_path = _image_path(images_dir, f.get("image_name", ""))
        rgb_src = _imread_rgb(source_path) if source_path is not None else None
        rgba = cv2.imread(f.get("rgba_path", ""), cv2.IMREAD_UNCHANGED)
        mask_hybrid = (rgba[..., -1] > 127).astype(np.uint8) * 255 if rgba is not None and rgba.ndim == 3 else None
        if rgb_src is None or mask_hybrid is None:
            continue

        label = (f"cam={f.get('cam_index')} | {f.get('image_name','?')} | "
                 f"az={f.get('azimuth', 0.0):+.1f} | "
                 f"fg={f.get('object_coverage', 0.0):.3f}")
        trip = _make_triptych(rgb_src, mask_hybrid, label)
        (debug_dir / "triptych").mkdir(parents=True, exist_ok=True)
        out_path = debug_dir / "triptych" / f"cam_{f.get('cam_index'):03d}.png"
        if cv2.imwrite(str(out_path), cv2.cvtColor(trip, cv2.COLOR_RGB2BGR)):
            triptych_files.append(str(out_path))

    # Contact sheet
    for f in frames[:contact_sheet_size]:
        source_path = _image_path(images_dir, f.get("image_name", ""))
        rgb_src = _imread_rgb(source_path) if source_path is not None else None
        if rgb_src is None:
            continue
        rgba = cv2.imread(f.get("rgba_path", ""), cv2.IMREAD_UNCHANGED)
        if rgba is not None and rgba.ndim == 3 and rgba.shape[-1] == 4:
            mask = (rgba[..., -1] > 127).astype(np.uint8) * 255
        else:
            mask = None
        if mask is None:
            continue
        sheet_items.append((rgb_src, mask,
                            f"cam={f.get('cam_index')} fg={f.get('object_coverage', 0):.2f}"))

    sheet = _make_contact_sheet(sheet_items)
    contact_sheet = None
    if sheet is not None:
        contact_sheet = debug_dir / "contact_sheet.png"
        if not cv2.imwrite(str(contact_sheet), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR)):
            contact_sheet = None

    summary = {
        "n_frames": n_total,
        "n_triptychs": len(triptych_files),
        "triptych_files": triptych_files,
        "contact_sheet": str(contact_sheet) if contact_sheet is not None else None,
        "object_coverage_mean": float(np.mean([f.get("object_coverage", 0.0) for f in frames])) if frames else 0.0,
        "object_coverage_min": float(np.min([f.get("object_coverage", 0.0) for f in frames])) if frames else 0.0,
        "object_coverage_max": float(np.max([f.get("object_coverage", 0.0) for f in frames])) if frames else 0.0,
    }
    with open(debug_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Extraction debug saved to: %s", debug_dir)
    return summary


def main():
    parser = argparse.ArgumentParser(description="ModuleTBD extraction visual debug.")
    parser.add_argument("--extraction_index", required=True,
                        help="Path to extraction_index.json")
    parser.add_argument("--images_dir", required=True,
                        help="Directory containing source images (e.g. images_4)")
    parser.add_argument("--output_root", default=None,
                        help="If given, writes to <output_root>/debug/extraction/")
    parser.add_argument("--max_triptychs", type=int, default=20)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")

    with open(args.extraction_index, "r") as f:
        manifest = json.load(f)

    if args.output_root:
        debug_dir = Path(args.output_root) / "debug" / "extraction"
    else:
        debug_dir = Path(args.extraction_index).parent / "debug"
    generate_debug_artifacts(
        manifest=manifest,
        images_dir=args.images_dir,
        debug_dir=debug_dir,
        max_triptychs=args.max_triptychs,
    )


if __name__ == "__main__":
    main()
