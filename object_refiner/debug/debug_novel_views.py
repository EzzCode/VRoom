"""Visual debug for the current hallucination manifest."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import cast, Any

import cv2
import numpy as np

_VROOM_ROOT = Path(__file__).resolve().parents[2]
if str(_VROOM_ROOT) not in sys.path:
    sys.path.insert(0, str(_VROOM_ROOT))

logger = logging.getLogger(__name__)


def _imread(path):
    return cv2.imread(str(path), cv2.IMREAD_UNCHANGED) if path else None


def _rgba_on_bg(rgba, bg=(245, 245, 245)):
    if rgba is None:
        return None
    if rgba.ndim == 2:
        return cv2.cvtColor(rgba, cv2.COLOR_GRAY2BGR)
    if rgba.shape[-1] == 4:
        alpha = rgba[..., -1:].astype(np.float32) / 255.0
        rgb = rgba[..., :3].astype(np.float32)
        out = rgb * alpha + np.full_like(rgb, bg, np.float32) * (1.0 - alpha)
        return out.astype(np.uint8)
    return rgba


def _resize_h(img, max_h=170):
    if img is None:
        return None
    img = np.asarray(img)
    h, w = img.shape[:2]
    scale = min(1.0, max_h / max(h, 1))
    if scale < 1.0:
        return cv2.resize(cast(Any, img), (max(1, int(w * scale)), max(1, int(h * scale))), cv2.INTER_AREA)  # type: ignore
    return img


def _label_band(img, text, bg):
    band = np.full((28, img.shape[1], 3), bg, np.uint8)
    cv2.putText(band, text, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([band, img])


def _pad_to(tile, height, width):
    out = np.full((height, width, 3), 230, np.uint8)
    out[:tile.shape[0], :tile.shape[1]] = tile
    return out


def _camera_azimuth(camera):
    if isinstance(camera, dict):
        return camera.get("azimuth_deg")
    return getattr(camera, "azimuth_deg", None)


def make_generated_grid(manifest, debug_dir, n_cols=5, thumb_h=170):
    tiles = []
    for frame in manifest.get("frames", []):
        img = _rgba_on_bg(_imread(frame.get("rgba_path")))
        if img is None:
            continue
        img = _resize_h(img, thumb_h)
        accepted = bool(frame.get("accepted", False))
        bg = (60, 160, 80) if accepted else (170, 70, 70)
        label = f"#{frame.get('index')} az={float(frame.get('azimuth_deg', 0.0)):+.1f} el={float(frame.get('elevation_deg', 0.0)):+.1f}"
        tiles.append(_label_band(img, label, bg))

    if not tiles:
        return None

    tile_h = max(tile.shape[0] for tile in tiles)
    tile_w = max(tile.shape[1] for tile in tiles)
    tiles = [_pad_to(tile, tile_h, tile_w) for tile in tiles]
    while len(tiles) % n_cols != 0:
        tiles.append(np.full((tile_h, tile_w, 3), 230, np.uint8))

    rows = [np.hstack(tiles[i:i + n_cols]) for i in range(0, len(tiles), n_cols)]
    grid = np.vstack(rows)
    out_path = debug_dir / "generated_grid.png"
    cv2.imwrite(str(out_path), grid)
    return out_path


def make_coverage_polar(manifest, scope_cameras, debug_dir):
    canvas = 640
    img = np.full((canvas, canvas, 3), 250, np.uint8)
    cx, cy = canvas // 2, canvas // 2
    r_outer = canvas // 2 - 30
    cv2.circle(img, (cx, cy), r_outer, (200, 200, 200), 1)
    cv2.line(img, (cx, cy - r_outer), (cx, cy + r_outer), (220, 220, 220), 1)
    cv2.line(img, (cx - r_outer, cy), (cx + r_outer, cy), (220, 220, 220), 1)

    def point(azimuth_deg, radius):
        angle = np.deg2rad(float(azimuth_deg))
        return int(cx + np.sin(angle) * radius), int(cy - np.cos(angle) * radius)

    for camera in scope_cameras or []:
        azimuth = _camera_azimuth(camera)
        if azimuth is None:
            continue
        cv2.circle(img, point(azimuth, r_outer * 0.85), 4, (180, 60, 60), -1)

    for frame in manifest.get("frames", []):
        color = (60, 180, 60) if frame.get("accepted") else (180, 60, 60)
        center = point(frame.get("azimuth_deg", 0.0), r_outer * 0.55)
        cv2.circle(img, center, 6, color, -1)

    cond = manifest.get("conditioning", {})
    if cond:
        cv2.drawMarker(img, point(cond.get("azimuth_deg", 0.0), r_outer * 0.85), (40, 120, 220), cv2.MARKER_STAR, 14, 2)

    cv2.putText(img, "Azimuth coverage", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1, cv2.LINE_AA)
    out_path = debug_dir / "coverage_polar.png"
    cv2.imwrite(str(out_path), img)
    return out_path


def generate_debug_artifacts(*, manifest, scope_cameras, debug_dir):
    debug_dir = Path(debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)
    make_generated_grid(manifest, debug_dir)
    make_coverage_polar(manifest, scope_cameras, debug_dir)

    # Copy input conditioning image if it exists in 03_novel_views
    try:
        obj_dir = debug_dir.parent.parent
        input_cond_src = obj_dir / "03_novel_views" / "input.png"
        if input_cond_src.exists():
            import shutil
            shutil.copy2(input_cond_src, debug_dir / "input.png")
            logger.info("Copied input conditioning image to debug novel_views: %s", debug_dir / "input.png")
    except Exception as exc:
        logger.warning("Failed to copy input conditioning image to debug: %s", exc)

    frames = manifest.get("frames", [])
    summary = {
        "n_views": int(manifest.get("n_views", len(frames))),
        "n_kept": int(manifest.get("n_kept", sum(1 for frame in frames if frame.get("accepted")))),
        "conditioning": manifest.get("conditioning", {}),
        "accepted_indices": [int(frame["index"]) for frame in frames if frame.get("accepted")],
    }
    with open(debug_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Novel-view debug saved to: %s", debug_dir)
    return summary


def main():
    parser = argparse.ArgumentParser(description="object_refiner novel-view visual debug.")
    parser.add_argument("--hallucination_index", required=True)
    parser.add_argument("--debug_dir", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
    with open(args.hallucination_index) as f:
        manifest = json.load(f)
    generate_debug_artifacts(manifest=manifest, scope_cameras=[], debug_dir=Path(args.debug_dir))


if __name__ == "__main__":
    main()
