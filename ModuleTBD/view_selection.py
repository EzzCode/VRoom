import json
import logging
import math
from pathlib import Path

import cv2
import numpy as np

from .utils.helpers import find_image, find_seg_map

logger = logging.getLogger(__name__)

WEIGHTS = {
    "front":  0.40, # prefer front-facing views
    "cover":  0.25, # how big the object is in frame max is COVER_TARGET
    "sharp":  0.20, # rank-normalized masked Laplacian variance
    "expose": 0.15,
}

COVER_TARGET = 0.30
COVER_FLOOR  = 0.02
COVER_CEIL   = 0.85

MIN_MASK_PIXELS  = 64     # frames with fewer foreground pixels are discarded
EXPOSURE_CLAMP   = 5.0    # penalty scale for blown-out / blacked-out pixels


def _largest_cc(mask, min_pixels=MIN_MASK_PIXELS):
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=4)
    if n_labels < 2:
        return np.zeros_like(mask, dtype=bool)
    areas = stats[1:, cv2.CC_STAT_AREA]
    k = int(np.argmax(areas)) + 1
    if stats[k, cv2.CC_STAT_AREA] < int(min_pixels):
        return np.zeros_like(mask, dtype=bool)
    return labels == k


def _close_and_fill(mask):
    if not np.any(mask):
        return np.zeros_like(mask, dtype=bool)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    h, w = closed.shape
    flood = closed.copy()
    ff = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, ff, (0, 0), 1)
    holes = flood == 0
    return (closed.astype(bool) | holes)


def _load_seg_mask(seg_map_dir, image_name, seg_label, shape):
    """Return a boolean mask for `seg_label` read from Module1's per-frame segmentation map."""
    if seg_map_dir is None or seg_label is None:
        return None
    seg_map_path = find_seg_map(Path(seg_map_dir), image_name)
    if seg_map_path is None:
        return None
    seg_map = cv2.imread(str(seg_map_path), cv2.IMREAD_UNCHANGED)
    if seg_map is None:
        return None
    if seg_map.ndim == 3:
        seg_map = seg_map[..., 0]
    height, width = shape
    if seg_map.shape[:2] != (height, width):
        seg_map = cv2.resize(seg_map, (width, height), interpolation=cv2.INTER_NEAREST)
    return seg_map == int(seg_label)


def extract_frame(scope, cam_index, images_dir, out_rgba_dir,
                  seg_map_dir, seg_label,
                  min_pixels=MIN_MASK_PIXELS):

    camera = scope.cameras[cam_index]
    image_name = camera["image_name"]
    images_path = find_image(images_dir, image_name)

    if images_path is None:
        logger.warning("No image for camera index %d image (%s) in %s", cam_index, image_name, images_dir)
        raise FileNotFoundError(f"No image for camera {cam_index} ({image_name}) in {images_dir}")

    img = cv2.imread(str(images_path), cv2.IMREAD_COLOR)
    if img is None:
        logger.warning("Failed to read %s", images_path)
        raise IOError(f"Failed to read {images_path}")

    height, width = img.shape[:2]
    img = img[:, :, ::-1]  # BGR to RGB

    raw_mask = _load_seg_mask(seg_map_dir, image_name, seg_label, (height, width))
    if raw_mask is None:
        raise RuntimeError(
            f"cam {cam_index} ({image_name}): seg map not found under {seg_map_dir} "
            f"for seg_label={seg_label}. Ensure Module1 seg maps cover all visible cameras."
        )

    mask = _largest_cc(_close_and_fill(raw_mask), min_pixels=min_pixels)

    n_pixels = int(mask.sum())
    if n_pixels < int(min_pixels):
        logger.warning("cam %d: mask too small (%d px) — skipping", cam_index, n_pixels)
        return None

    alpha_u8 = mask.astype(np.uint8) * 255
    rgba = np.concatenate([img, alpha_u8[..., None]], axis=-1)
    out_rgba_dir.mkdir(parents=True, exist_ok=True)
    rgba_path = out_rgba_dir / f"{cam_index:03d}__{image_name}.png"
    cv2.imwrite(str(rgba_path), cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))

    return {
        "cam_index": cam_index,
        "image_name": image_name,
        "fg_fraction": n_pixels / (height * width),
        "azimuth_deg": float(camera.get("azimuth_deg", float("nan"))),
        "elevation_deg": float(camera.get("elevation_deg", float("nan"))),
        "out_rgba_path": str(rgba_path),
        "sharpness": _sharp_metric(img, mask),
        "exposure": _exposure_score(img, mask),
    }


def run_extraction(scope, images_dir, output_dir,
                   seg_map_dir, seg_label,
                   min_pixels=MIN_MASK_PIXELS):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seg_map_dir = Path(seg_map_dir)

    frames = []
    for ci in scope.visible_cam_indices:
        try:
            frame = extract_frame(
                scope, ci,
                Path(images_dir),
                output_dir / "extracted",
                seg_map_dir=seg_map_dir,
                seg_label=seg_label,
                min_pixels=min_pixels,
            )
            if frame is not None:
                frames.append(frame)
        except Exception as e:
            logger.exception("cam %d failed: %s", ci, e)

    manifest = {"frames": frames}
    with open(output_dir / "extraction_index.json", "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Extracted %d/%d frames", len(frames), len(scope.visible_cam_indices))
    return manifest


def _front_score(az_deg, el_deg):
    if not math.isfinite(az_deg) or not math.isfinite(el_deg):
        logger.debug("Non-finite azimuth or elevation")
        return 0.0
    s_az = (1.0 + math.cos(math.radians(az_deg))) * 0.5
    s_el = max(0.0, math.cos(math.radians(el_deg)))
    return float(s_az * s_el)


def _cover_score(fg_fraction):
    if fg_fraction <= COVER_FLOOR or fg_fraction >= COVER_CEIL:
        logger.debug("Foreground fraction %.3f outside of [%f, %f]", fg_fraction, COVER_FLOOR, COVER_CEIL)
        return 0.0
    if fg_fraction <= COVER_TARGET:
        return (fg_fraction - COVER_FLOOR) / max(COVER_TARGET - COVER_FLOOR, 1e-6)
    return max(0.0, 1.0 - (fg_fraction - COVER_TARGET) / max(COVER_CEIL - COVER_TARGET, 1e-6))


def _sharp_metric(rgb, mask):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    vals = lap[mask]
    return float(np.var(vals)) if vals.size else 0.0


def _rank_normalize(values):
    values = np.asarray(values, np.float64)
    n = len(values)
    if n == 0:
        return values
    if n == 1:
        return np.array([1.0], np.float64)
    order = np.argsort(values)
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.linspace(0.0, 1.0, n)
    return ranks



def _exposure_score(rgb, mask):
    pix = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)[mask] / 255.0
    luma = max(0.0, 1.0 - 2.0 * abs(float(np.mean(pix)) - 0.5))
    return float(luma * (1.0 - min(1.0, float(np.mean(pix > 0.98)) * EXPOSURE_CLAMP))
                      * (1.0 - min(1.0, float(np.mean(pix < 0.02)) * EXPOSURE_CLAMP)))


def run_scoring(extraction_index_path, top_k=5):
    weights = WEIGHTS
    with open(extraction_index_path) as f:
        frames = json.load(f)["frames"]
    if not frames:
        return {"ranking": [], "top_k": []}

    collected = []
    for fr in frames:
        az = float(fr.get("azimuth_deg", float("nan")))
        if math.isfinite(az):
            az = ((az + 180.0) % 360.0) - 180.0
        el = float(fr.get("elevation_deg", 0.0))
        if not math.isfinite(el):
            el = 0.0
        collected.append((fr, az, el, float(fr["sharpness"]), float(fr["exposure"])))

    sharp_norm = _rank_normalize([item[3] for item in collected])

    results = []
    for (fr, az, el, sharp_raw, exposure), sharp in zip(collected, sharp_norm):
        comp = {
            "front":  _front_score(az, el),
            "cover":  _cover_score(float(fr["fg_fraction"])),
            "sharp":  float(sharp),
            "expose": exposure,
        }
        results.append({
            "cam_index": fr["cam_index"],
            "image_name": fr["image_name"],
            "out_rgba_path": fr["out_rgba_path"],
            "azimuth_deg": az,
            "elevation_deg": el,
            "fg_fraction": float(fr["fg_fraction"]),
            "sharpness": float(sharp_raw),
            "components": comp,
            "score": sum(weights[k] * comp[k] for k in weights),
        })

    results.sort(key=lambda x: -x["score"])
    for i, r in enumerate(results):
        r["rank"] = i + 1
    
    result = {
        "ranking": results,  # for debug generation
        "top_k": results[:top_k],
    }
    
    if result["top_k"]:
        best = result["top_k"][0]
        logger.info("Best frame: camera=%d azimuth degree=%.1f score=%.3f components=%s",
                    best["cam_index"], best["azimuth_deg"], best["score"],
                    {k: round(v, 2) for k, v in best["components"].items()})
    return result
