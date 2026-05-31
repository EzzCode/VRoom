import json
import logging
import math
from pathlib import Path

import cv2
import numpy as np

from .utils.helpers import find_image, find_tracked_id_map

logger = logging.getLogger(__name__)

WEIGHTS = {
    "front":  0.40, # prefer front-facing views
    "cover":  0.25, # how big the object is in frame. max is COVER_TARGET
    "sharp":  0.20,
    "expose": 0.15,
}

COVER_TARGET = 0.30
COVER_FLOOR  = 0.02
COVER_CEIL   = 0.85

MIN_MASK_PIXELS  = 64     
EXPOSURE_CLAMP   = 5.0    # penalty scale



def _load_object_mask(tracked_id_map_dir, image_name, tracked_object_id, shape):
    if tracked_id_map_dir is None or tracked_object_id is None:
        return None
    id_map_path = find_tracked_id_map(Path(tracked_id_map_dir), image_name)
    if id_map_path is None:
        return None
    id_map = cv2.imread(str(id_map_path), cv2.IMREAD_UNCHANGED)
    if id_map is None:
        return None
    if id_map.ndim == 3:
        id_map = id_map[..., 0]
    height, width = shape
    if id_map.shape[:2] != (height, width):
        id_map = cv2.resize(id_map, (width, height), interpolation=cv2.INTER_NEAREST)
    return id_map == int(tracked_object_id)


def run_extraction(scope, images_dir, output_dir, tracked_id_map_dir, tracked_object_id):
    output_dir = Path(output_dir)
    extracted_dir = output_dir / "extracted"
    extracted_dir.mkdir(parents=True, exist_ok=True)
    tracked_id_map_dir = Path(tracked_id_map_dir)

    frames = []
    for cam_index in scope.visible_cam_indices:
        camera = scope.cameras[cam_index]
        image_name = camera["image_name"]
        try:
            images_path = find_image(Path(images_dir), image_name)
            if images_path is None:
                raise FileNotFoundError(f"No image for camera {cam_index} ({image_name}) in {images_dir}")

            img = cv2.imread(str(images_path), cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError(f"Failed to read image {images_path}")

            height, width = img.shape[:2]
            img = img[:, :, ::-1]  # BGR to RGB

            mask = _load_object_mask(tracked_id_map_dir, image_name, tracked_object_id, (height, width))
            if mask is None:
                raise RuntimeError(
                    f"cam {cam_index} ({image_name}): tracked id-map for object {tracked_object_id} "
                    f"not found under {tracked_id_map_dir}"
                )

            n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=4)
            if n_labels < 2:
                continue

            largest_component_idx = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
            n_pixels = int(stats[largest_component_idx, cv2.CC_STAT_AREA])
            if n_pixels < MIN_MASK_PIXELS:
                logger.warning("cam %d: largest component too small (%d px)", cam_index, n_pixels)
                continue

            mask = labels == largest_component_idx

            # Create RGBA output with the object mask as alpha
            masked_frame = np.concatenate([img, (mask.astype(np.uint8) * 255)[..., None]], axis=-1)
            output_path = extracted_dir / f"{cam_index:03d}__{image_name}.png"
            cv2.imwrite(str(output_path), cv2.cvtColor(masked_frame, cv2.COLOR_RGBA2BGRA))

            # variance of Laplacian over the mask is higher more sharper
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            laplacian = cv2.Laplacian(gray, cv2.CV_32F)[mask]
            sharpness = float(np.var(laplacian)) if laplacian.size else 0.0
            
            #normalize values from 0 to 1
            img = gray.astype(np.float32)[mask] / 255.0
            #average brightness of the object 0.5 is perfect condition
            score = max(0.0, 1.0 - 2.0 * abs(float(np.mean(img)) - 0.5))
            #over/under exposure penalty
            exposure = float(score
                             * (1.0 - min(1.0, float(np.mean(img > 0.98)) * EXPOSURE_CLAMP))
                             * (1.0 - min(1.0, float(np.mean(img < 0.02)) * EXPOSURE_CLAMP)))

            frames.append({
                "cam_index": cam_index,
                "image_name": image_name,
                "object_coverage": n_pixels / (height * width),
                "azimuth": float(camera.get("azimuth_deg", float("nan"))),
                "elevation": float(camera.get("elevation_deg", float("nan"))),
                "rgba_path": str(output_path),
                "sharpness": sharpness,
                "exposure": exposure,
            })
        except Exception as e:
            logger.exception("cam %d failed: %s", cam_index, e)

    manifest = {"frames": frames}
    with open(output_dir / "extraction_index.json", "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Extracted %d/%d frames", len(frames), len(scope.visible_cam_indices))
    return manifest


def run_scoring(extraction_index_path, top_k=5):
    with open(extraction_index_path) as f:
        frames = json.load(f)["frames"]
    if not frames:
        return {"ranking": [], "top_k": []}

    # normalise sharpness for outliers 
    sharpness = np.asarray([float(f["sharpness"]) for f in frames], np.float64)
    n = len(frames)
    if n == 1:
        sharp_norm = np.array([1.0], np.float64)
    else:
        sort = np.argsort(sharpness)
        sharp_norm = np.empty(n, dtype=np.float64)
        sharp_norm[sort] = np.linspace(0.0, 1.0, n)

    results = []
    for frame, sharp in zip(frames, sharp_norm):
        #wrap angles to [-180, 180]
        azimuth = ((float(frame["azimuth"]) + 180.0) % 360.0) - 180.0
        elevation = float(frame["elevation"])

        #prefer 0 and 0
        if not math.isfinite(azimuth) or not math.isfinite(elevation):
            front = 0.0
        else:
            front = float(((1.0 + math.cos(math.radians(azimuth))) * 0.5)
                          * max(0.0, math.cos(math.radians(elevation))))

        # Cover is object is too small or fills the whole frame
        coverage = float(frame["object_coverage"])
        eps = 1e-6
        if coverage <= COVER_FLOOR or coverage >= COVER_CEIL:
            cover = 0.0
        elif coverage <= COVER_TARGET:
            cover = (coverage - COVER_FLOOR) / max(COVER_TARGET - COVER_FLOOR, eps)
        else:
            cover = max(0.0, 1.0 - (coverage - COVER_TARGET) / max(COVER_CEIL - COVER_TARGET, eps))

        exposure = float(frame["exposure"])
        score = WEIGHTS["front"] * front + WEIGHTS["cover"] * cover + WEIGHTS["sharp"] * sharp + WEIGHTS["expose"] * exposure
        results.append({
            "cam_index": frame["cam_index"],
            "image_name": frame["image_name"],
            "rgba_path": frame["rgba_path"],
            "azimuth": azimuth,
            "elevation": elevation,
            "object_coverage": coverage,
            "sharpness": float(frame["sharpness"]),
            "components": {"front": front, "cover": cover, "sharp": float(sharp), "expose": exposure},
            "score": score,
        })

    results.sort(key=lambda r: -r["score"])
    for rank, r in enumerate(results, 1):
        r["rank"] = rank

    result = {
        "ranking": results,  # for debug
        "top_k": results[:top_k],
    }

    if result["top_k"]:
        best = result["top_k"][0]
        logger.info("Best frame: camera=%d azimuth degree=%.1f score=%.3f components=%s",
                    best["cam_index"], best["azimuth"], best["score"],
                    {k: round(v, 2) for k, v in best["components"].items()})
    return result
