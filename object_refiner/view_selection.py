import json
import logging
import math
from pathlib import Path
from typing import cast, Any

import cv2
import numpy as np

from .utils.helpers import find_image, find_tracked_id_map

logger = logging.getLogger(__name__)

WEIGHTS = {
    "front":  0.40,
    "cover":  0.25,
    "sharp":  0.20,
    "expose": 0.15,
}

COVER_TARGET = 0.30
COVER_FLOOR  = 0.02
COVER_CEIL   = 0.85

MIN_MASK_PIXELS  = 64     
EXPOSURE_CLAMP   = 5.0    # penalty scale



def _render_mask(gaussians, object_label_id, camera_spec, shape):
    """Render GS alpha for *object_label_id* and return a bool mask of shape
    *shape* (H, W).  Returns None on any error."""
    try:
        import torch
        from .utils.gstrain_wrapper import make_camera as _make_cam, render_rgba as _render
        from .constants import ALPHA_THRESH
        cam = _make_cam(
            camera_spec["R"], camera_spec["T"], camera_spec["K"],
            camera_spec["width"], camera_spec["height"],
        )
        with torch.no_grad():
            alpha = _render(
                gaussians, cam,
                object_label_id=int(object_label_id),
                training=False, bg_white=False,
            )["alpha"].detach().cpu().numpy()
        if alpha.ndim == 3:
            alpha = alpha[0]
        gs_mask = alpha > ALPHA_THRESH
        H, W = shape
        if gs_mask.shape != (H, W):
            gs_mask = cv2.resize(
                gs_mask.astype(np.uint8), (W, H),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        return gs_mask
    except Exception as exc:  # noqa: BLE001
        logger.debug("GS alpha render failed for cam: %s", exc)
        return None


def _load_object_mask(tracked_id_map_dir, image_name, tracked_object_id, shape,
                       alias_tracked_ids=None):
    if tracked_id_map_dir is None or tracked_object_id is None:
        return None
    id_map_path = find_tracked_id_map(Path(tracked_id_map_dir), image_name)
    if id_map_path is None:
        return None
    id_map_matrix = cv2.imread(str(id_map_path), cv2.IMREAD_UNCHANGED)
    if id_map_matrix is None:
        return None
    id_map = np.asarray(id_map_matrix)
    if id_map.ndim == 3:
        id_map = cast(Any, id_map)[..., 0]
    height, width = shape
    if id_map.shape[:2] != (height, width):
        id_map = cv2.resize(id_map, (width, height), interpolation=cv2.INTER_NEAREST)
    all_ids = [int(tracked_object_id)]
    if alias_tracked_ids:
        all_ids += [int(a) for a in alias_tracked_ids]
    return np.isin(id_map, all_ids)


def run_extraction(scope, images_dir, output_dir, tracked_id_map_dir, tracked_object_id,
                   alias_tracked_ids=None, gaussians=None, object_label_id=None):
    output_dir = Path(output_dir)
    extracted_dir = output_dir / "extracted"
    extracted_dir.mkdir(parents=True, exist_ok=True)
    masks_dir = output_dir / "debug_masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    tracked_id_map_dir = Path(tracked_id_map_dir)
    use_hybrid = gaussians is not None and object_label_id is not None
    
    if alias_tracked_ids:
        logger.info(
            "Extraction: tracked_object_id=%d, alias_tracked_ids=%s",
            tracked_object_id, sorted(alias_tracked_ids),
        )

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

            mask = _load_object_mask(
                tracked_id_map_dir, image_name, tracked_object_id, (height, width),
                alias_tracked_ids=alias_tracked_ids,
            )
            if mask is None:
                raise RuntimeError(
                    f"cam {cam_index} ({image_name}): tracked id-map for object {tracked_object_id} "
                    f"not found under {tracked_id_map_dir}"
                )

            mask_tracker = mask.copy()
            mask_for_debug = None

            if use_hybrid:
                gs_mask = _render_mask(gaussians, object_label_id, camera, (height, width))
                if gs_mask is not None:
                    mask_for_debug = gs_mask
                    mask = np.logical_and(mask, gs_mask)
                else:
                    logger.debug("cam %d render failed using tracker mask only", cam_index)

           
            n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=4)
            if n_labels < 2:
                continue

            valid_components = [
                idx for idx in range(1, n_labels)
                if int(stats[idx, cv2.CC_STAT_AREA]) >= MIN_MASK_PIXELS
            ]
            if not valid_components:
                logger.warning("cam %d: no component meets min pixels (%d)", cam_index, MIN_MASK_PIXELS)
                continue

            mask = np.isin(labels, valid_components)
            n_pixels = int(mask.sum())

            # masks for debug 
            stem = f"{cam_index:03d}__{image_name}"
            tracker_mask_path = str(masks_dir / f"{stem}__tracker.npy")
            np.save(tracker_mask_path, mask_tracker.astype(np.uint8))
            gs_mask_path = None
            if mask_for_debug is not None:
                gs_mask_path = str(masks_dir / f"{stem}__gs.npy")
                np.save(gs_mask_path, mask_for_debug.astype(np.uint8))

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
            exposure = score * (1.0 - min(1.0, float(np.mean(img > 0.98)) * EXPOSURE_CLAMP)) * (1.0 - min(1.0, float(np.mean(img < 0.02)) * EXPOSURE_CLAMP))

            frames.append({
                "cam_index": cam_index,
                "image_name": image_name,
                "object_coverage": n_pixels / (height * width),
                "azimuth_deg": float(camera.get("azimuth_deg", float("nan"))),
                "elevation_deg": float(camera.get("elevation_deg", float("nan"))),
                "rgba_path": str(output_path),
                "tracker_mask_path": tracker_mask_path,
                "gs_mask_path": gs_mask_path,
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
        azimuth_deg = ((float(frame["azimuth_deg"]) + 180.0) % 360.0) - 180.0
        elevation_deg = float(frame["elevation_deg"])

        #prefer 0 and 0
        if not math.isfinite(azimuth_deg) or not math.isfinite(elevation_deg):
            front = 0.0
        else:
            front = ((1.0 + math.cos(math.radians(azimuth_deg))) * 0.5) * max(0.0, math.cos(math.radians(elevation_deg)))

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
            "azimuth_deg": azimuth_deg,
            "elevation_deg": elevation_deg,
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
        components = best.get("components")
        if isinstance(components, dict):
            logger.info("Best frame: camera=%d azimuth degree=%.1f score=%.3f components=%s",
                        best["cam_index"], best["azimuth_deg"], best["score"],
                        {k: round(v, 2) for k, v in components.items()})
    return result
