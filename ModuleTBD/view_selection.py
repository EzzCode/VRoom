import json
import logging
import math
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np

from .utils.gstrain_wrapper import make_camera, render_rgba
from .utils.helpers import find_image, find_id_map, auto_resolve_module1_id

logger = logging.getLogger(__name__)

WEIGHTS = {
    "front":  0.35, # prefer front-facing views
    "cover":  0.20, # how big the object is in frame max is COVER_TARGET
    "sharp":  0.20, # rank-normalized masked Laplacian variance
    "expose": 0.10, 
    "occl":   0.15, # higher fraction of trained model pixels kept in hybrid mask
}

COVER_TARGET = 0.30
COVER_FLOOR  = 0.02
COVER_CEIL   = 0.85

@dataclass
class ExtractedFrame:
    cam_index: int
    image_name: str
    image_path: str
    n_pixels_objgs: int
    n_pixels_hybrid: int
    bbox_xywh: list
    fg_fraction: float
    azimuth_deg: float
    elevation_deg: float
    out_rgba_path: str
    out_mask_path: str
    used_real_mask: bool


def extract_frame(scope, gaussians, pipe_config, cam_index, images_dir,
                  id_map_dir, module1_obj_id, out_rgba_dir, out_mask_dir):

    alpha_thresh = 0.4
    min_pixels = 64

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

    v_camera = make_camera(camera["R"], camera["T"], camera["K"], camera["width"], camera["height"])
    alpha = render_rgba(gaussians, v_camera, pipe_config, object_label_id=scope.object_label_id)
    alpha = alpha["alpha"].detach().cpu().numpy()

    mask_model = (alpha[0] if alpha.ndim == 3 else alpha) > alpha_thresh
    if mask_model.shape != (height, width):
        mask_model = cv2.resize(mask_model.astype(np.uint8), (width, height),
                                interpolation=cv2.INTER_NEAREST).astype(bool)

    used_real = False
    mask_hybrid = mask_model

    id_map_path = find_id_map(id_map_dir, image_name)
    if id_map_path is not None:
        id_map = cv2.imread(str(id_map_path), cv2.IMREAD_UNCHANGED)
        if id_map is not None:
            if id_map.shape[:2] != (height, width):
                id_map = cv2.resize(id_map, (width, height), interpolation=cv2.INTER_NEAREST)
            mask_hybrid = np.logical_and(mask_model, id_map == module1_obj_id)
            used_real = True

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_hybrid.astype(np.uint8), connectivity=4)
    if n_labels < 0:
        mask_hybrid = np.zeros_like(mask_hybrid, dtype=bool)
    areas = stats[1:, cv2.CC_STAT_AREA]
    k = int(np.argmax(areas)) + 1
    if stats[k, cv2.CC_STAT_AREA] < min_pixels:
        mask_hybrid = np.zeros_like(mask_hybrid, dtype=bool)
    else:
        mask_hybrid = labels == k

    n_pixels_hybrid = int(mask_hybrid.sum())
    if n_pixels_hybrid < min_pixels:
        logger.warning("cam %d: mask too small (%d px) — skipping", cam_index, n_pixels_hybrid)
        return None

    alpha_u8 = mask_hybrid.astype(np.uint8) * 255
    rgba = np.concatenate([img, alpha_u8[..., None]], axis=-1)
    ys, xs = np.where(mask_hybrid)
    bbox = [int(xs.min()), int(ys.min()), int(xs.max()) - int(xs.min()) + 1,
            int(ys.max()) - int(ys.min()) + 1]

    out_rgba_dir.mkdir(parents=True, exist_ok=True)
    out_mask_dir.mkdir(parents=True, exist_ok=True)
    rgba_path = out_rgba_dir / f"{cam_index:03d}__{image_name}.png"
    mask_path = out_mask_dir / f"{cam_index:03d}__{image_name}_mask.png"
    cv2.imwrite(str(rgba_path), cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
    cv2.imwrite(str(mask_path), alpha_u8)

    return ExtractedFrame(
        cam_index=cam_index,
        image_name=image_name,
        image_path=str(images_path),
        n_pixels_objgs=int(mask_model.sum()),
        n_pixels_hybrid=n_pixels_hybrid,
        bbox_xywh=bbox,
        fg_fraction=n_pixels_hybrid / (height * width),
        azimuth_deg=float(camera.get("azimuth_deg", float("nan"))),
        elevation_deg=float(camera.get("elevation_deg", float("nan"))),
        out_rgba_path=str(rgba_path),
        out_mask_path=str(mask_path),
        used_real_mask=used_real,
    )


def run_extraction(scope, gaussians, pipe_config, images_dir, output_dir,
                   id_map_dir=None, module1_obj_id=None):
 
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    id_map_dir = Path(id_map_dir) if id_map_dir else None

    if id_map_dir is not None and module1_obj_id is None:
        module1_obj_id = auto_resolve_module1_id(scope, gaussians, pipe_config, id_map_dir)
    else:
        logger.warning("Could not resolve Module1 id — falling back to ObjectGS mask only.")

    views = []
    for ci in scope.visible_cam_indices:
        try:
            v = extract_frame(scope, gaussians, pipe_config, ci,
                              Path(images_dir), id_map_dir, module1_obj_id,
                              output_dir / "extracted", output_dir / "masks")
            if v is not None:
                views.append(v)
        except Exception as e:
            logger.exception("cam %d failed: %s", ci, e)

    n_real = sum(1 for v in views if v.used_real_mask)
    manifest = {
        "object_id": int(scope.object_label_id),
        "module1_obj_id": int(module1_obj_id) if module1_obj_id is not None else None,
        "n_visible_cams": len(scope.visible_cam_indices),
        "n_extracted": len(views),
        "n_used_real_mask": n_real,
        "frames": [asdict(v) for v in views],
    }
    with open(output_dir / "extraction_index.json", "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Extracted %d/%d frames (%d with real mask)",
                len(views), len(scope.visible_cam_indices), n_real)
    return manifest


def _front_score(az_deg, el_deg):
    if not math.isfinite(az_deg) or not math.isfinite(el_deg):
        return 0.0
    s_az = (1.0 + math.cos(math.radians(az_deg))) * 0.5
    s_el = max(0.0, math.cos(math.radians(el_deg)))
    return float(s_az * s_el)


def _cover_score(fg_fraction):
    if fg_fraction <= COVER_FLOOR or fg_fraction >= COVER_CEIL:
        return 0.0
    if fg_fraction <= COVER_TARGET:
        return (fg_fraction - COVER_FLOOR) / max(COVER_TARGET - COVER_FLOOR, 1e-6)
    return max(0.0, 1.0 - (fg_fraction - COVER_TARGET) / max(COVER_CEIL - COVER_TARGET, 1e-6))


def _sharp_metric(rgb, mask):
    if mask.sum() < 32:
        return 0.0
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
    if mask.sum() < 32:
        return 0.0
    pix = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)[mask] / 255.0
    luma = max(0.0, 1.0 - 2.0 * abs(float(np.mean(pix)) - 0.5))
    return float(luma * (1.0 - min(1.0, float(np.mean(pix > 0.98)) * 5.0))
                      * (1.0 - min(1.0, float(np.mean(pix < 0.02)) * 5.0)))


def _occlusion_score(n_hybrid, n_objgs):
    return float(min(1.0, n_hybrid / max(n_objgs, 1)))



def run_scoring(extraction_index_path, top_k=5):
    weights = WEIGHTS
    with open(extraction_index_path) as f:
        frames = json.load(f)["frames"]
    if not frames:
        return {"weights": weights, "frames": [], "ranking": [], "top_k": []}

    collected = []
    for fr in frames:
        img = cv2.imread(fr["image_path"], cv2.IMREAD_COLOR)
        mask_u8 = cv2.imread(fr["out_mask_path"], cv2.IMREAD_GRAYSCALE)
        if img is None or mask_u8 is None:
            continue
        img = img[:, :, ::-1]  # BGR → RGB
        mask = mask_u8 > 127
        az = float(fr.get("azimuth_deg", float("nan")))
        if math.isfinite(az):
            az = ((az + 180.0) % 360.0) - 180.0
        el = float(fr.get("elevation_deg", 0.0))
        if not math.isfinite(el):
            el = 0.0
        collected.append((fr, img, mask, az, el, _sharp_metric(img, mask)))

    sharp_norm = _rank_normalize([item[5] for item in collected])

    results = []
    for (fr, img, mask, az, el, sharp_raw), sharp in zip(collected, sharp_norm):
        comp = {
            "front":  _front_score(az, el),
            "cover":  _cover_score(float(fr["fg_fraction"])),
            "sharp":  float(sharp),
            "expose": _exposure_score(img, mask),
            "occl":   _occlusion_score(int(fr["n_pixels_hybrid"]), int(fr["n_pixels_objgs"])),
        }
        results.append({
            "cam_index": fr["cam_index"],
            "image_name": fr["image_name"],
            "out_rgba_path": fr["out_rgba_path"],
            "azimuth_deg": az,
            "elevation_deg": el,
            "fg_fraction": float(fr["fg_fraction"]),
            "n_pixels_hybrid": int(fr["n_pixels_hybrid"]),
            "n_pixels_objgs": int(fr["n_pixels_objgs"]),
            "sharpness_raw": float(sharp_raw),
            "components": comp,
            "score": sum(weights[k] * comp[k] for k in weights),
        })

    results.sort(key=lambda x: -x["score"])
    for i, r in enumerate(results):
        r["rank"] = i + 1
    
    result = {
        "weights": weights,
        "n_frames": len(results),
        "ranking": results,
        "top_k": results[:top_k],
    }
    
    if result["top_k"]:
        best = result["top_k"][0]
        logger.info("Best frame: camera=%d azimuth degree=%.1f score=%.3f components=%s",
                    best["cam_index"], best["azimuth_deg"], best["score"],
                    {k: round(v, 2) for k, v in best["components"].items()})
    return result
