import json
import logging
import math
import re
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np

from .utils.gstrain_wrapper import make_camera, render_rgba
from .utils.scene_analysis import ObjectScope

logger = logging.getLogger(__name__)

# ── Score weights ──────────────────────────────────────────────────────────────

WEIGHTS = {
    "front":  0.35, # prefer front-facing views
    "cover":  0.20, # how big the object is in frame max is COVER_TARGET
    "sharp":  0.20,
    "expose": 0.10, 
    "occl":   0.15, # higher fraction of trained model pixels kept in hybrid mask
}

COVER_TARGET = 0.30
COVER_FLOOR  = 0.02
COVER_CEIL   = 0.85

# ── File helpers ───────────────────────────────────────────────────────────────

def _find_image(images_dir, img_name):
    if not images_dir.exists():
        return None
    for ext in (".jpg", ".JPG", ".jpeg", ".png", ".PNG"):
        c = images_dir / f"{img_name}{ext}"
        if c.exists():
            return c
    for f in images_dir.iterdir():
        if f.is_file() and f.stem == img_name:
            return f
    return None


def _find_id_map(id_map_dir, img_name):
    if not id_map_dir.exists():
        return None
    for ext in (".png", ".jpg", ".jpeg"):
        c = id_map_dir / f"{img_name}{ext}"
        if c.exists():
            return c
    # Replica-style rename
    renamed = img_name.replace("_rgb_", "_semantic_instance_").replace("rgb", "semantic_instance")
    if renamed != img_name:
        for ext in (".png", ".jpg", ".jpeg"):
            c = id_map_dir / f"{renamed}{ext}"
            if c.exists():
                return c
    # Trailing-digit fuzzy match
    m = re.search(r"(\d+)$", img_name)
    if m:
        suffix = m.group(1)
        for f in id_map_dir.iterdir():
            if f.is_file() and f.suffix.lower() in (".png", ".jpg", ".jpeg"):
                m2 = re.search(r"(\d+)$", f.stem)
                if m2 and m2.group(1) == suffix:
                    return f
    return None

# ── Mask helpers ───────────────────────────────────────────────────────────────

def _largest_cc(mask, min_pixels=64):
    """Return the largest 4-connected component in the mask"""
    n_labels, lbl, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=4)
    if n_labels <= 1:
        return np.zeros_like(mask, dtype=bool)
    areas = stats[1:, cv2.CC_STAT_AREA]
    k = int(np.argmax(areas)) + 1
    if stats[k, cv2.CC_STAT_AREA] < min_pixels:
        return np.zeros_like(mask, dtype=bool)
    return lbl == k


def _close_fill(mask, kernel=3):
    """ close then fill holes."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
    closed = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, k)
    h, w = closed.shape
    flood = closed.copy()
    cv2.floodFill(flood, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 1)
    holes = (flood == 0).astype(np.uint8)
    return (closed | holes) > 0

# ── Extraction ─────────────────────────────────────────────────────────────────

def auto_resolve_module1_id(scope, gaussians, pipe_config, id_map_dir, n_probe=5, tau_alpha=0.4):
    """Vote across cameras to find the Module1 instance id that best matches the ObjectGS silhouette."""
    if not id_map_dir.exists():
        return None
    indices = list(scope.visible_cam_indices)
    if len(indices) > n_probe:
        indices = indices[::max(1, len(indices) // n_probe)][:n_probe]

    votes = {}
    for ci in indices:
        cam_p = scope.cameras[ci]
        id_map_path = _find_id_map(id_map_dir, cam_p["image_name"])
        if id_map_path is None:
            continue
        id_map = cv2.imread(str(id_map_path), cv2.IMREAD_UNCHANGED)
        if id_map is None:
            continue
        cam = make_camera(cam_p["R"], cam_p["T"], cam_p["K"], cam_p["width"], cam_p["height"])
        alpha = render_rgba(gaussians, cam, pipe_config,
                            object_label_id=scope.object_label_id)["alpha"].detach().cpu().numpy()
        m_objgs = (alpha[0] if alpha.ndim == 3 else alpha) > tau_alpha
        H, W = m_objgs.shape
        if id_map.shape[:2] != (H, W):
            id_map = cv2.resize(id_map, (W, H), interpolation=cv2.INTER_NEAREST)
        for uid in np.unique(id_map):
            if int(uid) == 0:
                continue
            m_real = (id_map == uid)
            inter = float(np.logical_and(m_real, m_objgs).sum())
            union = float(np.logical_or(m_real, m_objgs).sum())
            votes[int(uid)] = votes.get(int(uid), 0.0) + inter / max(union, 1.0)

    if not votes:
        return None
    winner, best_score = max(votes.items(), key=lambda kv: kv[1])
    logger.info("Auto-resolved module1_id=%d (top votes: %s)", winner,
                {k: round(v, 3) for k, v in sorted(votes.items(), key=lambda kv: -kv[1])[:5]})
    return winner if best_score > 0 else None


@dataclass
class ExtractedView:
    cam_index: int
    image_name: str
    image_path: str
    n_pixels_objgs: int
    n_pixels_hybrid: int
    bbox_xywh: list
    fg_fraction: float
    azimuth_deg: float
    out_rgba_path: str
    out_mask_path: str
    used_real_mask: bool


def extract_frame(scope, gaussians, pipe_config, cam_index, images_dir,
                  id_map_dir, module1_obj_id, out_rgba_dir, out_mask_dir,
                  tau_alpha=0.4, min_pixels=64):
    cam_p = scope.cameras[cam_index]
    img_name = cam_p["image_name"]

    img_path = _find_image(images_dir, img_name)
    if img_path is None:
        logger.warning("No image for cam %d (%s) in %s", cam_index, img_name, images_dir)
        return None
    bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if bgr is None:
        logger.warning("Failed to read %s", img_path)
        return None
    H_img, W_img = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    cam = make_camera(cam_p["R"], cam_p["T"], cam_p["K"], cam_p["width"], cam_p["height"])
    alpha = render_rgba(gaussians, cam, pipe_config,
                        object_label_id=scope.object_label_id)["alpha"].detach().cpu().numpy()
    m_objgs = (alpha[0] if alpha.ndim == 3 else alpha) > tau_alpha
    if m_objgs.shape != (H_img, W_img):
        m_objgs = cv2.resize(m_objgs.astype(np.uint8), (W_img, H_img),
                             interpolation=cv2.INTER_NEAREST).astype(bool)

    used_real = False
    m_hybrid = m_objgs
    if id_map_dir is not None and module1_obj_id is not None:
        id_map_path = _find_id_map(id_map_dir, img_name)
        if id_map_path is not None:
            id_map = cv2.imread(str(id_map_path), cv2.IMREAD_UNCHANGED)
            if id_map is not None:
                if id_map.shape[:2] != (H_img, W_img):
                    id_map = cv2.resize(id_map, (W_img, H_img), interpolation=cv2.INTER_NEAREST)
                m_hybrid = np.logical_and(m_objgs, id_map == module1_obj_id)
                used_real = True

    m_hybrid = _largest_cc(_close_fill(m_hybrid), min_pixels)
    n_pix_hybrid = int(m_hybrid.sum())
    if n_pix_hybrid < min_pixels:
        logger.warning("cam %d: mask too small (%d px) — skipping", cam_index, n_pix_hybrid)
        return None

    alpha_u8 = m_hybrid.astype(np.uint8) * 255
    rgba = np.concatenate([rgb, alpha_u8[..., None]], axis=-1)
    ys, xs = np.where(m_hybrid)
    bbox = [int(xs.min()), int(ys.min()), int(xs.max()) - int(xs.min()) + 1,
            int(ys.max()) - int(ys.min()) + 1]

    out_rgba_dir.mkdir(parents=True, exist_ok=True)
    out_mask_dir.mkdir(parents=True, exist_ok=True)
    rgba_path = out_rgba_dir / f"{cam_index:03d}__{img_name}.png"
    mask_path = out_mask_dir / f"{cam_index:03d}__{img_name}_mask.png"
    cv2.imwrite(str(rgba_path), cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
    cv2.imwrite(str(mask_path), alpha_u8)

    return ExtractedView(
        cam_index=cam_index,
        image_name=img_name,
        image_path=str(img_path),
        n_pixels_objgs=int(m_objgs.sum()),
        n_pixels_hybrid=n_pix_hybrid,
        bbox_xywh=bbox,
        fg_fraction=n_pix_hybrid / (H_img * W_img),
        azimuth_deg=float(cam_p.get("azimuth_deg", float("nan"))),
        out_rgba_path=str(rgba_path),
        out_mask_path=str(mask_path),
        used_real_mask=used_real,
    )


def run_extraction(scope, gaussians, pipe_config, images_dir, output_dir,
                   id_map_dir=None, module1_obj_id=None,
                   tau_alpha=0.4, min_pixels=64, auto_resolve=True):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    id_map_dir = Path(id_map_dir) if id_map_dir else None

    if id_map_dir is not None and module1_obj_id is None and auto_resolve:
        module1_obj_id = auto_resolve_module1_id(scope, gaussians, pipe_config, id_map_dir)
    if id_map_dir is not None and module1_obj_id is None:
        logger.warning("Could not resolve Module1 id — falling back to ObjectGS mask only.")

    views = []
    for ci in scope.visible_cam_indices:
        try:
            v = extract_frame(scope, gaussians, pipe_config, ci,
                              Path(images_dir), id_map_dir, module1_obj_id,
                              output_dir / "extracted", output_dir / "masks",
                              tau_alpha, min_pixels)
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

# ── Frame scoring ──────────────────────────────────────────────────────────────

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


def _sharpness(rgb, mask):
    if mask.sum() < 32:
        return 0.0
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    vals = cv2.Laplacian(gray, cv2.CV_32F)[mask]
    return float(np.var(vals)) if vals.size > 0 else 0.0


def _exposure(rgb, mask):
    if mask.sum() < 32:
        return {"mean_luma": 0.0, "frac_saturated": 1.0, "frac_dark": 1.0}
    pix = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)[mask] / 255.0
    return {
        "mean_luma": float(np.mean(pix)),
        "frac_saturated": float(np.mean(pix > 0.98)),
        "frac_dark": float(np.mean(pix < 0.02)),
    }


def _exposure_score(em):
    luma = max(0.0, 1.0 - 2.0 * abs(em["mean_luma"] - 0.5))
    return float(luma * (1.0 - min(1.0, em["frac_saturated"] * 5.0))
                     * (1.0 - min(1.0, em["frac_dark"] * 5.0)))


def _occl_score(n_hybrid, n_objgs):
    return float(min(1.0, n_hybrid / max(n_objgs, 1))) if n_objgs > 0 else 0.0


def _rank_normalize(values):
    n = len(values)
    if n == 0:
        return values
    if n == 1:
        return np.array([1.0])
    order = np.argsort(values)
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.linspace(0.0, 1.0, n)
    return ranks


def score_frames(extraction_index_path, weights=None, top_k=5):
    weights = weights or WEIGHTS
    with open(extraction_index_path) as f:
        frames = json.load(f)["frames"]
    if not frames:
        return {"weights": weights, "frames": [], "ranking": [], "top_k": []}

    scored = []
    for fr in frames:
        bgr = cv2.imread(fr["image_path"], cv2.IMREAD_COLOR)
        mask_u8 = cv2.imread(fr["out_mask_path"], cv2.IMREAD_GRAYSCALE)
        if bgr is None or mask_u8 is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mask = mask_u8 > 127
        az = float(fr.get("azimuth_deg", float("nan")))
        if math.isfinite(az):
            az = ((az + 180.0) % 360.0) - 180.0
        scored.append({
            "cam_index": fr["cam_index"],
            "image_name": fr["image_name"],
            "out_rgba_path": fr["out_rgba_path"],
            "azimuth_deg": az,
            "fg_fraction": float(fr["fg_fraction"]),
            "n_pixels_hybrid": int(fr["n_pixels_hybrid"]),
            "n_pixels_objgs": int(fr["n_pixels_objgs"]),
            "_rgb": rgb,
            "_mask": mask,
            "_sharp_raw": _sharpness(rgb, mask),
            "_exposure": _exposure(rgb, mask),
        })

    sharp_norm = _rank_normalize(np.array([s["_sharp_raw"] for s in scored]))

    results = []
    for s, sh in zip(scored, sharp_norm):
        comp = {
            "front":  _front_score(s["azimuth_deg"], 0.0),
            "cover":  _cover_score(s["fg_fraction"]),
            "sharp":  float(sh),
            "expose": _exposure_score(s["_exposure"]),
            "occl":   _occl_score(s["n_pixels_hybrid"], s["n_pixels_objgs"]),
        }
        results.append({
            "cam_index": s["cam_index"],
            "image_name": s["image_name"],
            "out_rgba_path": s["out_rgba_path"],
            "azimuth_deg": s["azimuth_deg"],
            "fg_fraction": s["fg_fraction"],
            "n_pixels_hybrid": s["n_pixels_hybrid"],
            "n_pixels_objgs": s["n_pixels_objgs"],
            "components": comp,
            "score": sum(weights[k] * comp[k] for k in weights),
        })

    results.sort(key=lambda x: -x["score"])
    for i, r in enumerate(results):
        r["rank"] = i + 1

    return {
        "weights": weights,
        "n_frames": len(results),
        "ranking": results,
        "top_k": results[:top_k],
    }


def run_scoring(extraction_index_path, output_dir, top_k=5):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = score_frames(extraction_index_path, top_k=top_k)
    with open(output_dir / "scores.json", "w") as f:
        json.dump(result, f, indent=2)
    if result["top_k"]:
        best = result["top_k"][0]
        logger.info("Best frame: cam=%d az=%.1f score=%.3f comps=%s",
                    best["cam_index"], best["azimuth_deg"], best["score"],
                    {k: round(v, 2) for k, v in best["components"].items()})
    return result
