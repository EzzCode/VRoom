"""Hybrid Object Extraction.

For each visible training camera of an object, produce a clean RGBA tile of
just the object cut out of the real photograph. Two-source mask:

    M_objgs  = render(ObjectGS, object_label_id=k).alpha > tau_alpha
    M_real   = (id_map == module1_obj_id)              (Module-1 SAM-based)
    M_hybrid = largest_cc(M_objgs ∩ M_real, min_pixels=64)

Source modes (CLI):
    1) id_map_dir + module1_obj_id explicit
    2) id_map_dir, module1_obj_id auto-resolved via silhouette IoU
    3) None → degenerate fallback: M_hybrid = M_objgs (still useful)

The resulting RGBA composites the *real-photo* RGB onto a white background
where M_hybrid is true. We never use the ObjectGS rendered RGB for training
data — that's the whole point of the rebuild.

Outputs at <out_root>/obj_<id>/01_extraction/::

    extracted/<seq>__<cam_id>__<img_name>.png    RGBA, original camera resolution
    masks/<seq>__<cam_id>__<img_name>_mask.png   uint8 0/255
    extraction_index.json                        manifest

Run via pipeline orchestrator (recommended)::

    python -m object_isolation.run_pipeline \\
        --model_path temp_deps/ObjectGS/outputs/3dovs/.../2026-03-19_04-01-38 \\
        --object_id 8 \\
        --scene_dir data/3dovs/bed \\
        --output_root object_isolation/outputs
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
import logging
import re
import sys
from typing import Optional

import cv2
import numpy as np

_VROOM_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_VROOM_ROOT) not in sys.path:
    sys.path.insert(0, str(_VROOM_ROOT))

from .gs_renderer import create_camera, render_rgba
from .object_scope import ObjectScope

logger = logging.getLogger(__name__)


# ── Filename resolution ────────────────────────────────────────────────────────

def _resolve_id_map_path(id_map_dir: Path, img_name: str) -> Optional[Path]:
    """Try to find the id-map file for `img_name` under `id_map_dir`.

    Strategies (in order):
        1) <id_map_dir>/<img_name>.<png|jpg>
        2) Replica-style: replace 'rgb' → 'semantic_instance' or '_rgb_' → '_semantic_instance_'
        3) Match by trailing digits (e.g. 'train_rgb_0120' ↔ '..._0120.png')
    """
    if not id_map_dir.exists():
        return None
    for ext in (".png", ".jpg", ".jpeg"):
        cand = id_map_dir / f"{img_name}{ext}"
        if cand.exists():
            return cand
    # Replica-style rename
    renamed = img_name.replace("_rgb_", "_semantic_instance_").replace("rgb", "semantic_instance")
    if renamed != img_name:
        for ext in (".png", ".jpg", ".jpeg"):
            cand = id_map_dir / f"{renamed}{ext}"
            if cand.exists():
                return cand
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


# ── Module-1 ID auto-resolution ────────────────────────────────────────────

def auto_resolve_module1_id(scope, gaussians, pipe_config, id_map_dir: Path,
                            n_probe_cams: int = 5,
                            tau_alpha: float = 0.4) -> Optional[int]:
    """Probe a few visible cams. For each, build M_objgs and find the
    Module-1 instance id whose mask has the highest IoU with M_objgs.
    Vote across cams. Return the winning id (excluding 0=bg).
    """
    if not id_map_dir.exists():
        logger.info("id_map_dir does not exist: %s", id_map_dir)
        return None

    indices = list(scope.visible_cam_indices)
    if len(indices) > n_probe_cams:
        step = max(1, len(indices) // n_probe_cams)
        indices = indices[::step][:n_probe_cams]

    votes: dict[int, float] = {}

    for ci in indices:
        cam_p = scope.cameras[ci]
        id_map_path = _resolve_id_map_path(id_map_dir, cam_p['img_name'])
        if id_map_path is None:
            continue
        id_map = cv2.imread(str(id_map_path), cv2.IMREAD_UNCHANGED)
        if id_map is None:
            continue

        # Render ObjectGS alpha for this object only.
        cam = create_camera(
            cam_p['R'], cam_p['T'], cam_p['K'], cam_p['width'], cam_p['height'],
        )
        res = render_rgba(
            gaussians, cam, pipe_config,
            bg_white=True, object_label_id=scope.object_label_id,
        )
        alpha = res['alpha'].detach().cpu().numpy()
        if alpha.ndim == 3:
            alpha = alpha[0]
        m_objgs = alpha > tau_alpha

        # Resize id_map to render resolution if needed.
        H, W = m_objgs.shape
        if id_map.shape[:2] != (H, W):
            id_map = cv2.resize(id_map, (W, H), interpolation=cv2.INTER_NEAREST)

        unique_ids = np.unique(id_map)
        for uid in unique_ids:
            if int(uid) == 0:
                continue  # background
            m_real = (id_map == uid)
            inter = float(np.logical_and(m_real, m_objgs).sum())
            union = float(np.logical_or(m_real, m_objgs).sum())
            iou = inter / max(union, 1.0)
            votes[int(uid)] = votes.get(int(uid), 0.0) + iou

    if not votes:
        return None
    winner = max(votes.items(), key=lambda kv: kv[1])
    logger.info("Auto-resolved module1_obj_id=%d (votes: %s)",
                winner[0],
                {k: round(v, 3) for k, v in sorted(votes.items(), key=lambda kv: -kv[1])[:5]})
    return winner[0] if winner[1] > 0 else None


# ── Mask post-processing ────────────────────────────────────────────────────

def _largest_cc(mask: np.ndarray, min_pixels: int = 64) -> np.ndarray:
    """Keep the largest 4-connected component if it has ≥ min_pixels;
    otherwise return all-zeros."""
    m_u8 = mask.astype(np.uint8)
    n_labels, lbl, stats, _ = cv2.connectedComponentsWithStats(m_u8, connectivity=4)
    if n_labels <= 1:
        return np.zeros_like(m_u8, dtype=bool)
    # stats[0] = background
    areas = stats[1:, cv2.CC_STAT_AREA]
    if len(areas) == 0:
        return np.zeros_like(m_u8, dtype=bool)
    k = int(np.argmax(areas)) + 1
    if int(stats[k, cv2.CC_STAT_AREA]) < min_pixels:
        return np.zeros_like(m_u8, dtype=bool)
    return lbl == k


def _close_and_fill(mask: np.ndarray, kernel: int = 3) -> np.ndarray:
    """Tiny morphological close + flood-fill holes."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
    closed = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, k)
    # Flood-fill holes: invert, flood from corners, invert back.
    h, w = closed.shape
    flood = closed.copy()
    ff = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, ff, (0, 0), 1)
    holes = (flood == 0).astype(np.uint8)
    return ((closed | holes) > 0)


# ── Per-frame extraction ────────────────────────────────────────────────────

@dataclass
class FrameExtraction:
    cam_index: int
    img_name: str
    image_path: str
    width: int
    height: int
    n_pixels_objgs: int
    n_pixels_real: int
    n_pixels_hybrid: int
    bbox_xywh: list  # in original image pixels
    fg_fraction: float
    azimuth_V_deg: float
    out_rgba_path: str
    out_mask_path: str
    used_real_mask: bool


def _find_image_file(images_dir: Path, img_name: str) -> Optional[Path]:
    if not images_dir.exists():
        return None
    for ext in (".jpg", ".JPG", ".jpeg", ".png", ".PNG"):
        c = images_dir / f"{img_name}{ext}"
        if c.exists():
            return c
    # Fuzzy match by stem
    for f in images_dir.iterdir():
        if f.is_file() and f.stem == img_name:
            return f
    return None


def extract_frame(scope: ObjectScope, gaussians, pipe_config,
                  cam_index: int,
                  images_dir: Path,
                  id_map_dir: Optional[Path],
                  module1_obj_id: Optional[int],
                  out_rgba_dir: Path,
                  out_mask_dir: Path,
                  tau_alpha: float = 0.4,
                  min_pixels: int = 64) -> Optional[FrameExtraction]:
    cam_p = scope.cameras[cam_index]
    img_name = cam_p['img_name']
    img_path = _find_image_file(images_dir, img_name)
    if img_path is None:
        logger.warning("No real image for cam %d (img_name=%s) under %s", cam_index, img_name, images_dir)
        return None
    bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if bgr is None:
        logger.warning("Failed to read %s", img_path)
        return None
    H_img, W_img = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # ObjectGS alpha at the camera's native resolution.
    cam = create_camera(
        cam_p['R'], cam_p['T'], cam_p['K'], cam_p['width'], cam_p['height'],
    )
    res = render_rgba(
        gaussians, cam, pipe_config,
        bg_white=True, object_label_id=scope.object_label_id,
    )
    alpha = res['alpha'].detach().cpu().numpy()
    if alpha.ndim == 3:
        alpha = alpha[0]
    H_r, W_r = alpha.shape
    m_objgs = alpha > tau_alpha
    if m_objgs.shape != (H_img, W_img):
        m_objgs_rs = cv2.resize(m_objgs.astype(np.uint8), (W_img, H_img),
                                interpolation=cv2.INTER_NEAREST).astype(bool)
    else:
        m_objgs_rs = m_objgs

    # Module-1 mask if available.
    used_real = False
    m_real = None
    if id_map_dir is not None and module1_obj_id is not None:
        id_map_path = _resolve_id_map_path(id_map_dir, img_name)
        if id_map_path is not None:
            id_map = cv2.imread(str(id_map_path), cv2.IMREAD_UNCHANGED)
            if id_map is not None:
                if id_map.shape[:2] != (H_img, W_img):
                    id_map = cv2.resize(id_map, (W_img, H_img),
                                        interpolation=cv2.INTER_NEAREST)
                m_real = (id_map == module1_obj_id)
                used_real = True

    # Hybrid combine.
    if used_real:
        m_hybrid = np.logical_and(m_objgs_rs, m_real)
    else:
        m_hybrid = m_objgs_rs

    # Clean up: largest CC, close + fill.
    m_hybrid = _close_and_fill(m_hybrid, kernel=3)
    m_hybrid = _largest_cc(m_hybrid, min_pixels=min_pixels)

    n_pix_hybrid = int(m_hybrid.sum())
    if n_pix_hybrid < min_pixels:
        logger.warning("cam %d: hybrid mask too small (%d px) — skipping", cam_index, n_pix_hybrid)
        return None

    # Build RGBA: real RGB + alpha=mask*255.
    alpha_u8 = (m_hybrid.astype(np.uint8) * 255)
    rgba = np.concatenate([rgb, alpha_u8[..., None]], axis=-1)

    # bbox in pixels.
    ys, xs = np.where(m_hybrid)
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    bbox = [x0, y0, x1 - x0, y1 - y0]

    # Save.
    out_rgba_dir.mkdir(parents=True, exist_ok=True)
    out_mask_dir.mkdir(parents=True, exist_ok=True)
    out_rgba_name = f"{cam_index:03d}__{img_name}.png"
    out_mask_name = f"{cam_index:03d}__{img_name}_mask.png"
    out_rgba_path = out_rgba_dir / out_rgba_name
    out_mask_path = out_mask_dir / out_mask_name
    cv2.imwrite(str(out_rgba_path), cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
    cv2.imwrite(str(out_mask_path), alpha_u8)

    return FrameExtraction(
        cam_index=cam_index,
        img_name=img_name,
        image_path=str(img_path),
        width=W_img,
        height=H_img,
        n_pixels_objgs=int(m_objgs_rs.sum()),
        n_pixels_real=int(m_real.sum()) if used_real else -1,
        n_pixels_hybrid=n_pix_hybrid,
        bbox_xywh=bbox,
        fg_fraction=float(n_pix_hybrid) / float(H_img * W_img),
        azimuth_V_deg=float(cam_p.get('azimuth_V_deg', float('nan'))),
        out_rgba_path=str(out_rgba_path),
        out_mask_path=str(out_mask_path),
        used_real_mask=used_real,
    )


# ── Top-level ───────────────────────────────────────────────────────────────

def run_extraction(scope: ObjectScope, gaussians, pipe_config, *,
                   images_dir: Path,
                   id_map_dir: Optional[Path],
                   module1_obj_id: Optional[int],
                   output_dir: Path,
                   tau_alpha: float = 0.4,
                   min_pixels: int = 64,
                   auto_resolve: bool = True) -> dict:
    """Run extraction over all visible cams; return manifest dict."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_rgba_dir = output_dir / "extracted"
    out_mask_dir = output_dir / "masks"

    # Tag azimuths onto cam dicts so we can read them later.
    for ci, az in zip(scope.visible_cam_indices,
                      [scope.cameras[ci].get('azimuth_V_deg', None)
                       for ci in scope.visible_cam_indices]):
        if az is None and 'azimuth_V_deg' not in scope.cameras[ci]:
            # Will be filled by scope discovery already; if not, leave NaN.
            pass

    # ID resolution.
    if id_map_dir is not None and module1_obj_id is None and auto_resolve:
        module1_obj_id = auto_resolve_module1_id(scope, gaussians, pipe_config, id_map_dir)
    if id_map_dir is not None and module1_obj_id is None:
        logger.warning("Could not determine Module-1 obj id — falling back to ObjectGS-alpha-only.")

    extractions: list[FrameExtraction] = []
    for ci in scope.visible_cam_indices:
        try:
            r = extract_frame(scope, gaussians, pipe_config,
                              cam_index=ci,
                              images_dir=images_dir,
                              id_map_dir=id_map_dir,
                              module1_obj_id=module1_obj_id,
                              out_rgba_dir=out_rgba_dir,
                              out_mask_dir=out_mask_dir,
                              tau_alpha=tau_alpha,
                              min_pixels=min_pixels)
            if r is not None:
                extractions.append(r)
        except Exception as e:  # pragma: no cover
            logger.exception("cam %d failed: %s", ci, e)

    n_used_real = sum(1 for e in extractions if e.used_real_mask)
    manifest = {
        "object_id": int(scope.object_label_id),
        "module1_obj_id": int(module1_obj_id) if module1_obj_id is not None else None,
        "id_map_dir": str(id_map_dir) if id_map_dir is not None else None,
        "images_dir": str(images_dir),
        "tau_alpha": tau_alpha,
        "min_pixels": min_pixels,
        "n_visible_cams": len(scope.visible_cam_indices),
        "n_extracted": len(extractions),
        "n_used_real_mask": n_used_real,
        "frames": [e.__dict__ for e in extractions],
    }
    with open(output_dir / "extraction_index.json", "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Extraction complete: %d/%d frames (real_mask used in %d)",
                len(extractions), len(scope.visible_cam_indices), n_used_real)
    return manifest
