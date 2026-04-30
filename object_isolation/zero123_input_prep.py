"""Phase 2.2 — Zero123++ input preparation.

Take the picked reference RGBA tile (already cropped, square, letterboxed
by Phase 1) and produce the exact 320x320 (or 512x512) RGB-on-white image
that ``sudo-ai/zero123plus`` expects:

    * Largest connected component on the alpha mask (drop floaters).
    * Re-tighten the bbox after the floater removal.
    * Center the object on a square white canvas with a configurable margin.
    * Composite onto pure white (the Zero123++ training distribution).

The result is saved to ``<obj_dir>/zero123_input.png`` and the canvas
parameters are recorded so downstream phases can map between Zero123++
canvas pixels and real-world rays.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CanvasInfo:
    """Recording the (pre)processing applied to the reference tile."""

    canvas_size: int            # output canvas side in px
    object_bbox_in_canvas: list  # [x0, y0, x1, y1]
    margin_frac: float
    largest_cc_pixel_count: int
    background: str = "white"


def _largest_connected_component(alpha: np.ndarray, alpha_thr: float) -> np.ndarray:
    """Return a uint8 mask of only the largest connected component (8-conn)."""
    bin_mask = (alpha > alpha_thr).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
    if n <= 1:
        return bin_mask
    # stats[0] is background; pick the largest non-background CC by area
    areas = stats[1:, cv2.CC_STAT_AREA]
    best = 1 + int(np.argmax(areas))
    return (labels == best).astype(np.uint8)


def _bbox_of_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def prepare_zero123_input(
    obj_dir: str,
    canvas_size: int = 320,
    margin_frac: float = 0.10,
    alpha_thr: float = 0.05,
) -> dict:
    """Read ``<obj_dir>/reference.png``, produce ``<obj_dir>/zero123_input.png``,
    and persist the canvas info into ``<obj_dir>/zero123_input.json``.

    ``canvas_size``: 320 matches the v1.2 model default; 512 also works.
    ``margin_frac``: fraction of canvas left as white margin around the
    object's tightest bbox. The Objaverse training data has ~10-15% margin.
    """
    obj_dir_p = Path(obj_dir)
    ref_path = obj_dir_p / "reference.png"
    if not ref_path.exists():
        raise FileNotFoundError(f"reference.png missing at {ref_path}; run --phase reference first.")

    img = cv2.imread(str(ref_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"Failed to read {ref_path}")
    if img.shape[2] != 4:
        raise ValueError(f"reference.png must be RGBA, got shape {img.shape}")

    bgr = img[..., :3]
    alpha = img[..., 3].astype(np.float32) / 255.0

    # 1. Largest CC -> drops floater fragments / mis-rendered satellites
    cc = _largest_connected_component(alpha, alpha_thr=alpha_thr)
    if cc.sum() < 16:
        raise RuntimeError(
            f"Reference image has only {int(cc.sum())} foreground pixels after "
            "largest-component cleanup; pick a different reference."
        )

    # 2. Tight bbox of the cleaned alpha
    x0, y0, x1, y1 = _bbox_of_mask(cc)
    bw = x1 - x0
    bh = y1 - y0
    side = max(bw, bh)

    # 3. Resize the object so it occupies (1 - 2*margin_frac) of the canvas
    target_obj_side = int(round((1.0 - 2.0 * margin_frac) * canvas_size))
    scale = target_obj_side / float(side)
    new_w = max(1, int(round(bw * scale)))
    new_h = max(1, int(round(bh * scale)))

    obj_bgr = bgr[y0:y1, x0:x1]
    obj_alpha = (cc[y0:y1, x0:x1].astype(np.float32))  # already 0/1 mask
    obj_alpha = obj_alpha * alpha[y0:y1, x0:x1]        # preserve soft edges

    obj_bgr_r = cv2.resize(obj_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    obj_alpha_r = cv2.resize(obj_alpha, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 4. Composite onto a white canvas, centered
    canvas = np.full((canvas_size, canvas_size, 3), 255, dtype=np.uint8)
    cx0 = (canvas_size - new_w) // 2
    cy0 = (canvas_size - new_h) // 2
    cx1 = cx0 + new_w
    cy1 = cy0 + new_h
    a = obj_alpha_r[..., None]
    canvas[cy0:cy1, cx0:cx1] = (
        obj_bgr_r.astype(np.float32) * a + canvas[cy0:cy1, cx0:cx1].astype(np.float32) * (1.0 - a)
    ).astype(np.uint8)

    out_img_path = obj_dir_p / "zero123_input.png"
    cv2.imwrite(str(out_img_path), canvas)

    info = CanvasInfo(
        canvas_size=int(canvas_size),
        object_bbox_in_canvas=[int(cx0), int(cy0), int(cx1), int(cy1)],
        margin_frac=float(margin_frac),
        largest_cc_pixel_count=int(cc.sum()),
    )
    with open(obj_dir_p / "zero123_input.json", "w", encoding="utf-8") as f:
        json.dump(asdict(info), f, indent=2)
    logger.info(
        "Zero123++ input ready: %s (canvas=%d, obj_bbox=%s)",
        out_img_path, canvas_size, info.object_bbox_in_canvas,
    )
    return asdict(info)
