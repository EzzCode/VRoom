"""
Visual debug for Phase 3 (extraction).

Outputs (under <output_root>/<scene>/<obj>/debug_phase03/):
    extraction_index.json (copied)
    triptych/<cam_id>__<img_name>.png   real | M_real | M_objgs | M_hybrid | composite
    contact_sheet.png                    grid of all extracted RGBAs on white
"""
from __future__ import annotations

from pathlib import Path
import json
import logging
import math
import sys

import cv2
import numpy as np
import torch

_VROOM_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_VROOM_ROOT) not in sys.path:
    sys.path.insert(0, str(_VROOM_ROOT))

from object_isolation.core.gs_renderer import (
    create_camera, render_rgba,
)
from object_isolation.core.object_scope import discover_object_scope
from object_isolation.core.extraction import (
    run_extraction, _resolve_id_map_path, _find_image_file,
)

logger = logging.getLogger(__name__)


def _ensure3(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    return img


def _resize_to(img: np.ndarray, size_wh: tuple) -> np.ndarray:
    return cv2.resize(img, size_wh, interpolation=cv2.INTER_NEAREST)


def make_triptychs(scope, gaussians, pipe_config,
                   images_dir: Path, id_map_dir, module1_obj_id,
                   manifest: dict, out_dir: Path,
                   max_panels: int = 8, panel_w: int = 360,
                   tau_alpha: float = 0.4):
    """For up to max_panels frames, build a 5-panel strip:
        real RGB | M_real | M_objgs | M_hybrid | RGBA composite
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = manifest['frames']
    if len(frames) > max_panels:
        step = max(1, len(frames) // max_panels)
        frames = frames[::step][:max_panels]

    saved: list[Path] = []

    for fr in frames:
        ci = fr['cam_index']
        img_name = fr['img_name']
        cam_p = scope.cameras[ci]

        bgr = cv2.imread(fr['image_path'], cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        rgb_real = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        H_img, W_img = rgb_real.shape[:2]

        # Re-render objgs alpha (cheap) for visualization.
        cam = create_camera(cam_p['R'], cam_p['T'], cam_p['K'],
                           cam_p['width'], cam_p['height'])
        res = render_rgba(gaussians, cam, pipe_config, bg_white=False,
                         object_label_id=scope.object_label_id)
        alpha = res['alpha'].detach().cpu().numpy()
        if alpha.ndim == 3:
            alpha = alpha[0]
        m_objgs = alpha > tau_alpha
        if m_objgs.shape != (H_img, W_img):
            m_objgs_rs = cv2.resize(m_objgs.astype(np.uint8), (W_img, H_img),
                                    interpolation=cv2.INTER_NEAREST).astype(bool)
        else:
            m_objgs_rs = m_objgs

        # M_real
        m_real_vis = np.zeros((H_img, W_img), dtype=bool)
        if id_map_dir is not None and module1_obj_id is not None:
            id_map_path = _resolve_id_map_path(Path(id_map_dir), img_name)
            if id_map_path is not None:
                id_map = cv2.imread(str(id_map_path), cv2.IMREAD_UNCHANGED)
                if id_map is not None:
                    if id_map.shape[:2] != (H_img, W_img):
                        id_map = cv2.resize(id_map, (W_img, H_img),
                                            interpolation=cv2.INTER_NEAREST)
                    m_real_vis = (id_map == module1_obj_id)

        # Hybrid mask from saved file.
        m_hybrid_u8 = cv2.imread(fr['out_mask_path'], cv2.IMREAD_GRAYSCALE)
        if m_hybrid_u8 is None:
            continue
        m_hybrid = m_hybrid_u8 > 127

        # Composite (real RGB on white).
        composite = np.where(m_hybrid[..., None], rgb_real, np.full_like(rgb_real, 255))

        # Build panel strip.
        panel_h = int(round(panel_w * H_img / W_img))
        size = (panel_w, panel_h)

        def _to_panel(arr, label, color=(255, 255, 255)):
            img = cv2.resize(_ensure3(arr), size, interpolation=cv2.INTER_AREA)
            cv2.putText(img, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            return img

        panel_real = _to_panel(rgb_real, "real")
        panel_mreal = _to_panel((m_real_vis.astype(np.uint8) * 255), "M_real" + ("" if m_real_vis.any() else " (NONE)"))
        panel_mobjgs = _to_panel((m_objgs_rs.astype(np.uint8) * 255), "M_objgs")
        panel_mhybrid = _to_panel((m_hybrid.astype(np.uint8) * 255), "M_hybrid")
        panel_comp = _to_panel(composite, "composite")

        strip = np.concatenate([panel_real, panel_mreal, panel_mobjgs, panel_mhybrid, panel_comp], axis=1)

        header = np.full((28, strip.shape[1], 3), 245, dtype=np.uint8)
        cv2.putText(header,
                    f"cam={ci} | img={img_name} | hybrid_px={int(m_hybrid.sum())} | used_real={fr['used_real_mask']} | az_V={fr['azimuth_V_deg']:.1f}",
                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1, cv2.LINE_AA)
        out = np.concatenate([header, strip], axis=0)

        out_path = out_dir / f"{ci:03d}__{img_name}.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
        saved.append(out_path)

    return saved


def make_contact_sheet(manifest: dict, out_path: Path,
                       tile_w: int = 192, cols: int = 6, max_tiles: int = 60):
    """Grid of all extracted objects on a checkered background."""
    frames = manifest['frames'][:max_tiles]
    if not frames:
        return None
    tiles = []
    for fr in frames:
        rgba = cv2.imread(fr['out_rgba_path'], cv2.IMREAD_UNCHANGED)
        if rgba is None:
            continue
        # Crop to bbox with 10% pad.
        x, y, w, h = fr['bbox_xywh']
        pad = int(0.1 * max(w, h))
        x0 = max(0, x - pad); y0 = max(0, y - pad)
        x1 = min(rgba.shape[1], x + w + pad); y1 = min(rgba.shape[0], y + h + pad)
        crop = rgba[y0:y1, x0:x1]
        # Composite onto checkerboard.
        cb = _checker(crop.shape[1], crop.shape[0])
        a = crop[..., 3:4].astype(np.float32) / 255.0
        rgb = cv2.cvtColor(crop[..., :3], cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        comp = a * rgb + (1 - a) * (cb.astype(np.float32) / 255.0)
        comp_u8 = (comp * 255 + 0.5).astype(np.uint8)
        # Resize.
        ratio = tile_w / max(comp_u8.shape[1], 1)
        new_h = max(1, int(round(comp_u8.shape[0] * ratio)))
        thumb = cv2.resize(comp_u8, (tile_w, new_h), interpolation=cv2.INTER_AREA)
        # Annotate.
        label = f"#{fr['cam_index']} az={fr['azimuth_V_deg']:.0f}"
        cv2.putText(thumb, label, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(thumb, label, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(thumb)

    if not tiles:
        return None
    # Pad to common height.
    max_h = max(t.shape[0] for t in tiles)
    tiles = [_pad_h(t, max_h) for t in tiles]
    rows = []
    for i in range(0, len(tiles), cols):
        row_tiles = tiles[i:i + cols]
        # Pad final row.
        while len(row_tiles) < cols:
            row_tiles.append(np.full_like(row_tiles[0], 240))
        rows.append(np.concatenate(row_tiles, axis=1))
    sheet = np.concatenate(rows, axis=0)
    cv2.imwrite(str(out_path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    return out_path


def _pad_h(img: np.ndarray, target_h: int) -> np.ndarray:
    if img.shape[0] >= target_h:
        return img
    pad = np.full((target_h - img.shape[0], img.shape[1], 3), 240, dtype=np.uint8)
    return np.concatenate([img, pad], axis=0)


def _checker(w: int, h: int, sq: int = 16) -> np.ndarray:
    out = np.full((h, w, 3), 220, dtype=np.uint8)
    for y in range(0, h, sq):
        for x in range(0, w, sq):
            if ((x // sq) + (y // sq)) % 2 == 0:
                out[y:y + sq, x:x + sq] = 240
    return out


def run_debug(model_path: str, object_id: int, scene_dir: str,
              output_root: str,
              id_map_dir: str = "auto",
              module1_obj_id=None,
              tau_alpha: float = 0.4,
              min_pixels: int = 64) -> dict:
    out_dir = Path(output_root) / f"obj_{object_id}"
    phase3_dir = out_dir / "phase3"
    debug_dir = out_dir / "debug_phase03"
    debug_dir.mkdir(parents=True, exist_ok=True)

    scope, world_local, local_sv3d, gaussians, pipe = discover_object_scope(
        model_path=model_path, object_label_id=object_id,
    )

    scene_p = Path(scene_dir)
    images_dir = scene_p / "images"

    # id_map dir auto-detection.
    resolved_id_map_dir = None
    if id_map_dir == "auto":
        candidates = [
            scene_p / "tracked" / "id_maps",
            scene_p / "semantic_instance",
            scene_p / "object_mask",
        ]
        for c in candidates:
            if c.exists() and any(c.iterdir()):
                resolved_id_map_dir = c
                break
        if resolved_id_map_dir is None:
            logger.info("No id_map dir auto-detected; falling back to ObjectGS-alpha-only.")
    elif id_map_dir.lower() in ("none", "null", ""):
        resolved_id_map_dir = None
    else:
        resolved_id_map_dir = Path(id_map_dir)

    manifest = run_extraction(scope, gaussians, pipe,
                              images_dir=images_dir,
                              id_map_dir=resolved_id_map_dir,
                              module1_obj_id=module1_obj_id,
                              output_dir=phase3_dir,
                              tau_alpha=tau_alpha,
                              min_pixels=min_pixels,
                              auto_resolve=True)

    # Visual debug.
    make_triptychs(scope, gaussians, pipe,
                   images_dir, resolved_id_map_dir, manifest.get('module1_obj_id'),
                   manifest, debug_dir / "triptych",
                   max_panels=8, tau_alpha=tau_alpha)
    make_contact_sheet(manifest, debug_dir / "contact_sheet.png")

    summary = {
        "manifest_path": str(phase3_dir / "extraction_index.json"),
        "module1_obj_id_resolved": manifest.get('module1_obj_id'),
        "id_map_dir": str(resolved_id_map_dir) if resolved_id_map_dir else None,
        "n_extracted": manifest['n_extracted'],
        "n_visible_cams": manifest['n_visible_cams'],
        "n_used_real_mask": manifest['n_used_real_mask'],
    }
    with open(debug_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Phase 3 debug saved to: %s", debug_dir)
    return summary


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Phase 3 extraction visual debug.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--object_id", required=True, type=int)
    parser.add_argument("--scene_dir", required=True,
                        help="Scene directory containing images/ and optionally tracked/id_maps/ or semantic_instance/")
    parser.add_argument("--output_root", default="object_isolation/outputs")
    parser.add_argument("--id_map_dir", default="auto",
                        help="'auto' (default) | 'none' | explicit path")
    parser.add_argument("--module1_obj_id", type=int, default=None,
                        help="Override Module-1 instance id; auto-resolved if omitted.")
    parser.add_argument("--tau_alpha", type=float, default=0.4)
    parser.add_argument("--min_pixels", type=int, default=64)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
    run_debug(args.model_path, args.object_id, args.scene_dir,
              args.output_root, id_map_dir=args.id_map_dir,
              module1_obj_id=args.module1_obj_id,
              tau_alpha=args.tau_alpha, min_pixels=args.min_pixels)


if __name__ == "__main__":
    main()
