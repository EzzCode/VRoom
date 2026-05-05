"""Visual debug for the Phase 6 object-isolation training pipeline.

Outputs under <output_root>/obj_<id>/debug/:
    alignment_audit_strip.png    before / after / ref / overlap for every hallucinated view
    supervision_contact_sheet.png  real + retained supervision images
    training_loss.png            training loss curve
    compare_sheet.png            object/full-scene before-after comparison grids
    summary.json
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

_VROOM_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_VROOM_ROOT) not in sys.path:
    sys.path.insert(0, str(_VROOM_ROOT))

from object_isolation.run_phases import run as _run_pipeline

logger = logging.getLogger(__name__)


def _putlbl(img, text, org, fg=(255, 255, 255), bg=(0, 0, 0), scale=0.5, thick=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, bg, thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, fg, thick, cv2.LINE_AA)


def _read_bgr(path_value) -> np.ndarray | None:
    if not path_value:
        return None
    path = Path(path_value)
    if not path.is_absolute():
        path = _VROOM_ROOT / path
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def _mask_from_image(path_value, size_wh: tuple[int, int] | None = None) -> np.ndarray | None:
    img = _read_bgr(path_value)
    if img is None:
        return None
    if size_wh is not None and img.shape[1::-1] != size_wh:
        img = cv2.resize(img, size_wh, interpolation=cv2.INTER_AREA)
    return img.mean(axis=2) < 250


def _resize_tile(img: np.ndarray | None, tile: int) -> np.ndarray:
    if img is None:
        return np.full((tile, tile, 3), 230, dtype=np.uint8)
    return cv2.resize(img, (tile, tile), interpolation=cv2.INTER_AREA)


def _overlay_masks(mask: np.ndarray | None, ref_mask: np.ndarray | None, tile: int) -> np.ndarray:
    out = np.full((tile, tile, 3), 245, dtype=np.uint8)
    if mask is None or ref_mask is None:
        return out
    if ref_mask.shape != mask.shape:
        ref_mask = cv2.resize(ref_mask.astype(np.uint8), (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    mask = cv2.resize(mask.astype(np.uint8), (tile, tile), interpolation=cv2.INTER_NEAREST) > 0
    ref_mask = cv2.resize(ref_mask.astype(np.uint8), (tile, tile), interpolation=cv2.INTER_NEAREST) > 0
    out[mask & ~ref_mask] = (60, 60, 220)
    out[ref_mask & ~mask] = (60, 200, 60)
    out[mask & ref_mask] = (60, 220, 220)
    return out


def make_alignment_audit_strip(audit_path: Path, out_path: Path, tile: int = 180) -> Path | None:
    if not audit_path.exists():
        return None
    with open(audit_path, "r", encoding="utf-8") as f:
        audit = json.load(f)
    frames = audit.get("frames", [])
    if not frames:
        return None

    # 4 tiles per row: original | aligned | ref | overlap
    row_h = tile + 42
    info_w = 340
    n_tiles = 4
    canvas = np.full((40 + row_h * len(frames), n_tiles * tile + info_w, 3), 245, dtype=np.uint8)
    _putlbl(canvas, "Phase 6 alignment audit", (10, 26), fg=(20, 20, 20), bg=(255, 255, 255), scale=0.58)
    _putlbl(canvas, "before | after | ref | overlap   red=halluc-only  green=ref-only  yellow=both",
            (260, 26), fg=(70, 70, 70), bg=(255, 255, 255), scale=0.40)

    for idx, fr in enumerate(frames):
        y0 = 40 + idx * row_h

        original_path = fr.get("original_image_path") or fr.get("image_path")
        aligned_path  = fr.get("image_path")
        ref_path      = fr.get("reference_path")

        original_img = _read_bgr(original_path)
        aligned_img  = _read_bgr(aligned_path)
        ref_img      = _read_bgr(ref_path)

        transform = str(fr.get("alignment_transform", "identity"))
        accepted  = bool(fr.get("accepted"))

        # column 0: original (before alignment)
        canvas[y0:y0 + tile, 0:tile] = _resize_tile(original_img, tile)
        _putlbl(canvas, "original", (4, y0 + tile + 14),
                fg=(80, 80, 80), bg=(255, 255, 255), scale=0.35)

        # column 1: after alignment (may be same as original if identity)
        is_identity = (transform == "identity") or (original_path == aligned_path)
        col1 = _resize_tile(aligned_img, tile)
        if not is_identity:
            # green border around the aligned tile to signal a transform was applied
            cv2.rectangle(col1, (0, 0), (tile - 1, tile - 1), (60, 180, 60), 3)
        canvas[y0:y0 + tile, tile:2 * tile] = col1
        label1 = transform[:18] if not is_identity else "identity"
        _putlbl(canvas, label1, (tile + 4, y0 + tile + 14),
                fg=(40, 130, 40) if not is_identity else (80, 80, 80),
                bg=(255, 255, 255), scale=0.35)

        # column 2: ObjectGS reference render
        canvas[y0:y0 + tile, 2 * tile:3 * tile] = _resize_tile(ref_img, tile)
        _putlbl(canvas, "ref", (2 * tile + 4, y0 + tile + 14),
                fg=(80, 80, 80), bg=(255, 255, 255), scale=0.35)

        # column 3: overlap mask
        mask     = _mask_from_image(aligned_path)
        ref_mask = _mask_from_image(ref_path, size_wh=(mask.shape[1], mask.shape[0]) if mask is not None else None)
        canvas[y0:y0 + tile, 3 * tile:4 * tile] = _overlay_masks(mask, ref_mask, tile)
        _putlbl(canvas, "overlap", (3 * tile + 4, y0 + tile + 14),
                fg=(80, 80, 80), bg=(255, 255, 255), scale=0.35)

        # info column
        keep  = accepted
        color = (40, 150, 40) if keep else (40, 40, 190)
        status = "KEEP" if keep else "DROP"
        info_x = n_tiles * tile + 12
        _putlbl(canvas, f"#{fr.get('frame_index')} {status}", (info_x, y0 + 24), fg=color, bg=(255, 255, 255), scale=0.55)
        _putlbl(canvas, f"az={float(fr.get('azimuth_V_deg', 0.0)):+.1f}", (info_x, y0 + 48), fg=(45, 45, 45), bg=(255, 255, 255), scale=0.43)
        _putlbl(canvas, f"mask IoU={float(fr.get('mask_iou', 0.0)):.3f}  bbox={float(fr.get('bbox_iou', 0.0)):.3f}", (info_x, y0 + 72), fg=(45, 45, 45), bg=(255, 255, 255), scale=0.43)
        _putlbl(canvas, f"centroid={float(fr.get('centroid_distance_norm', 0.0)):.3f}  area={float(fr.get('area_ratio', 0.0)):.3f}", (info_x, y0 + 96), fg=(45, 45, 45), bg=(255, 255, 255), scale=0.43)
        reasons = ",".join(fr.get("reject_reasons", []))[:52]
        _putlbl(canvas, reasons if reasons else "masks overlap cleanly", (info_x, y0 + 122), fg=color, bg=(255, 255, 255), scale=0.38)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)
    return out_path


def make_supervision_contact_sheet(manifest_path: Path, out_path: Path, tile: int = 160, cols: int = 8) -> Path | None:
    if not manifest_path.exists():
        return None
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    views = manifest.get("views", [])
    if not views:
        return None
    rows = int(np.ceil(len(views) / cols))
    canvas = np.full((40 + rows * (tile + 28), cols * tile, 3), 240, dtype=np.uint8)
    _putlbl(canvas, "Phase 6 retained supervision views", (10, 26), fg=(20, 20, 20), bg=(255, 255, 255), scale=0.58)
    for idx, view in enumerate(views):
        row, col = divmod(idx, cols)
        x0 = col * tile
        y0 = 40 + row * (tile + 28)
        img = _resize_tile(_read_bgr(view.get("image_path")), tile)
        canvas[y0:y0 + tile, x0:x0 + tile] = img
        source = str(view.get("source", "?"))
        color = (80, 80, 80) if source == "real" else (40, 140, 40)
        _putlbl(canvas, f"{source[:4]} #{view.get('frame_index')}", (x0 + 5, y0 + tile + 18), fg=color, bg=(255, 255, 255), scale=0.42)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)
    return out_path


def make_loss_plot(summary_path: Path, out_path: Path, width: int = 900, height: int = 360) -> Path | None:
    if not summary_path.exists():
        return None
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    losses = [float(x) for x in summary.get("loss_history", [])]
    if not losses:
        return None
    canvas = np.full((height, width, 3), 250, dtype=np.uint8)
    _putlbl(canvas, "Phase 7 training loss", (14, 28), fg=(20, 20, 20), bg=(255, 255, 255), scale=0.58)
    plot = canvas[54:height - 34, 60:width - 24]
    plot[:] = 255
    lo, hi = min(losses), max(losses)
    if abs(hi - lo) < 1e-8:
        hi = lo + 1.0
    pts = []
    for idx, val in enumerate(losses):
        x = int(round(idx * (plot.shape[1] - 1) / max(len(losses) - 1, 1)))
        y = int(round((hi - val) * (plot.shape[0] - 1) / (hi - lo)))
        pts.append((x + 60, y + 54))
    if len(pts) > 1:
        cv2.polylines(canvas, [np.asarray(pts, dtype=np.int32)], False, (40, 120, 210), 2, cv2.LINE_AA)
    for text, y in [(f"max {hi:.4f}", 64), (f"min {lo:.4f}", height - 42), (f"final {losses[-1]:.4f}", 45)]:
        _putlbl(canvas, text, (8, y), fg=(65, 65, 65), bg=(255, 255, 255), scale=0.42)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)
    return out_path


def make_compare_sheet(renders_dir: Path, out_path: Path, tile_w: int = 360) -> Path | None:
    images = []
    for sub in ("object_only", "full_scene"):
        for path in sorted((renders_dir / sub).glob("compare_*.png")):
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is not None:
                ratio = tile_w / max(img.shape[1], 1)
                tile_h = max(1, int(round(img.shape[0] * ratio)))
                img = cv2.resize(img, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
                _putlbl(img, sub, (8, 20), fg=(255, 255, 255), bg=(0, 0, 0), scale=0.45)
                images.append(img)
    if not images:
        return None
    max_h = max(img.shape[0] for img in images)
    padded = []
    for img in images:
        if img.shape[0] < max_h:
            pad = np.full((max_h - img.shape[0], img.shape[1], 3), 245, dtype=np.uint8)
            img = np.concatenate([img, pad], axis=0)
        padded.append(img)
    sheet = np.concatenate(padded, axis=0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet)
    return out_path


def run_debug(
    model_path: str,
    object_id: int,
    output_root: str = "object_isolation/outputs",
    iterations: int = 1200,
    hallucination_weight: float = 1.0,
    real_weight: float = 1.0,
    novel_rgb_weight: float = 1.0,
    hallucination_rgb_scale: float = 1.0,
    depth_weight: float = 0.1,
    depth_start_iter: int = 100,
    depth_front_weight: float = 1.0,
    depth_back_weight: float = 0.15,
    fov_y_deg: float = 50.0,
    colmap_init_target_points: int = 8000,
    enable_densification: bool = False,
    max_anchor_count: int = 20000,
    densify_grad_threshold: float = 0.00005,
    densify_extra_ratio: float = 0.08,
    n_compare_views: int = 8,
    no_run: bool = False,
) -> dict:
    output_root_p = Path(output_root)
    obj_dir = output_root_p / f"obj_{int(object_id)}"
    debug_dir = obj_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    if not no_run:
        _run_pipeline(
            model_path=model_path,
            output_root=output_root_p,
            object_ids=[int(object_id)],
            iterations=int(iterations),
            hallucination_weight=float(hallucination_weight),
            real_weight=float(real_weight),
            novel_rgb_weight=float(novel_rgb_weight),
            hallucination_rgb_scale=float(hallucination_rgb_scale),
            depth_weight=float(depth_weight),
            depth_start_iter=int(depth_start_iter),
            depth_front_weight=float(depth_front_weight),
            depth_back_weight=float(depth_back_weight),
            fov_y_deg=float(fov_y_deg),
            colmap_init_target_points=int(colmap_init_target_points),
            enable_densification=bool(enable_densification),
            max_anchor_count=int(max_anchor_count),
            densify_grad_threshold=float(densify_grad_threshold),
            densify_extra_ratio=float(densify_extra_ratio),
            n_compare_views=int(n_compare_views),
            skip_compare=False,
        )

    outputs = {
        "alignment_audit_strip": make_alignment_audit_strip(
            obj_dir / "alignment_audit.json", debug_dir / "alignment_audit_strip.png"),
        "supervision_contact_sheet": make_supervision_contact_sheet(
            obj_dir / "supervision_manifest.json", debug_dir / "supervision_contact_sheet.png"),
        "training_loss": make_loss_plot(
            obj_dir / "training_summary.json", debug_dir / "training_loss.png"),
        "compare_sheet": make_compare_sheet(
            obj_dir / "renders", debug_dir / "compare_sheet.png"),
    }
    summary = {
        "object_id": int(object_id),
        "model_path": str(model_path),
        "output_root": str(output_root_p),
        "debug_dir": str(debug_dir),
        "outputs": {key: str(value) if value is not None else None for key, value in outputs.items()},
    }
    with open(debug_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Phase 6 debug visuals saved to: %s", debug_dir)
    return summary


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Phase 6 training pipeline debug visualizer.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--object_id", required=True, type=int)
    parser.add_argument("--output_root", default="object_isolation/outputs")
    parser.add_argument("--iterations", type=int, default=1200)
    parser.add_argument("--hallucination_weight", type=float, default=1.0)
    parser.add_argument("--real_weight", type=float, default=1.0)
    parser.add_argument("--novel_rgb_weight", type=float, default=1.0)
    parser.add_argument("--hallucination_rgb_scale", type=float, default=1.0)
    parser.add_argument("--depth_weight", type=float, default=0.1)
    parser.add_argument("--depth_start_iter", type=int, default=100)
    parser.add_argument("--depth_front_weight", type=float, default=1.0)
    parser.add_argument("--depth_back_weight", type=float, default=0.15)
    parser.add_argument("--fov_y_deg", type=float, default=50.0)
    parser.add_argument("--colmap_init_target_points", type=int, default=8000)
    parser.add_argument("--enable_densification", action="store_true")
    parser.add_argument("--max_anchor_count", type=int, default=20000)
    parser.add_argument("--densify_grad_threshold", type=float, default=0.00005)
    parser.add_argument("--densify_extra_ratio", type=float, default=0.08)
    parser.add_argument("--n_compare_views", type=int, default=8)
    parser.add_argument("--no_run", action="store_true",
                        help="Only regenerate debug panels from existing outputs (skip pipeline).")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
    run_debug(
        args.model_path,
        args.object_id,
        args.output_root,
        iterations=args.iterations,
        hallucination_weight=args.hallucination_weight,
        real_weight=args.real_weight,
        novel_rgb_weight=args.novel_rgb_weight,
        hallucination_rgb_scale=args.hallucination_rgb_scale,
        depth_weight=args.depth_weight,
        depth_start_iter=args.depth_start_iter,
        depth_front_weight=args.depth_front_weight,
        depth_back_weight=args.depth_back_weight,
        fov_y_deg=args.fov_y_deg,
        colmap_init_target_points=args.colmap_init_target_points,
        enable_densification=args.enable_densification,
        max_anchor_count=args.max_anchor_count,
        densify_grad_threshold=args.densify_grad_threshold,
        densify_extra_ratio=args.densify_extra_ratio,
        n_compare_views=args.n_compare_views,
        no_run=args.no_run,
    )


if __name__ == "__main__":
    main()