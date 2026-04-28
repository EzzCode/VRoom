import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def _largest_component(mask: np.ndarray, min_pixels: int = 64) -> np.ndarray:
    mask_u8 = (mask.astype(np.uint8) > 0).astype(np.uint8)
    if int(mask_u8.sum()) < int(min_pixels):
        return mask_u8.astype(bool)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if n_labels <= 1:
        return mask_u8.astype(bool)
    areas = stats[1:, cv2.CC_STAT_AREA]
    if areas.size == 0:
        return mask_u8.astype(bool)
    keep = int(np.argmax(areas) + 1)
    out = labels == keep
    if int(out.sum()) < int(min_pixels):
        return mask_u8.astype(bool)
    return out


def _bbox_from_mask(mask: np.ndarray):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max() + 1)
    y0, y1 = int(ys.min()), int(ys.max() + 1)
    return x0, y0, x1, y1


def _mask_stats(mask: np.ndarray):
    h, w = mask.shape[:2]
    area = int(mask.sum())
    ratio = float(area / max(h * w, 1))
    bbox = _bbox_from_mask(mask)
    if bbox is None:
        return {
            "area_px": area,
            "area_ratio": ratio,
            "bbox": None,
            "bbox_area_ratio": 0.0,
        }
    x0, y0, x1, y1 = bbox
    bw = max(0, x1 - x0)
    bh = max(0, y1 - y0)
    bbox_ratio = float((bw * bh) / max(h * w, 1))
    return {
        "area_px": area,
        "area_ratio": ratio,
        "bbox": [int(x0), int(y0), int(x1), int(y1)],
        "bbox_area_ratio": bbox_ratio,
        "bbox_w": int(bw),
        "bbox_h": int(bh),
    }


def _overlay_mask(rgb: np.ndarray, mask: np.ndarray, color=(0, 255, 0)):
    out = rgb.copy()
    if out.dtype != np.uint8:
        out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
    alpha = 0.35
    overlay = np.zeros_like(out)
    overlay[:, :] = np.array(color, dtype=np.uint8)
    m = mask.astype(bool)
    out[m] = cv2.addWeighted(out[m], 1.0 - alpha, overlay[m], alpha, 0)
    return out


def _label(img: np.ndarray, text: str):
    out = img.copy()
    cv2.rectangle(out, (8, 8), (out.shape[1] - 8, 52), (0, 0, 0), -1)
    cv2.putText(out, text, (14, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def _fit(img: np.ndarray, size: int = 512):
    h, w = img.shape[:2]
    scale = min(size / max(w, 1), size / max(h, 1))
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size, 3), 245, dtype=np.uint8)
    x0 = (size - nw) // 2
    y0 = (size - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def _make_sheet(images, cols=3, tile=512):
    rows = int(np.ceil(len(images) / max(cols, 1)))
    sheet = np.full((rows * tile, cols * tile, 3), 230, dtype=np.uint8)
    for i, img in enumerate(images):
        r = i // cols
        c = i % cols
        y0, x0 = r * tile, c * tile
        sheet[y0:y0 + tile, x0:x0 + tile] = _fit(img, tile)
    return sheet


def main():
    parser = argparse.ArgumentParser(description="Audit object input/mask scale before replenishment")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--object_id", type=int, required=True)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--thresholds", default="0.30,0.45,0.60")
    parser.add_argument("--tight_alpha_threshold", type=float, default=0.55)
    parser.add_argument("--tight_crop_margin_frac", type=float, default=0.03)
    parser.add_argument("--tight_fill_ratio", type=float, default=0.58)
    args = parser.parse_args()

    from target_replenishment.core.objectgs_bridge import load_gaussians, get_anchor_positions
    from target_replenishment.core.perspective_graph import build_perspective_graph
    from target_replenishment.core.coverage_analyzer import analyze_coverage
    from target_replenishment.core.novel_view_generator import render_object_for_input, _prepare_input

    model_path = Path(args.model_path)
    out_dir = Path(args.output_dir) if args.output_dir else Path("scale_debug_obj9") / "input_scale_audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    gaussians, pipe_config = load_gaussians(str(model_path), args.iteration)
    anchor_xyz_global = get_anchor_positions(gaussians)
    graph = build_perspective_graph(str(model_path / "cameras.json"), anchor_xyz_global, overlap_method="visibility")

    labels = gaussians.label_ids.squeeze(-1).detach().cpu().numpy().astype(np.int32)
    obj_mask = labels == int(args.object_id)
    if not obj_mask.any():
        raise ValueError(f"Object ID {args.object_id} not found")

    object_anchors = anchor_xyz_global[obj_mask]
    coverage = analyze_coverage(object_anchors, graph.cameras, up_axis="auto")
    best_cam = coverage.best_input_cam

    input_render = render_object_for_input(
        gaussians,
        pipe_config,
        object_center=coverage.object_center,
        object_radius=coverage.object_radius,
        input_cam_position=best_cam["position"],
        up_vector=coverage.up_vector,
        object_id=int(args.object_id),
        render_size=512,
        reference_K=best_cam.get("K"),
        reference_width=best_cam.get("width"),
        reference_height=best_cam.get("height"),
    )

    rgb = input_render["rgb"]
    alpha = input_render["alpha"].astype(np.float32)
    if alpha.max() > 1.0:
        alpha = alpha / 255.0

    cv2.imwrite(str(out_dir / "input_view.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out_dir / "input_alpha.png"), np.clip(alpha * 255.0, 0, 255).astype(np.uint8))

    thresholds = [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]

    images = []
    stats = {
        "model_path": str(model_path),
        "object_id": int(args.object_id),
        "best_input_camera_id": best_cam.get("id"),
        "best_input_camera_name": best_cam.get("image_name", best_cam.get("name")),
        "threshold_masks": {},
    }

    images.append(_label(rgb, "input_view"))
    alpha_vis = cv2.cvtColor(np.clip(alpha * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    images.append(_label(alpha_vis, "input_alpha"))

    for t in thresholds:
        m = _largest_component(alpha > float(np.clip(t, 0.01, 0.99)), min_pixels=64)
        stats["threshold_masks"][f"{t:.2f}"] = _mask_stats(m)
        over = _overlay_mask(rgb, m, color=(0, 220, 0))
        images.append(_label(over, f"alpha>{t:.2f}"))

    prepared_current = _prepare_input(
        rgb,
        alpha,
        alpha_threshold=0.30,
        crop_margin_frac=0.08,
        fill_ratio=0.78,
    )
    prepared_tight = _prepare_input(
        rgb,
        alpha,
        alpha_threshold=float(args.tight_alpha_threshold),
        crop_margin_frac=float(args.tight_crop_margin_frac),
        fill_ratio=float(args.tight_fill_ratio),
    )

    prepared_current_512 = cv2.resize(prepared_current, (512, 512), interpolation=cv2.INTER_NEAREST)
    prepared_tight_512 = cv2.resize(prepared_tight, (512, 512), interpolation=cv2.INTER_NEAREST)

    m_current = _largest_component(np.mean(prepared_current_512.astype(np.float32), axis=2) < 250.0, min_pixels=64)
    m_tight = _largest_component(np.mean(prepared_tight_512.astype(np.float32), axis=2) < 250.0, min_pixels=64)

    stats["prepared_current"] = {
        "alpha_threshold": 0.30,
        "crop_margin_frac": 0.08,
        "fill_ratio": 0.78,
        **_mask_stats(m_current),
    }
    stats["prepared_tight"] = {
        "alpha_threshold": float(args.tight_alpha_threshold),
        "crop_margin_frac": float(args.tight_crop_margin_frac),
        "fill_ratio": float(args.tight_fill_ratio),
        **_mask_stats(m_tight),
    }

    images.append(_label(prepared_current_512, "prepared_current"))
    images.append(_label(prepared_tight_512, "prepared_tight"))
    images.append(_label(_overlay_mask(prepared_current_512, m_current, color=(255, 140, 0)), "current_mask"))
    images.append(_label(_overlay_mask(prepared_tight_512, m_tight, color=(255, 140, 0)), "tight_mask"))

    sheet = _make_sheet(images, cols=3, tile=512)
    cv2.imwrite(str(out_dir / "scale_audit_sheet.png"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))

    with (out_dir / "scale_audit.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"Saved: {out_dir / 'scale_audit_sheet.png'}")
    print(f"Saved: {out_dir / 'scale_audit.json'}")


if __name__ == "__main__":
    main()
