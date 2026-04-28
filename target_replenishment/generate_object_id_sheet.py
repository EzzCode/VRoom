import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch

from target_replenishment.core.objectgs_bridge import (
    create_virtual_camera,
    load_gaussians,
    render_view,
)


def _sample_camera_indices(n_cameras: int, max_scan: int) -> list:
    if n_cameras <= 0:
        return []
    if max_scan <= 0 or n_cameras <= max_scan:
        return list(range(n_cameras))
    return np.linspace(0, n_cameras - 1, num=max_scan, dtype=int).tolist()


def _build_scaled_camera(cam_meta: dict, max_side: int):
    w0 = int(cam_meta["width"])
    h0 = int(cam_meta["height"])
    fx0 = float(cam_meta["fx"])
    fy0 = float(cam_meta["fy"])

    scale = min(1.0, float(max_side) / float(max(w0, h0)))
    w = max(64, int(round(w0 * scale)))
    h = max(64, int(round(h0 * scale)))

    k = np.array(
        [
            [fx0 * scale, 0.0, w / 2.0],
            [0.0, fy0 * scale, h / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    r = np.asarray(cam_meta["rotation"], dtype=np.float32).reshape(3, 3)
    t = np.asarray(cam_meta["position"], dtype=np.float32).reshape(3)
    return create_virtual_camera(r, t, k, w, h)


def _render_object_rgb_alpha(gaussians, pipe_config, camera, object_id: int):
    bg = torch.ones(3, dtype=torch.float32, device="cuda")
    out = render_view(
        gaussians,
        camera,
        pipe_config,
        bg_color=bg,
        object_label_id=int(object_id),
    )
    rgb = (out["rgb"].permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    alpha = out["alpha"].squeeze(0).cpu().numpy()
    return rgb, alpha


def _fit_tile(img_rgb: np.ndarray, tile_size: int) -> np.ndarray:
    h, w = img_rgb.shape[:2]
    if h <= 0 or w <= 0:
        return np.full((tile_size, tile_size, 3), 255, dtype=np.uint8)

    scale = min(tile_size / float(w), tile_size / float(h))
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_AREA)

    tile = np.full((tile_size, tile_size, 3), 245, dtype=np.uint8)
    x0 = (tile_size - nw) // 2
    y0 = (tile_size - nh) // 2
    tile[y0:y0 + nh, x0:x0 + nw] = resized
    return tile


def _draw_label(tile: np.ndarray, text: str):
    cv2.rectangle(tile, (8, 8), (tile.shape[1] - 8, 56), (0, 0, 0), thickness=-1)
    cv2.putText(
        tile,
        text,
        (16, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def generate_object_id_sheet(
    model_path: str,
    output_path: str,
    iteration: int = -1,
    tile_size: int = 320,
    max_scan_cameras: int = 24,
    include_zero: bool = False,
):
    model_dir = Path(model_path)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    gaussians, pipe_config = load_gaussians(str(model_dir), iteration)

    labels = gaussians.label_ids.squeeze(-1).detach().cpu().numpy().astype(np.int32)
    obj_ids = np.unique(labels).tolist()
    if not include_zero:
        obj_ids = [x for x in obj_ids if x != 0]

    if not obj_ids:
        raise RuntimeError("No object IDs found in model labels.")

    cameras_path = model_dir / "cameras.json"
    if not cameras_path.exists():
        raise FileNotFoundError(f"Missing cameras file: {cameras_path}")
    cameras = json.loads(cameras_path.read_text(encoding="utf-8"))
    if not cameras:
        raise RuntimeError("cameras.json is empty.")

    sampled = _sample_camera_indices(len(cameras), max_scan_cameras)

    tiles = []
    stats = []

    for object_id in obj_ids:
        best_score = -1.0
        best_rgb = None

        for cam_idx in sampled:
            cam = _build_scaled_camera(cameras[cam_idx], max_side=640)
            rgb, alpha = _render_object_rgb_alpha(gaussians, pipe_config, cam, object_id)
            score = float(alpha.sum())
            if score > best_score:
                best_score = score
                best_rgb = rgb

        if best_rgb is None:
            best_rgb = np.full((tile_size, tile_size, 3), 255, dtype=np.uint8)

        tile = _fit_tile(best_rgb, tile_size)
        _draw_label(tile, f"ID {int(object_id)}")
        cv2.rectangle(tile, (0, 0), (tile_size - 1, tile_size - 1), (180, 180, 180), 1)
        tiles.append(tile)
        stats.append((int(object_id), best_score))

    n = len(tiles)
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))

    sheet = np.full((rows * tile_size, cols * tile_size, 3), 235, dtype=np.uint8)
    for idx, tile in enumerate(tiles):
        r = idx // cols
        c = idx % cols
        y0 = r * tile_size
        x0 = c * tile_size
        sheet[y0:y0 + tile_size, x0:x0 + tile_size] = tile

    cv2.imwrite(str(out_path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))

    legend_path = out_path.with_suffix(".txt")
    with legend_path.open("w", encoding="utf-8") as f:
        f.write("Object ID sheet legend\n")
        f.write(f"Model: {model_dir}\n")
        f.write(f"Output: {out_path}\n")
        f.write(f"Objects: {len(obj_ids)}\n")
        f.write("\n")
        for object_id, score in stats:
            f.write(f"ID {object_id}: best_alpha_sum={score:.2f}\n")

    print(f"Saved object ID sheet: {out_path}")
    print(f"Saved legend: {legend_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate one image sheet showing all object IDs.")
    parser.add_argument("--model_path", required=True, help="Path to ObjectGS model output folder")
    parser.add_argument(
        "--output",
        default=None,
        help="Output PNG path. Default: <model_path>/object_id_sheet.png",
    )
    parser.add_argument("--iteration", type=int, default=-1, help="Iteration to load")
    parser.add_argument("--tile_size", type=int, default=320, help="Square tile size in pixels")
    parser.add_argument(
        "--max_scan_cameras",
        type=int,
        default=24,
        help="Number of sampled cameras per object while searching for best visibility",
    )
    parser.add_argument(
        "--include_zero",
        action="store_true",
        help="Include label 0 in the sheet",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_path)
    out_path = Path(args.output) if args.output else (model_dir / "object_id_sheet.png")

    generate_object_id_sheet(
        model_path=str(model_dir),
        output_path=str(out_path),
        iteration=args.iteration,
        tile_size=args.tile_size,
        max_scan_cameras=args.max_scan_cameras,
        include_zero=args.include_zero,
    )


if __name__ == "__main__":
    main()
