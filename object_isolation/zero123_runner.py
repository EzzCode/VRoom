"""Zero123++ v1.2 backend (sudo-ai/zero123plus) via HuggingFace diffusers.

The pipeline takes a square RGB conditioning image and produces a 640x960
(width x height) RGB grid: 2 columns x 3 rows of 320x320 tiles in the
Zero123++ v1.2 schedule (row-major):

    (0,0) az=30   el=20      (0,1) az=90   el=-10
    (1,0) az=150  el=20      (1,1) az=210  el=-10
    (2,0) az=270  el=20      (2,1) az=330  el=-10

We DO NOT change Zero123++'s schedule here; ``pose_alignment`` already
maps these (az, el) to world poses.

Output:
    obj_<id>/novel_views/
        tile_0.png   ...  tile_5.png    (RGBA, 320x320 — alpha by background subtract)
        tile_grid.png                   (raw 960x640 model output)
        novel_views_meta.json           (per-tile pose schedule + RGBA path)
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ── Backend-agnostic interface ──────────────────────────────────────────────


@dataclass
class Zero123Output:
    """Wraps the raw 6-tile output."""

    tile_grid_rgb: np.ndarray          # (H, W, 3) uint8, 960h x 640w for v1.2
    tile_size: int                     # 320 for v1.2
    grid_rows: int = 3
    grid_cols: int = 2


def _slice_tiles(grid: np.ndarray, tile_size: int, rows: int, cols: int) -> list[np.ndarray]:
    """Cut the model output into ``rows*cols`` row-major RGB tiles."""
    H, W = grid.shape[:2]
    if H != rows * tile_size or W != cols * tile_size:
        raise ValueError(
            f"Zero123++ output shape {grid.shape} doesn't match "
            f"({rows}x{cols} tiles of {tile_size}x{tile_size})"
        )
    tiles: list[np.ndarray] = []
    for r in range(rows):
        for c in range(cols):
            tiles.append(grid[
                r * tile_size:(r + 1) * tile_size,
                c * tile_size:(c + 1) * tile_size,
            ])
    return tiles


def _alpha_from_bg(rgb: np.ndarray, bg_thr: int = 12, soften_px: int = 1) -> np.ndarray:
    """Estimate per-tile alpha by chroma-keying against the auto-detected
    background colour.

    Zero123++ v1.2 actually outputs against a near-uniform GREY background
    (~171/171/171), not white. We therefore sample the background colour
    from the image's four corners (10x10 windows, taking the per-channel
    median of all 400 pixels) and treat any pixel within ``bg_thr`` of
    that colour as background.

    The largest non-background connected component is kept as foreground
    to drop disconnected floaters around the object.
    """
    import cv2  # local import keeps module importable on machines without OpenCV in CI
    H, W = rgb.shape[:2]
    s = min(10, H // 4, W // 4)
    if s < 1:
        s = 1
    corners = np.concatenate([
        rgb[:s, :s].reshape(-1, 3),
        rgb[:s, -s:].reshape(-1, 3),
        rgb[-s:, :s].reshape(-1, 3),
        rgb[-s:, -s:].reshape(-1, 3),
    ], axis=0)
    bg = np.median(corners, axis=0).astype(np.int32)  # (3,)
    diff = np.abs(rgb.astype(np.int32) - bg[None, None, :]).max(axis=2)
    fg = (diff > int(bg_thr)).astype(np.uint8)
    if fg.sum() == 0:
        return np.zeros(rgb.shape[:2], dtype=np.float32)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    if n > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        best = 1 + int(np.argmax(areas))
        fg = (labels == best).astype(np.uint8)
    if soften_px > 0:
        fg = cv2.GaussianBlur(fg.astype(np.float32), (0, 0), soften_px)
    return np.clip(fg.astype(np.float32), 0.0, 1.0)


# Back-compat shim — old name kept for callers that imported it.
_alpha_from_white_bg = _alpha_from_bg


# ── HuggingFace backend ─────────────────────────────────────────────────────


def _load_pipeline(device: str, dtype_str: str):
    """Load the sudo-ai/zero123plus pipeline lazily so the module can be
    imported on machines without diffusers."""
    import torch
    from diffusers import DiffusionPipeline

    dtype = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}[dtype_str]
    pipe = DiffusionPipeline.from_pretrained(
        "sudo-ai/zero123plus-v1.2",
        custom_pipeline="sudo-ai/zero123plus-pipeline",
        torch_dtype=dtype,
    )
    pipe.to(device)
    return pipe


def run_zero123plus_v12(
    cond_image_path: str,
    num_inference_steps: int = 75,
    guidance_scale: float = 4.0,
    seed: Optional[int] = 42,
    device: str = "cuda",
    dtype: str = "float16",
) -> Zero123Output:
    """Run the v1.2 pipeline once on the conditioning image, returning the
    raw 6-tile grid RGB array (640x960x3 uint8).
    """
    import torch
    from PIL import Image

    pipe = _load_pipeline(device=device, dtype_str=dtype)
    cond = Image.open(cond_image_path).convert("RGB")

    generator = torch.Generator(device=device).manual_seed(int(seed)) if seed is not None else None
    result = pipe(
        cond,
        num_inference_steps=int(num_inference_steps),
        guidance_scale=float(guidance_scale),
        generator=generator,
    )
    pil_img = result.images[0]
    grid = np.asarray(pil_img.convert("RGB"))
    # v1.2 grid: 3 rows x 2 cols of 320x320 -> shape (960, 640, 3)
    return Zero123Output(tile_grid_rgb=grid, tile_size=grid.shape[1] // 2, grid_rows=3, grid_cols=2)


# ── Driver: integrate with pose_alignment + persist ─────────────────────────


def generate_novel_views(
    obj_dir: str,
    backend: str = "plus_v12",
    num_inference_steps: int = 75,
    guidance_scale: float = 4.0,
    seed: Optional[int] = 42,
    device: str = "cuda",
    dtype: str = "float16",
    white_thr: int = 12,
) -> dict:
    """Generate novel views for the object in ``obj_dir`` using the prepared
    Zero123++ input image. Writes ``novel_views/tile_0..5.png`` (RGBA),
    ``novel_views/tile_grid.png``, and ``novel_views/novel_views_meta.json``.
    """
    import cv2

    obj_dir_p = Path(obj_dir)
    cond_path = obj_dir_p / "zero123_input.png"
    poses_path = obj_dir_p / "novel_views" / "poses.json"
    if not cond_path.exists():
        raise FileNotFoundError(
            f"zero123_input.png missing at {cond_path}; run --phase prep first."
        )
    if not poses_path.exists():
        raise FileNotFoundError(
            f"poses.json missing at {poses_path}; run --phase align first."
        )

    if backend != "plus_v12":
        raise NotImplementedError(f"Unknown Zero123++ backend: {backend}")

    out = run_zero123plus_v12(
        cond_image_path=str(cond_path),
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        seed=seed,
        device=device,
        dtype=dtype,
    )

    novel_dir = obj_dir_p / "novel_views"
    novel_dir.mkdir(parents=True, exist_ok=True)

    # Save raw grid (BGR for cv2)
    grid_bgr = cv2.cvtColor(out.tile_grid_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(novel_dir / "tile_grid.png"), grid_bgr)

    # Slice into 6 tiles (row-major)
    tiles_rgb = _slice_tiles(out.tile_grid_rgb, out.tile_size, out.grid_rows, out.grid_cols)

    # Load the pose schedule we computed in Phase 3
    with open(poses_path, "r", encoding="utf-8") as f:
        poses = json.load(f)
    if len(poses) != len(tiles_rgb):
        raise ValueError(
            f"poses.json has {len(poses)} entries but Zero123++ produced "
            f"{len(tiles_rgb)} tiles."
        )

    # Build per-tile alpha + write RGBA + record metadata
    meta_entries = []
    for i, (tile_rgb, pose) in enumerate(zip(tiles_rgb, poses)):
        alpha = _alpha_from_bg(tile_rgb, bg_thr=white_thr)
        rgba = np.concatenate(
            [tile_rgb, np.clip(alpha * 255.0, 0, 255).astype(np.uint8)[..., None]], axis=2,
        )
        # cv2 expects BGRA
        bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
        tile_path = novel_dir / f"tile_{i}.png"
        cv2.imwrite(str(tile_path), bgra)

        meta_entries.append({
            "tile_index": i,
            "image_path": f"novel_views/{tile_path.name}",
            "tile_az_deg": pose["tile_az_deg"],
            "tile_el_deg": pose["tile_el_deg"],
            "R_w2c": pose["R_w2c"],
            "T_w2c": pose["T_w2c"],
            "K": pose["K"],
            "width": pose["width"],
            "height": pose["height"],
            "visible_pixel_count": int((alpha > 0.05).sum()),
            "mean_alpha": float(alpha.mean()),
        })

    meta = {
        "backend": backend,
        "num_inference_steps": int(num_inference_steps),
        "guidance_scale": float(guidance_scale),
        "seed": seed,
        "tiles": meta_entries,
    }
    with open(novel_dir / "novel_views_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    logger.info(
        "Generated %d novel-view tiles -> %s",
        len(meta_entries), novel_dir,
    )
    return meta
