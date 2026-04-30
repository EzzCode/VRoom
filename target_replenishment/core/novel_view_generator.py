"""
Novel View Generator — Zero123++ multi-view generation from a single object image.

Generates 6 consistent views at 320×320 (upscaled to 512×512) from a single
rendered object view using Zero123++ v1.2.

Zero123++ generates a 640×960 grid (2 cols × 3 rows) of 6 views with fixed
azimuth offsets [30, 90, 150, 210, 270, 330]° from the input view.

Public API:
    load_zero123pp(device) -> Pipeline
    render_object_for_input(gaussians, pipe_config, coverage_result, ...) -> dict
    generate_novel_views(pipeline, image, alpha, ...) -> list[dict]
"""

__all__ = ['load_zero123pp', 'render_object_for_input', 'generate_novel_views']

import logging
import numpy as np
import cv2
from PIL import Image

logger = logging.getLogger(__name__)

_PIPELINE = None

# Zero123++ v1.2 azimuth offsets from input view (degrees)
ZERO123PP_AZIMUTHS = [30, 90, 150, 210, 270, 330]
# Zero123++ v1.2 elevation offsets (degrees), alternating by view index.
ZERO123PP_ELEVATIONS = [20, -10, 20, -10, 20, -10]


def load_zero123pp(device: str = "cuda"):
    """Load the Zero123++ v1.2 multi-view generation pipeline.

    Returns:
        Zero123++ diffusion pipeline on the specified device.
    """
    global _PIPELINE
    if _PIPELINE is not None:
        return _PIPELINE

    import torch
    from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler

    model_id = "sudo-ai/zero123plus-v1.2"
    logger.info(f"Loading Zero123++ model: {model_id}")

    pipeline = DiffusionPipeline.from_pretrained(
        model_id,
        custom_pipeline="sudo-ai/zero123plus-pipeline",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipeline.scheduler.config, timestep_spacing='trailing'
    )
    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)

    _PIPELINE = pipeline
    logger.info("Zero123++ v1.2 loaded successfully.")
    return pipeline


def render_object_for_input(
    gaussians,
    pipe_config,
    object_center: np.ndarray,
    object_radius: float,
    input_cam_position: np.ndarray,
    up_vector: np.ndarray,
    object_id: int,
    render_size: int = 512,
    reference_K: np.ndarray = None,
    reference_width: int = None,
    reference_height: int = None,
):
    """Render a clean, centered, full-object view using a virtual camera.

    Two FoV strategies (selected automatically):
      A. If `reference_K`, `reference_width`, `reference_height` are provided
         (recommended): reuse the training camera's actual FoV by rescaling
         its intrinsics to `render_size`. This guarantees a consistent
         object scale across objects regardless of `object_radius` or
         `up_vector` estimation noise — fixes the "Object N renders at the
         wrong scale" symptom that happens when `_estimate_up_vector`
         picks a different axis for different objects.
      B. Otherwise, compute FoV from `object_radius` so the object fills
         ~70% of the frame. Sensitive to `object_radius` accuracy.

    The camera position is fixed to `input_cam_position` and the camera
    orientation comes from `look_at(input_cam_position, object_center, up_vector)`,
    so the object is always centered.

    Returns:
        dict with 'rgb' (H,W,3 uint8), 'alpha' (H,W float32),
        'camera_R', 'camera_T', 'camera_K'.
    """
    import torch
    from target_replenishment.core.objectgs_bridge import create_virtual_camera, render_view
    from target_replenishment.core.view_alignment import look_at

    # Position camera at the same location as best training camera
    cam_pos = input_cam_position.astype(np.float32)
    dist_to_center = float(np.linalg.norm(cam_pos - object_center))

    use_reference_K = (
        reference_K is not None
        and reference_width is not None
        and reference_height is not None
        and reference_width > 0
        and reference_height > 0
    )

    if use_reference_K:
        # Rescale training cam K to render_size while preserving its FoV.
        # FoV stays identical → object scale is determined by the real lens
        # and the (already-correct) cam-to-object distance, not by object_radius.
        sx = render_size / float(reference_width)
        sy = render_size / float(reference_height)
        fx = float(reference_K[0, 0]) * sx
        fy = float(reference_K[1, 1]) * sy
        K = np.array([
            [fx, 0.0, render_size / 2.0],
            [0.0, fy, render_size / 2.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)
        fov_x = 2.0 * np.arctan(render_size / (2.0 * fx))
        logger.info(
            f"Virtual camera (training-K reuse): dist={dist_to_center:.2f}, "
            f"fov_x={np.degrees(fov_x):.1f}°, fx={fx:.1f}, fy={fy:.1f}, "
            f"render_size={render_size}"
        )
    else:
        # Legacy path: synthesize FoV from object angular size with 20% margin.
        angular_size = 2 * np.arctan(object_radius / max(dist_to_center, 1e-6))
        fov = angular_size / 0.7
        fov = np.clip(fov, np.radians(20), np.radians(90))
        fx = (render_size / 2) / np.tan(fov / 2)
        K = np.array([
            [fx, 0, render_size / 2],
            [0, fx, render_size / 2],
            [0, 0, 1]
        ], dtype=np.float32)
        logger.info(
            f"Virtual camera (synth-FoV): dist={dist_to_center:.2f}, "
            f"fov={np.degrees(fov):.1f}°, fx={fx:.1f}, "
            f"render_size={render_size}"
        )

    # Look at object center
    R, T = look_at(cam_pos, object_center, up_vector)

    cam = create_virtual_camera(R, T, K, render_size, render_size)
    bg_white = torch.ones(3, dtype=torch.float32, device="cuda")
    result = render_view(gaussians, cam, pipe_config, bg_white, object_label_id=object_id)

    rgb_np = (result['rgb'].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    alpha_np = result['alpha'].squeeze(0).cpu().numpy()

    return {
        'rgb': rgb_np,
        'alpha': alpha_np,
        'camera_R': R,
        'camera_T': T,
        'camera_K': K,
    }


def generate_novel_views(
    pipeline,
    input_image: np.ndarray,
    alpha_mask: np.ndarray = None,
    num_inference_steps: int = 75,
    seed: int = 42,
    input_alpha_threshold: float = 0.30,
    input_crop_margin_frac: float = 0.08,
    input_fill_ratio: float = 0.78,
) -> list:
    """Generate 6 novel views from a single object image using Zero123++.

    Args:
        pipeline: Loaded Zero123++ pipeline.
        input_image: (H, W, 3) uint8 — rendered isolated object.
        alpha_mask: (H, W) float32 — object alpha from 2DGS render.
                    Used to composite on white background.
        num_inference_steps: Diffusion steps (default 75 for quality).
        seed: Random seed.
        input_alpha_threshold: Alpha threshold used to build the foreground mask.
        input_crop_margin_frac: Crop margin around the foreground bbox.
        input_fill_ratio: Fraction of the canvas occupied by the centered object.

    Returns:
        List of 6 dicts:
            'rgb': (512, 512, 3) uint8
            'azimuth_offset_deg': float (degrees relative to input view)
    """
    import torch

    prepared = _prepare_input(
        input_image,
        alpha_mask,
        alpha_threshold=float(input_alpha_threshold),
        crop_margin_frac=float(input_crop_margin_frac),
        fill_ratio=float(input_fill_ratio),
    )
    input_pil = Image.fromarray(prepared)

    logger.info(f"Running Zero123++ inference ({num_inference_steps} steps)...")
    generator = torch.Generator(device=pipeline.device).manual_seed(seed)

    result = pipeline(
        input_pil,
        num_inference_steps=num_inference_steps,
        generator=generator,
    )

    if hasattr(result, 'images'):
        output_image = result.images[0]
    else:
        output_image = result[0]

    output_np = np.array(output_image)
    views = _split_grid(output_np)

    logger.info(f"Generated {len(views)} novel views at {views[0]['rgb'].shape[:2]}")
    return views


def _prepare_input(
    image: np.ndarray,
    alpha: np.ndarray = None,
    target_size: int = 320,
    alpha_threshold: float = 0.30,
    crop_margin_frac: float = 0.08,
    fill_ratio: float = 0.78,
) -> np.ndarray:
    """Composite on white bg and resize.

    Since render_object_for_input already produces a properly centered, framed
    image, we just need to composite on white and resize to Zero123++ input size.
    """
    H, W = image.shape[:2]

    if alpha is not None:
        alpha_2d = alpha.squeeze() if alpha.ndim > 2 else alpha
        alpha_2d = alpha_2d.astype(np.float32)
        if alpha_2d.max() > 1.0:
            alpha_2d = alpha_2d / 255.0
        alpha_2d = np.clip(alpha_2d, 0.0, 1.0)

        # Higher threshold (was 0.05) to drop semi-transparent shells that
        # leave grey halos around the object — Zero123++ interprets them as
        # near-background colour and bakes the grey into its outputs.
        alpha_threshold = float(np.clip(alpha_threshold, 0.01, 0.99))
        crop_margin_frac = float(np.clip(crop_margin_frac, 0.0, 0.50))
        fill_ratio = float(np.clip(fill_ratio, 0.10, 0.95))

        fg_mask = _largest_component_mask(alpha_2d > alpha_threshold, min_pixels=64)
        # Hard-binarize alpha INSIDE the kept fg mask so semi-transparent
        # edges become opaque (any non-zero alpha → 1.0). This makes the
        # composite look like a clean cutout, not a soft alpha matte.
        alpha_clean = fg_mask.astype(np.float32)

        alpha_3ch = alpha_clean[..., np.newaxis]
        white_bg = np.ones_like(image, dtype=np.float32) * 255.0
        composited = (image.astype(np.float32) * alpha_3ch + white_bg * (1.0 - alpha_3ch))
        composited = np.clip(composited, 0, 255).astype(np.uint8)

        if fg_mask.any():
            ys, xs = np.where(fg_mask)
            y0, y1 = ys.min(), ys.max() + 1
            x0, x1 = xs.min(), xs.max() + 1
            margin = int(round(crop_margin_frac * max(H, W)))
            y0 = max(0, y0 - margin)
            y1 = min(H, y1 + margin)
            x0 = max(0, x0 - margin)
            x1 = min(W, x1 + margin)

            crop_rgb = composited[y0:y1, x0:x1]
            crop_mask = fg_mask[y0:y1, x0:x1]
            canvas_size = max(H, W)
            composited = _center_object_on_canvas(
                crop_rgb,
                crop_mask,
                canvas_size=canvas_size,
                fill_ratio=fill_ratio,
            )
    else:
        composited = image.copy()

    # Already square from render_object_for_input, just resize
    from PIL import Image as PILImage
    pil = PILImage.fromarray(composited)
    resized = pil.resize((target_size, target_size), PILImage.LANCZOS)

    logger.info(
        "Input prepared: %dx%d -> %dx%d (alpha_thresh=%.2f, margin=%.3f, fill=%.2f)",
        W, H, target_size, target_size, alpha_threshold, crop_margin_frac, fill_ratio,
    )
    return np.array(resized)


def _split_grid(grid_image: np.ndarray) -> list:
    """Split Zero123++ v1.2 output grid into individual views.

    Zero123++ v1.2 outputs a 640×960 (W×H) image = 2 cols × 3 rows = 6 views.
    """
    H, W = grid_image.shape[:2]

    # Auto-detect: find the layout that gives the most square cells
    layouts = [(2, 3), (3, 2), (6, 1), (1, 6)]
    best_layout = None
    best_ratio = float('inf')

    for n_cols, n_rows in layouts:
        cell_w = W // n_cols
        cell_h = H // n_rows
        if cell_w < 50 or cell_h < 50:
            continue
        ratio = max(cell_w / cell_h, cell_h / cell_w)
        if ratio < best_ratio:
            best_ratio = ratio
            best_layout = (n_cols, n_rows)

    if best_layout is None:
        logger.warning(f"Cannot determine grid layout for {W}x{H}")
        return [{
            'rgb': grid_image,
            'azimuth_offset_deg': 0,
            'elevation_offset_deg': 0,
        }]

    n_cols, n_rows = best_layout
    cell_w = W // n_cols
    cell_h = H // n_rows

    views = []
    for row in range(n_rows):
        for col in range(n_cols):
            idx = row * n_cols + col
            if idx >= 6:
                break

            y0 = row * cell_h
            x0 = col * cell_w
            cell = grid_image[y0:y0+cell_h, x0:x0+cell_w]

            # Remove Zero123++ solid grey background. Zero123++ paints BG
            # in the [165..180]^3 range with low chroma; diffusion noise +
            # JPEG-style decoding push some pixels outside an exact-match
            # threshold, leaving visible halos. Use HSV: low saturation +
            # mid-range value catches the full grey spread without eating
            # into desaturated object pixels.
            cell_hsv = cv2.cvtColor(cell, cv2.COLOR_RGB2HSV)
            sat = cell_hsv[..., 1]
            val = cell_hsv[..., 2]
            bg_mask = (sat < 25) & (val > 140) & (val < 210)
            # Also catch the canonical [170,169,170] core in case the HSV
            # conversion misses a few edge pixels.
            bg_diff = np.abs(cell.astype(np.int32) - [170, 169, 170])
            bg_mask |= (bg_diff.sum(axis=-1) < 30)
            cell[bg_mask] = [255, 255, 255]

            # Remove tiny floating components by keeping only the largest object.
            fg_mask = np.any(np.abs(cell.astype(np.int32) - 255) > 8, axis=-1)
            fg_clean = _largest_component_mask(fg_mask, min_pixels=64)
            cell[~fg_clean] = [255, 255, 255]

            # Upscale to 512×512
            if cell.shape[0] != 512 or cell.shape[1] != 512:
                pil = Image.fromarray(cell)
                cell = np.array(pil.resize((512, 512), Image.LANCZOS))

            views.append({
                'rgb': cell,
                'azimuth_offset_deg': ZERO123PP_AZIMUTHS[idx],
                'elevation_offset_deg': ZERO123PP_ELEVATIONS[idx],
            })

    logger.info(
        f"Split grid {W}x{H} into {len(views)} views "
        f"({n_cols}×{n_rows} layout, {cell_w}x{cell_h} per cell)"
    )
    return views


def _largest_component_mask(mask: np.ndarray, min_pixels: int = 16) -> np.ndarray:
    """Keep largest connected component in a binary mask."""
    mask_u8 = (mask.astype(np.uint8) > 0).astype(np.uint8)
    if mask_u8.sum() < min_pixels:
        return mask_u8.astype(bool)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num_labels <= 1:
        return mask_u8.astype(bool)

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = int(np.argmax(areas) + 1)
    keep = labels == largest_label
    if keep.sum() < min_pixels:
        return mask_u8.astype(bool)
    return keep


def _center_object_on_canvas(
    crop_rgb: np.ndarray,
    crop_mask: np.ndarray,
    canvas_size: int,
    fill_ratio: float,
) -> np.ndarray:
    """Place cropped object on centered white canvas with target fill ratio."""
    ys, xs = np.where(crop_mask)
    if len(xs) == 0:
        canvas = np.ones((canvas_size, canvas_size, 3), dtype=np.uint8) * 255
        return canvas

    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    obj_rgb = crop_rgb[y0:y1, x0:x1]
    obj_mask = crop_mask[y0:y1, x0:x1].astype(np.uint8)

    h, w = obj_mask.shape
    if h <= 0 or w <= 0:
        canvas = np.ones((canvas_size, canvas_size, 3), dtype=np.uint8) * 255
        return canvas

    target_side = max(1, int(round(fill_ratio * canvas_size)))
    scale = target_side / float(max(h, w))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    obj_rgb_rs = cv2.resize(obj_rgb, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    obj_mask_rs = cv2.resize(obj_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST) > 0

    canvas = np.ones((canvas_size, canvas_size, 3), dtype=np.uint8) * 255
    x_off = (canvas_size - new_w) // 2
    y_off = (canvas_size - new_h) // 2

    patch = canvas[y_off:y_off + new_h, x_off:x_off + new_w]
    patch[obj_mask_rs] = obj_rgb_rs[obj_mask_rs]
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = patch
    return canvas