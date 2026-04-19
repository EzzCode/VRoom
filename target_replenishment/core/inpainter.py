"""
Inpainter — Stable Diffusion inpainting with optional propagated priors.

Uses stabilityai/stable-diffusion-2-inpainting via the diffusers library.
No ControlNets, IP-Adapters, or LoRAs — single generative model only.

Public API:
    load_inpainter(device, model_id)  -> InpaintingPipeline
    inpaint_view(pipeline, rgb, mask, prior, prompt, seed) -> np.ndarray
"""

__all__ = ['load_inpainter', 'inpaint_view']

import logging
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "runwayml/stable-diffusion-inpainting"

def load_inpainter(device: str = "cuda", model_id: str = None):
    """Load the SD Inpainting pipeline with ControlNet depth guidance.

    Returns:
        StableDiffusionControlNetInpaintPipeline on the specified device.
    """
    import torch
    from diffusers import AutoPipelineForInpainting, ControlNetModel, StableDiffusionControlNetInpaintPipeline

    model_id = model_id or _DEFAULT_MODEL
    logger.info(f"Loading ControlNet and inpainting model: {model_id}")

    try:
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11f1p_sd15_depth", 
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )
        pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            model_id,
            controlnet=controlnet,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            safety_checker=None
        )
    except Exception as e:
        logger.error(f"Failed to load ControlNet Pipeline. Error: {e}. Falling back to default.")
        pipe = AutoPipelineForInpainting.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        )

    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    # Optimize memory
    if device == "cuda":
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass  # xformers not available

    logger.info("Inpainting pipeline loaded.")
    return pipe


def inpaint_view(
    pipeline,
    rgb: np.ndarray,
    mask: np.ndarray,
    propagated_prior: np.ndarray = None,
    depth: np.ndarray = None,
    prompt: str = "",
    negative_prompt: str = "blurry, low quality, distorted, artifacts",
    seed: int = 42,
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
    strength: float = 1.0,
) -> np.ndarray:
    """Inpaint the masked region of an RGB image using ControlNet geometry guidance.

    Args:
        pipeline: Loaded StableDiffusionControlNetInpaintPipeline.
        rgb: (H, W, 3) uint8 — rendered view with defects.
        mask: (H, W) uint8 — binary repair mask (1 = inpaint here).
        propagated_prior: (H, W, 3) uint8 — warped content from neighbor.
        depth: (H, W) float32 — depth map for ControlNet geometry preservation.
        ...

    Returns:
        (H, W, 3) uint8 — inpainted image with soft blending boundaries.
    """
    import torch
    import cv2

    H, W = rgb.shape[:2]

    # Composite propagated prior into the image if available
    input_rgb = rgb.copy()
    if propagated_prior is not None:
        prior_mask = (propagated_prior.sum(axis=2) > 0).astype(np.float32)
        fill_region = (mask > 0) & (prior_mask > 0)
        
        blurred_mask = cv2.GaussianBlur(prior_mask, (11, 11), 0)
        alpha_blend = (blurred_mask * prior_mask)[..., np.newaxis]
        
        composited = input_rgb.astype(np.float32) * (1.0 - alpha_blend) + propagated_prior.astype(np.float32) * alpha_blend
        input_rgb[mask > 0] = np.clip(composited[mask > 0], 0, 255).astype(np.uint8)
        
        logger.info(f"Composited {fill_region.sum()} prior pixels")

    target_size = _round_to_multiple(max(H, W), 8)
    img_pil = Image.fromarray(input_rgb).resize((target_size, target_size), Image.LANCZOS)
    mask_pil = Image.fromarray(mask * 255).resize((target_size, target_size), Image.NEAREST)
    
    kwargs = {}
    if depth is not None and hasattr(pipeline, 'controlnet'):
        # Normalize depth for ControlNet ([0, 255] RGB)
        d_min, d_max = depth.min(), depth.max()
        norm_depth = (depth - d_min) / max(d_max - d_min, 1e-8)
        depth_rgb = np.stack([norm_depth * 255]*3, axis=-1).astype(np.uint8)
        depth_pil = Image.fromarray(depth_rgb).resize((target_size, target_size), Image.LANCZOS)
        kwargs['control_image'] = depth_pil

    generator = torch.Generator(device=pipeline.device).manual_seed(seed)

    result = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=img_pil,
        mask_image=mask_pil,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        strength=strength,
        generator=generator,
        **kwargs
    ).images[0]

    # Resize back to original
    result_np = np.array(result.resize((W, H), Image.LANCZOS))

    # Soft Blend output to avoid hallucinatory VAE seams being forwarded to the optimizer
    mask_soft = cv2.GaussianBlur(mask.astype(np.float32), (21, 21), 0)[..., np.newaxis]
    output = (rgb.astype(np.float32) * (1.0 - mask_soft) + result_np.astype(np.float32) * mask_soft).astype(np.uint8)

    logger.info(f"Inpainted {mask.sum()} px (prompt='{prompt[:50]}') with soft blending")
    return output


def _round_to_multiple(n: int, m: int) -> int:
    """Round up to nearest multiple of m, with min 512."""
    n = max(n, 512)
    return ((n + m - 1) // m) * m
