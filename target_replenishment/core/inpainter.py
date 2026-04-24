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

_DEFAULT_MODEL = "sd2-community/stable-diffusion-2-inpainting"

def load_inpainter(device: str = "cuda", model_id: str = None):
    """Load the SD Inpainting pipeline.

    Returns:
        diffusers.AutoPipelineForInpainting on the specified device.
    """
    import torch
    from diffusers import AutoPipelineForInpainting

    model_id = model_id or _DEFAULT_MODEL
    logger.info(f"Loading inpainting model: {model_id}")

    pipe = AutoPipelineForInpainting.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        variant="fp16" if device == "cuda" else None,
    )
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)


    # Optimize memory
    if device == "cuda":
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass  # xformers not available

    logger.info("Inpainting model loaded.")
    return pipe


def inpaint_view(
    pipeline,
    rgb: np.ndarray,
    mask: np.ndarray,
    propagated_prior: np.ndarray = None,
    prompt: str = "",
    negative_prompt: str = "blurry, low quality, distorted, artifacts",
    seed: int = 42,
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
    strength: float = 1.0,
) -> np.ndarray:
    """Inpaint the masked region of an RGB image.

    Args:
        pipeline: Loaded StableDiffusionInpaintPipeline.
        rgb: (H, W, 3) uint8 — rendered view with defects.
        mask: (H, W) uint8 — binary repair mask (1 = inpaint here).
        propagated_prior: (H, W, 3) uint8 — warped content from neighbor.
            If provided, composited into the image before inpainting to give
            the diffusion model a visual prior for consistency.
        prompt: Text prompt for the inpainter.
        negative_prompt: Negative text prompt.
        seed: Random seed for reproducibility.
        num_inference_steps: Diffusion steps.
        guidance_scale: Classifier-free guidance scale.
        strength: Denoising strength (1.0 = full inpainting).

    Returns:
        (H, W, 3) uint8 — inpainted image.
    """
    import torch

    H, W = rgb.shape[:2]

    # Composite propagated prior into the image if available
    input_rgb = rgb.copy()
    if propagated_prior is not None:
        import cv2
        prior_mask = (propagated_prior.sum(axis=2) > 0).astype(np.float32)
        fill_region = (mask > 0) & (prior_mask > 0)
        
        # Soften the edges of the prior compositing using Gaussian blur so the SD VAE doesn't hallucinate 
        # artifacts along harsh pixel-perfect boundaries.
        blurred_mask = cv2.GaussianBlur(prior_mask, (11, 11), 0)
        # Mask the alpha by the original prior to prevent blending with black background pixels
        alpha_blend = (blurred_mask * prior_mask)[..., np.newaxis]
        
        composited = input_rgb.astype(np.float32) * (1.0 - alpha_blend) + propagated_prior.astype(np.float32) * alpha_blend
        input_rgb[mask > 0] = np.clip(composited[mask > 0], 0, 255).astype(np.uint8)
        
        logger.info(f"Composited {fill_region.sum()} prior pixels into input with softened edges")

    # SD Inpainting expects 512×512 or multiples of 8
    target_size = _round_to_multiple(max(H, W), 8)
    img_pil = Image.fromarray(input_rgb).resize((target_size, target_size), Image.LANCZOS)
    mask_pil = Image.fromarray(mask * 255).resize((target_size, target_size), Image.NEAREST)

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
    ).images[0]

    # Resize back to original
    result_np = np.array(result.resize((W, H), Image.LANCZOS))

    # Blend: only replace masked pixels (keep original content outside mask)
    output = rgb.copy()
    output[mask > 0] = result_np[mask > 0]

    logger.info(f"Inpainted {mask.sum()} px (prompt='{prompt[:50]}')")
    return output


def _round_to_multiple(n: int, m: int) -> int:
    """Round up to nearest multiple of m, with min 512."""
    n = max(n, 512)
    return ((n + m - 1) // m) * m
