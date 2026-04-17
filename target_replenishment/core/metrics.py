"""
Metrics — Quality evaluation utilities for image comparison.

Provides PSNR, SSIM, and optional LPIPS metrics.

Public API:
    compute_psnr(img1, img2) -> float
    compute_ssim(img1, img2) -> float
    compute_lpips(img1, img2, net) -> float  (requires lpips package)
"""

__all__ = ['compute_psnr', 'compute_ssim', 'compute_lpips']

import numpy as np
import cv2


def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Peak Signal-to-Noise Ratio between two uint8 images.

    Returns:
        PSNR in dB. Higher = more similar. Typical range: 20-40 dB.
    """
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse < 1e-10:
        return float('inf')
    return float(20 * np.log10(255.0 / np.sqrt(mse)))


def compute_ssim(
    img1: np.ndarray,
    img2: np.ndarray,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    """Structural Similarity Index between two images.

    Returns:
        SSIM in [0, 1]. Higher = more similar.
    """
    if img1.ndim == 3:
        # Channel-average
        scores = [_ssim_gray(img1[:, :, c], img2[:, :, c], k1, k2) for c in range(img1.shape[2])]
        return float(np.mean(scores))
    return float(_ssim_gray(img1, img2, k1, k2))


def _ssim_gray(img1: np.ndarray, img2: np.ndarray, k1: float, k2: float) -> float:
    C1 = (k1 * 255) ** 2
    C2 = (k2 * 255) ** 2
    img1f = img1.astype(np.float64)
    img2f = img2.astype(np.float64)

    mu1 = cv2.GaussianBlur(img1f, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2f, (11, 11), 1.5)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(img1f ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2f ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1f * img2f, (11, 11), 1.5) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(np.mean(ssim_map))


def compute_lpips(
    img1: np.ndarray,
    img2: np.ndarray,
    net: str = 'vgg',
) -> float:
    """Learned Perceptual Image Patch Similarity.

    Requires: pip install lpips

    Returns:
        LPIPS score (lower = more similar). Typical range: 0.0-0.5.
    """
    import torch
    import lpips
    fn = lpips.LPIPS(net=net).cuda()

    t1 = torch.from_numpy(img1).permute(2, 0, 1).float().unsqueeze(0).cuda() / 127.5 - 1.0
    t2 = torch.from_numpy(img2).permute(2, 0, 1).float().unsqueeze(0).cuda() / 127.5 - 1.0
    with torch.no_grad():
        score = fn(t1, t2)
    return float(score.item())
