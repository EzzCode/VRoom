import cv2
import numpy as np


def _largest_component_bbox(img: np.ndarray, bg_color, diff_thresh: float = 10.0,
                             min_pixels: int = 64):
    """Return (min_x, min_y, max_x, max_y) of the largest connected non-bg component.

    Floaters / drips outside the main object would inflate a global bbox and pull
    alignment in the wrong direction. This restricts the bbox to the dominant
    object blob.
    """
    diff = np.abs(img.astype(np.float32) - np.array(bg_color, dtype=np.float32))
    mask = (np.sum(diff, axis=-1) > diff_thresh).astype(np.uint8)
    if mask.sum() == 0:
        return None
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n_labels <= 1:
        return None
    # stats: [x, y, w, h, area]; row 0 is background.
    areas = stats[1:, cv2.CC_STAT_AREA]
    if areas.size == 0 or areas.max() < min_pixels:
        return None
    best = 1 + int(np.argmax(areas))
    x, y, w, h, _a = stats[best]
    return (int(x), int(y), int(x + w - 1), int(y + h - 1))


def align_image_to_render_bbox(target_rgb: np.ndarray, rendered_rgb: np.ndarray,
                                bg_color=(255, 255, 255), return_diag: bool = False,
                                scale_mode: str = "cover"):
    """Warp target_rgb (Zero123 output) to match the 2D bounding box of rendered_rgb.

    Both images are assumed uint8 with a solid bg color. Bbox is taken from the
    largest connected non-bg component in each image to ignore floaters/drips.

        If `return_diag=True`, returns (aligned_rgb, dx, dy, scale) where dx/dy are
        the centre offsets (rendered − target, in pixels) and scale is the applied
        isotropic scale.

        scale_mode:
            - "cover": scale by max(rw/tw, rh/th), so target covers render bbox.
            - "contain": scale by min(rw/tw, rh/th), so target stays inside render bbox.
    """
    tgt_bbox = _largest_component_bbox(target_rgb, bg_color)
    rnd_bbox = _largest_component_bbox(rendered_rgb, bg_color)

    if tgt_bbox is None or rnd_bbox is None:
        if return_diag:
            return target_rgb.copy(), 0.0, 0.0, 1.0
        return target_rgb.copy()

    tx1, ty1, tx2, ty2 = tgt_bbox
    rx1, ry1, rx2, ry2 = rnd_bbox

    tw, th = max(1, tx2 - tx1), max(1, ty2 - ty1)
    rw, rh = max(1, rx2 - rx1), max(1, ry2 - ry1)

    sx = rw / tw
    sy = rh / th
    if str(scale_mode).lower() == "contain":
        scale = min(sx, sy)
    else:
        scale = max(sx, sy)

    # Calculate center points
    tcx, tcy = tx1 + tw / 2.0, ty1 + th / 2.0
    rcx, rcy = rx1 + rw / 2.0, ry1 + rh / 2.0

    # Build affine transform matrix
    # 1. Translate target center to origin
    # 2. Scale
    # 3. Translate to render center
    M = np.zeros((2, 3), dtype=np.float64)
    M[0, 0] = scale
    M[1, 1] = scale
    M[0, 2] = rcx - scale * tcx
    M[1, 2] = rcy - scale * tcy

    # Warp image, keeping the background color
    h, w = target_rgb.shape[:2]
    aligned_rgb = cv2.warpAffine(
        target_rgb, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=bg_color
    )

    if return_diag:
        dx = float(rcx - tcx)
        dy = float(rcy - tcy)
        return aligned_rgb, dx, dy, float(scale)
    return aligned_rgb