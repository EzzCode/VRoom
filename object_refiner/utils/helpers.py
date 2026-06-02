import json
import logging
from pathlib import Path
import cv2
import torch
import numpy as np
import re

# TODO(label-alignment): remove this import once tracked_object_id == object_label_id end-to-end
from object_refiner.utils.gstrain_wrapper import make_camera, render_rgba, render_rgba
from object_refiner.utils.sv3d_wrapper import HallucinatedView

logger = logging.getLogger(__name__)
SV3D_VIEW_COUNT      = 21
_VROOM_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path_value, *, manifest_dir):
    p = Path(path_value)
    if p.is_absolute():
        return p
    for candidate in (manifest_dir / p, Path.cwd() / p, _VROOM_ROOT / p):
        if candidate.exists():
            return candidate
    return _VROOM_ROOT / p

def normalize(vector: np.ndarray):
    eps = 1e-8
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm < eps:
        raise ValueError(f"Cannot normalize zero or non-finite vector: {vector}")
    return vector / norm

def find_image(images_dir, img_name):
    if not images_dir.exists():
        return None
    for ext in (".jpg", ".JPG", ".jpeg", ".png", ".PNG"):
        c = images_dir / f"{img_name}{ext}"
        if c.exists():
            return c
    for f in images_dir.iterdir():
        if f.is_file() and f.stem == img_name:
            return f
    return None


def find_tracked_id_map(tracked_id_map_dir, img_name):
    """Return the per-frame tracked id-map file for `img_name`, or None.

    The tracked id-map is produced by masks_and_tracking's object_tracker — a single-channel
    PNG where each pixel value is the instance label assigned by the tracker.
    Distinct from raw SAM output, which is a stack of per-mask binary arrays
    with no temporal consistency.
    """
    if not tracked_id_map_dir.exists():
        return None
    for ext in (".png", ".jpg", ".jpeg"):
        c = tracked_id_map_dir / f"{img_name}{ext}"
        if c.exists():
            return c
    # Replica-style rename (rgb -> semantic_instance)
    renamed = img_name.replace("_rgb_", "_semantic_instance_").replace("rgb", "semantic_instance")
    if renamed != img_name:
        for ext in (".png", ".jpg", ".jpeg"):
            c = tracked_id_map_dir / f"{renamed}{ext}"
            if c.exists():
                return c
    # Trailing-digit fuzzy match
    m = re.search(r"(\d+)$", img_name)
    if m:
        suffix = m.group(1)
        for f in tracked_id_map_dir.iterdir():
            if f.is_file() and f.suffix.lower() in (".png", ".jpg", ".jpeg"):
                m2 = re.search(r"(\d+)$", f.stem)
                if m2 and m2.group(1) == suffix:
                    return f
    return None

# TODO(label-alignment): delete this entire function once tracked_object_id == object_label_id end-to-end;
#   caller in __main__.py passes tracked_object_id=object_label_id directly and this is no longer needed.
def vote_tracked_object_id(scope, gaussians, pipe_config, tracked_id_map_dir, n_probe=5, tau_alpha=0.4):
    """IoU-vote the GS alpha mask against the per-frame tracked id-maps to find
    the tracked_object_id (instance label) that best matches the Gaussian model's
    silhouette. This is the same integer label that vote.py writes into the
    labeled COLMAP point cloud."""
    if not tracked_id_map_dir.exists():
        return None
    indices = list(scope.visible_cam_indices)
    if len(indices) > n_probe:
        indices = indices[::max(1, len(indices) // n_probe)][:n_probe]

    votes = {}
    for ci in indices:
        cam_p = scope.cameras[ci]
        id_map_path = find_tracked_id_map(tracked_id_map_dir, cam_p["image_name"])
        if id_map_path is None:
            continue
        id_map = cv2.imread(str(id_map_path), cv2.IMREAD_UNCHANGED)
        if id_map is None:
            continue
        cam = make_camera(cam_p["R"], cam_p["T"], cam_p["K"], cam_p["width"], cam_p["height"])
        alpha = render_rgba(gaussians, cam, pipe_config,
                            object_label_id=scope.object_label_id)["alpha"].detach().cpu().numpy()
        gs_mask = (alpha[0] if alpha.ndim == 3 else alpha) > tau_alpha
        H, W = gs_mask.shape
        if id_map.shape[:2] != (H, W):
            id_map = cv2.resize(id_map, (W, H), interpolation=cv2.INTER_NEAREST)
        for label in np.unique(id_map):
            if int(label) == 0:
                continue
            m_id = (id_map == label)
            inter = float(np.logical_and(m_id, gs_mask).sum())
            union = float(np.logical_or(m_id, gs_mask).sum())
            votes[int(label)] = votes.get(int(label), 0.0) + inter / max(union, 1.0)

    if not votes:
        return None
    winner, best_score = max(votes.items(), key=lambda kv: kv[1])
    logger.info("Voted tracked_object_id=%d (top labels: %s)", winner,
                {k: round(v, 3) for k, v in sorted(votes.items(), key=lambda kv: -kv[1])[:5]})
    return winner if best_score > 0 else None

def load_cache(out_dir, cond_azimuth_deg, cond_elevation_deg):
    out_dir       = Path(out_dir)
    manifest_path = out_dir / "generation.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No cache manifest at {manifest_path}. Re-run without reuse_sv3d=True.")

    with open(manifest_path) as f:
        manifest = json.load(f)

    n_views = int(manifest.get("n_views", -1))
    if n_views != SV3D_VIEW_COUNT:
        raise RuntimeError(
            f"Cached manifest has n_views={n_views} but backend expects {SV3D_VIEW_COUNT}. "
            "Delete the cache or re-run without reuse_sv3d=True."
        )

    cond = manifest.get("conditioning", {})
    m_az = float(cond.get("azimuth_deg", float("nan")))
    m_el = float(cond.get("elevation_deg", float("nan")))
    daz  = abs(((m_az - cond_azimuth_deg + 180.0) % 360.0) - 180.0)
    if daz > 0.5 or abs(m_el - cond_elevation_deg) > 0.5:
        raise RuntimeError(
            f"Cached conditioning az/el=({m_az:.2f}, {m_el:.2f}) differs from "
            f"current ({cond_azimuth_deg:.2f}, {cond_elevation_deg:.2f}). Re-run without reuse_sv3d=True."
        )

    views = []
    for entry in sorted(manifest.get("frames", []), key=lambda e: int(e["index"])):
        stem = f"{entry['index']:02d}__az{round(entry['azimuth_deg']):+04d}"
        path = out_dir / "sv3d" / f"{stem}.png"
        if not path.exists():
            raise FileNotFoundError(f"Missing cached frame: {path}. Re-run without reuse_sv3d=True.")
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Cannot read cached frame: {path}")
        views.append(HallucinatedView(
            rgb=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
            azimuth_deg=float(entry["azimuth_deg"]),
            elevation_deg=float(entry["elevation_deg"]),
        ))
    if not views:
        raise RuntimeError(f"Manifest at {manifest_path} has no frames.")
    logger.info("Cache reuse: loaded %d frames from %s.", len(views), manifest_path)
    return views


import functools
import torch.nn.functional as F

def ssim_loss(prediction: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    @functools.lru_cache(maxsize=4)
    def kernel(size: int, channels: int, device_str: str, dtype):
        dist = torch.distributions.Normal(loc=size // 2, scale=1.5)
        coords = torch.arange(size, dtype=torch.float32)
        weights = dist.log_prob(coords).exp()
        weights = weights / weights.sum()
        kernel2d = weights[:, None] @ weights[None, :]
        return kernel2d.unsqueeze(0).unsqueeze(0).expand(channels, 1, size, size).contiguous().to(device=device_str, dtype=dtype)

    if prediction.dim() == 3:
        prediction = prediction.unsqueeze(0)
        target = target.unsqueeze(0)
    prediction = prediction.clamp(0.0, 1.0)
    target = target.clamp(0.0, 1.0)
    channels = prediction.shape[1]
    padding = window_size // 2
    window = kernel(window_size, channels, str(prediction.device), prediction.dtype)
    mu_a = F.conv2d(prediction, window, padding=padding, groups=channels)
    mu_b = F.conv2d(target, window, padding=padding, groups=channels)
    sigma_a = F.conv2d(prediction * prediction, window, padding=padding, groups=channels) - mu_a.pow(2)
    sigma_b = F.conv2d(target * target, window, padding=padding, groups=channels) - mu_b.pow(2)
    sigma_ab = F.conv2d(prediction * target, window, padding=padding, groups=channels) - (mu_a * mu_b)
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    numerator = (2.0 * mu_a * mu_b + c1) * (2.0 * sigma_ab + c2)
    denominator = (mu_a.pow(2) + mu_b.pow(2) + c1) * (sigma_a + sigma_b + c2)
    return 1.0 - (numerator / denominator).mean()