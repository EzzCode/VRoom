"""
ObjectGS Bridge — Adapter between VRoom target replenishment and ObjectGS internals.

Handles:
  - Loading a trained ObjectGS model (anchor PLY + MLP checkpoints)
  - Constructing virtual cameras for rendering
  - Rendering with full anchor ID attribution (pixel → parent anchor)
  - Spatial hole detection without MLP invocation
  - Extracting anchor geometry for scoring

Usage:
    from target_replenishment.core.objectgs_bridge import (
        load_gaussians, render_view, create_virtual_camera,
        get_anchor_positions, build_anchor_id_map
    )
"""

import sys
import json
import logging
import numpy as np
import torch
import cv2
import yaml
from pathlib import Path
from types import SimpleNamespace

logger = logging.getLogger(__name__)

# ── ObjectGS import setup ────────────────────────────────────────────────────

_VROOM_ROOT = Path(__file__).resolve().parent.parent.parent
_OBJECTGS_DIR = _VROOM_ROOT / "temp_deps" / "ObjectGS"

if not _OBJECTGS_DIR.exists():
    raise ImportError(
        f"ObjectGS not found at {_OBJECTGS_DIR}. "
        "Clone it: git clone https://github.com/RuijieZhu94/ObjectGS.git temp_deps/ObjectGS"
    )

if str(_OBJECTGS_DIR) not in sys.path:
    sys.path.insert(0, str(_OBJECTGS_DIR))

from scene.base_model import GaussianModel            # noqa: E402
from utils.general_utils import parse_cfg              # noqa: E402
from utils.graphics_utils import getProjectionMatrix   # noqa: E402
from utils.semantic_utils import OneHotEncoder         # noqa: E402
from gaussian_renderer.render import render as objectgs_render  # noqa: E402
from gaussian_renderer.render import prefilter_voxel   # noqa: E402


# ── Model Loading ────────────────────────────────────────────────────────────

def load_gaussians(model_path: str, iteration: int = -1) -> tuple:
    """Load a trained ObjectGS model (anchor PLY + MLP checkpoints).

    Args:
        model_path: ObjectGS training output folder containing config.yaml
                    and point_cloud/iteration_XXXXX/.
                    fallback to final_model/ if not found or model.
        iteration: Which iteration to load (-1 = latest).

    Returns:
        (gaussians, pipe_config)
    """
    model_path = Path(model_path)
    config_path = model_path / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    lp, op, pp = parse_cfg(cfg)

    pc_base = model_path / "point_cloud"
    final_model_dir = model_path / "final_model"

    # Optional direct-export fallback (model files stored at model root)
    root_model_dir = model_path if (model_path / "point_cloud.ply").exists() else None

    def _iter_index(d: Path):
        try:
            return int(d.name.split("_")[-1])
        except ValueError:
            return None

    if iteration == -1:
        iter_dirs = []
        if pc_base.exists():
            iter_dirs = [
                d for d in pc_base.iterdir()
                if d.is_dir() and d.name.startswith("iteration_") and _iter_index(d) is not None
            ]
            iter_dirs = sorted(iter_dirs, key=_iter_index)

        if iter_dirs:
            iter_dir = iter_dirs[-1]
            iteration = _iter_index(iter_dir)
        elif final_model_dir.exists():
            iter_dir = final_model_dir
            iteration = -1
        elif root_model_dir is not None:
            iter_dir = root_model_dir
            iteration = -1
        else:
            raise FileNotFoundError(
                f"No ObjectGS checkpoint found. Checked:\n"
                f" - {pc_base}/iteration_*\n"
                f" - {final_model_dir}\n"
                f" - {model_path}/point_cloud.ply"
            )
    else:
        # Support both unpadded and zero-padded iteration folder names
        candidates = [
            pc_base / f"iteration_{iteration}",
            pc_base / f"iteration_{iteration:05d}",
        ]
        iter_dir = next((c for c in candidates if c.exists()), None)

        if iter_dir is None:
            if final_model_dir.exists():
                logger.warning(
                    f"Requested iteration {iteration} not found; falling back to final_model."
                )
                iter_dir = final_model_dir
            elif root_model_dir is not None:
                logger.warning(
                    f"Requested iteration {iteration} not found; falling back to model root export."
                )
                iter_dir = root_model_dir
            else:
                raise FileNotFoundError(
                    f"Iteration {iteration} not found in {pc_base} and no fallback model found."
                )

    ply_path = iter_dir / "point_cloud.ply"
    if not ply_path.exists():
        raise FileNotFoundError(f"PLY not found: {ply_path}")

    logger.info(f"Loading ObjectGS model from {iter_dir} (iteration {iteration})")

    model_config = lp.model_config
    gaussians = GaussianModel(**model_config['kwargs'])
    gaussians.load_ply(str(ply_path))
    gaussians.load_mlp_checkpoints(str(iter_dir))
    gaussians.id_encoder = OneHotEncoder(gaussians.label_ids)

    rep_candidates = [
        iter_dir / "replenishment.json",
        model_path / "replenishment.json",
        iter_dir.parent.parent / "replenishment.json" if iter_dir.parent.name == "point_cloud" else None,
    ]
    for rep_path in [p for p in rep_candidates if p is not None]:
        if not rep_path.exists():
            continue
        try:
            with open(rep_path, "r", encoding="utf-8") as f:
                rep_data = json.load(f)
            if 'n_original_anchors' in rep_data:
                gaussians.n_original_anchors = int(rep_data['n_original_anchors'])
            if 'override_view_dir' in rep_data:
                gaussians.override_view_dir = torch.tensor(
                    rep_data['override_view_dir'], dtype=torch.float32, device="cuda"
                )
            if 'seeded_opacity_gates' in rep_data:
                gates = torch.tensor(
                    rep_data['seeded_opacity_gates'], dtype=torch.float32, device="cuda"
                ).reshape(-1, 1)
                gaussians.replenishment_seed_opacity_gate = gates
            if 'seeded_opacity_lifts' in rep_data:
                lifts = torch.tensor(
                    rep_data['seeded_opacity_lifts'], dtype=torch.float32, device="cuda"
                )
                if lifts.numel() == 0:
                    lifts = lifts.reshape(0, 1)
                elif lifts.ndim == 1:
                    lifts = lifts.reshape(-1, 1)
                else:
                    lifts = lifts.reshape(lifts.shape[0], -1)
                gaussians.replenishment_seed_opacity_lift = lifts
            logger.info("Loaded replenishment metadata from %s", rep_path)
            break
        except Exception as exc:
            logger.warning("Failed to load replenishment metadata from %s: %s", rep_path, exc)
    
    # Initialize the optimizer strictly matching original training configuration 
    # to prevent gradient explosiveness and catastrophic forgetting.
    if hasattr(op, 'position_lr_init'):
        gaussians.training_setup(op)
        
    gaussians.eval()
    gaussians.explicit_gs = False

    logger.info(
        f"Loaded: {gaussians.get_anchor.shape[0]} anchors, "
        f"n_offsets={gaussians.n_offsets}, gs_attr={gaussians.gs_attr}"
    )
    return gaussians, pp


# ── Camera Construction ──────────────────────────────────────────────────────

class VirtualCamera:
    """Lightweight camera compatible with ObjectGS's render function."""

    def __init__(self, R: np.ndarray, T: np.ndarray, K: np.ndarray, width: int, height: int, uid: int = 0):
        self.uid = uid
        self.R = R
        self.T = T
        self.image_width = width
        self.image_height = height
        self.resolution_scale = 1.0

        self.fx, self.fy = K[0, 0], K[1, 1]
        self.cx, self.cy = K[0, 2], K[1, 2]
        self.FoVx = 2 * np.arctan(width / (2 * self.fx))
        self.FoVy = 2 * np.arctan(height / (2 * self.fy))

        Rt = np.eye(4, dtype=np.float32)
        Rt[:3, :3] = R
        Rt[:3, 3] = T.flatten()
        self.world_view_transform = torch.tensor(Rt).transpose(0, 1).cuda()

        self.projection_matrix = getProjectionMatrix(
            znear=0.01, zfar=100.0, fovX=self.FoVx, fovY=self.FoVy
        ).transpose(0, 1).cuda()

        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))
        ).squeeze(0)

        self.camera_center = self.world_view_transform.inverse()[3, :3]


def create_virtual_camera(R, T, K, width, height):
    """Build a VirtualCamera from raw numpy matrices."""
    return VirtualCamera(R, T.flatten(), K, width, height)


# ── Rendering with Anchor ID Attribution ─────────────────────────────────────

def render_view(gaussians, camera, pipe_config, bg_color=None, object_label_id=None):
    """Render a single view through the full ObjectGS pipeline.

    Returns dict with 'rgb', 'alpha', 'depth', 'normal', 'semantics',
    'parent_anchor_ids', 'means2d', 'radii'.
    """
    if bg_color is None:
        bg_color = torch.zeros(3, dtype=torch.float32, device="cuda")

    gaussians.set_anchor_mask(camera.camera_center, camera.resolution_scale)
    if hasattr(pipe_config, 'add_prefilter') and pipe_config.add_prefilter:
        visible_mask = prefilter_voxel(camera, gaussians).squeeze()
    else:
        visible_mask = gaussians._anchor_mask

    object_mask = None
    if object_label_id is not None:
        object_mask = (gaussians.label_ids.squeeze() == object_label_id)

    with torch.no_grad():
        render_pkg = objectgs_render(
            camera, gaussians, pipe_config, bg_color,
            visible_mask=visible_mask, training=False, object_mask=object_mask,
        )

    result = {
        'rgb': torch.clamp(render_pkg['render'], 0.0, 1.0),
        'alpha': render_pkg['render_alphas'],
        'depth': render_pkg.get('render_depth'),
        'normal': render_pkg.get('render_normals'),
        'semantics': render_pkg.get('render_semantics'),
        'parent_anchor_ids': render_pkg.get('parent_anchor_ids'),
        'means2d': render_pkg.get('viewspace_points'),
        'radii': render_pkg.get('radii'),
    }
    return result


def build_anchor_id_map(render_result: dict, H: int, W: int, n_anchors: int) -> np.ndarray:
    """Build per-pixel anchor ID map from render output.

    Uses the 2D projected means and radii of each generated Gaussian,
    combined with parent_anchor_ids, to assign each pixel to the anchor
    whose generated Gaussian contributed most (largest radius / closest).

    Returns:
        (H, W) int32 array. -1 = no anchor covers this pixel.
    """
    parent_ids = render_result['parent_anchor_ids']
    means2d = render_result['means2d']
    radii = render_result['radii']

    if parent_ids is None or means2d is None or radii is None:
        logger.warning("Anchor ID data not available — returning empty map")
        return np.full((H, W), -1, dtype=np.int32)

    parent_ids_np = parent_ids.cpu().numpy()              # (N_gaussians,)
    means2d_np = means2d.squeeze(0).cpu().numpy()         # (N_gaussians, 2)
    radii_np = radii.cpu().numpy().astype(np.float32)     # (N_gaussians,)

    # Only keep Gaussians with positive radii
    valid = radii_np > 0
    if valid.sum() == 0:
        return np.full((H, W), -1, dtype=np.int32)

    valid_ids = parent_ids_np[valid]
    valid_means = means2d_np[valid]
    valid_radii = radii_np[valid]

    # Build map: for each pixel, which anchor's Gaussian is nearest with largest radius
    anchor_map = np.full((H, W), -1, dtype=np.int32)
    weight_map = np.zeros((H, W), dtype=np.float32)

    for i in range(len(valid_ids)):
        cx, cy = int(valid_means[i, 0]), int(valid_means[i, 1])
        r = int(valid_radii[i])
        if r <= 0:
            continue

        # Bounding box of this splat
        x0 = max(0, cx - r)
        x1 = min(W, cx + r + 1)
        y0 = max(0, cy - r)
        y1 = min(H, cy + r + 1)

        weight = valid_radii[i]  # proxy: larger radius = more influence
        # Stamp anchor ID where this Gaussian dominates
        mask = weight_map[y0:y1, x0:x1] < weight
        anchor_map[y0:y1, x0:x1] = np.where(mask, valid_ids[i], anchor_map[y0:y1, x0:x1])
        weight_map[y0:y1, x0:x1] = np.where(mask, weight, weight_map[y0:y1, x0:x1])

    logger.info(f"Anchor ID map: {(anchor_map >= 0).sum()} / {H*W} pixels covered, "
                f"{len(np.unique(anchor_map[anchor_map >= 0]))} unique anchors")
    return anchor_map


def project_anchor_silhouette(
    camera,
    anchor_positions: np.ndarray,
    object_radius: float,
    height: int = None,
    width: int = None,
    blur_size: int = 9,
    radius_scale: float = 0.18,
) -> np.ndarray:
    """Project 3D anchors into a soft 2D silhouette mask.

    The mask is a coarse occupancy prior used to supervise alpha/coverage.
    It is intentionally smooth so the optimizer can fill holes without
    overfitting to single projected points.
    """
    if anchor_positions is None or len(anchor_positions) == 0:
        h = int(height or camera.image_height)
        w = int(width or camera.image_width)
        return np.zeros((h, w), dtype=np.float32)

    h = int(height or camera.image_height)
    w = int(width or camera.image_width)

    anchors = np.asarray(anchor_positions, dtype=np.float32)
    if anchors.ndim != 2 or anchors.shape[1] != 3:
        raise ValueError(f"anchor_positions must have shape (N, 3), got {anchors.shape}")

    pts_h = np.concatenate([anchors, np.ones((anchors.shape[0], 1), dtype=np.float32)], axis=1)
    view = camera.world_view_transform.detach().cpu().numpy().astype(np.float32)
    proj = camera.projection_matrix.detach().cpu().numpy().astype(np.float32)

    clip = pts_h @ view @ proj
    w_coord = clip[:, 3]
    valid = np.isfinite(w_coord) & (w_coord > 1e-6)
    if not np.any(valid):
        return np.zeros((h, w), dtype=np.float32)

    clip = clip[valid]
    ndc = clip[:, :3] / np.clip(clip[:, 3:4], 1e-6, None)

    in_view = (
        np.isfinite(ndc).all(axis=1)
        & (ndc[:, 0] >= -1.5) & (ndc[:, 0] <= 1.5)
        & (ndc[:, 1] >= -1.5) & (ndc[:, 1] <= 1.5)
    )
    if not np.any(in_view):
        return np.zeros((h, w), dtype=np.float32)

    ndc = ndc[in_view]
    cam_space = (pts_h[valid][in_view] @ view)
    depth = np.clip(cam_space[:, 2], 1e-3, None)

    pixels = np.empty((ndc.shape[0], 2), dtype=np.float32)
    pixels[:, 0] = (ndc[:, 0] * 0.5 + 0.5) * (w - 1)
    pixels[:, 1] = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * (h - 1)

    fx = float(camera.fx)
    fy = float(camera.fy)
    focal = max((fx + fy) * 0.5, 1.0)
    object_radius = float(object_radius) if object_radius is not None else 1.0

    # Convert the object radius to a conservative projected footprint.
    radius_scale = float(np.clip(radius_scale, 0.02, 1.0))
    base_radius = np.clip((object_radius / np.median(depth)) * focal * radius_scale, 2.0, 48.0)
    mask = np.zeros((h, w), dtype=np.float32)

    for (x, y), z in zip(pixels, depth):
        radius = int(np.clip(base_radius * np.clip(np.median(depth) / z, 0.75, 1.75), 2.0, 64.0))
        cx = int(round(x))
        cy = int(round(y))
        if cx < -radius or cy < -radius or cx >= w + radius or cy >= h + radius:
            continue
        cv2.circle(mask, (cx, cy), radius, 1.0, -1)

    if blur_size > 1:
        blur_size = int(blur_size)
        if blur_size % 2 == 0:
            blur_size += 1
        mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)

    if mask.max() > 0:
        mask = mask / mask.max()

    return np.clip(mask, 0.0, 1.0)


# ── Spatial Hole Detection (No MLP invocation) ──────────────────────────────

# ── Geometry Helpers ─────────────────────────────────────────────────────────

def get_anchor_positions(gaussians) -> np.ndarray:
    """Anchor xyz as numpy (N, 3) float32."""
    return gaussians.get_anchor.detach().cpu().numpy()


def get_anchor_scales(gaussians) -> np.ndarray:
    """Anchor scales as numpy (N, 6) float32 (log-space)."""
    return gaussians._scaling.detach().cpu().numpy()


def get_anchor_rotations(gaussians) -> np.ndarray:
    """Anchor rotations as numpy (N, 4) float32 quaternions."""
    return gaussians._rotation.detach().cpu().numpy()


def get_label_ids(gaussians) -> np.ndarray:
    """Per-anchor label IDs as numpy (N,) uint8."""
    return gaussians.label_ids.squeeze().detach().cpu().numpy().astype(np.uint8)
