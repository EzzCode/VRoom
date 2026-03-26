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
import logging
import numpy as np
import torch
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
    if iteration == -1:
        iter_dirs = sorted(
            [d for d in pc_base.iterdir() if d.is_dir() and d.name.startswith("iteration_")],
            key=lambda d: int(d.name.split("_")[-1])
        )
        if not iter_dirs:
            raise FileNotFoundError(f"No iteration dirs in {pc_base}")
        iter_dir = iter_dirs[-1]
        iteration = int(iter_dir.name.split("_")[-1])
    else:
        iter_dir = pc_base / f"iteration_{iteration}"

    ply_path = iter_dir / "point_cloud.ply"
    if not ply_path.exists():
        raise FileNotFoundError(f"PLY not found: {ply_path}")

    logger.info(f"Loading ObjectGS model from {iter_dir} (iteration {iteration})")

    model_config = lp.model_config
    gaussians = GaussianModel(**model_config['kwargs'])
    gaussians.load_ply(str(ply_path))
    gaussians.load_mlp_checkpoints(str(iter_dir))
    gaussians.id_encoder = OneHotEncoder(gaussians.label_ids)
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


# ── Spatial Hole Detection (No MLP invocation) ──────────────────────────────

def detect_spatial_holes(
    gaussians,
    camera,
    coverage_threshold: float = 0.3,
) -> np.ndarray:
    """Detect holes by projecting anchor bounding boxes to screen space.

    This is a lightweight spatial check — no MLPs, no rasterization.
    Anchors have known spatial extent from their voxel scale (_scaling[:,:3]).
    If a region of the screen has zero anchor bboxes intersecting it,
    it is a guaranteed geometry hole.

    Args:
        gaussians: Loaded GaussianModel.
        camera: VirtualCamera.
        coverage_threshold: Fraction below which a tile is "hole".

    Returns:
        (H, W) float32 array: per-pixel coverage score [0, 1].
        0 = no anchor covers this pixel, 1 = well-covered.
    """
    H, W = camera.image_height, camera.image_width
    anchor_xyz = get_anchor_positions(gaussians)
    anchor_scales = np.exp(get_anchor_scales(gaussians)[:, :3])  # voxel extents

    R, T = camera.R, camera.T.flatten()
    K = np.array([[camera.fx, 0, camera.cx], [0, camera.fy, camera.cy], [0, 0, 1]], dtype=np.float32)

    # Project anchor centers
    cam_pts = (R @ anchor_xyz.T).T + T[np.newaxis, :]
    z = cam_pts[:, 2]
    valid = z > 0.01

    coverage = np.zeros((H, W), dtype=np.float32)
    if valid.sum() == 0:
        return coverage

    # Project to pixels
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    u = fx * cam_pts[valid, 0] / z[valid] + cx
    v = fy * cam_pts[valid, 1] / z[valid] + cy

    # Compute pixel radii from anchor voxel scale
    max_scale = anchor_scales[valid].max(axis=1)  # largest voxel extent
    pixel_radii = (fx * max_scale / z[valid]).astype(int)
    pixel_radii = np.clip(pixel_radii, 1, 50)

    # Stamp coverage
    for i in range(len(u)):
        px, py = int(u[i]), int(v[i])
        r = pixel_radii[i]
        x0, x1 = max(0, px - r), min(W, px + r + 1)
        y0, y1 = max(0, py - r), min(H, py + r + 1)
        coverage[y0:y1, x0:x1] += 1.0

    # Normalize to [0, 1]
    if coverage.max() > 0:
        coverage = np.clip(coverage / coverage.max(), 0, 1)

    n_hole_pixels = (coverage < coverage_threshold).sum()
    logger.info(f"Spatial coverage: {n_hole_pixels}/{H*W} pixels below threshold {coverage_threshold}")
    return coverage


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
