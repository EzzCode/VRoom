"""Thin rendering wrapper for ObjectGS — used by the object-isolation pipeline.

Imports DIRECTLY from ``temp_deps/ObjectGS``; has NO dependency on
``target_replenishment``.

Public API
----------
VirtualCamera(R, T, K, width, height)
    Lightweight camera object compatible with ObjectGS render.

create_camera(R, T, K, width, height) -> VirtualCamera
    Convenience constructor.

render_rgba(gaussians, cam, pipe_config, bg_white=True, object_label_id=None)
    -> {'rgb': (3,H,W) float32 cuda, 'alpha': (H,W) float32 cuda}

prefilter_anchors(gaussians, cam) -> bool mask (N,) cuda
    Voxel prefilter — returns visible anchor mask.
"""

from __future__ import annotations

import sys
import logging
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

# ── ObjectGS path setup ──────────────────────────────────────────────────────

_VROOM_ROOT = Path(__file__).resolve().parent.parent.parent
_OBJECTGS_DIR = _VROOM_ROOT / "temp_deps" / "ObjectGS"

if not _OBJECTGS_DIR.exists():
    raise ImportError(
        f"ObjectGS not found at {_OBJECTGS_DIR}. "
        "Clone it: git clone https://github.com/RuijieZhu94/ObjectGS.git temp_deps/ObjectGS"
    )

if str(_OBJECTGS_DIR) not in sys.path:
    sys.path.insert(0, str(_OBJECTGS_DIR))

from utils.graphics_utils import getProjectionMatrix                   # noqa: E402
from gaussian_renderer.render import render as _ogs_render             # noqa: E402
from gaussian_renderer.render import prefilter_voxel as _prefilter     # noqa: E402


# ── VirtualCamera ────────────────────────────────────────────────────────────

class VirtualCamera:
    """Lightweight camera compatible with ObjectGS render / prefilter_voxel."""

    def __init__(
        self,
        R: np.ndarray,
        T: np.ndarray,
        K: np.ndarray,
        width: int,
        height: int,
        uid: int = 0,
    ):
        self.uid = uid
        self.R = np.asarray(R, dtype=np.float32)
        self.T = np.asarray(T, dtype=np.float32).flatten()
        self.image_width = int(width)
        self.image_height = int(height)
        self.resolution_scale = 1.0

        K = np.asarray(K, dtype=np.float32)
        self.fx, self.fy = float(K[0, 0]), float(K[1, 1])
        self.cx, self.cy = float(K[0, 2]), float(K[1, 2])
        self.FoVx = 2.0 * np.arctan(width / (2.0 * self.fx))
        self.FoVy = 2.0 * np.arctan(height / (2.0 * self.fy))

        # world_view_transform: column-major [R|T] (ObjectGS stores transposed)
        Rt = np.eye(4, dtype=np.float32)
        Rt[:3, :3] = self.R
        Rt[:3, 3] = self.T
        self.world_view_transform = torch.tensor(Rt, dtype=torch.float32).transpose(0, 1).cuda()

        self.projection_matrix = (
            getProjectionMatrix(znear=0.01, zfar=100.0, fovX=self.FoVx, fovY=self.FoVy)
            .transpose(0, 1)
            .cuda()
        )

        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0)
            .bmm(self.projection_matrix.unsqueeze(0))
            .squeeze(0)
        )

        # Camera centre in world (for prefilter_voxel / set_anchor_mask).
        self.camera_center = self.world_view_transform.inverse()[3, :3]


def create_camera(
    R: np.ndarray,
    T: np.ndarray,
    K: np.ndarray,
    width: int,
    height: int,
    uid: int = 0,
) -> VirtualCamera:
    """Build a VirtualCamera from raw numpy R_w2c / T_w2c / K matrices."""
    return VirtualCamera(R, T, K, width, height, uid=uid)


# ── Rendering helpers ────────────────────────────────────────────────────────

def prefilter_anchors(gaussians, cam: VirtualCamera) -> torch.Tensor:
    """Return a boolean mask (N_anchors,) marking anchors visible from *cam*.

    Calls ObjectGS's voxel prefilter if ``pipe_config.add_prefilter`` is
    enabled (inferred from model), otherwise falls back to the model's own
    ``_anchor_mask``.  Always safe to call before render.
    """
    gaussians.set_anchor_mask(cam.camera_center, cam.resolution_scale)
    try:
        mask = _prefilter(cam, gaussians).squeeze()
    except Exception:
        mask = gaussians._anchor_mask
    return mask


def render_rgba(
    gaussians,
    cam: VirtualCamera,
    pipe_config,
    *,
    bg_white: bool = True,
    object_label_id: int | None = None,
    exclude_object_label_id: int | None = None,
    training: bool = False,
) -> dict:
    """Render *gaussians* from *cam* and return ``{'rgb', 'alpha'}``.

    Parameters
    ----------
    gaussians:
        Loaded ``GaussianModel`` (either ``gs_attr='2D'`` or ``'3D'``).
    cam:
        ``VirtualCamera`` built with :func:`create_camera`.
    pipe_config:
        Pipeline config returned by ``discover_object_scope`` / ``parse_cfg``.
    bg_white:
        If ``True`` use a white background; black otherwise.
    object_label_id:
        When given, only render anchors whose ``label_ids == object_label_id``.
    exclude_object_label_id:
        When given, render every anchor except this label. Ignored if
        ``object_label_id`` is set.
    training:
        Passed through to ``objectgs_render``.  Set ``True`` during gradient
        accumulation; ``False`` for comparison renders.

    Returns
    -------
    dict with:
    * ``'rgb'``   — ``(3, H, W)`` float32 cuda tensor in [0, 1].
    * ``'alpha'`` — ``(H, W)``   float32 cuda tensor in [0, 1].
    """
    bg_val = 1.0 if bg_white else 0.0
    bg = torch.full((3,), bg_val, dtype=torch.float32, device="cuda")

    gaussians.set_anchor_mask(cam.camera_center, cam.resolution_scale)
    if hasattr(pipe_config, "add_prefilter") and pipe_config.add_prefilter:
        try:
            visible_mask = _prefilter(cam, gaussians).squeeze()
        except Exception:
            visible_mask = gaussians._anchor_mask
    else:
        visible_mask = gaussians._anchor_mask

    object_mask = None
    if object_label_id is not None:
        object_mask = (gaussians.label_ids.squeeze() == int(object_label_id))
    elif exclude_object_label_id is not None:
        object_mask = (gaussians.label_ids.squeeze() != int(exclude_object_label_id))

    pkg = _ogs_render(
        cam,
        gaussians,
        pipe_config,
        bg,
        visible_mask=visible_mask,
        training=bool(training),
        object_mask=object_mask,
    )

    # render returns different shapes for 2DGS vs 3DGS; both expose 'render'
    # (3, H, W) and 'render_alphas'.
    rgb = torch.clamp(pkg["render"], 0.0, 1.0)        # (3, H, W)
    alpha_raw = pkg["render_alphas"]                   # (1, H, W) or (H, W)
    if alpha_raw.ndim == 3:
        alpha = alpha_raw[0]                           # (H, W)
    else:
        alpha = alpha_raw

    return {"rgb": rgb, "alpha": alpha}
