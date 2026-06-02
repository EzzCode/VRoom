import logging
import numpy as np
import torch

from gstrain.gaussian_renderer.render import prefilter_voxel as _prefilter
from gstrain.gaussian_renderer.render import render as _render
from gstrain.vroom_core.utils.geometry import projection_matrix


class Camera:
    def __init__(self, R, T, K, width, height, uid=0):
        self.uid = uid
        self.R = np.asarray(R, dtype=np.float32)
        self.T = np.asarray(T, dtype=np.float32).flatten()
        self.image_width = int(width)
        self.image_height = int(height)
        self.resolution_scale = 1.0

        K = np.asarray(K, dtype=np.float32)
        self.fx = float(K[0, 0])
        self.fy = float(K[1, 1])
        self.cx = float(K[0, 2])
        self.cy = float(K[1, 2])
        self.FoVx = 2.0 * np.arctan(width / (2.0 * self.fx))
        self.FoVy = 2.0 * np.arctan(height / (2.0 * self.fy))

        Rt = np.eye(4, dtype=np.float32)
        Rt[:3, :3] = self.R
        Rt[:3, 3] = self.T
        self.world_view_transform = torch.tensor(Rt).transpose(0, 1).cuda()
        self.projection_matrix = (
            projection_matrix(znear=0.01, zfar=100.0, fov_x=self.FoVx, fov_y=self.FoVy)
            .transpose(0, 1)
            .cuda()
        )
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0)
            .bmm(self.projection_matrix.unsqueeze(0))
            .squeeze(0)
        )
        self.camera_center = self.world_view_transform.inverse()[3, :3]


def make_camera(R, T, K, width, height, uid=0):
    return Camera(R, T, K, width, height, uid=uid)


def prefilter_anchors(gaussians, cam):
    """Return a boolean mask (N,) of anchors visible from cam."""
    gaussians.set_anchor_mask(cam.camera_center, cam.resolution_scale)
    try:
        return _prefilter(cam, gaussians).squeeze()
    except Exception:
        return gaussians._anchor_mask


def render_rgba(gaussians, cam, pipe_config, bg_white=True, object_label_id=None, exclude_label_id=None, training=False):
    """Render gaussians from cam. Returns {'rgb': (3,H,W) cuda, 'alpha': (H,W) cuda}."""
    bg = torch.full((3,), 1.0 if bg_white else 0.0, dtype=torch.float32, device="cuda")

    gaussians.set_anchor_mask(cam.camera_center, cam.resolution_scale)
    if getattr(pipe_config, "add_prefilter", False):
        try:
            visible_mask = _prefilter(cam, gaussians).squeeze()
        except Exception:
            visible_mask = gaussians._anchor_mask
    else:
        visible_mask = gaussians._anchor_mask

    object_mask = None
    if object_label_id is not None:
        object_mask = (gaussians.label_ids.squeeze() == int(object_label_id))
    elif exclude_label_id is not None:
        object_mask = (gaussians.label_ids.squeeze() != int(exclude_label_id))

    pkg = _render(cam, gaussians, pipe_config, bg,
                  visible_mask=visible_mask, training=bool(training),
                  object_mask=object_mask)

    rgb = torch.clamp(pkg["render"], 0.0, 1.0)
    alpha_raw = pkg["render_alphas"]
    alpha = alpha_raw[0] if alpha_raw.ndim == 3 else alpha_raw
    return {"rgb": rgb, "alpha": alpha}
