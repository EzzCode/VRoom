import logging
import numpy as np
import torch

from gstrain.vroom_core.utilities.render import apply_frustum_culling as _culling
from gstrain.vroom_core.utilities.render import render as _render
from gstrain.vroom_core.utilities.utils import projection_matrix


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
    try:
        return _culling(cam, gaussians.anchor_cloud, gaussians.gaussian_type).squeeze()
    except Exception:
        return gaussians._anchor_mask


def render_rgba(
    gaussians,
    cam,
    pipe_config,
    bg_white=True,
    object_label_id=None,
    exclude_label_id=None,
    training=False,
    visible_mask=None,
):
    """Render gaussians from cam. Returns {'rgb': (3,H,W) cuda, 'alpha': (H,W) cuda}."""
    bg = torch.full((3,), 1.0 if bg_white else 0.0, dtype=torch.float32, device="cuda")

    if visible_mask is None:
        try:
            visible_mask = _culling(cam, gaussians.anchor_cloud, gaussians.gaussian_type).squeeze()
        except Exception:
            visible_mask = gaussians._anchor_mask

    # Combine with label/object mask if requested
    object_mask = None
    if object_label_id is not None:
        object_mask = (gaussians.label_ids.squeeze() == int(object_label_id))
    elif exclude_label_id is not None:
        object_mask = (gaussians.label_ids.squeeze() != int(exclude_label_id))

    if object_mask is not None:
        visible_mask = visible_mask & object_mask.to(visible_mask.device)

    # Decode anchors into neural Gaussians
    decoded_output = gaussians.decoder.forward_pass(
        anchor_cloud=gaussians.anchor_cloud,
        visible_anchors_mask=visible_mask,
        camera=cam,
    )

    # Prepare positions and rotations
    from gstrain.vroom_core.core.training.orchestration import prepare_gaussian_space_props
    gaussian_positions, normalized_rotations = prepare_gaussian_space_props(
        anchor_cloud=gaussians.anchor_cloud,
        visible_anchors_mask=visible_mask,
        negative_opacity_filter=decoded_output["negative_opacity_filter"],
        rotations_pred=decoded_output["rotations"],
    )

    # Render
    pkg = _render(
        viewpoint_camera=cam,
        decoded_output=decoded_output,
        gaussian_positions=gaussian_positions,
        normalized_rotations=normalized_rotations,
        bg_color=bg,
        gaussian_type=gaussians.gaussian_type,
        render_mode=gaussians.render_mode,
        tile_size_2dgs=gaussians.tile_size_2dgs,
    )

    rgb = torch.clamp(pkg["render"], 0.0, 1.0)
    alpha_raw = pkg["render_alphas"]
    alpha = alpha_raw[0] if alpha_raw.ndim == 3 else alpha_raw

    return_pkg = {"rgb": rgb, "alpha": alpha}
    if training:
        return_pkg.update({
            "render": rgb,
            "render_alphas": alpha_raw,
            "scaling": pkg["scaling"],
            "opacity": pkg["opacity"],
            "rendered_2d_points": pkg["rendered_2d_points"],
            "visible_anchors_mask": visible_mask,
            "negative_opacity_filter": decoded_output["negative_opacity_filter"],
            "render_depth": pkg.get("render_depth"),
        })
        if gaussians.gaussian_type == "2D":
            return_pkg.update({
                "render_normals": pkg.get("render_normals"),
                "render_normals_from_depth": pkg.get("render_normals_from_depth"),
                "render_distort": pkg.get("render_distort"),
            })
    return return_pkg
