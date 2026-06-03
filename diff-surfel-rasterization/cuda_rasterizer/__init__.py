import torch.nn as nn
import torch
import torch.nn.functional as F
from torch import Tensor

import math
from typing import NamedTuple
from typing import Optional, Tuple, Dict

from . import _C
from .utils import cpu_deep_copy_tuple, _compute_cam2clip_mat, _depth_to_normal

# Helper functions

# Supported template sizes for feature chunking
_SUPPORTED_CHANNELS = [1, 3, 4, 8, 16, 32]


def _next_supported(n: int) -> int:
    """
    Find the smallest supported channel count >= n.
    If n is larger than the largest supported channel count, return the largest supported channel count.
    """
    for s in _SUPPORTED_CHANNELS:
        if s >= n:
            return s
    return _SUPPORTED_CHANNELS[-1]


class _SurfelRasterizationSettings(NamedTuple):
    """
    Surfel rasterizer settings.
    """

    glob_scale_mod: float
    w2cam_mat: Tensor
    w2clip_mat: Tensor
    img_W: int
    img_H: int
    debug: bool


# Autograd Functions


class _RasterizerFirstPass(torch.autograd.Function):
    """
    First pass autograd function.
    """

    @staticmethod
    @torch.cuda.amp.custom_fwd(
        cast_inputs=torch.float32
    )  # Ensure receieved inputs are float32
    def forward(
        ctx,
        points_world_space,
        scale_vecs,
        quats,
        opacities,
        colors_feat,
        background,
        trick_projected_centers,  # for tensor leaf trick (unused intentionally)
        raster_settings,
    ):
        # Prepare the args for CUDA rasterizer
        args = (
            points_world_space,
            scale_vecs,
            raster_settings.glob_scale_mod,
            quats,
            opacities,
            raster_settings.w2cam_mat,
            raster_settings.w2clip_mat,
            raster_settings.img_W,
            raster_settings.img_H,
            colors_feat,
            background,
            raster_settings.debug,
        )

        # Call rasterizer
        (
            n_isects,
            rendered_color_feat,
            rendered_aux,
            projected_centers,
            asymmetric_radii,
            splat2pix_mats,
            normal_opacity,
            sorted_surfel_indices,
            tile_ranges,
            contrib_state,
            transmittance_and_moments,
        ) = _C.rasterize_surfels_fwd(*args)

        # Save backward pass context
        ctx.raster_settings = raster_settings
        ctx.n_isects = n_isects
        ctx.save_for_backward(
            # Inputs
            colors_feat,
            background,
            points_world_space,
            scale_vecs,
            quats,
            # Preprocessing
            projected_centers,
            asymmetric_radii,
            splat2pix_mats,
            normal_opacity,
            # Binning
            sorted_surfel_indices,
            # Image
            tile_ranges,
            contrib_state,
            transmittance_and_moments,
        )

        return (
            rendered_color_feat,
            rendered_aux,
            projected_centers,
            asymmetric_radii,
            splat2pix_mats,
            normal_opacity,
            sorted_surfel_indices,
            tile_ranges,
            contrib_state,
            transmittance_and_moments,
        )

    @staticmethod
    @torch.cuda.amp.custom_bwd  # Ensure receieved inputs are of same precision as fwd
    def backward(
        ctx,
        grad_rendered_color_feat,  # Image rendering loss
        grad_rendered_aux,  # Aux outputs gradients
        dumm_grad_projected_centers,  # None (doesn't carry gradients but was a fwd pass output)
        dumm_grad_asymmetric_radii,  # None
        dumm_grad_splat2pix_mats,  # None
        dumm_grad_normal_opacity,  # None
        dumm_grad_sorted_indices,  # None
        dumm_grad_tile_ranges,  # None
        dumm_grad_contrib_state,  # None
        dumm_grad_transmittance,  # None
    ):
        # Unpack saved context
        raster_settings = ctx.raster_settings
        n_isects = ctx.n_isects
        (
            # Inputs
            colors_feat,
            background,
            points_world_space,
            scale_vecs,
            quats,
            # Preprocessing
            projected_centers,
            asymmetric_radii,
            splat2pix_mats,
            normal_opacity,
            # Binning
            sorted_surfel_indices,
            # Image
            tile_ranges,
            contrib_state,
            transmittance_and_moments,
        ) = ctx.saved_tensors

        # Prepare the args for CUDA rasterizer
        args = (
            # Forward pass saved state
            points_world_space,
            scale_vecs,
            raster_settings.glob_scale_mod,
            quats,
            raster_settings.w2cam_mat,
            raster_settings.w2clip_mat,
            raster_settings.img_W,
            raster_settings.img_H,
            colors_feat,
            background,
            # Saved forward pass buffers
            # Preprocess buffers
            projected_centers,
            asymmetric_radii,
            splat2pix_mats,
            normal_opacity,
            # Binning buffers
            sorted_surfel_indices,
            # Image
            tile_ranges,
            contrib_state,
            transmittance_and_moments,
            # Input gradients
            grad_rendered_color_feat,
            grad_rendered_aux,
            raster_settings.debug,
        )

        # Call rasterizer
        (
            grad_points_world_space,
            grad_scale_vecs,
            grad_quats,
            grad_projected_centers,
            grad_splat2pix_mats,
            grad_opacity,
            grad_colors_feat,
        ) = _C.rasterize_surfels_bwd(*args)

        # Ensure that the order of returning outputs is the same
        # as the order of the forward function's inputs.
        # Regarding the tensor leaf trick, now trick_projected_centers
        # actually has gradients.
        return (
            grad_points_world_space,
            grad_scale_vecs,
            grad_quats,
            grad_opacity,
            grad_colors_feat,
            None,  # background
            grad_projected_centers,  # tensor leaf trick
            None,  # raster settings
        )


class _RasterizerSubsequent(torch.autograd.Function):
    """
    Subsequent passes autograd function.
    """

    @staticmethod
    @torch.cuda.amp.custom_fwd(
        cast_inputs=torch.float32
    )  # Ensure receieved inputs are float32
    def forward(
        ctx,
        colors_feat,
        background,
        projected_centers,
        splat2pix_mats,
        normal_opacity,
        sorted_surfel_indices,
        tile_ranges,
        contrib_state,
        transmittance_and_moments,
        rendered_aux,
        raster_settings,
    ):
        # Prepare the args for CUDA rasterizer
        args = (
            raster_settings.img_W,
            raster_settings.img_H,
            colors_feat,
            background,
            # Saved from first pass
            projected_centers,
            splat2pix_mats,
            normal_opacity,
            sorted_surfel_indices,
            tile_ranges,
            # Re-used output buffers
            contrib_state,
            transmittance_and_moments,
            rendered_aux,
            raster_settings.debug,
        )

        # Call rasterizer
        rendered_color_feat = _C.rasterize_surfels_fwd_subsequent(*args)

        # Save backward pass context
        ctx.raster_settings = raster_settings
        ctx.save_for_backward(
            # Inputs
            colors_feat,
            background,
            # Preprocessing
            projected_centers,
            splat2pix_mats,
            normal_opacity,
            # Binning
            sorted_surfel_indices,
            # Image
            tile_ranges,
            contrib_state,
            transmittance_and_moments,
        )

        return rendered_color_feat

    @staticmethod
    @torch.cuda.amp.custom_bwd  # Ensure receieved inputs are of same precision as fwd
    def backward(
        ctx,
        grad_rendered_color_feat,  # Image rendering loss
    ):
        # Unpack saved context
        raster_settings = ctx.raster_settings
        (
            # Inputs
            colors_feat,
            background,
            # Preprocessing
            projected_centers,
            splat2pix_mats,
            normal_opacity,
            # Binning
            sorted_surfel_indices,
            # Image
            tile_ranges,
            contrib_state,
            transmittance_and_moments,
        ) = ctx.saved_tensors

        # First pass and subsequent are still coupled.
        # We need to pass a dummy tensor to hold gradients for aux outputs.
        grad_rendered_aux = torch.zeros(
            (7, raster_settings.img_H, raster_settings.img_W),
            device=grad_rendered_color_feat.device,
        )

        # Prepare the args for CUDA rasterizer
        args = (
            # Forward pass saved state
            raster_settings.img_W,
            raster_settings.img_H,
            colors_feat,
            background,
            # Saved forward pass buffers
            # Preprocess buffers
            projected_centers,
            splat2pix_mats,
            normal_opacity,
            # Binning buffers
            sorted_surfel_indices,
            # Image
            tile_ranges,
            contrib_state,
            transmittance_and_moments,
            # Input gradients
            grad_rendered_color_feat,
            grad_rendered_aux,
            raster_settings.debug,
        )

        # Call rasterizer
        (
            grad_points_world_space,
            grad_scale_vecs,
            grad_quats,
            grad_projected_centers,
            grad_splat2pix_mats,
            grad_opacity,
            grad_colors_feat,
        ) = _C.rasterize_surfels_bwd_subsequent(*args)

        # Ensure that the order of returning outputs is the same
        # as the order of the forward function's inputs.
        # Regarding the tensor leaf trick, now trick_projected_centers
        # actually has gradients.
        return (
            grad_colors_feat,
            # Rest are dummy gradients
            None,  # background
            None,  # projected_centers
            None,  # splat2pix_mats
            None,  # normal_opacity
            None,  # sorted_surfel_indices
            None,  # tile_ranges
            None,  # contrib_state
            None,  # transmittance_and_moments
            None,  # rendered_aux
            None,  # raster_settings
        )


# Rasterizer Dispatchers


def _rasterize_surfels_first_pass(
    points_world_space,
    scale_vecs,
    quats,
    opacities,
    colors_feat,
    background,
    trick_projected_centers,  # for tensor leaf trick
    raster_settings,
):
    """
    Dispatch rasterizer for the first pass
    """
    # Call forward function for first pass
    return _RasterizerFirstPass.apply(
        points_world_space,
        scale_vecs,
        quats,
        opacities,
        colors_feat,
        background,
        trick_projected_centers,  # for tensor leaf trick
        raster_settings,
    )


def _rasterize_surfels_subsequent(
    colors_feat,
    background,
    projected_centers,
    splat2pix_mats,
    normal_opacity,
    sorted_surfel_indices,
    tile_ranges,
    contrib_state,
    transmittance_and_moments,
    rendered_aux,
    raster_settings,
):
    """
    Dispatch rasterizer for the subsquent passes
    """
    # Call forward function for subsequent passes
    return _RasterizerSubsequent.apply(
        colors_feat,
        background,
        projected_centers,
        splat2pix_mats,
        normal_opacity,
        sorted_surfel_indices,
        tile_ranges,
        contrib_state,
        transmittance_and_moments,
        rendered_aux,
        raster_settings,
    )


class _SurfelRasterizer(nn.Module):
    """
    Main Surfel rasterizer entry point
    """

    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def fwd_first_pass(
        self,
        points_world_space,
        scale_vecs,
        quats,
        opacities,
        colors_feat,
        background,
        trick_projected_centers,  # for tensor leaf trick
    ):
        return _rasterize_surfels_first_pass(
            points_world_space,
            scale_vecs,
            quats,
            opacities,
            colors_feat,
            background,
            trick_projected_centers,  # for tensor leaf trick
            self.raster_settings,
        )

    def fwd_subsequent(
        self,
        colors_feat,
        background,
        projected_centers,
        splat2pix_mats,
        normal_opacity,
        sorted_surfel_indices,
        tile_ranges,
        contrib_state,
        transmittance_and_moments,
        rendered_aux,
    ):
        return _rasterize_surfels_subsequent(
            colors_feat,
            background,
            projected_centers,
            splat2pix_mats,
            normal_opacity,
            sorted_surfel_indices,
            tile_ranges,
            contrib_state,
            transmittance_and_moments,
            rendered_aux,
            self.raster_settings,
        )


# Currently not supporting batched rendering / stero.
# Would require to batch projected_centers and modify the main training loop
# to become compatible.
def rasterize_2dgs(
    points_world_space: Tensor,  # [P, 3]
    quats: Tensor,  # [P, 4]
    scale_vecs: Tensor,  # [P, 2] or [P, 3]
    opacities: Tensor,  # [P]
    colors_feat: Tensor,  # [P, Channels]
    w2cam_mats: Tensor,  # [C, 4, 4]
    cam_intrinsics: Tensor,  # [C, 3, 3]
    img_W: int,
    img_H: int,
    near_plane: float = 0.01,
    far_plane: float = 100.0,
    backgrounds: Optional[Tensor] = None,
    depth_ratio: int = 0,
    **kwargs,
) -> Tuple[Tuple, Dict]:
    """
    High-level CUDA differential 2DGS rasterizer API.
    """
    # Find the number of different cameras used (typically only 1)
    cam_count = len(w2cam_mats)

    # Find the used device (should be CUDA / VRAM)
    device = points_world_space.device

    # Find channel count for the concatenated colors+features
    channel_count = colors_feat.shape[-1]

    # Normalize quaternions and ensure contiguous memory
    quats = F.normalize(quats, dim=-1).contiguous()

    # Ensure scale vectors are 2D (since surfels are 2D) and contiguous
    scale_vecs = scale_vecs[:, :2].contiguous()

    # Safely ensure that opacities dims are [N, 1]
    opacities = opacities.contiguous()
    if opacities.dim() == 1:
        opacities = opacities.unsqueeze(-1)

    # Loop over different cameras (images) and rasterize
    rendered_color_feat = []
    rendered_aux = []
    asymmetric_radii = []

    for cam_id in range(cam_count):
        # Compute camera FOV from camera intrinsics
        fovx = 2 * math.atan(img_W / (2 * cam_intrinsics[cam_id, 0, 0].item()))
        fovy = 2 * math.atan(img_H / (2 * cam_intrinsics[cam_id, 1, 1].item()))

        # Compute different space transformation matrices.
        # Transpose them into col-major order for GLM.
        w2cam_mat = w2cam_mats[cam_id].transpose(0, 1)

        cam2clip_mat = _compute_cam2clip_mat(
            near_plane=near_plane,
            far_plane=far_plane,
            fovx=fovx,
            fovy=fovy,
            device=device,
        ).transpose(0, 1)

        w2clip_mat = w2cam_mat @ cam2clip_mat

        # Create the rasterizer settings object
        raster_settings = _SurfelRasterizationSettings(
            glob_scale_mod=1.0,
            w2cam_mat=w2cam_mat,
            w2clip_mat=w2clip_mat,
            img_W=img_W,
            img_H=img_H,
            debug=False,
        )

        # Create the rasterizer object
        rasterizer = _SurfelRasterizer(raster_settings)

        # Prepare output buffers
        # 1. General Outputs
        rendered_color_feat_ = []
        rendered_aux_ = None

        # 2. Preprocessing Intermediates
        asymmetric_radii_ = None
        splat2pix_mats = None
        normal_opacity = None

        # 3. Binning Intermediates
        sorted_surfel_indices = None

        # 4. Image intermediates
        tile_ranges = None
        contrib_state = None
        transmittance_and_moments = None

        # 5. Tensor leaf trick.
        # Trick pytorch inputting this to the autograd function class
        # causing it to receive the calculated gradients for projected_centers
        # in backward pass.
        trick_projected_centers = torch.zeros(
            (points_world_space.shape[0], 2), requires_grad=True, device=device
        )

        # Rasterize and perform multiple passes if channel count of colors+features requires.
        channel_idx = 0
        while channel_idx < channel_count:
            # Calculate the supported stride for the remaining number of channels
            rem_channels = channel_count - channel_idx
            if rem_channels <= 0:  # Sanity check
                break
            stride = _next_supported(rem_channels)

            # Slice and pad colors+features tensor to fit the supported stride
            _colors_feat = colors_feat[..., channel_idx : channel_idx + stride]
            shape_before = _colors_feat.shape  # Track pre-slice shape
            if _colors_feat.shape[-1] < stride:
                pad = torch.zeros(
                    _colors_feat.shape[0],
                    stride - _colors_feat.shape[-1],
                    device=device,
                ).detach()
                _colors_feat = torch.cat([_colors_feat, pad], dim=-1)

            # Slice and pad background tensor to fit the supported stride
            if backgrounds is not None:
                background = backgrounds[cam_id][channel_idx : channel_idx + stride]
                if background.shape[0] < stride:
                    background = torch.cat(
                        [
                            background,
                            torch.zeros(stride - background.shape[0], device=device),
                        ]
                    )
            else:
                background = torch.zeros(stride, device=device)

            # Call rasterizer
            if channel_idx == 0:
                # Use first-pass methods
                (
                    # 1. General Outputs
                    _rendered_color_feat_,
                    rendered_aux_,
                    projected_centers,
                    # 2. Preprocessing Intermediates
                    asymmetric_radii_,
                    splat2pix_mats,
                    normal_opacity,
                    # 3. Binning Intermediates
                    sorted_surfel_indices,
                    # 4. Image intermediates
                    tile_ranges,
                    contrib_state,
                    transmittance_and_moments,
                ) = rasterizer.fwd_first_pass(
                    points_world_space=points_world_space,
                    scale_vecs=scale_vecs,
                    quats=quats,
                    opacities=opacities,
                    colors_feat=_colors_feat,
                    background=background,
                    trick_projected_centers=trick_projected_centers,  # for tensor leaf trick
                )
            else:
                # Use subsequent-pass methods
                _rendered_color_feat_ = rasterizer.fwd_subsequent(
                    colors_feat=_colors_feat,
                    background=background,
                    # Saved from First Pass
                    projected_centers=projected_centers,
                    splat2pix_mats=splat2pix_mats,
                    normal_opacity=normal_opacity,
                    sorted_surfel_indices=sorted_surfel_indices,
                    tile_ranges=tile_ranges,
                    # Re-used output buffers
                    contrib_state=contrib_state,
                    transmittance_and_moments=transmittance_and_moments,
                    rendered_aux=rendered_aux_,
                )

            # Remove padding from output rendered colors+features
            if shape_before[-1] < stride:
                _rendered_color_feat_ = _rendered_color_feat_[: shape_before[-1], :, :]

            # Store with the rest of rendered colors+features
            rendered_color_feat_.append(_rendered_color_feat_)

            channel_idx += stride

        # Prepare correct dimensions and join all rendered colors+features from all cameras
        # rendered_color_feat_ dimensions: [#passes, pass's stride, H, W]
        rendered_color_feat_ = torch.cat(
            rendered_color_feat_, dim=0
        )  # [channels, H, W]
        rendered_color_feat_ = rendered_color_feat_.permute(1, 2, 0)  # [H, W, channels]

        # Collect outputs from all cameras
        rendered_color_feat.append(rendered_color_feat_)
        rendered_aux.append(rendered_aux_)
        asymmetric_radii.append(asymmetric_radii_)

    # Stack the outputs from all cameras
    rendered_color_feat = torch.stack(
        rendered_color_feat, dim=0
    )  # [cam_count, H, W, channels]
    rendered_aux = torch.stack(
        rendered_aux, dim=0
    )  # [cam_count, AUX_CHANNEL_COUNT, H, W]
    asymmetric_radii = torch.stack(asymmetric_radii, dim=0)  # [cam_count, ...]

    # Split auxiliary render outputs
    rendered_aux = rendered_aux.permute(
        0, 2, 3, 1
    )  # [cam_count, H, W, AUX_CHANNEL_COUNT]
    rendered_expected_depth = rendered_aux[..., 0:1]
    rendered_alphas = rendered_aux[..., 1:2]
    rendered_normal = rendered_aux[..., 2:5]
    rendered_median_depth = rendered_aux[..., 5:6]
    rendered_distortion = rendered_aux[..., 6:7]

    # Rotate normals from camera to world space

    # 1. Extract rotation portion of world 2 cam matrix.
    # Transpose of orthogonal matrix is its inverse
    cam2w_mats = w2cam_mats[:, :3, :3].transpose(1, 2)

    # 2. Multiply each pixel's rotation vector by
    # the cam 2 world space rotation matrix
    rendered_normal = torch.einsum(
        "...ij,...j->...i",
        cam2w_mats.unsqueeze(1).unsqueeze(
            1
        ),  # Shape [C, 1, 1, 3, 3] to broadcast over H, W
        rendered_normal,
    )

    # Gradient masking to prevent cross-pixel gradient leakage and normalization singularities.
    # Threshold caps the gradients to prevent explosion.
    normalized_depth = rendered_expected_depth / rendered_alphas.clamp(min=1e-6)
    _mask = (rendered_alphas > 0.005).float().detach()
    rendered_expected_depth = normalized_depth * _mask + normalized_depth.detach() * (
        1.0 - _mask
    )

    # Ensure no NaN outputs for depths
    rendered_expected_depth = torch.nan_to_num(
        rendered_expected_depth, nan=0.0, posinf=0.0, neginf=0.0
    )
    rendered_median_depth = torch.nan_to_num(
        rendered_median_depth, nan=0.0, posinf=0.0, neginf=0.0
    )

    # Choose between expected depth (unbounded scenes) and median depth (bounded scenes)
    rendered_depth = (
        rendered_expected_depth * (1 - depth_ratio)
        + rendered_median_depth * depth_ratio
    )

    # Compute normals from depth
    normal_from_depth = _depth_to_normal(
        rendered_depth, torch.linalg.inv(w2cam_mats), cam_intrinsics
    )
    normal_from_depth = normal_from_depth * rendered_alphas.detach()

    # Concatenate colors+features with depth for compatibility
    rendered_color_feat_depth = torch.cat([rendered_color_feat, rendered_depth], dim=-1)

    # Compute max radius of ellipsoid per surfel

    # 1. Unpack upper and lower bits x, y radii
    radius_x = asymmetric_radii & 0xFFFF
    radius_y = (asymmetric_radii >> 16) & 0xFFFF

    # 2. Get the max radius per surfel and cast to float
    max_radius = torch.maximum(radius_x, radius_y).float()

    # Prepare outputs
    meta = {
        "normals_rend": rendered_normal,
        "normals_surf": normal_from_depth,
        "render_distloss": rendered_distortion,
        "means2d": trick_projected_centers,  # Tensor leaf trick. Actually has gradients
        "width": img_W,
        "height": img_H,
        "radii": max_radius,
        "n_cameras": cam_count,
        "gaussian_ids": None,
    }

    return (rendered_color_feat_depth, rendered_alphas), meta


def frustum_cull_2dgs(
    points_world_space: Tensor,  # [P, 3]
    quats: Tensor,  # [P, 4]
    scale_vecs: Tensor,  # [P, 2] or [P, 3]
    w2cam_mats: Tensor,  # [C, 4, 4]
    cam_intrinsics: Tensor,  # [C, 3, 3]
    img_W: int,
    img_H: int,
    near_plane: float = 0.01,
    far_plane: float = 100.0,
) -> Tuple[Tensor]:
    """
    High-level CUDA frustum culling API.
    Culls surfels that aren't in the camera's view.
    """

    # Find the number of different cameras used (typically only 1)
    cam_count = len(w2cam_mats)

    # Find the used device (should be CUDA / VRAM)
    device = points_world_space.device

    # Normalize quaternions and ensure contiguous memory
    quats = F.normalize(quats, dim=-1).contiguous()

    # Ensure scale vectors are 2D (since surfels are 2D) and contiguous
    scale_vecs = scale_vecs[:, :2].contiguous()

    # Loop over different cameras and cull surfels

    all_radii = []
    for cam_id in range(cam_count):
        # Compute camera FOV from camera intrinsics
        fovx = 2 * math.atan(img_W / (2 * cam_intrinsics[cam_id, 0, 0].item()))
        fovy = 2 * math.atan(img_H / (2 * cam_intrinsics[cam_id, 1, 1].item()))

        # Compute different space transformation matrices.
        # Transpose them into col-major order for GLM.
        w2cam_mat = w2cam_mats[cam_id].transpose(0, 1)

        cam2clip_mat = _compute_cam2clip_mat(
            near_plane=near_plane,
            far_plane=far_plane,
            fovx=fovx,
            fovy=fovy,
            device=device,
        ).transpose(0, 1)

        w2clip_mat = w2cam_mat @ cam2clip_mat

        # Call surfel culling kernel
        radii = _C.frustum_cull_surfels(
            points_world_space,
            scale_vecs,
            1.0,  # glob_scale_mod
            quats,
            w2cam_mat,
            w2clip_mat,
            img_W,
            img_H,
            False,  # debug
        )
        all_radii.append(radii)

    return (torch.stack(all_radii, dim=0),)  # [C, P]
