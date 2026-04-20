#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from typing import NamedTuple
import torch.nn as nn
import torch
from torch import Tensor
from typing import Optional, Tuple, Dict
from . import _C

def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)

def rasterize_gaussians(
    means3D,
    means2D,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    raster_settings,
):
    return _RasterizeGaussians.apply(
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    )

class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    ):

        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg, 
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.debug
        )

        # Invoke C++/CUDA rasterizer
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                num_rendered, color, depth, radii, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            num_rendered, color, depth, radii, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer)
        return color, radii, depth

    @staticmethod
    def backward(ctx, grad_out_color, grad_radii, grad_depth):

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (raster_settings.bg,
                means3D, 
                radii, 
                colors_precomp, 
                scales, 
                rotations, 
                raster_settings.scale_modifier, 
                cov3Ds_precomp, 
                raster_settings.viewmatrix, 
                raster_settings.projmatrix, 
                raster_settings.tanfovx, 
                raster_settings.tanfovy, 
                grad_out_color,
                grad_depth,
                sh, 
                raster_settings.sh_degree, 
                raster_settings.campos,
                geomBuffer,
                num_rendered,
                binningBuffer,
                imgBuffer,
                raster_settings.debug)

        # Compute gradients for relevant tensors by invoking backward method
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
             grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)

        # C++ Backend P=0 Gradient Bug: If P=0, dL_dout_color has 3 channels because forward outputted 3 channels.
        # This causes grad_colors_precomp to be born as [0, 3] inside C++. But PyTorch autograd demands [0, 16].
        # We must manually pacify autograd by cleanly buffering the missing channels.
        if num_rendered == 0 and grad_colors_precomp is not None and colors_precomp is not None:
            if grad_colors_precomp.shape[1] < colors_precomp.shape[1]:
                padding = torch.zeros(grad_colors_precomp.shape[0], colors_precomp.shape[1] - grad_colors_precomp.shape[1], device=grad_colors_precomp.device)
                grad_colors_precomp = torch.cat([grad_colors_precomp, padding], dim=1)

        grads = (
            grad_means3D,
            grad_means2D,
            grad_sh,
            grad_colors_precomp,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            None,
        )

        return grads

class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int 
    tanfovx : float
    tanfovy : float
    bg : torch.Tensor
    scale_modifier : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    sh_degree : int
    campos : torch.Tensor
    prefiltered : bool
    debug : bool

class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean 
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)
            
        return visible

    def forward(self, means3D, means2D, opacities, shs = None, colors_precomp = None, scales = None, rotations = None, cov3D_precomp = None):
        
        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')
        
        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')
        
        if shs is None:
            shs = torch.Tensor([]).cuda()
        if colors_precomp is None:
            colors_precomp = torch.Tensor([]).cuda()

        if scales is None:
            scales = torch.Tensor([]).cuda()
        if rotations is None:
            rotations = torch.Tensor([]).cuda()
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([]).cuda()
        

        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            means3D,
            means2D,
            shs,
            colors_precomp,
            opacities,
            scales, 
            rotations,
            cov3D_precomp,
            raster_settings, 
        )


import math
import torch.nn.functional as F

# Supported template sizes for feature chunking
_SUPPORTED_CHANNELS = [1, 3, 4, 8, 16, 32]

def _next_supported(n) -> int:
    """Find the smallest supported channel count >= n. 
    If n is larger than the largest supported channel count, return the largest supported channel count.
    """
    for s in _SUPPORTED_CHANNELS:
        if s >= n:
            return s
    return _SUPPORTED_CHANNELS[-1]

def _get_projection_matrix(znear, zfar, fovX, fovY, device):
    """Build OpenGL-style projection matrix from FoV."""
    tanHalfFovY = math.tan(fovY / 2)
    tanHalfFovX = math.tan(fovX / 2)
    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4, device=device)
    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = 1.0
    P[2, 2] = zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def _depth_to_normal(depth, viewmats_inv, Ks):
    """Compute surface normals from depth map via finite differences."""
    # depth: [1, H, W, 1], viewmats_inv: [1, 4, 4], Ks: [1, 3, 3]
    H, W = depth.shape[1], depth.shape[2]
    K = Ks[0]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    
    d = depth[0, :, :, 0]  # [H, W]
    
    # Create pixel grid
    u = torch.arange(W, device=depth.device, dtype=depth.dtype)
    v = torch.arange(H, device=depth.device, dtype=depth.dtype)
    u, v = torch.meshgrid(u, v, indexing='xy')
    
    # Unproject to camera space
    x = (u - cx) / fx * d
    y = (v - cy) / fy * d
    pts = torch.stack([x, y, d], dim=-1)  # [H, W, 3]
    
    # Finite differences
    dx = pts[1:-1, 2:] - pts[1:-1, :-2]  # horizontal
    dy = pts[2:, 1:-1] - pts[:-2, 1:-1]  # vertical
    normal = torch.cross(dx, dy, dim=-1)
    normal = F.normalize(normal, dim=-1)
    
    # Pad back to full size
    normal = F.pad(normal, (0, 0, 1, 1, 1, 1), mode='constant', value=0)
    
    # Transform to world space
    R_inv = viewmats_inv[0, :3, :3]
    normal = torch.einsum('ij,hwj->hwi', R_inv, normal)
    
    return normal.unsqueeze(0)  # [1, H, W, 3]

def depth_to_points(
    depths: Tensor, camtoworlds: Tensor, Ks: Tensor, z_depth: bool = True
) -> Tensor:
    """Convert depth maps to 3D points

    Args:
        depths: Depth maps [..., H, W, 1]
        camtoworlds: Camera-to-world transformation matrices [..., 4, 4]
        Ks: Camera intrinsics [..., 3, 3]
        z_depth: Whether the depth is in z-depth (True) or ray depth (False)

    Returns:
        points: 3D points in the world coordinate system [..., H, W, 3]
    """
    assert depths.shape[-1] == 1, f"Invalid depth shape: {depths.shape}"
    assert camtoworlds.shape[-2:] == (
        4,
        4,
    ), f"Invalid viewmats shape: {camtoworlds.shape}"
    assert Ks.shape[-2:] == (3, 3), f"Invalid Ks shape: {Ks.shape}"
    assert (
        depths.shape[:-3] == camtoworlds.shape[:-2] == Ks.shape[:-2]
    ), f"Shape mismatch! depths: {depths.shape}, viewmats: {camtoworlds.shape}, Ks: {Ks.shape}"

    device = depths.device
    height, width = depths.shape[-3:-1]

    x, y = torch.meshgrid(
        torch.arange(width, device=device),
        torch.arange(height, device=device),
        indexing="xy",
    )  # [H, W]

    fx = Ks[..., 0, 0]  # [...]
    fy = Ks[..., 1, 1]  # [...]
    cx = Ks[..., 0, 2]  # [...]
    cy = Ks[..., 1, 2]  # [...]

    # camera directions in camera coordinates
    camera_dirs = F.pad(
        torch.stack(
            [
                (x - cx[..., None, None] + 0.5) / fx[..., None, None],
                (y - cy[..., None, None] + 0.5) / fy[..., None, None],
            ],
            dim=-1,
        ),
        (0, 1),
        value=1.0,
    )  # [..., H, W, 3]

    # ray directions in world coordinates
    directions = torch.einsum(
        "...ij,...hwj->...hwi", camtoworlds[..., :3, :3], camera_dirs
    )  # [..., H, W, 3]
    origins = camtoworlds[..., :3, -1]  # [..., 3]

    if not z_depth:
        directions = F.normalize(directions, dim=-1)

    points = origins[..., None, None, :] + depths * directions
    return points

def depth_to_normal(
    depths: Tensor, camtoworlds: Tensor, Ks: Tensor, z_depth: bool = True
) -> Tensor:
    """Convert depth maps to surface normals

    Args:
        depths: Depth maps [..., H, W, 1]
        camtoworlds: Camera-to-world transformation matrices [..., 4, 4]
        Ks: Camera intrinsics [..., 3, 3]
        z_depth: Whether the depth is in z-depth (True) or ray depth (False)

    Returns:
        normals: Surface normals in the world coordinate system [..., H, W, 3]
    """
    points = depth_to_points(depths, camtoworlds, Ks, z_depth=z_depth)  # [..., H, W, 3]
    dx = torch.cat(
        [points[..., 2:, 1:-1, :] - points[..., :-2, 1:-1, :]], dim=-3
    )  # [..., H-2, W-2, 3]
    dy = torch.cat(
        [points[..., 1:-1, 2:, :] - points[..., 1:-1, :-2, :]], dim=-2
    )  # [..., H-2, W-2, 3]
    normals = F.normalize(torch.cross(dx, dy, dim=-1), dim=-1)  # [..., H-2, W-2, 3]
    normals = F.pad(normals, (0, 0, 1, 1, 1, 1), value=0.0)  # [..., H, W, 3]
    return normals

def rasterization_2dgs(
    means,          # [N, 3]
    quats,          # [N, 4]
    scales,         # [N, 3] — only first 2 used
    opacities,      # [N]
    colors,         # [N, D] or [N, K, 3] for SH
    viewmats,       # [C, 4, 4]
    Ks,             # [C, 3, 3]
    width,
    height,
    near_plane=0.01,
    far_plane=100.0,
    backgrounds=None,  # [C, D]
    packed=False,
    sh_degree=None,
    render_mode="RGB",
    features=None,     # [N, F] — per-Gaussian semantic features
    tile_size=16,      # unused (hardcoded to 16 in diff-surfel)
    **kwargs,
):
    """
    gsplat-compatible wrapper around diff-surfel-rasterization.
    
    Returns:
        render_colors, render_alphas, render_normals, render_normals_from_depth,
        render_distort, render_median, render_features, info
    """
    device = means.device
    C = len(viewmats)  # number of cameras (usually 1)
    
    # Normalize quats (diff-surfel doesn't do this internally)
    quats = F.normalize(quats, dim=-1).contiguous()
    # diff-surfel CUDA kernel reads scales as glm::vec2 (stride-2).
    # Must pass exactly (N, 2)
    scales_2d = scales[:, :2].contiguous()
    
    all_render_colors = []
    all_render_features = []
    all_infos = []
    
    for cid in range(C):
        K = Ks[cid]
        fx, fy = K[0, 0].item(), K[1, 1].item()
        cx, cy = K[0, 2].item(), K[1, 2].item()
        
        # K → FoV
        FoVx = 2 * math.atan(width / (2 * fx))
        FoVy = 2 * math.atan(height / (2 * fy))
        tanfovx = math.tan(FoVx * 0.5)
        tanfovy = math.tan(FoVy * 0.5)
        
        # Build projection matrix
        world_view_transform = viewmats[cid].transpose(0, 1)
        projection_matrix = _get_projection_matrix(
            near_plane, far_plane, FoVx, FoVy, device
        ).transpose(0, 1)
        full_proj_transform = (
            world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))
        ).squeeze(0)
        camera_center = world_view_transform.inverse()[3, :3]
        
        background = (
            backgrounds[cid] if backgrounds is not None
            else torch.zeros(3, device=device)
        )
        
        raster_settings = GaussianRasterizationSettings(
            image_height=height,
            image_width=width,
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=background,
            scale_modifier=1.0,
            viewmatrix=world_view_transform,
            projmatrix=full_proj_transform,
            sh_degree=0 if sh_degree is None else sh_degree,
            campos=camera_center,
            prefiltered=False,
            debug=False,
        )
        
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)
        means2D = torch.zeros_like(means, requires_grad=True, device=device)
        
        # Determine colors
        if sh_degree is not None and colors.dim() == 3:
            shs = colors
            colors_precomp = None
        else:
            shs = None
            colors_precomp = colors
        
        render_feat = None
        use_single_pass = (
            features is not None 
            and shs is None 
            and colors_precomp is not None
            and (3 + features.shape[-1] <= _SUPPORTED_CHANNELS[-1])
        )
        
        # force standard fallback dual-pass. diff-surfel 2DGS kernel math fundamentally destructs normal/distort geometry matrices for C > 3.
        # --- 1. Rasterize RGB Natively (C=3) ---
        color, radii, allmap = rasterizer(
            means3D=means,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=(opacities[:, None] if opacities.dim() == 1 else opacities).contiguous(),
            scales=scales_2d,
            rotations=quats,
        )
        
        # --- 2. Rasterize Features ---
        render_feat = None
        if features is not None:
            # Drop fragmentations
            torch.cuda.empty_cache() 
            
            F_dim = features.shape[-1]
            padded_c = _next_supported(F_dim)
            
            chunk = features
            if padded_c > F_dim:
                chunk = torch.cat([chunk, torch.zeros(chunk.shape[0], padded_c - F_dim, device=device)], dim=-1).detach().contiguous()
                
            bg_feat = torch.zeros(padded_c, device=device)
            feat_settings = GaussianRasterizationSettings(
                image_height=height,
                image_width=width,
                tanfovx=tanfovx,
                tanfovy=tanfovy,
                bg=bg_feat,
                scale_modifier=1.0,
                viewmatrix=world_view_transform,
                projmatrix=full_proj_transform,
                sh_degree=0,
                campos=camera_center,
                prefiltered=False,
                debug=False,
            )
            feat_rasterizer = GaussianRasterizer(raster_settings=feat_settings)
            
            # Evaluate geometry gradients cleanly isolated from allmap regularizers
            feat_color, _, _ = feat_rasterizer(
                means3D=means,
                means2D=means2D,
                shs=None,
                colors_precomp=chunk,
                opacities=(opacities[:, None] if opacities.dim() == 1 else opacities).contiguous(),
                scales=scales_2d,
                rotations=quats,
            )
            
            # P=0 Pad fix natively for the feature return
            if feat_color.shape[0] < padded_c:
                padding = torch.zeros(padded_c - feat_color.shape[0], height, width, device=device)
                feat_color = torch.cat([feat_color, padding], dim=0)
                
            render_feat = feat_color[:F_dim]
                
        all_render_colors.append(color)
        all_render_features.append(render_feat)
        all_infos.append({'radii': radii, 'means2d': means2D, 'allmap': allmap})
    
    # Unpack allmap [7, H, W] → individual outputs (using camera 0)
    allmap = all_infos[0]['allmap']  # [7, H, W]
    render_depth = allmap[0:1]           # [1, H, W]
    render_alpha = allmap[1:2]           # [1, H, W]
    render_normal = allmap[2:5]          # [3, H, W]
    render_median = allmap[5:6]          # [1, H, W]
    render_distort = allmap[6:7]         # [1, H, W]
    
    # Convert to gsplat output format: [1, H, W, C]
    render_colors_out = all_render_colors[0].unsqueeze(0).permute(0, 2, 3, 1)  # [1, H, W, 3]
    
    # Add depth to colors if render_mode requires it
    if render_mode in ["RGB+D", "RGB+ED"]:
        depth_hw = render_depth.permute(1, 2, 0).unsqueeze(0)  # [1, H, W, 1]
        if render_mode == "RGB+ED":
            depth_hw = depth_hw / render_alpha.permute(1, 2, 0).unsqueeze(0).clamp(min=1e-10)
        render_colors_out = torch.cat([render_colors_out, depth_hw], dim=-1)
    
    render_alphas_out = render_alpha.permute(1, 2, 0).unsqueeze(0)  # [1, H, W, 1]
    
    # Normals: camera space → world space
    R_cam = viewmats[0, :3, :3].T
    render_normals_cam = render_normal.permute(1, 2, 0)  # [H, W, 3]
    render_normals_out = (render_normals_cam @ R_cam).unsqueeze(0)  # [1, H, W, 3]
    
    # Normals from depth
    depth_for_normal = render_depth.permute(1, 2, 0).unsqueeze(0)  # [1, H, W, 1]
    if render_mode == "RGB+ED":
        depth_for_normal = depth_for_normal / render_alphas_out.clamp(min=1e-10)
    render_normals_from_depth = _depth_to_normal(
        depth_for_normal, torch.linalg.inv(viewmats), Ks
    )
    
    render_distort_out = render_distort.permute(1, 2, 0).unsqueeze(0)  # [1, H, W, 1]
    render_median_out = render_median.permute(1, 2, 0).unsqueeze(0)    # [1, H, W, 1]
    
    # Features: [F, H, W] → [1, H, W, F]
    render_features_out = None
    if all_render_features[0] is not None:
        render_features_out = all_render_features[0].permute(1, 2, 0).unsqueeze(0)
    
    info = {
        'radii': all_infos[0]['radii'].unsqueeze(0),
        'means2d': all_infos[0]['means2d'],   # leaf tensor — .grad populated during backward
    }
    
    return (
        render_colors_out,
        render_alphas_out,
        render_normals_out,
        render_normals_from_depth,
        render_distort_out,
        render_median_out,
        render_features_out,
        info,
    )


def rasterization_2dgs_inria_wrapper(
    means: Tensor,  # [N, 3]
    quats: Tensor,  # [N, 4]
    scales: Tensor,  # [N, 3]
    opacities: Tensor,  # [N]
    colors: Tensor,  # [N, D] or [N, K, 3]
    viewmats: Tensor,  # [C, 4, 4]
    Ks: Tensor,  # [C, 3, 3]
    width: int,
    height: int,
    near_plane: float = 0.01,
    far_plane: float = 100.0,
    eps2d: float = 0.3,
    sh_degree: Optional[int] = None,
    backgrounds: Optional[Tensor] = None,
    depth_ratio: int = 0,
    **kwargs,
) -> Tuple[Tuple, Dict]:
    """Wrapper for 2DGS's rasterization backend which is based on Inria's backend.

    Install the 2DGS rasterization backend from
        https://github.com/hbb1/diff-surfel-rasterization

    Credit to Jeffrey Hu https://github.com/jefequien

    """
    from diff_surfel_rasterization import (
        GaussianRasterizationSettings,
        GaussianRasterizer,
    )

    assert eps2d == 0.3, "This is hard-coded in CUDA to be 0.3"
    C = len(viewmats)
    device = means.device
    channels = colors.shape[-1]

    # rasterization from inria does not do normalization internally
    quats = F.normalize(quats, dim=-1).contiguous()  # [N, 4]
    scales = scales[:, :2].contiguous()  # [N, 2]

    render_colors = []
    for cid in range(C):
        FoVx = 2 * math.atan(width / (2 * Ks[cid, 0, 0].item()))
        FoVy = 2 * math.atan(height / (2 * Ks[cid, 1, 1].item()))
        tanfovx = math.tan(FoVx * 0.5)
        tanfovy = math.tan(FoVy * 0.5)

        world_view_transform = viewmats[cid].transpose(0, 1)
        projection_matrix = _get_projection_matrix(
            znear=near_plane, zfar=far_plane, fovX=FoVx, fovY=FoVy, device=device
        ).transpose(0, 1)
        full_proj_transform = (
            world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))
        ).squeeze(0)
        camera_center = world_view_transform.inverse()[3, :3]

        means2D = torch.zeros_like(means, requires_grad=True, device=device)

        render_colors_ = []
        radii = None
        allmap = None

        # Create the rasterizer
        stride = _next_supported(channels)
        if backgrounds is not None:
            bg = backgrounds[cid]
            if bg.shape[0] < stride:
                bg = torch.cat([bg, torch.zeros(stride - bg.shape[0], device=device)])
        else:
            bg = torch.zeros(stride, device=device)

        raster_settings = GaussianRasterizationSettings(
            image_height=height,
            image_width=width,
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg,
            scale_modifier=1.0,
            viewmatrix=world_view_transform,
            projmatrix=full_proj_transform,
            sh_degree=0 if sh_degree is None else sh_degree,
            campos=camera_center,
            prefiltered=False,
            debug=False,
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        color_idx = 0
        while color_idx < channels:
            # stride = _next_supported(channels - color_idx)
            _colors = colors[..., color_idx : color_idx + stride]
            shape_before = _colors.shape
            if _colors.shape[-1] < stride:
                pad = torch.zeros(
                    _colors.shape[0], stride - _colors.shape[-1], device=device
                ).detach()
                _colors = torch.cat([_colors, pad], dim=-1)

            

            # Call the rasterizer
            if color_idx == 0:
                # This is the first time we call the rasterizer, so we need to get the radii and allmap
                _render_colors_, radii, allmap = rasterizer(
                    means3D=means,
                    means2D=means2D,
                    shs=_colors if colors.dim() == 3 else None,
                    colors_precomp=_colors if colors.dim() == 2 else None,
                    opacities=opacities[:, None].contiguous(),
                    scales=scales,
                    rotations=quats,
                    cov3D_precomp=None,
                )
            else:
                # We don't need the radii and allmap for the subsequent calls
                # torch.cuda.empty_cache() # Drop fragmentations
                _render_colors_, _, _ = rasterizer(
                    means3D=means,
                    means2D=means2D,
                    shs=_colors if colors.dim() == 3 else None,
                    colors_precomp=_colors if colors.dim() == 2 else None,
                    opacities=opacities[:, None].contiguous(),
                    scales=scales,
                    rotations=quats,
                    cov3D_precomp=None,
                )
            if shape_before[-1] < stride:
                _render_colors_ = _render_colors_[:shape_before[-1], :, :]
            render_colors_.append(_render_colors_)
            color_idx += stride

        # render_colors_ is [#passes, pass's stride, H, W]
        render_colors_ = torch.cat(render_colors_, dim=0) # [channels, H, W]
        render_colors_ = render_colors_.permute(1, 2, 0)  # [H, W, channels]
        render_colors.append(render_colors_)
    render_colors = torch.stack(render_colors, dim=0)

    # additional maps
    allmap = allmap.permute(1, 2, 0).unsqueeze(0)  # [1, H, W, C]
    render_depth_expected = allmap[..., 0:1]
    render_alphas = allmap[..., 1:2]
    render_normal = allmap[..., 2:5]
    render_depth_median = allmap[..., 5:6]
    render_dist = allmap[..., 6:7]

    render_normal = render_normal @ (world_view_transform[:3, :3].T)
    render_depth_expected = render_depth_expected / render_alphas
    render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)
    render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)

    # render_depth is either median or expected by setting depth_ratio to 1 or 0
    # for bounded scene, use median depth, i.e., depth_ratio = 1;
    # for unbounded scene, use expected depth, i.e., depth_ratio = 0, to reduce disk aliasing.
    render_depth = (
        render_depth_expected * (1 - depth_ratio) + (depth_ratio) * render_depth_median
    )

    normals_surf = depth_to_normal(render_depth, torch.linalg.inv(viewmats), Ks)
    normals_surf = normals_surf * (render_alphas).detach()

    render_colors = torch.cat([render_colors, render_depth], dim=-1)

    meta = {
        "normals_rend": render_normal,
        "normals_surf": normals_surf,
        "render_distloss": render_dist,
        "means2d": means2D,
        "width": width,
        "height": height,
        "radii": radii.unsqueeze(0),
        "n_cameras": C,
        "gaussian_ids": None,
    }
    return (render_colors, render_alphas), meta