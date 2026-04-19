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
    """Find the smallest supported channel count >= n."""
    for s in _SUPPORTED_CHANNELS:
        if s >= n:
            return s
    raise ValueError(f"Feature chunk too large: {n}. Max supported: {_SUPPORTED_CHANNELS[-1]}")

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
    P[3, 2] = -1.0
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
    C = viewmats.shape[0]  # number of cameras (usually 1)
    
    # Normalize quats (diff-surfel doesn't do this internally)
    quats = F.normalize(quats, dim=-1).contiguous()
    # diff-surfel CUDA kernel reads scales as glm::vec2 (stride-2).
    # Must pass exactly (N, 2) contiguous.
    scales_2d = scales[:, :2].contiguous()
    means = means.contiguous()
    colors = colors.contiguous()
    
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
            backgrounds[cid][:3] if backgrounds is not None
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
        means2D.retain_grad()
        
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
        
        if use_single_pass:
            # 1. Pad colors_precomp + features to a supported channel count 
            F_dim = features.shape[-1]
            actual_c = 3 + F_dim
            padded_c = _next_supported(actual_c)
            
            # Combine RGB and detached features (prevent gradients back to semantic decoder)
            combined_chunk = torch.cat([colors_precomp, features.detach()], dim=-1)
            if padded_c > actual_c:
                combined_chunk = torch.cat([
                    combined_chunk,
                    torch.zeros(combined_chunk.shape[0], padded_c - actual_c, device=device)
                ], dim=-1)
                
            # 2. Pad background 
            bg_combined = torch.cat([
                background, 
                torch.zeros(padded_c - 3, device=device)
            ])
            
            # 3. Update settings and reconstruct rasterizer
            raster_settings = GaussianRasterizationSettings(
                image_height=height,
                image_width=width,
                tanfovx=tanfovx,
                tanfovy=tanfovy,
                bg=bg_combined,
                scale_modifier=1.0,
                viewmatrix=world_view_transform,
                projmatrix=full_proj_transform,
                sh_degree=0,
                campos=camera_center,
                prefiltered=False,
                debug=False,
            )
            rasterizer = GaussianRasterizer(raster_settings=raster_settings)
            
            # 4. Rasterize everything in one unified pass
            color_feat, radii, allmap = rasterizer(
                means3D=means,
                means2D=means2D,
                shs=None,
                colors_precomp=combined_chunk,
                opacities=(opacities[:, None] if opacities.dim() == 1 else opacities).contiguous(),
                scales=scales_2d,
                rotations=quats,
            )
            
            # 5. Split output
            color = color_feat[:3]
            render_feat = color_feat[3:3+F_dim]
            
        else:
            # --- Rasterize RGB (Fallback / Standard path) ---
            color, radii, allmap = rasterizer(
                means3D=means,
                means2D=means2D,
                shs=shs,
                colors_precomp=colors_precomp,
                opacities=(opacities[:, None] if opacities.dim() == 1 else opacities).contiguous(),
                scales=scales_2d,
                rotations=quats,
            )
            # color: [3, H, W], allmap: [7, H, W]
            
            # --- Rasterize features in chunks ---
            torch.cuda.empty_cache() # reduce fragmentation before large feature 
            if features is not None:
                F_dim = features.shape[-1]
                chunk_size = _SUPPORTED_CHANNELS[-1]  # 32
                feat_chunks = []
                
                for i in range(0, F_dim, chunk_size):
                    chunk = features[:, i:i+chunk_size].detach()
                    actual_c = chunk.shape[-1]
                    padded_c = _next_supported(actual_c)
                    
                    if padded_c > actual_c:
                        chunk = torch.cat([
                            chunk,
                            torch.zeros(chunk.shape[0], padded_c - actual_c, device=device)
                        ], dim=-1)
                    
                    # Need matching background
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
                    feat_rasterizer = GaussianRasterizer(feat_settings)
                    
                    # Detach geometry parameters here to prevent double gradients!
                    feat_color, _, _ = feat_rasterizer(
                        means3D=means.detach(),
                        means2D=torch.zeros_like(means, device=device),
                        shs=None,
                        colors_precomp=chunk,
                        opacities=(opacities[:, None] if opacities.dim() == 1 else opacities).contiguous().detach(),
                        scales=scales_2d.detach(),
                        rotations=quats.detach(),
                    )
                    # feat_color: [padded_c, H, W] — trim padding
                    feat_chunks.append(feat_color[:actual_c])
                
                render_feat = torch.cat(feat_chunks, dim=0)  # [F_dim, H, W]
                
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
