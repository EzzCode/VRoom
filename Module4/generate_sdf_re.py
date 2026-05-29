import numpy as np
import torch

def generate_tsdf_single_cam(depth_map, intrinsics, extrinsics,
                             grid_shape, voxel_size, trunc_margin, 
                             world_points_h, color_image=None):
    """
    Generates TSDF for a single camera view using PyTorch (GPU-accelerated).
    
    Parameters:
    - depth_map: torch.Tensor, depth map of the current view
    - intrinsics: torch.Tensor, camera intrinsics matrix
    - extrinsics: torch.Tensor, camera extrinsics matrix
    - grid_shape: tuple (Nx, Ny, Nz), shape of the TSDF grid
    - voxel_size: float, size of each voxel
    - trunc_margin: float, truncation margin for TSDF
    - world_points_h: torch.Tensor, Pre-computed (N³, 4) homogeneous world-point matrix.
      Pre-computed once in fuse_tsdf and shared across all cameras to avoid redundant work.
    - color_image: (H, W, 3) torch.Tensor, optional, color image of the current view

    Returns:
    - final_grid: (Nx, Ny, Nz) PyTorch tensor containing the TSDF values for each voxel.
    - color_grid: (Nx, Ny, Nz, 3) PyTorch tensor containing the sampled RGB colors, or None.
    - final_weights: (Nx, Ny, Nz) PyTorch tensor containing the dynamic weights for each voxel.
    """
    
    device = world_points_h.device # ensure all on same device
    # 1. Transform world points to camera space
    cam_points_h = world_points_h @ extrinsics.T  # (N³, 4) x (4, 4) -> (N³, 4)
    # Extract the 3D coordinates from homogeneous coordinates
    cam_points = cam_points_h[:, :3]  # (N³, 3)

    # 2. Extract depth (z)
    corner_depths = cam_points[:, 2]  # (N³,)

    # 3. Transform from camera to image space 
    image_points = cam_points @ intrinsics.T  # (N³, 3) x (3, 3) -> (N³, 3)

    # 4. Divide by depth (Z) to perform perspective projection.
    # This projects the 3D camera-space points onto the 2D image plane,
    # giving the exact (u, v) pixel coordinates where each voxel lands.
    pixel_u = image_points[:, 0] / corner_depths  # (N³,)
    pixel_v = image_points[:, 1] / corner_depths  # (N³,)

    # 5. Filter out blind spots
    H, W = depth_map.shape # get height and width
    # Subtract 1 to avoid errors in bilinear interpolation if pixel
    # coordinates are on the boundary of the image.
    valid_mask = (
                 (pixel_u >= 0) & (pixel_u < W - 1) &
                 (pixel_v >= 0) & (pixel_v < H - 1) &
                 (corner_depths > 0)
                 )
    
    # 6. Bilinear Interpolation
    u_valid = pixel_u[valid_mask]
    v_valid = pixel_v[valid_mask]
    # Find 4 corners for bilinear interpolation

    # Since depth maps are discrete, we need to find the 4 pixel corners
    # surrounding the projected point (u, v) and perform bilinear interpolation.
    # A u,v point in the image will be surrounded by 4 pixels from the depth map:
    #  U0, V0 ---- U1, V0
    #    |           |
    #  U0, V1 ---- U1, V1
  
    # Subtract 2 because we will access u1 and v1 which are +1 from u0 and v0,
    # and its 0-indexed, so we need to ensure u0 and v0 are at least 0 and at
    # most W-2 and H-2 respectively.
    # Clamp to prevent any out-of-bounds CUDA errors (extra safety).
    u0 = torch.clamp(torch.floor(u_valid).long(), 0, W - 2)
    v0 = torch.clamp(torch.floor(v_valid).long(), 0, H - 2)
    u1 = u0 + 1
    v1 = v0 + 1

    # Interpolation weights
    # The weight of a corner is equal to the area of the opposite rectangle
    # formed by the other three corners.
    wa = (u1.float() - u_valid) * (v1.float() - v_valid) # top-left weight
    wb = (u_valid - u0.float()) * (v1.float() - v_valid) # top-right weight
    wc = (u1.float() - u_valid) * (v_valid - v0.float()) # bottom-left weight
    wd = (u_valid - u0.float()) * (v_valid - v0.float()) # bottom-right weight

    # Assign 4 corners
    # v first because depth maps are matrices accessed as row, column (v, u)
    d00 = depth_map[v0, u0] # top-left depth
    d01 = depth_map[v0, u1] # top-right depth
    d10 = depth_map[v1, u0] # bottom-left depth
    d11 = depth_map[v1, u1] # bottom-right depth

    # Calculate sampled_depths: the camera's interpolated depth measurement
    # for the exact pixel where each valid 3D grid point projected. 
    # This represents the actual surface distance, which we will subtract from 
    # the grid point's theoretical distance (corner_depths) to compute the final SDF.
    
    # In simple terms, this is the depth value of the surface seen by the camera,
    # and corner_depths is the depth of the voxel corner in camera space.

    # For points that lie on same line of sight, they will have same sampled_depths
    # because they project to same pixel. Later on, we will calculate the SDF as
    # the difference between corner_depths and sampled_depths.

    # Size is N³ becasue the grid is N x N x N, however, it's flattened for GPU processing.
    # All grid is processed, but points that lie in same line will have same sampled_depths
    # because they project to same pixel.
    # Technically size is V not N³ because we only keep valid points, but for simplicity N³.
    depth_check = (d00 > 0) & (d01 > 0) & (d10 > 0) & (d11 > 0)
    sampled_depths = wa * d00 + wb * d01 + wc * d10 + wd * d11 # (N³,)

    sampled_depths = sampled_depths[depth_check] # (N³,)

    # Create mask that filters out both out-of-bounds and 0-depth pixels
    final_valid_mask = valid_mask.clone()
    final_valid_mask[valid_mask] = depth_check

    # 7. Compute SDF
    valid_cam_points = cam_points[final_valid_mask] # (V, 3)
    valid_corner_depths = corner_depths[final_valid_mask] # (V,)
    # Calculate ray lengths from camera to each valid grid point in camera space.
    ray_lengths = torch.linalg.norm(valid_cam_points, dim=1) # (V,)
    # Use directional cosine for better results.
    directional_cosine = valid_corner_depths / ray_lengths
    # Initialize all voxels with a dummy "empty space" value (1e6)
    # This guarantees that even with a large trunc_margin, unseen voxels will have 1.0 TSDF.
    sdf_values = torch.ones(world_points_h.shape[0], device=device, dtype=torch.float32) * 1e6
    # Compute SDF only for valid points
    sdf_values[final_valid_mask] = (sampled_depths - valid_corner_depths) * directional_cosine

    # 8. Truncation
    valid_sdf = sdf_values[final_valid_mask]
    # Clamp to truncation margin and normalize to [-1, 1]
    tsdf_valid = torch.clamp(valid_sdf, -trunc_margin, trunc_margin) / trunc_margin
    tsdf_values = torch.ones(world_points_h.shape[0], device=device, dtype=torch.float32)
    tsdf_values[final_valid_mask] = tsdf_valid

    # 9. Dynamic Weighting
    # Assign higher weights to measurements taken close to camera
    weights = torch.zeros(world_points_h.shape[0], device=device, dtype=torch.float32)
    # Carving Mask: We only assign weight to empty space (>0) or surface (approx. 0). 
    # We ignore deep interior (-1) because cameras cannot see through solid walls
    carving_mask = final_valid_mask & (tsdf_values > -0.99)
    # Inverse square weighting
    weights[carving_mask] = 1.0 / (corner_depths[carving_mask] ** 2 + 1e-5)

    # 9. Reshape
    final_grid = tsdf_values.view(grid_shape)
    final_weights = weights.view(grid_shape)

    # 10. Color Sampling
    color_grid = None
    if color_image is not None:
        color_values = torch.zeros((world_points_h.shape[0], 3), device=device, dtype=torch.float32)
        
        c00 = color_image[v0, u0]
        c01 = color_image[v0, u1]
        c10 = color_image[v1, u0]
        c11 = color_image[v1, u1]
        
        # unsqueeze(-1) allows multiplying the weights by the color values (3 channels)
        sampled_colors = (
            c00 * wa.unsqueeze(-1) + 
            c01 * wb.unsqueeze(-1) + 
            c10 * wc.unsqueeze(-1) + 
            c11 * wd.unsqueeze(-1)
        )
        
        color_values[final_valid_mask] = sampled_colors[depth_check]
        color_grid = color_values.view(*grid_shape, 3)

    return final_grid, color_grid, final_weights

def fuse_tsdf(depth_maps, intrinsics_list, extrinsics_list, grid_shape, 
              voxel_size, trunc_margin, color_images=None, grid_origin=None):
    """
    Fuses multiple TSDFs from different camera views into a single global TSDF grid
    (GPU-accelerated).

    Parameters:
    - depth_maps: list of 2D numpy arrays, one per camera.
    - intrinsics_list: list of 3x3 numpy arrays, camera intrinsics per camera.
    - extrinsics_list: list of 4x4 numpy arrays, camera extrinsics per camera.
    - grid_shape: tuple (Nx, Ny, Nz), shape of the TSDF voxel grid.
    - voxel_size: float, the size of each voxel in world units.
    - trunc_margin: float, the truncation margin for the TSDF.
    - color_images: (optional) list of (H, W, 3) numpy arrays, one per camera.
    - grid_origin: (optional) (3,) array, world-space corner where the grid starts.

    Returns:
    - fused_tsdf: (Nx, Ny, Nz) numpy array of fused TSDF values.
    - fused_colors: (Nx, Ny, Nz, 3) numpy array of fused RGB colors, or None.
    - obs_count: (Nx, Ny, Nz) numpy array counting how many cameras observed each voxel.
    """