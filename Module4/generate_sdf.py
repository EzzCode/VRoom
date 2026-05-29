import numpy as np
import torch

def generate_tsdf_single_camera(depth_map_t, intrinsics_t, extrinsics_t, grid_shape, voxel_size, trunc_margin, color_image_t=None, depth_trunc=None, world_points_h_t=None):
    """
    GPU-Accelerated TSDF generation using PyTorch.
    
    Inputs:
    - depth_map_t: 2D PyTorch tensor of depth values from the camera.
    - intrinsics_t: 3x3 Camera lens matrix (PyTorch tensor).
    - extrinsics_t: 4x4 Camera pose matrix (World-to-Camera, PyTorch tensor).
    - grid_shape: Tuple (Nx, Ny, Nz) for the grid size.
    - voxel_size: Physical size of one voxel in coordinate units.
    - trunc_margin: The maximum +/- distance to cap the SDF (the "T" in TSDF).
    - color_image_t: (optional) (H, W, 3) RGB image tensor from the same camera.
    - depth_trunc: (optional) Maximum depth value to consider. Pixels with depth beyond
      this are ignored, preventing noisy far-away surfaces from corrupting the TSDF.
    - world_points_h_t: (optional) Pre-computed (N³, 4) homogeneous world-point matrix.
      If provided, skips computation (saves time when the same grid is used across multiple cameras)
      
    Returns:
    - final_grid: (Nx, Ny, Nz) PyTorch tensor containing the TSDF values for each voxel.
    - color_grid: (Nx, Ny, Nz, 3) PyTorch tensor containing the sampled RGB colors, or None.
    - final_weights: (Nx, Ny, Nz) PyTorch tensor containing the dynamic weights for each voxel.
    """
    device = world_points_h_t.device

    # Step 1: Extrinsics (World Space to Camera Space)
    # world_points_h_t shape: (N^3, 4)
    # extrinsics_t shape: (4, 4) 
    cam_points = torch.matmul(world_points_h_t, extrinsics_t.T)[:, :3]  # (N³, 3)
    corner_depths = cam_points[:, 2]  # z-values                          (N³,) 

    # Step 2: Intrinsics (Camera Space to Image Space)
    # cam_points shape: (N³, 3)
    # intrinsics_t shape: (3, 3)
    image_points = torch.matmul(cam_points, intrinsics_t.T)             # (N³, 3)

    # Divide by depth (Z) to perform perspective projection.
    # This projects the 3D camera-space points onto the 2D image plane,
    # giving the exact (u, v) pixel coordinates where each voxel lands.
    pixel_u = image_points[:, 0] / image_points[:, 2]
    pixel_v = image_points[:, 1] / image_points[:, 2]

    # Step 3: Filter out blind spots
    H, W = depth_map_t.shape

    # We subtract 1 to avoid errors in bilinear interpolation
    # if pixel coordinates are on the boundary of the image.
    bounds_mask = (
        (pixel_u >= 0) & (pixel_u < W - 1) &
        (pixel_v >= 0) & (pixel_v < H - 1) &
        (corner_depths > 0)
    )

    # Bilinear Interpolation
    u_valid = pixel_u[bounds_mask]
    v_valid = pixel_v[bounds_mask]

    # Calculate the 4 surrounding pixel indices
    # Clamp to prevent any out-of-bounds CUDA errors (extra safety)
    u0 = torch.clamp(torch.floor(u_valid).long(), 0, W - 2)
    v0 = torch.clamp(torch.floor(v_valid).long(), 0, H - 2)
    u1 = u0 + 1
    v1 = v0 + 1

    # Interpolation weights
    wa = (u1.double() - u_valid) * (v1.double() - v_valid)
    wb = (u_valid - u0.double()) * (v1.double() - v_valid)
    wc = (u1.double() - u_valid) * (v_valid - v0.double())
    wd = (u_valid - u0.double()) * (v_valid - v0.double())

    d00 = depth_map_t[v0, u0]
    d01 = depth_map_t[v0, u1]
    d10 = depth_map_t[v1, u0]
    d11 = depth_map_t[v1, u1]

    # Check if all 4 surrounding pixels have valid depth values
    depth_check = (d00 > 0) & (d01 > 0) & (d10 > 0) & (d11 > 0)
    if depth_trunc is not None:
        depth_check &= (d00 < depth_trunc) & (d01 < depth_trunc) & (d10 < depth_trunc) & (d11 < depth_trunc)

    sampled_depths = wa * d00 + wb * d01 + wc * d10 + wd * d11

    valid_mask = bounds_mask.clone()
    valid_mask[bounds_mask] = depth_check

    # Step 4: SDF Calculation
    # Step 4: SDF Calculation
    # Compare the distance to voxel vs the depth camera's measurement.
    valid_cam_points = cam_points[valid_mask]

    # Straight-line distance from camera to voxel (the ray length)
    ray_lengths = torch.linalg.norm(valid_cam_points, dim=1)

    # Depth maps store Z-depth (perpendicular to image plane), not ray length.
    # Multiplying the Z-depth SDF by cos(theta) projects it onto the ray direction,
    # giving a consistent surface across views and avoiding holes at oblique angles.
    directional_cosine = corner_depths[valid_mask] / ray_lengths

    # Initialize all voxels with a dummy "empty space" value (1e6)
    # This guarantees that even with a huge trunc_margin, unseen voxels will always normalize to exactly 1.0 TSDF.
    sdf_values = torch.ones(world_points_h_t.shape[0], device=device, dtype=torch.float64) * 1e6

    # SDF = (Camera's Measurement - Voxel's Z-depth) * cos(theta)
    # Positive SDF: Voxel is in empty space (in front of the surface).
    # Zero SDF: Voxel is exactly on the surface.
    # Negative SDF: Voxel is buried inside the solid object.
    final_sampled_depths = sampled_depths[depth_check]
    sdf_values[valid_mask] = (final_sampled_depths - corner_depths[valid_mask]) * directional_cosine

    # Step 5: Truncation
    valid_sdf = sdf_values[valid_mask]
    # Clamp the SDF values to the truncation margin 
    # and divide by the truncation margin to normalize them.
    # This ensures that the TSDF values are always between -1 and 1.
    tsdf_valid = torch.clamp(valid_sdf, -trunc_margin, trunc_margin) / trunc_margin

    tsdf_values = torch.ones(world_points_h_t.shape[0], device=device, dtype=torch.float64)
    tsdf_values[valid_mask] = tsdf_valid

    # Dynamic Weighting
    # We assign higher confidence (weight) to measurements taken close to the camera.
    weights = torch.zeros(world_points_h_t.shape[0], device=device, dtype=torch.float64)
    
    # Carving Mask: We only assign weight to empty space (>0) or surface (~0). 
    # We ignore deep interior (-1) because cameras cannot see through solid walls
    carving_mask = valid_mask & (tsdf_values > -0.99)
    
    # Inverse-Square Law: Depth cameras are exponentially less accurate far away.
    # A camera 1m away gets a weight of 1.0. A camera 4m away gets a weight of 0.06.
    # The +1e-5 prevents divide-by-zero crashes.
    weights[carving_mask] = 1.0 / (corner_depths[carving_mask] ** 2 + 1e-5)

    # Reshape and Return
    final_grid = tsdf_values.view(grid_shape)
    final_weights = weights.view(grid_shape)

    color_grid = None
    if color_image_t is not None:
        color_values = torch.zeros((world_points_h_t.shape[0], 3), device=device, dtype=torch.float64)
        
        c00 = color_image_t[v0, u0]
        c01 = color_image_t[v0, u1]
        c10 = color_image_t[v1, u0]
        c11 = color_image_t[v1, u1]
        
        # unsqueeze(-1) allows multiplying the weights by the color values (3 channels)
        sampled_colors = (
            c00 * wa.unsqueeze(-1) + 
            c01 * wb.unsqueeze(-1) + 
            c10 * wc.unsqueeze(-1) + 
            c11 * wd.unsqueeze(-1)
        )
        
        color_values[valid_mask] = sampled_colors[depth_check]
        color_grid = color_values.view(*grid_shape, 3)

    return final_grid, color_grid, final_weights


def fuse_tsdf(depth_maps, intrinsics_list, extrinsics_list, grid_shape, voxel_size, trunc_margin, color_images=None, grid_origin=None, depth_trunc=None):
    """
    GPU-Accelerated multi-camera fusion using PyTorch.
    Inputs and outputs remain as standard NumPy arrays for compatibility with the rest of the pipeline.
    
    Each camera only sees part of the scene. Fusion combines all views so that
    voxels seen by multiple cameras get averaged, producing a more complete and
    accurate reconstruction.
    
    Inputs:
    - depth_maps: list of 2D depth arrays, one per camera.
    - intrinsics_list: list of 3x3 intrinsics matrices, one per camera.
    - extrinsics_list: list of 4x4 extrinsics matrices, one per camera.
    - grid_shape: Tuple (Nx, Ny, Nz) for the grid size.
    - voxel_size: Physical size of one voxel in coordinate units.
    - trunc_margin: The maximum +/- distance to cap the SDF.
    - color_images: (optional) list of (H, W, 3) RGB images, one per camera.
    - grid_origin: (optional) (3,) array, the world-space corner where the grid starts.
    - depth_trunc: (optional) Maximum depth to consider per camera.
    
    Returns:
    - fused_tsdf: (Nx, Ny, Nz) numpy array of fused TSDF values.
    - fused_colors: (Nx, Ny, Nz, 3) numpy array of fused RGB colors, or None.
    - obs_count: (Nx, Ny, Nz) numpy array counting how many cameras observed each voxel.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  [PyTorch TSDF] Accelerated on {device.type.upper()} (float64 precision)")

    tsdf_sum = torch.zeros(grid_shape, device=device, dtype=torch.float64)
    weight_sum = torch.zeros(grid_shape, device=device, dtype=torch.float64)
    obs_count = torch.zeros(grid_shape, device=device, dtype=torch.int32)
    
    if color_images is not None:
        color_sum = torch.zeros((*grid_shape, 3), device=device, dtype=torch.float64)

    num_cameras = len(depth_maps)

    _grid_origin = grid_origin if grid_origin is not None else np.array([0.0, 0.0, 0.0])
    
    # Pre-compute the world-point grid + homogeneous coords once on the GPU.
    # All cameras share the same grid, so we avoid recomputing meshgrid + hstack
    # for every camera (saves N³ allocations per camera).
    coords_x = torch.arange(grid_shape[0], device=device, dtype=torch.float64) * voxel_size + _grid_origin[0]
    coords_y = torch.arange(grid_shape[1], device=device, dtype=torch.float64) * voxel_size + _grid_origin[1]
    coords_z = torch.arange(grid_shape[2], device=device, dtype=torch.float64) * voxel_size + _grid_origin[2]
    
    gx, gy, gz = torch.meshgrid(coords_x, coords_y, coords_z, indexing='ij')

    # dim = -1 because we stack the 3 coordinates [x, y, z] along the last dimension
    # This makes the shape (Nx, Ny, Nz, 3), where each element is an [x, y, z] coordinate
    world_points = torch.stack([gx, gy, gz], dim=-1).view(-1, 3)

    # Add a column of 1s to make [x, y, z] -> [x, y, z, 1] (homogeneous coordinates)
    ones = torch.ones((world_points.shape[0], 1), device=device, dtype=torch.float64)
    # Precomputed world points homogeneous coordinates, shape (N³, 4)
    precomputed_wph_t = torch.cat([world_points, ones], dim=1)

    for i in range(num_cameras):
        # Move inputs to GPU
        depth_t = torch.from_numpy(depth_maps[i]).to(device=device, dtype=torch.float64)
        intrinsics_t = torch.from_numpy(intrinsics_list[i]).to(device=device, dtype=torch.float64)
        extrinsics_t = torch.from_numpy(extrinsics_list[i]).to(device=device, dtype=torch.float64)
        
        colors_t = None
        if color_images is not None:
            colors_t = torch.from_numpy(color_images[i]).to(device=device, dtype=torch.float64)

        tsdf_i, color_i, weight_i = generate_tsdf_single_camera(
            depth_t, intrinsics_t, extrinsics_t,
            grid_shape, voxel_size, trunc_margin, color_image_t=colors_t,
            depth_trunc=depth_trunc,
            world_points_h_t=precomputed_wph_t
        )

        tsdf_sum += tsdf_i * weight_i # grid of sdf values
        weight_sum += weight_i # grid of weights
        obs_count += (weight_i > 0).to(torch.int32) # grid of observation counts
        
        if color_images is not None and color_i is not None:
            color_sum += color_i * weight_i.unsqueeze(-1)

    # Average: for each corner, divide the accumulated TSDF by the sum of weights.
    # Corners seen by no camera stay at 1.0 (free space / no data).
    mask = weight_sum > 0
    fused_tsdf = torch.ones_like(tsdf_sum)
    fused_tsdf[mask] = tsdf_sum[mask] / weight_sum[mask] # weighted average of sdf values

    fused_colors = None
    if color_images is not None:
        fused_colors = torch.zeros_like(color_sum)
        fused_colors[mask] = color_sum[mask] / weight_sum[mask].unsqueeze(-1) # weighted average of colors

    print(f"  Fusion complete. Corners observed by at least 1 camera: "
          f"{mask.sum().item()} / {np.prod(grid_shape)}")

    # Move results back to CPU as NumPy arrays
    return fused_tsdf.cpu().numpy(), (fused_colors.cpu().numpy() if fused_colors is not None else None), obs_count.cpu().numpy()
