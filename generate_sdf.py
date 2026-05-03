import numpy as np

def generate_tsdf_single_camera(depth_map, intrinsics, extrinsics, grid_shape, voxel_size, trunc_margin, color_image=None, grid_origin=None, depth_trunc=None, world_points_h=None):
    """
    Improved TSDF generation with:
    1. Bilinear Interpolation for smoother surfaces
    2. Euclidean Ray Distance instead of Z-depth difference
    3. Depth-dependent weight calculation
    
    Project a 3D voxel grid onto a 2D depth map to calculate TSDF.
    
    Inputs:
    - depth_map: 2D array of depth values from the camera.
    - intrinsics: 3x3 Camera lens matrix.
    - extrinsics: 4x4 Camera pose matrix (World-to-Camera).
    - grid_shape: Tuple (N, N, N) for the grid size.
    - voxel_size: Physical size of one voxel in coordinate units.
    - trunc_margin: The maximum +/- distance to cap the SDF (the "T" in TSDF).
    - color_image: (optional) (H, W, 3) RGB image from the same camera. If provided,
      colors are sampled at the same pixel coordinates as depth.
    - grid_origin: (optional) (3,) array, the world-space corner where the grid starts.
      Defaults to (0, 0, 0).
    - depth_trunc: (optional) Maximum depth value to consider. Pixels with depth beyond
      this are ignored, preventing noisy far-away surfaces from corrupting the TSDF.
    - world_points_h: (optional) Pre-computed (N³, 4) homogeneous world-point matrix.
      If provided, skips the meshgrid + hstack computation (saves time when the same
      grid is used across multiple cameras).
    """
    if grid_origin is None:
        grid_origin = np.array([0.0, 0.0, 0.0])
    
    # =====================================================================
    # Step 1: Create the 3D Grid
    # Generate the (X, Y, Z) physical coordinates for the 3D grid.
    # Note: Because we are using Marching Cubes, these coordinates represent 
    # the exact corners of the voxels, not the voxels themselves. 
    # A grid of N=64 corners will generate 63x63x63 hollow voxel spaces.
    # =====================================================================
    
    N = grid_shape[0] # number of corners (vertices)

    # =====================================================================
    # Step 1: Create the 3D Grid (or reuse pre-computed one)
    # =====================================================================
    # If world_points_h was pre-computed and passed in, skip the expensive
    # meshgrid + hstack. This saves a lot of time when fusing multiple cameras
    # that all use the same grid (which is the common case).
    if world_points_h is None:
        coords_x = np.arange(grid_shape[0]) * voxel_size + grid_origin[0]
        coords_y = np.arange(grid_shape[1]) * voxel_size + grid_origin[1]
        coords_z = np.arange(grid_shape[2]) * voxel_size + grid_origin[2]
        x, y, z = np.meshgrid(coords_x, coords_y, coords_z, indexing='ij') # 3D grid of coordinates

        # ==========================================================================================
        # np.meshgrid created 3 separate 3D cubes of numbers (an X-cube, a Y-cube, and a Z-cube).
        # if x cube was a dict:
        # (example with N=3, voxel_size=0.5)
        # x_cube_dictionary = {
        # KEY (The 3D Grid Index)   :  VALUE (The Physical X Measurement)
        # "(Floor 0, Row 0, Col 0)"   :  0.0, 
        # "(Floor 0, Row 0, Col 1)"   :  0.5,
        # "(Floor 0, Row 0, Col 2)"   :  1.0,
        
        # "(Floor 1, Row 0, Col 0)"   :  0.0, 
        # "(Floor 1, Row 0, Col 1)"   :  0.5,
        # "(Floor 1, Row 0, Col 2)"   :  1.0,
        # ... all 262,144 spots ...
        # }
        # ==========================================================================================
        # Flatten the 3d grid for gpu matrix math
        # np.stack([x, y, z], axis=-1):
        # stack() grabs those 3 cubes and glues them together at the deepest level (axis=-1).
        # by deepest level, we mean the last dimension.
        # so instead of 3 separate arrays, we have one solid 3D block where every single spot 
        # holds a [X, Y, Z] coordinate package. Shape becomes (N, N, N, 3).
        # ==========================================================================================

        world_points = np.stack([x, y, z], axis=-1).reshape(-1, 3)

        # ==========================================================================================
        # flattened world_points array:
        # (example with N=64, voxel_size=0.05)
        # Shape: (262144, 3) -> A flat list where every row is exactly one [X, Y, Z] coordinate.
        # index         [   X,      Y,      Z   ]    Physical Location in the Box
        # --------      -------------------------    -------------------------------------------
        # [0]           [ 0.00,   0.00,   0.00 ]  <- Bottom, Front, Left corner (Origin)
        # [1]           [ 0.05,   0.00,   0.00 ]  <- Moving right along the X-axis...
        # [2]           [ 0.10,   0.00,   0.00 ]  
        # ...
        # [63]          [ 3.15,   0.00,   0.00 ]  <- Hit the right wall. Time to step back on Y-axis.
        # [64]          [ 0.00,   0.05,   0.00 ]  <- Start of Row 2 (X resets to 0, Y moves back)
        # [65]          [ 0.05,   0.05,   0.00 ]  
        # ...
        # [4095]        [ 3.15,   3.15,   0.00 ]  <- Finished the entire Floor 0. Time to step up.
        # [4096]        [ 0.00,   0.00,   0.05 ]  <- Start of Floor 1 (Z moves up)
        # ...
        # [262143]      [ 3.15,   3.15,   3.15 ]  <- The last dot: Top, Back, Right corner.
        # ==========================================================================================
        
        # =====================================================================
        # Step 2a: Add a column of 1s to make [X, Y, Z] -> [X, Y, Z, 1] (homogeneous coordinates)
        # This lets the 4x4 matrix apply both rotation AND translation in one multiplication.
        ones = np.ones((world_points.shape[0], 1))           # shape: (N³, 1)
        world_points_h = np.hstack([world_points, ones])     # shape: (N³, 4)

    # Extract world_points (first 3 cols) for later use if needed
    world_points = world_points_h[:, :3]

    # =====================================================================
    # Step 2: World Space to Camera Space (Extrinsics)
    # Move the points from the global world into the camera's local view.
    # =====================================================================

    # Step 2b: Multiply by the extrinsics matrix (4x4) to transform into camera space.
    # extrinsics is (4, 4), world_points_h.T is (4, N³) -> result is (4, N³) -> transpose to (N³, 4)
    # We only keep the first 3 rows (X, Y, Z in camera space), discard the homogeneous row.
    cam_points = (extrinsics @ world_points_h.T).T[:, :3]  # shape: (N³, 3)

    # The Z-coordinate in camera space = how far each point is along the camera's viewing direction.
    corner_depths = cam_points[:, 2]                        # shape: (N³,) 

    # =====================================================================
    # Step 3: Camera Space to Image Space (Intrinsics)
    # Smash the 3D points onto the 2D image plane.

    # =====================================================================
    # Multiply the 3x3 intrinsics matrix by the camera-space points to get the pixel coordinates.
    # intrinsics is (3, 3), cam_points.T is (3, N³) -> result is (3, N³) -> transpose to (N³, 3)
    image_points = (intrinsics @ cam_points.T).T  # shape: (N³, 3)

    # Keep as floats for bilinear interpolation
    # Perspective Divide: divide X and Y by Z to project onto the 2D image plane.
    # This is what makes faraway things look smaller, just like a real camera.
    pixel_u = image_points[:, 0] / image_points[:, 2]
    pixel_v = image_points[:, 1] / image_points[:, 2]

    # =====================================================================
    # Step 4: Filter out the blind spots
    # Some voxels will project outside the photograph (e.g., pixel_u = -50).
    # Create a boolean mask to filter out points that fall outside the 
    # depth_map's width/height, or have a voxel_depth <= 0 (behind the camera).
    # =====================================================================
    H, W = depth_map.shape  # height & width of the depth image (in pixels)

    # Strict bounds for bilinear interpolation (need to access u+1, v+1 safely)
    bounds_mask = (
        (pixel_u >= 0) & (pixel_u < W - 1) &
        (pixel_v >= 0) & (pixel_v < H - 1) &
        (corner_depths > 0)
    )

    # --- IMPROVEMENT 3: Bilinear Interpolation ---
    u_valid = pixel_u[bounds_mask]
    v_valid = pixel_v[bounds_mask]

    u0 = np.floor(u_valid).astype(int)
    v0 = np.floor(v_valid).astype(int)
    u1 = u0 + 1
    v1 = v0 + 1

    # Interp weights
    wa = (u1 - u_valid) * (v1 - v_valid)
    wb = (u_valid - u0) * (v1 - v_valid)
    wc = (u1 - u_valid) * (v_valid - v0)
    wd = (u_valid - u0) * (v_valid - v0)

    # Gather 4 corners
    d00 = depth_map[v0, u0]
    d01 = depth_map[v0, u1]
    d10 = depth_map[v1, u0]
    d11 = depth_map[v1, u1]

    # Only keep points where ALL 4 neighbors have valid depth
    depth_check = (d00 > 0) & (d01 > 0) & (d10 > 0) & (d11 > 0)
    if depth_trunc is not None:
        depth_check &= (d00 < depth_trunc) & (d01 < depth_trunc) & (d10 < depth_trunc) & (d11 < depth_trunc)

    # Interpolate depth
    sampled_depths = wa * d00 + wb * d01 + wc * d10 + wd * d11

    valid_mask = np.zeros(world_points.shape[0], dtype=bool)
    valid_mask[bounds_mask] = depth_check

    # =====================================================================
    # Step 5: The Core Math (SDF Calculation)
    # For the valid pixels, look up their depth in the actual depth_map.
    # Calculate: SDF = (Surface Depth from Image) - (Voxel Depth)
    # =====================================================================
    # --- IMPROVEMENT 2: Euclidean Ray Distance ---
    # Find directional cosine of the camera ray for each valid voxel
    valid_cam_points = cam_points[valid_mask]
    ray_lengths = np.linalg.norm(valid_cam_points, axis=1)
    directional_cosine = corner_depths[valid_mask] / ray_lengths

    # Initialize with large positive values to mark as "unseen / uninitialized"
    sdf_values = np.ones(world_points.shape[0]) * 999.0

    # Calculate precise Euclidean TSDF
    final_sampled_depths = sampled_depths[depth_check]
    sdf_values[valid_mask] = (final_sampled_depths - corner_depths[valid_mask]) * directional_cosine

    # =====================================================================
    # Step 6: Truncation
    # Cap the sdf_values between +trunc_margin and -trunc_margin.
    # Then, divide by trunc_margin so the final values sit neatly between -1.0 and 1.0.
    # =====================================================================
    # Truncate to [-1, 1]
    # We clip the computed values, but keep the "999" (unseen) values intact temporarily
    valid_sdf = sdf_values[valid_mask]
    tsdf_valid = np.clip(valid_sdf, -trunc_margin, trunc_margin) / trunc_margin
    
    tsdf_values = np.ones(world_points.shape[0]) # Default to +1.0
    tsdf_values[valid_mask] = tsdf_valid

    # --- IMPROVEMENT 4: Dynamic Weighting (Depth-Dependent) ---
    weights = np.zeros(world_points.shape[0])
    
    # Weight is > 0 ONLY for valid mask points that are NOT truncated behind the surface (-1)
    # This enables IMPROVEMENT 1: Free space carving (points with tsdf == +1.0 in front of surface get weight)
    carving_mask = valid_mask & (tsdf_values > -0.99)
    weights[carving_mask] = 1.0 / (corner_depths[carving_mask] ** 2 + 1e-5)

    # =====================================================================
    # Step 7: Reshape and Return
    # Take flat list of calculations and reshape it back into the 
    # original 3D grid shape (N, N, N).
    # =====================================================================
    final_grid = tsdf_values.reshape(grid_shape)  # (N³,) -> (N, N, N)
    final_weights = weights.reshape(grid_shape)

    # =====================================================================
    # Step 8: Sample Colors (optional)
    # If a color image was provided, look up the RGB color for each valid corner
    # at the same pixel (v, u) used for depth. Unseen corners get black (0, 0, 0).
    # black shouldn't cause problem because weighting is used
    # =====================================================================
    # Bilinear Interpolate Colors (Optional)
    color_grid = None
    if color_image is not None:
        color_values = np.zeros((world_points.shape[0], 3))
        
        c00 = color_image[v0, u0]
        c01 = color_image[v0, u1]
        c10 = color_image[v1, u0]
        c11 = color_image[v1, u1]
        
        sampled_colors = (
            c00 * wa[:, np.newaxis] + 
            c01 * wb[:, np.newaxis] + 
            c10 * wc[:, np.newaxis] + 
            c11 * wd[:, np.newaxis]
        )
        
        color_values[valid_mask] = sampled_colors[depth_check]
        color_grid = color_values.reshape(*grid_shape, 3)

    return final_grid, color_grid, final_weights


def fuse_tsdf(depth_maps, intrinsics_list, extrinsics_list, grid_shape, voxel_size, trunc_margin, color_images=None, grid_origin=None, depth_trunc=None):
    """
    Improved multi-camera fusion using the generated grids and dynamic weights.
    
    Each camera only sees part of the scene. Fusion combines all views so that
    voxels seen by multiple cameras get averaged, producing a more complete and
    accurate reconstruction.
    
    Inputs:
    - depth_maps: list of 2D depth arrays, one per camera.
    - intrinsics_list: list of 3x3 intrinsics matrices, one per camera.
    - extrinsics_list: list of 4x4 extrinsics matrices, one per camera.
    - grid_shape: Tuple (N, N, N) for the grid size.
    - voxel_size: Physical size of one voxel in coordinate units.
    - trunc_margin: The maximum +/- distance to cap the SDF.
    - color_images: (optional) list of (H, W, 3) RGB images, one per camera.
    - grid_origin: (optional) (3,) array, the world-space corner where the grid starts.
    - depth_trunc: (optional) Maximum depth to consider per camera.
    
    Returns:
    - fused_tsdf: (N, N, N) numpy array of fused TSDF values.
    - fused_colors: (N, N, N, 3) numpy array of fused RGB colors, or None.
    """
    # Accumulators: sum of TSDF values and count of how many cameras saw each corner.
    tsdf_sum = np.zeros(grid_shape)
    weight_sum = np.zeros(grid_shape)
    obs_count = np.zeros(grid_shape, dtype=int)  # how many cameras observed each voxel
    if color_images is not None:
        color_sum = np.zeros((*grid_shape, 3))

    num_cameras = len(depth_maps)

    # Pre-compute the world-point grid + homogeneous coords ONCE.
    # All cameras share the same grid, so we avoid recomputing meshgrid + hstack
    # for every camera (saves N³ allocations per camera).
    _grid_origin = grid_origin if grid_origin is not None else np.array([0.0, 0.0, 0.0])
    coords_x = np.arange(grid_shape[0]) * voxel_size + _grid_origin[0]
    coords_y = np.arange(grid_shape[1]) * voxel_size + _grid_origin[1]
    coords_z = np.arange(grid_shape[2]) * voxel_size + _grid_origin[2]
    gx, gy, gz = np.meshgrid(coords_x, coords_y, coords_z, indexing='ij')
    world_points = np.stack([gx, gy, gz], axis=-1).reshape(-1, 3)
    ones = np.ones((world_points.shape[0], 1))
    precomputed_wph = np.hstack([world_points, ones])  # (N³, 4)

    for i in range(num_cameras):
        print(f"  Processing camera {i + 1}/{num_cameras} (Improved)...")

        color_img_i = color_images[i] if color_images is not None else None
        tsdf_i, color_i, weight_i = generate_tsdf_single_camera(
            depth_maps[i], intrinsics_list[i], extrinsics_list[i],
            grid_shape, voxel_size, trunc_margin, color_image=color_img_i,
            grid_origin=grid_origin, depth_trunc=depth_trunc,
            world_points_h=precomputed_wph
        )

        # Uses the improved weights returned by the single camera generator
        # Automatically handles Free Space Carving & Depth drop-off
        tsdf_sum += tsdf_i * weight_i
        weight_sum += weight_i
        obs_count += (weight_i > 0).astype(int)
        
        if color_images is not None and color_i is not None:
            color_sum += color_i * weight_i[..., np.newaxis]

    # Average: for each corner, divide the accumulated TSDF by the sum of weights.
    # Corners seen by no camera stay at 1.0 (free space / no data).
    mask = weight_sum > 0
    fused_tsdf = np.ones_like(tsdf_sum)
    fused_tsdf[mask] = tsdf_sum[mask] / weight_sum[mask]

    # Average colors the same way
    fused_colors = None
    if color_images is not None:
        fused_colors = np.zeros_like(color_sum)
        fused_colors[mask] = color_sum[mask] / weight_sum[mask, np.newaxis]

    print(f"  Fusion complete. Corners observed by at least 1 camera: "
          f"{np.sum(mask)} / {np.prod(grid_shape)}")

    return fused_tsdf, fused_colors, obs_count
