import torch
import numpy as np

def generate_tsdf_single_camera(depth_map, intrinsics, extrinsics, grid_shape, voxel_size, trunc_margin, color_image=None, grid_origin=None, depth_trunc=None):
    """
    Project a 3D voxel grid onto a 2D depth map to calculate TSDF.
    
    Inputs:
    - depth_map: 2D array of depth values from the camera.
    - intrinsics: 3x3 Camera lens matrix.
    - extrinsics: 4x4 Camera pose matrix (World-to-Camera).
    - grid_shape: Tuple (N, N, N) for the grid size.
    - voxel_size: Physical size of one voxel in meters.
    - trunc_margin: The maximum +/- distance to cap the SDF (the "T" in TSDF).
    - color_image: (optional) (H, W, 3) RGB image from the same camera. If provided,
      colors are sampled at the same pixel coordinates as depth.
    - grid_origin: (optional) (3,) array, the world-space corner where the grid starts.
      Defaults to (0, 0, 0).
    - depth_trunc: (optional) Maximum depth value to consider. Pixels with depth beyond
      this are ignored, preventing noisy far-away surfaces from corrupting the TSDF.
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
    # Step 2: World Space to Camera Space (Extrinsics)
    # Move the points from the global world into the camera's local view.

    # =====================================================================
    # Step 2a: Add a column of 1s to make [X, Y, Z] -> [X, Y, Z, 1] (homogeneous coordinates)
    # This lets the 4x4 matrix apply both rotation AND translation in one multiplication.
    ones = np.ones((world_points.shape[0], 1))           # shape: (N³, 1)
    world_points_h = np.hstack([world_points, ones])     # shape: (N³, 4)

    # Step 2b: Multiply by the extrinsics matrix (4x4) to transform into camera space.
    # extrinsics is (4, 4), world_points_h.T is (4, N³) -> result is (4, N³) -> transpose to (N³, 4)
    # We only keep the first 3 rows (X, Y, Z in camera space), discard the homogeneous row.
    cam_points = (extrinsics @ world_points_h.T).T[:, :3]  # shape: (N³, 3)

    # The Z-coordinate in camera space = how far each point is along the camera's viewing direction.
    corner_depths = cam_points[:, 2]                        # shape: (N³,) 
    # corner_depths is a 1D array of length N³.

    # =====================================================================
    # Step 3: Camera Space to Image Space (Intrinsics)
    # Smash the 3D points onto the 2D image plane.

    # =====================================================================
    # Multiply the 3x3 intrinsics matrix by the camera-space points to get the pixel coordinates.
    # intrinsics is (3, 3), cam_points.T is (3, N³) -> result is (3, N³) -> transpose to (N³, 3)
    image_points = (intrinsics @ cam_points.T).T  # shape: (N³, 3)

    # Perspective Divide: divide X and Y by Z to project onto the 2D image plane.
    # This is what makes faraway things look smaller, just like a real camera.
    pixel_u = (image_points[:, 0] / image_points[:, 2]).astype(int)  # horizontal pixel column
    pixel_v = (image_points[:, 1] / image_points[:, 2]).astype(int)  # vertical pixel row

    # =====================================================================
    # Step 4: Filter out the blind spots
    # Some voxels will project outside the photograph (e.g., pixel_u = -50).
    # Create a boolean mask to filter out points that fall outside the 
    # depth_map's width/height, or have a voxel_depth <= 0 (behind the camera).
    # =====================================================================
    H, W = depth_map.shape  # height & width of the depth image (in pixels)

    bounds_mask = (
        (pixel_u >= 0) & (pixel_u < W) &   # inside left/right bounds
        (pixel_v >= 0) & (pixel_v < H) &   # inside top/bottom bounds
        (corner_depths > 0)                  # in front of the camera, not behind it
    )

    # Fix: also reject pixels where depth_map = 0 (camera saw nothing there).
    # Without this, SDF = 0 - voxel_depth = large negative, which Marching Cubes
    # incorrectly interprets as "inside an object", producing false frustum-shaped walls.
    # We check depth > 0 only on the subset that passed bounds_mask, then expand back.
    # 0 depth means the camera ray at that pixel didn't hit any surface (e.g. empty background)
    valid_mask = np.zeros(world_points.shape[0], dtype=bool)
    sampled_depths = depth_map[pixel_v[bounds_mask], pixel_u[bounds_mask]]
    depth_check = sampled_depths > 0
    # If depth_trunc is set, also reject pixels where depth exceeds the max range.
    # This prevents far-away noisy surfaces from corrupting the TSDF.
    if depth_trunc is not None:
        depth_check = depth_check & (sampled_depths < depth_trunc)
    valid_mask[bounds_mask] = depth_check

    # =====================================================================
    # Step 5: The Core Math (SDF Calculation)
    # For the valid pixels, look up their depth in the actual depth_map.
    # Calculate: SDF = (Surface Depth from Image) - (Voxel Depth)
    # =====================================================================
    # Start with a default value of 1.0 (meaning "far from surface, in free space")
    # any positive value means the voxel is in front of the surface (free space)
    # any negative value means the voxel is behind the surface (inside the object)
    # we picked 1 because it's the maximum possible value for the truncated SDF
    sdf_values = np.ones(world_points.shape[0])  # shape: (N³,) 

    # For the valid points only, look up the actual surface depth from the depth image
    # at the pixel (v, u) each point projects to, then subtract the voxel's depth.
    # Positive = voxel is in front of the surface (free space)
    # Negative = voxel is behind the surface (inside the object)
    surface_depths = depth_map[pixel_v[valid_mask], pixel_u[valid_mask]]
    sdf_values[valid_mask] = surface_depths - corner_depths[valid_mask]

    # =====================================================================
    # Step 6: Truncation
    # Cap the sdf_values between +trunc_margin and -trunc_margin.
    # Then, divide by trunc_margin so the final values sit neatly between -1.0 and 1.0.
    # =====================================================================
    # np.clip chops off any value beyond the margin (e.g. sdf=5.0 with margin=0.1 becomes 0.1)
    # Dividing by trunc_margin normalizes to [-1.0, +1.0]
    tsdf_values = np.clip(sdf_values, -trunc_margin, trunc_margin) / trunc_margin

    # =====================================================================
    # Step 7: Reshape and Return
    # Take flat list of calculations and reshape it back into the 
    # original 3D grid shape (N, N, N).
    # =====================================================================
    final_grid = tsdf_values.reshape(grid_shape)  # (N³,) -> (N, N, N)

    # =====================================================================
    # Step 8: Sample Colors (optional)
    # If a color image was provided, look up the RGB color for each valid corner
    # at the same pixel (v, u) used for depth. Unseen corners get black (0, 0, 0).
    # black shouldn't cause problem because weighting is used
    # =====================================================================
    color_grid = None
    if color_image is not None:
        color_values = np.zeros((world_points.shape[0], 3))  # shape: (N³, 3)
        color_values[valid_mask] = color_image[pixel_v[valid_mask], pixel_u[valid_mask]]
        color_grid = color_values.reshape(*grid_shape, 3)    # (N³, 3) -> (N, N, N, 3)

    return final_grid, color_grid


def fuse_tsdf(depth_maps, intrinsics_list, extrinsics_list, grid_shape, voxel_size, trunc_margin, color_images=None, grid_origin=None, depth_trunc=None):
    """
    Fuse multiple single-camera TSDFs into one grid using weighted averaging.
    
    Each camera only sees part of the scene. Fusion combines all views so that
    voxels seen by multiple cameras get averaged, producing a more complete and
    accurate reconstruction.
    
    Inputs:
    - depth_maps: list of 2D depth arrays, one per camera.
    - intrinsics_list: list of 3x3 intrinsics matrices, one per camera.
    - extrinsics_list: list of 4x4 extrinsics matrices, one per camera.
    - grid_shape: Tuple (N, N, N) for the grid size.
    - voxel_size: Physical size of one voxel in meters.
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
    if color_images is not None:
        color_sum = np.zeros((*grid_shape, 3))  # (N, N, N, 3)

    num_cameras = len(depth_maps)

    for i in range(num_cameras):
        print(f"  Processing camera {i + 1}/{num_cameras}...")

        # Get the TSDF (and optionally color) from this single camera
        color_img_i = color_images[i] if color_images is not None else None
        tsdf_i, color_i = generate_tsdf_single_camera(
            depth_maps[i], intrinsics_list[i], extrinsics_list[i],
            grid_shape, voxel_size, trunc_margin, color_image=color_img_i,
            grid_origin=grid_origin, depth_trunc=depth_trunc
        )

        # Weight = 1 only for corners that are within the truncation band (|TSDF| < 1.0).
        # This excludes two types of useless corners:
        #   - TSDF = +1.0: unseen corners (default) or observed but far in free space (clamped)
        #   - TSDF = -1.0: observed but far behind the surface (clamped)
        # Without the abs check, -1.0 values from one camera would corrupt near-surface
        # values from another camera (e.g. averaging 0.0 and -1.0 = -0.5, creating holes).
        # TLDR: Average only TSDF values of corners that were seen by the camera and are inside
        # the truncation band.
        weight_i = (np.abs(tsdf_i) < 1.0).astype(float)

        tsdf_sum += tsdf_i * weight_i
        weight_sum += weight_i
        if color_images is not None and color_i is not None:
            # Accumulate colors with the same weights as TSDF
            # Because weight_i is (N,N,N) and color_i is (N,N,N,3), meaning its a 3D grid
            # of size N, where every point holds an RGB color
            # we need to broadcast weight_i to (N,N,N,1) to multiply with color_i
            color_sum += color_i * weight_i[..., np.newaxis]  # broadcast (N,N,N) -> (N,N,N,1)

    # Average: for each corner, divide the accumulated TSDF by the number of cameras that saw it.
    # Corners seen by no camera stay at 1.0 (free space / no data).
    fused_tsdf = np.where(weight_sum > 0, tsdf_sum / weight_sum, 1.0)

    # Average colors the same way
    fused_colors = None
    if color_images is not None:
        # Broadcast weight_sum (N,N,N) to match color_sum (N,N,N,3)
        fused_colors = np.where(weight_sum[..., np.newaxis] > 0,
                                color_sum / weight_sum[..., np.newaxis], 0.0)

    print(f"  Fusion complete. Corners observed by at least 1 camera: "
          f"{np.sum(weight_sum > 0)} / {np.prod(grid_shape)}")

    return fused_tsdf, fused_colors