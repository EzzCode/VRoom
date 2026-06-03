import numpy as np

from mc_tables import CORNER_OFFSETS, EDGE_CORNERS, EdgeMasks, TriangleTable

def run_marching_cubes(grid, N, color_grid = None):
    """
    Parameters:
    - grid: (N, N, N) array of TSDF values for each voxel corner
    - N: grid resolution
    - color_grid: (N, N, N, 3) array of RGB colors for each voxel corner
    Returns:
    - vertices: numpy array of shape (V, 3), vertex positions in grid coordinates.
    - triangles: numpy array of shape (T, 3), triangle vertex indices.
    - vertex_colors: numpy array of shape (V, 3), or None if color_grid not provided.
    """
    M = N - 1  # number of voxels in each dimension

    # 1. Extract corner TSDF values

    # List of 8 TSDF values for each corner in all voxels (i.e. voxel_corner_vals[0] 
    # contains TSDF value of corner 0 for all (prob. 127) voxels).
    voxel_corner_vals = []
    for dx, dy, dz in CORNER_OFFSETS:
        # dx:M+dx, dy:M+dy, dz:M+dz gives key. grid[key] gives tsdf val at corner for
        # all voxels.
        x_start = dx
        x_end   = dx + M
        
        y_start = dy
        y_end   = dy + M
        
        z_start = dz
        z_end   = dz + M
        
        # Extract TSDF values
        corner_data = grid[x_start:x_end, y_start:y_end, z_start:z_end]
        
        voxel_corner_vals.append(corner_data)
    
    # 2. Build voxel configuration indices

    # 8-bit index for each voxel where bit i is 1 if corner i is inside the surface (tsdf < 0).
    # Voxel is identified by its botto left corner.
    voxel_indices = np.zeros((M, M, M), dtype=np.int32)
    for i in range(8): # for each corner
        corner_is_inside = voxel_corner_vals[i] < 0
        corner_is_inside_int = corner_is_inside.astype(np.int32)
        shifted = corner_is_inside_int << i
        voxel_indices |= shifted
    # Now we have info on each corner (inside or outside/ +ve or -ve) in voxel_indices.

    # 3. Find active voxels (voxels with zero-crossing)
    active_mask = (voxel_indices > 0) & (voxel_indices < 255)
    active_x, active_y, active_z = np.where(active_mask) # get indices of active voxels
    num_active = len(active_x)
    if num_active == 0:
        return [], [], [] if color_grid is not None else None
    # Get voxel index and 8 corner TSDF values for active voxels
    active_voxel_indices = voxel_indices[active_x, active_y, active_z] # (num_active,) array of voxel confguration indices
    # EdgeMasks is a list of 256 integers, where for each of the 256 possible voxel confguration
    # in voxel_indices, the 12-bit integer in EdgeMasks gives which edges are intersected by the
    # surface.
    edge_masks_arr = np.array(EdgeMasks, dtype=np.int32) # convert to np for vectorization
    active_edge_masks = edge_masks_arr[active_voxel_indices]

    active_corners = np.empty((num_active, 8))                          # (num_active, 8)
    for i in range(8):
        # Get TSDF values for corner i of all active voxels
        active_corners[:, i] = voxel_corner_vals[i][active_x, active_y, active_z]

    # 4. Edge Interpolation

    vertex_chunks = [] # list of arrays of vertices from each edge
    vertex_color_chunks = [] if color_grid is not None else None
    # edge_vertex_ids: contains one row for each voxel and 12 columns for each edge.
    # initialize with -1 indicating no vertex assigned yet. When we find a cut edge,
    # we assign the corresponding vertex id to the correct position in this array.
    edge_vertex_ids = np.full((num_active, 12), -1, dtype=np.int32)
    current_vertex_id = 0

    for edge_index in range(12):
        # Use bitwise and to check if edge_index th bit is 1
        # Shift 1 to left by edge index to get sth like 00010000 if edge_index is 4.
        # Then bitwise and with active_edge_masks gives nonzero if edge_index th bit is 1.
        edge_is_cut = (active_edge_masks & (1 << edge_index)) > 0
        cut_voxel_indices = np.where(edge_is_cut)[0] # indices of voxels where this edge is cut

        if len(cut_voxel_indices) == 0:
            continue
        # Get coordinates of cut voxels
        cut_voxel_x = active_x[cut_voxel_indices]
        cut_voxel_y = active_y[cut_voxel_indices]
        cut_voxel_z = active_z[cut_voxel_indices]
        # Edge_Corners gives the two corners that form the edge, get htem
        corner1_idx = EDGE_CORNERS[edge_index][0]
        corner2_idx = EDGE_CORNERS[edge_index][1]
        # Get their TSDF values
        tsdf_1 = active_corners[cut_voxel_indices, corner1_idx]
        tsdf_2 = active_corners[cut_voxel_indices, corner2_idx]
        # Apply linear interpolation
        # Find t (between 0 and 1) where the surface intersects the edge.
        t = tsdf_1 / (tsdf_1 - tsdf_2 + 1e-5)
        # Get the 3D coordinates of the two corners
        corner1_coords = np.array(CORNER_OFFSETS[corner1_idx])
        corner2_coords = np.array(CORNER_OFFSETS[corner2_idx])
        # Get the 3D coordinates of the cut points using linear interpolation
        p_relative_to_voxel = corner1_coords + t[:, np.newaxis] * (corner2_coords - corner1_coords)
        # Add the voxel's global grid coordinates to place the vertex correctly in the 3D world
        global_p = p_relative_to_voxel + np.column_stack((cut_voxel_x, cut_voxel_y, cut_voxel_z))
        # Interpolate colors
        if color_grid is not None:
            color1 = color_grid[cut_voxel_x + corner1_coords[0], cut_voxel_y + corner1_coords[1], cut_voxel_z + corner1_coords[2]]
            color2 = color_grid[cut_voxel_x + corner2_coords[0], cut_voxel_y + corner2_coords[1], cut_voxel_z + corner2_coords[2]]
            vertex_color = color1 * (1 - t[:, np.newaxis]) + color2 * t[:, np.newaxis]

        # Add to vertices list and assign vertex ids
        vertex_chunks.append(global_p)
        if color_grid is not None:
            vertex_color_chunks.append(vertex_color)
        num_new_vertices = len(cut_voxel_indices)
        vertex_ids = np.arange(current_vertex_id, current_vertex_id + num_new_vertices)
        edge_vertex_ids[cut_voxel_indices, edge_index] = vertex_ids
        current_vertex_id += num_new_vertices

    vertices = np.concatenate(vertex_chunks, axis=0) # (total_num_vertices, 3)
    if color_grid is not None:
        vertex_colors = np.concatenate(vertex_color_chunks, axis=0)
    else:
        vertex_colors = None

    # 5. Triangle Construction
    triangles = []
    for i in range(num_active):
        voxel_config_index = active_voxel_indices[i] # get 8-bit configuraion
        # Look up in TriangleTable to get which edges form triangles for this configuration
        # This returns an array of edge indices, where every group of 3 consecutive edge
        # indices forms a triangle. Length is multiple of 3, and -1 indicates end of list.
        traingle_edge_indices = TriangleTable[voxel_config_index]
        for j in range(0, len(traingle_edge_indices), 3):
            edge_1 = traingle_edge_indices[j]
            if edge_1 == -1:
                break
            edge_2 = traingle_edge_indices[j + 1]
            edge_3 = traingle_edge_indices[j + 2]
            # Get the vertex ids for these edges from edge_vertex_ids.
            # This gives us the ids of the vertices
            v1_id = edge_vertex_ids[i, edge_1]
            v2_id = edge_vertex_ids[i, edge_2]
            v3_id = edge_vertex_ids[i, edge_3]
            triangles.append((v1_id, v2_id, v3_id))

    if len(triangles) == 0:
        triangles = np.empty((0, 3), dtype=np.int64)
    else:
        triangles = np.array(triangles, dtype=np.int64)
    return vertices, triangles, vertex_colors