import numpy as np

def compute_depth_trunc(depth_maps, semantic_maps, label_id, percentile = 99, margin = 1.1):
    """Auto-compute depth truncation value for a specific object."""
    all_valid_pixels = []
    for depth_map, semantic_map in zip(depth_maps, semantic_maps):
        mask = (semantic_map == label_id) & (depth_map > 0)
        
        if mask.any():
            all_valid_pixels.append(depth_map[mask])
    
    if not all_valid_pixels:
        return 3.0  # safe fallback
    
    merged_valid_pixels = np.concatenate(all_valid_pixels)

    # Cut off at the specified percentile to remove outliers,
    # then multiply by margin to give a slightly larger truncation value for safety.
    percentile_val = np.percentile(merged_valid_pixels, percentile, axis = 0)
    final_trunc = float(percentile_val * margin)

    return final_trunc

def unproject_to_3d(depth_map, mask, intrinsic, extrinsic):
    """Unproject masked depth pixels to 3D world coordinates."""
    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]

    vs, us = np.where(mask & (depth_map > 0))
    if len(vs) == 0:
        return np.zeros((0, 3))
    depths = depth_map[vs, us]

    cam_x = (us - cx) * depths / fx
    cam_y = (vs - cy) * depths / fy
    cam_z = depths
    cam_pts = np.stack([cam_x, cam_y, cam_z], axis=-1)  # size (M, 3) where M = number of valid pixels
    # axis = -1 so that cam_pts[i] = [x, y, z]
    rotation_matrix = extrinsic[:3, :3]
    translation_vector = extrinsic[:3, 3]

    world_pts = (cam_pts - translation_vector) @ rotation_matrix

    return world_pts

def remove_small_components(vertices, triangles, vertex_colors=None, min_ratio=0.05):
    """Remove small disconnected mesh parts"""
    if (len(triangles) == 0): # if mesh empty
        return vertices, triangles, vertex_colors

    triangles_arr = np.array(triangles)
    vertices_arr = np.array(vertices)

    # Weld duplicate vertices before running union-find.
    # Marching cubes produces duplicate vertices at shared edges between adjacent
    # voxels, making the mesh a triangle soup with no shared vertex ids between
    # neighbouring triangles. Without welding, union-find sees every voxel as its
    # own disconnected island.
    unique_verts, first_occ, inverse_map = np.unique(vertices_arr, axis=0,
                                                      return_index=True,
                                                      return_inverse=True)
    welded_triangles_arr = inverse_map[triangles_arr] # remap triangle vertex indices to welded vertex ids
    num_verts = len(unique_verts)

    parents_arr = np.arange(num_verts) # create array containing vertex ids,
    # Initially, each vertex is its own parent

    # Union-find algorithm
    # Find largest connected part of object's mesh, then remove other parts that are
    # smaller than a certain threshold (islands that are noise)

    def find_absolute_parent(x):
        while parents_arr[x] != x:
            parents_arr[x] = parents_arr[parents_arr[x]]  # path compression
            x = parents_arr[x]
        return x

    def union(a,b): # find absolute parents of a,b and set one as the parent of the other
        abs_parent_a = find_absolute_parent(a)
        abs_parent_b = find_absolute_parent(b)

        if (abs_parent_a != abs_parent_b):
            parents_arr[abs_parent_b] = abs_parent_a

        return

    # Connect vertices of a triangle together
    for v0, v1, v2 in welded_triangles_arr:
        union(v0, v1)
        union(v1, v2)

    # Find the biggest island (by triangle count)
    tri_roots = np.zeros(len(welded_triangles_arr), dtype=np.int32)
    for i in range(len(welded_triangles_arr)):
        tri_roots[i] = find_absolute_parent(welded_triangles_arr[i][0])

    unique_abs_parents, num_triangles = np.unique(tri_roots, return_counts=True)
    max_island_size = np.max(num_triangles)
    threshold = int(max_island_size * min_ratio)

    # Delete the small parts
    surviving_unique_parents = unique_abs_parents[num_triangles >= threshold]
    surviving_triangles_mask = np.isin(tri_roots, surviving_unique_parents)
    unremapped_triangles = welded_triangles_arr[surviving_triangles_mask]

    # Create a translation dict to map old vertex indices to new ones after
    # removing small components
    used_verts = np.unique(unremapped_triangles)
    translation_dict = np.full(num_verts, -1, dtype=np.int32)
    translation_dict[used_verts] = np.arange(len(used_verts))

    surviving_vertices = unique_verts[used_verts]
    surviving_triangles = translation_dict[unremapped_triangles]

    removed = len(triangles_arr) - len(surviving_triangles)
    if removed > 0:
        print(f"  Cleanup: removed {removed} triangles in small components")

    if vertex_colors is not None:
        # first_occ maps each welded vertex back to one of its original duplicates
        surviving_vertex_colors = np.array(vertex_colors)[first_occ[used_verts]]
    else:
        surviving_vertex_colors = None

    return surviving_vertices, surviving_triangles, surviving_vertex_colors

