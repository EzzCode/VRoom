import numpy as np


def remove_small_components(vertices, triangles, vertex_colors=None, min_ratio=0.05):
    """Remove small disconnected mesh fragments.
    
    Uses union-find on triangle connectivity to find connected components,
    then keeps only components with at least `min_ratio` of the largest
    component's triangle count.
    
    Returns filtered (vertices, triangles, vertex_colors).
    """
    if len(triangles) == 0:
        return vertices, triangles, vertex_colors
    
    tri_arr = np.array(triangles)
    n_verts = len(vertices)
    
    # Union-Find
    parent = np.arange(n_verts)
    
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    
    # Connect vertices that share a triangle
    for v0, v1, v2 in tri_arr:
        union(v0, v1)
        union(v1, v2)
    
    # Find component for each triangle (use first vertex's root)
    tri_roots = np.array([find(t[0]) for t in tri_arr])
    
    # Count triangles per component
    unique_roots, counts = np.unique(tri_roots, return_counts=True)
    max_count = counts.max()
    threshold = int(max_count * min_ratio)
    
    # Keep components above threshold
    keep_roots = set(unique_roots[counts >= threshold])
    keep_mask = np.array([r in keep_roots for r in tri_roots])
    
    kept_tris = tri_arr[keep_mask]
    
    # Remap vertices (only keep referenced ones)
    used_verts = np.unique(kept_tris)
    vert_map = np.full(n_verts, -1, dtype=int)
    vert_map[used_verts] = np.arange(len(used_verts))
    
    new_vertices = [vertices[i] for i in used_verts]
    new_triangles = [(vert_map[a], vert_map[b], vert_map[c]) for a, b, c in kept_tris]
    
    new_colors = None
    if vertex_colors is not None:
        new_colors = [vertex_colors[i] for i in used_verts]
    
    removed = len(triangles) - len(new_triangles)
    if removed > 0:
        print(f"  Cleanup: removed {removed} triangles in small components")
    
    return new_vertices, new_triangles, new_colors


def compute_depth_trunc(depth_maps, semantic_maps, label_id, percentile=99, margin=1.1):
    """Auto-compute depth_trunc from actual masked depth values for a given label.
    
    Takes the percentile-th value of all valid depths belonging to this label
    and multiplies by margin to give a tight but safe cutoff.
    """
    all_valid = []
    for d, sem in zip(depth_maps, semantic_maps):
        mask = (sem == label_id) & (d > 0)
        if mask.any():
            all_valid.append(d[mask])
    if not all_valid:
        return 3.0  # safe fallback
    merged = np.concatenate(all_valid)
    # margin is used to make the truncation margin slightly larger than the actual
    # percentile value to account for potential noise in the depth measurements.
    trunc = float(np.percentile(merged, percentile) * margin)
    return trunc


def unproject_to_3d(depth_map, mask, intrinsics, extrinsics):
    """Unproject masked depth pixels to 3D world coordinates."""
    fy = intrinsics[1, 1]
    fx = intrinsics[0, 0]
    cy = intrinsics[1, 2]
    cx = intrinsics[0, 2]

    vs, us = np.where(mask & (depth_map > 0))
    if len(vs) == 0:
        return np.zeros((0, 3))

    depths = depth_map[vs, us]

    # Pixel to camera coordinates
    cam_x = (us - cx) * depths / fx
    cam_y = (vs - cy) * depths / fy
    cam_z = depths
    cam_pts = np.stack([cam_x, cam_y, cam_z], axis=-1)  # (M, 3) where M = number of valid pixels

    # Camera to world coordinates
    R = extrinsics[:3, :3]
    t = extrinsics[:3, 3]
    # world = R^T @ (cam - t)
    world_pts = (cam_pts - t) @ R  # equivalent to R^T @ each point

    return world_pts
