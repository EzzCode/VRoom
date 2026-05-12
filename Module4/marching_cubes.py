import numpy as np

from mc_tables import CORNER_OFFSETS, EDGE_CORNERS, EdgeMasks, TriangleTable

# ============================================================================
# Marching Cubes Algorithm (lookup tables in mc_tables.py)
# ============================================================================

def run_marching_cubes(voxel_grid, N, color_grid=None):
    """
    Run the Marching Cubes algorithm on a voxel grid.
    
    Fully vectorized NumPy implementation with no per-voxel Python loops.
    
    Inputs:
    - voxel_grid: (N, N, N) numpy array of SDF values.
    - N: grid resolution (number of corners along each axis).
    - color_grid: (optional) (N, N, N, 3) numpy array of RGB colors per corner.
    
    Returns:
    - vertices: list of (x, y, z) tuples.
    - triangles: list of (v0, v1, v2) index tuples.
    - vertex_colors: list of (r, g, b) tuples, or None if color_grid not provided.
    """

    # =========================================================================
    # Vectorized Implementation
    #
    # Instead of looping over every voxel one at a time with Python for-loops,
    # we process all voxels simultaneously using NumPy array operations.
    # Faster because NumPy runs in C under the hood.
    #
    # The key ideas:
    #   1. Extract the 8 corner SDF values for all (N-1)³ voxels at once
    #      using array slicing (no loops needed).
    #   2. Build all cube indices simultaneously with vectorized bitwise ops.
    #   3. For each of the 12 edges, find all voxels where that edge is
    #      intersected, and batch-interpolate their vertices.
    #   4. Assemble triangles by iterating over the 256 MC cases (not voxels).
    # =========================================================================

    M = N - 1  # number of voxels along each axis

    # =========================================================================
    # Step 1: Extract corner SDF values for all voxels at once
    # =========================================================================
    # Corner i is at offset (i&1, (i>>1)&1, (i>>2)&1) relative to voxel origin.
    # For each corner, we slice the full grid to get an (M, M, M) array of
    # that corner's SDF value across every voxel.
    corner_vals = []
    for i in range(8):
        dx = i & 1
        dy = (i >> 1) & 1
        dz = (i >> 2) & 1
        corner_vals.append(voxel_grid[dx:dx+M, dy:dy+M, dz:dz+M])
    # corner_vals[i] has shape (M, M, M) - the SDF at corner i for every voxel

    # =========================================================================
    # Step 2: Build cube indices for all voxels at once
    # =========================================================================
    # cube_index is an 8-bit integer where bit i is set if corner i < 0 (inside).
    # We build this with vectorized bitwise OR across all corners.
    cube_indices = np.zeros((M, M, M), dtype=np.int32)
    for i in range(8):
        cube_indices |= ((corner_vals[i] < 0).astype(np.int32) << i)

    # =========================================================================
    # Step 3: Find active voxels (where the surface passes through)
    # =========================================================================
    # Convert EdgeMasks to a NumPy array for vectorized lookup.
    edge_masks_arr = np.array(EdgeMasks, dtype=np.int32)
    all_edge_masks = edge_masks_arr[cube_indices]  # (M, M, M)
    active_mask = all_edge_masks > 0  # voxels with at least one intersected edge

    # Get (x, y, z) coordinates of all active voxels
    active_x, active_y, active_z = np.where(active_mask)
    n_active = len(active_x)

    if n_active == 0:
        return [], [], ([] if color_grid is not None else None)

    # Gather the cube index and 8 corner SDF values for active voxels only
    active_cube_idx = cube_indices[active_x, active_y, active_z]     # (n_active,)
    active_corners = np.empty((n_active, 8))                          # (n_active, 8)
    for i in range(8):
        active_corners[:, i] = corner_vals[i][active_x, active_y, active_z]
    active_emasks = all_edge_masks[active_x, active_y, active_z]     # (n_active,)

    # =========================================================================
    # Step 4: Vectorized edge interpolation
    # =========================================================================
    # For each of the 12 edges, find which active voxels have that edge
    # intersected, interpolate the vertex position (and color), and record
    # the global vertex ID so triangles can reference it later.
    #
    # edge_vertex_ids[voxel_index, edge_index] = global vertex index
    # This replaces the per-voxel `edge_vertices = [None]*12` sticky-note list.
    edge_vertex_ids = np.full((n_active, 12), -1, dtype=np.int64)

    all_vertices = []     # will be concatenated at the end
    all_colors = []       # (only if color_grid is provided)
    vertex_offset = 0     # running count of total vertices emitted so far

    corner_offsets_arr = np.array(CORNER_OFFSETS, dtype=np.float64)  # (8, 3)

    for edge_i in range(12):
        # Which active voxels have this edge intersected?
        hit = (active_emasks & (1 << edge_i)) != 0  # (n_active,) bool
        n_hit = np.sum(hit)
        if n_hit == 0:
            continue

        # The two corners that define this edge
        c_a, c_b = EDGE_CORNERS[edge_i]
        val_a = active_corners[hit, c_a]  # (n_hit,)
        val_b = active_corners[hit, c_b]  # (n_hit,)

        # Interpolation factor t: where does the zero-crossing fall along the edge?
        denom = val_a - val_b
        t = np.where(np.abs(denom) < 1e-10, 0.5, val_a / denom)  # (n_hit,)

        # Corner offsets for the two endpoints of this edge
        off_a = corner_offsets_arr[c_a]  # (3,)
        off_b = corner_offsets_arr[c_b]  # (3,)

        # Interpolated vertex positions (in grid coordinates)
        hx = active_x[hit].astype(np.float64)
        hy = active_y[hit].astype(np.float64)
        hz = active_z[hit].astype(np.float64)

        vx = hx + off_a[0] + t * (off_b[0] - off_a[0])
        vy = hy + off_a[1] + t * (off_b[1] - off_a[1])
        vz = hz + off_a[2] + t * (off_b[2] - off_a[2])

        edge_verts = np.stack([vx, vy, vz], axis=1)  # (n_hit, 3)
        all_vertices.append(edge_verts)

        # Record the global vertex IDs for these edge intersections
        ids = np.arange(vertex_offset, vertex_offset + n_hit, dtype=np.int64)
        edge_vertex_ids[hit, edge_i] = ids
        vertex_offset += n_hit

        # Interpolate colors if provided
        if color_grid is not None:
            ix = active_x[hit]
            iy = active_y[hit]
            iz = active_z[hit]
            da = CORNER_OFFSETS[c_a]
            db = CORNER_OFFSETS[c_b]
            col_a = color_grid[ix + da[0], iy + da[1], iz + da[2]]  # (n_hit, 3)
            col_b = color_grid[ix + db[0], iy + db[1], iz + db[2]]  # (n_hit, 3)
            interp_col = col_a + t[:, np.newaxis] * (col_b - col_a)  # (n_hit, 3)
            all_colors.append(interp_col)

    # Concatenate all vertices into a single array
    if len(all_vertices) == 0:
        return [], [], ([] if color_grid is not None else None)

    vertices_arr = np.concatenate(all_vertices, axis=0)  # (total_verts, 3)

    # =========================================================================
    # Step 5: Assemble triangles
    # =========================================================================
    # Instead of looping over every active voxel and walking its TriangleTable
    # entry one by one, we group voxels by cube_index (only 256 possible values)
    # and emit all triangles for each case in one batch.
    tri_list_all = []

    for case_idx in range(256):
        tri_entry = TriangleTable[case_idx]
        if tri_entry[0] == -1:
            continue  # no triangles for this case

        # Find active voxels with this cube index
        case_mask = active_cube_idx == case_idx
        n_case = np.sum(case_mask)
        if n_case == 0:
            continue

        # Read edge indices from the triangle table in groups of 3
        case_evids = edge_vertex_ids[case_mask]  # (n_case, 12)

        i = 0
        while i < len(tri_entry) and tri_entry[i] != -1:
            e0, e1, e2 = tri_entry[i], tri_entry[i+1], tri_entry[i+2]
            v0 = case_evids[:, e0]  # (n_case,)
            v1 = case_evids[:, e1]
            v2 = case_evids[:, e2]
            tris = np.stack([v0, v1, v2], axis=1)  # (n_case, 3)
            tri_list_all.append(tris)
            i += 3

    # =========================================================================
    # Step 6: Convert to output format
    # =========================================================================
    # Convert numpy arrays to lists of tuples (matching the original API)
    vertices = [tuple(v) for v in vertices_arr]

    if len(tri_list_all) > 0:
        triangles_arr = np.concatenate(tri_list_all, axis=0)
        triangles = [tuple(t) for t in triangles_arr]
    else:
        triangles = []

    vertex_colors = None
    if color_grid is not None:
        if len(all_colors) > 0:
            colors_arr = np.concatenate(all_colors, axis=0)
            vertex_colors = [tuple(c) for c in colors_arr]
        else:
            vertex_colors = []

    # =========================================================================
    # CONCEPTUAL SUMMARY (same algorithm, vectorized execution):
    #
    # 1. The warehouse (vertices):
    #    All vertex coordinates are computed in bulk by edge, then concatenated.
    #
    # 2. The sticky notes (edge_vertex_ids):
    #    A (n_active, 12) array maps each active voxel's 12 edges to global
    #    vertex IDs - replacing the per-voxel `edge_vertices = [None]*12`.
    #
    # 3. The intersection (Interpolation):
    #    For each edge, all intersected voxels are interpolated simultaneously
    #    with vectorized NumPy ops (no per-voxel Python loop).
    #
    # 4. Drawing triangles:
    #    We group voxels by cube_index and emit all triangles for each MC case
    #    in one batch (256 iterations, not (N-1)³ iterations).
    # =========================================================================

    return vertices, triangles, vertex_colors


def export_obj(vertices, triangles, output_path, vertex_colors=None):
    """Export mesh to OBJ file. Optionally includes vertex colors (v x y z r g b)."""
    with open(output_path, "w") as f:
        f.write("# Marching Cubes output\n")
        f.write(f"# {len(vertices)} vertices, {len(triangles)} triangles\n\n")

        # Write vertices (with optional RGB color appended)
        if vertex_colors is not None:
            # Safety: pad colors if shorter than vertices (prevents zip truncation)
            if len(vertex_colors) < len(vertices):
                print(f"  WARNING: {len(vertices)} vertices but {len(vertex_colors)} colors, padding with gray")
                vertex_colors = list(vertex_colors) + [(0.5, 0.5, 0.5)] * (len(vertices) - len(vertex_colors))
            for (vx, vy, vz), (r, g, b) in zip(vertices, vertex_colors):
                f.write(f"v {vx} {vy} {vz} {r:.4f} {g:.4f} {b:.4f}\n")
        else:
            for vx, vy, vz in vertices:
                f.write(f"v {vx} {vy} {vz}\n")

        f.write("\n")

        # Write faces (OBJ uses 1-based indices), skip invalid ones
        n_verts = len(vertices)
        skipped = 0
        for v0, v1, v2 in triangles:
            if v0 >= n_verts or v1 >= n_verts or v2 >= n_verts:
                skipped += 1
                continue
            f.write(f"f {v0 + 1} {v1 + 1} {v2 + 1}\n")
        if skipped > 0:
            print(f"  WARNING: Skipped {skipped} faces with invalid vertex indices")

    print(f"Mesh exported to: {output_path}")


# ============================================================================
# Run the sphere example when this file is executed directly
# ============================================================================
if __name__ == "__main__":
    import os

    # Sphere example
    N = 64
    voxel_grid_numpy = np.zeros((N, N, N))

    center = N / 2
    radius = 20

    for i in range(N):
        for j in range(N):
            for k in range(N):
                distance = np.sqrt((i - center)**2 + (j - center)**2 + (k - center)**2)
                voxel_grid_numpy[i, j, k] = radius - distance

    import matplotlib.pyplot as plt

    # Grab the slice exactly halfway through the Z-axis (index 32)
    middle_slice = voxel_grid_numpy[:, :, int(N/2)]

    # Plot it using a color map where 0 is white, positive is red, negative is blue
    # plt.figure(figsize=(6, 6))
    # plt.title("2D Slice of the 3D Voxel Grid")
    # plt.imshow(middle_slice, cmap='RdBu', origin='lower')
    # plt.colorbar(label="SDF Value (Distance to surface)")

    # # Draw a contour line exactly where the value is 0 (the surface)
    # plt.contour(middle_slice, levels=[0], colors='black', linewidths=2)

    # plt.show()

    # Run Marching Cubes
    vertices, triangles, _ = run_marching_cubes(voxel_grid_numpy, N)
    print(f"Marching Cubes complete: {len(vertices)} vertices, {len(triangles)} triangles")

    # Export
    output_path = os.path.join(os.path.dirname(__file__), "output_mesh.obj")
    export_obj(vertices, triangles, output_path)