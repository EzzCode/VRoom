import struct
import numpy as np

def export_ply_binary(vertices, triangles, filename, vertex_colors=None):
    """
    Exports a mesh to a binary PLY file without relying on external libraries like Open3D.
    This saves massive amounts of space compared to OBJ files and loads instantly.
    
    Uses bulk NumPy operations instead of per-element struct.pack loops for speed.
    
    Inputs:
    - vertices: List of (x, y, z) tuples.
    - triangles: List of (v1, v2, v3) index tuples.
    - filename: Output path.
    - vertex_colors: (Optional) List of (r, g, b) tuples scaled [0, 1].
    """
    has_colors = vertex_colors is not None and len(vertex_colors) == len(vertices)

    # 1. Write the ASCII Header
    # PLY headers must be exact, ending with 'end_header\n'
    header = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {len(vertices)}",
        "property float x",
        "property float y",
        "property float z",
    ]

    if has_colors:
        header.extend([
            "property uchar red",
            "property uchar green",
            "property uchar blue",
        ])

    header.extend([
        f"element face {len(triangles)}",
        "property list uchar int vertex_indices",
        "end_header\n"
    ])

    # Open file in binary write mode
    with open(filename, 'wb') as f:
        # Write header
        f.write('\n'.join(header).encode('ascii'))

        # 2. Write Vertices (Binary) - bulk NumPy operation instead of per-vertex loop
        verts_arr = np.array(vertices, dtype=np.float32)  # (V, 3)

        if has_colors:
            # Convert [0,1] float colors to [0,255] uint8
            colors_arr = np.clip(np.array(vertex_colors, dtype=np.float64) * 255, 0, 255).astype(np.uint8)  # (V, 3)
            
            # Interleave: [x y z r g b] per vertex using a structured dtype
            # Each vertex record = 3 floats (12 bytes) + 3 uint8s (3 bytes) = 15 bytes
            vertex_record = np.dtype([('pos', '<f4', 3), ('col', 'u1', 3)])
            buf = np.empty(len(vertices), dtype=vertex_record)
            buf['pos'] = verts_arr
            buf['col'] = colors_arr
            f.write(buf.tobytes())
        else:
            f.write(verts_arr.tobytes())

        # 3. Write Faces (Binary) - bulk NumPy operation instead of per-face loop
        # PLY face format: [number_of_vertices(uint8), v1(int32), v2(int32), v3(int32)]
        # = 1 + 12 = 13 bytes per face
        face_record = np.dtype([('count', 'u1'), ('indices', '<i4', 3)])
        face_buf = np.empty(len(triangles), dtype=face_record)
        face_buf['count'] = 3
        face_buf['indices'] = np.array(triangles, dtype=np.int32)
        f.write(face_buf.tobytes())
