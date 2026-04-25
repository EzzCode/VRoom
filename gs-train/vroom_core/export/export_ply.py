import struct

def export_ply_binary(vertices, triangles, filename, vertex_colors=None):
    """
    Exports a mesh to a binary PLY file without relying on external libraries like Open3D.
    This saves massive amounts of space compared to OBJ files and loads instantly.
    
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

        # 2. Write Vertices (Binary)
        for i in range(len(vertices)):
            v = vertices[i]
            # Pack 3 floats (x, y, z) into binary format '<3f'
            f.write(struct.pack('<3f', v[0], v[1], v[2]))
            
            if has_colors:
                c = vertex_colors[i]
                # Scale [0, 1] to [0, 255]
                r, g, b = int(c[0]*255), int(c[1]*255), int(c[2]*255)
                # Pack 3 unsigned chars (r, g, b) into binary format '<3B'
                f.write(struct.pack('<3B', r, g, b))

        # 3. Write Faces (Binary)
        for t in triangles:
            # PLY face format: [number_of_vertices, v1, v2, v3]
            # Pack 1 unsigned char (3) and 3 integers (indices) into '<B3i'
            f.write(struct.pack('<B3i', 3, t[0], t[1], t[2]))
