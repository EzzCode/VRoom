import numpy as np

def export_ply_binary(vertices, triangles, output_path, vertex_colors=None):
    """Exports a mesh to a binary PLY file. PLY files are much smaller than OBJ."""

    header = "" # initialize header string
    header += "ply\n" # file type
    header += "format binary_little_endian 1.0\n" # format

    header += "element vertex " # number of vertices
    header += str(len(vertices)) + "\n"

    # properties
    header += "property float x\n"
    header += "property float y\n"
    header += "property float z\n"

    if vertex_colors is not None and len(vertex_colors) == len(vertices):
        header += "property uchar red\n"
        header += "property uchar green\n"
        header += "property uchar blue\n"

    header += "element face " # number of faces
    header += str(len(triangles)) + "\n"

    # For every face, read uchar-8-bit int (number of points in shape, always 3),
    # then 32-bit integers (vertex IDs)
    header += "property list uchar int vertex_indices\n"
    header += "end_header\n"

    # Open target file in binary mode and write header encoded to ASCII
    with open(output_path, 'wb') as f:
        f.write(header.encode('ascii'))

        # Convert list to np array and write to file
        vertices_arr = np.array(vertices, dtype=np.float32) 

        # Convert vertex colors to np array and write to file
        # Format: [V1_X, V1_Y, V1_Z, V1_Red, V1_Green, V1_Blue] [V2_X, V2_Y, V2_Z, V2_Red...]
        if vertex_colors is not None and len(vertex_colors) == len(vertices):
            # Convert colors from [0,1] float to [0,255] uint8, use clip for safety
            vertex_colors_uint8_arr = np.clip((np.array(vertex_colors) * 255), 0, 255).astype(np.uint8)

            vertex_record = np.dtype([('pos', '<f4', 3), ('col', 'u1', 3)]) # define structure

            buffer = np.empty(len(vertices), dtype=vertex_record)
            buffer['pos'] = vertices_arr
            buffer['col'] = vertex_colors_uint8_arr

            f.write(buffer.tobytes())
        else:
            f.write(vertices_arr.tobytes())

        # Write faces
        # PLY face format: [number_of_vertices(uint8), v1(int32), v2(int32), v3(int32)]
        # = 1 + 12 = 13 bytes per face
        face_record = np.dtype([('count', 'u1'), ('indices', '<i4', 3)])
        face_buffer = np.empty(len(triangles), dtype=face_record)
            
        face_buffer['count'] = 3
        face_buffer['indices'] = np.array(triangles, dtype=np.int32)
        f.write(face_buffer.tobytes())