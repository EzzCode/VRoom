import numpy as np

# ============================================================================
# Marching Cubes Lookup Tables
# ============================================================================
#
# Corner ordering follows:
#     i = cube index [0, 7]
#     x = (i & 1) >> 0
#     y = (i & 2) >> 1
#     z = (i & 4) >> 2
#
# Vertex layout:
#
#            6             7
#            +-------------+
#          / |           / |
#        /   |         /   |
#    2 +-----+-------+  3  |
#      |   4 +-------+-----+ 5
#      |   /         |   /
#      | /           | /
#    0 +-------------+ 1

# 3D offsets for each corner relative to voxel origin (x, y, z)
# Corner i is at ( i&1, (i&2)>>1, (i&4)>>2 )
CORNER_OFFSETS = [
    (0, 0, 0),  # corner 0
    (1, 0, 0),  # corner 1
    (0, 1, 0),  # corner 2
    (1, 1, 0),  # corner 3
    (0, 0, 1),  # corner 4
    (1, 0, 1),  # corner 5
    (0, 1, 1),  # corner 6
    (1, 1, 1),  # corner 7
]

# Pair of vertex indices for each edge on the cube
EDGE_CORNERS = [
    (0, 1),   # edge 0
    (1, 3),   # edge 1
    (3, 2),   # edge 2
    (2, 0),   # edge 3
    (4, 5),   # edge 4
    (5, 7),   # edge 5
    (7, 6),   # edge 6
    (6, 4),   # edge 7
    (0, 4),   # edge 8
    (1, 5),   # edge 9
    (3, 7),   # edge 10
    (2, 6),   # edge 11
]

# For each MC case, a mask of edge indices that need to be split
EdgeMasks = [
    0x0, 0x109, 0x203, 0x30a, 0x80c, 0x905, 0xa0f, 0xb06,
    0x406, 0x50f, 0x605, 0x70c, 0xc0a, 0xd03, 0xe09, 0xf00,
    0x190, 0x99, 0x393, 0x29a, 0x99c, 0x895, 0xb9f, 0xa96,
    0x596, 0x49f, 0x795, 0x69c, 0xd9a, 0xc93, 0xf99, 0xe90,
    0x230, 0x339, 0x33, 0x13a, 0xa3c, 0xb35, 0x83f, 0x936,
    0x636, 0x73f, 0x435, 0x53c, 0xe3a, 0xf33, 0xc39, 0xd30,
    0x3a0, 0x2a9, 0x1a3, 0xaa, 0xbac, 0xaa5, 0x9af, 0x8a6,
    0x7a6, 0x6af, 0x5a5, 0x4ac, 0xfaa, 0xea3, 0xda9, 0xca0,
    0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc, 0x1c5, 0x2cf, 0x3c6,
    0xcc6, 0xdcf, 0xec5, 0xfcc, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
    0x950, 0x859, 0xb53, 0xa5a, 0x15c, 0x55, 0x35f, 0x256,
    0xd56, 0xc5f, 0xf55, 0xe5c, 0x55a, 0x453, 0x759, 0x650,
    0xaf0, 0xbf9, 0x8f3, 0x9fa, 0x2fc, 0x3f5, 0xff, 0x1f6,
    0xef6, 0xfff, 0xcf5, 0xdfc, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
    0xb60, 0xa69, 0x963, 0x86a, 0x36c, 0x265, 0x16f, 0x66,
    0xf66, 0xe6f, 0xd65, 0xc6c, 0x76a, 0x663, 0x569, 0x460,
    0x460, 0x569, 0x663, 0x76a, 0xc6c, 0xd65, 0xe6f, 0xf66,
    0x66, 0x16f, 0x265, 0x36c, 0x86a, 0x963, 0xa69, 0xb60,
    0x5f0, 0x4f9, 0x7f3, 0x6fa, 0xdfc, 0xcf5, 0xfff, 0xef6,
    0x1f6, 0xff, 0x3f5, 0x2fc, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
    0x650, 0x759, 0x453, 0x55a, 0xe5c, 0xf55, 0xc5f, 0xd56,
    0x256, 0x35f, 0x55, 0x15c, 0xa5a, 0xb53, 0x859, 0x950,
    0x7c0, 0x6c9, 0x5c3, 0x4ca, 0xfcc, 0xec5, 0xdcf, 0xcc6,
    0x3c6, 0x2cf, 0x1c5, 0xcc, 0xbca, 0xac3, 0x9c9, 0x8c0,
    0xca0, 0xda9, 0xea3, 0xfaa, 0x4ac, 0x5a5, 0x6af, 0x7a6,
    0x8a6, 0x9af, 0xaa5, 0xbac, 0xaa, 0x1a3, 0x2a9, 0x3a0,
    0xd30, 0xc39, 0xf33, 0xe3a, 0x53c, 0x435, 0x73f, 0x636,
    0x936, 0x83f, 0xb35, 0xa3c, 0x13a, 0x33, 0x339, 0x230,
    0xe90, 0xf99, 0xc93, 0xd9a, 0x69c, 0x795, 0x49f, 0x596,
    0xa96, 0xb9f, 0x895, 0x99c, 0x29a, 0x393, 0x99, 0x190,
    0xf00, 0xe09, 0xd03, 0xc0a, 0x70c, 0x605, 0x50f, 0x406,
    0xb06, 0xa0f, 0x905, 0x80c, 0x30a, 0x203, 0x109, 0x0,
]

# When TriangleTable gives you [0, 3, 8, -1], it is giving you a set of instructions:
# "Draw a single triangle. Put the first corner of the triangle on Edge 0, the second corner
#  on Edge 3, and the third corner on Edge 8."

# Step A: The Discovery (EdgeMasks)
# The code looks at EdgeMasks[1] and sees 0x109.
# The binary math tells code: "the surface slices through Edge 0, Edge 3, and Edge 8."

# Step B: The Interpolation (Finding the Coordinates)
# Because the code knows Edge 0 is sliced, it triggers interpolate_vertex function.

# It looks at the actual SDF numbers on the two ends of Edge 0.

# It calculates the exact continuous coordinate, for example (10.5, 12.0, 15.0).

# It repeats this for Edge 3 and gets (10.0, 12.2, 15.0).

# It repeats this for Edge 8 and gets (10.0, 12.0, 15.8).

# Step C: The Temporary Storage
# The code needs a place to hold these coordinates while it finishes looking at the cube.
# It puts them in a temporary mini-list called edge_vertices that has 12 empty slots
# (one for each edge).

# edge_vertices[0] = (10.5, 12.0, 15.0)

# edge_vertices[3] = (10.0, 12.2, 15.0)

# edge_vertices[8] = (10.0, 12.0, 15.8)

# Step D: Drawing the Triangle (TriangleTable)
# Now, the code looks at TriangleTable[1] and reads the instruction: [0, 3, 8, -1].
# It translates that instruction by looking inside its temporary storage:

# Get the coordinate saved in slot 0 -> (10.5, 12.0, 15.0)

# Get the coordinate saved in slot 3 -> (10.0, 12.2, 15.0)

# Get the coordinate saved in slot 8 -> (10.0, 12.0, 15.8)

# For each MC case, a list of triangles specified as triples of edge indices, terminated by -1
TriangleTable = [
    [-1],
    [0, 3, 8, -1],
    [0, 9, 1, -1],
    [3, 8, 1, 1, 8, 9, -1],
    [2, 11, 3, -1],
    [8, 0, 11, 11, 0, 2, -1],
    [3, 2, 11, 1, 0, 9, -1],
    [11, 1, 2, 11, 9, 1, 11, 8, 9, -1],
    [1, 10, 2, -1],
    [0, 3, 8, 2, 1, 10, -1],
    [10, 2, 9, 9, 2, 0, -1],
    [8, 2, 3, 8, 10, 2, 8, 9, 10, -1],
    [11, 3, 10, 10, 3, 1, -1],
    [10, 0, 1, 10, 8, 0, 10, 11, 8, -1],
    [9, 3, 0, 9, 11, 3, 9, 10, 11, -1],
    [8, 9, 11, 11, 9, 10, -1],
    [4, 8, 7, -1],
    [7, 4, 3, 3, 4, 0, -1],
    [4, 8, 7, 0, 9, 1, -1],
    [1, 4, 9, 1, 7, 4, 1, 3, 7, -1],
    [8, 7, 4, 11, 3, 2, -1],
    [4, 11, 7, 4, 2, 11, 4, 0, 2, -1],
    [0, 9, 1, 8, 7, 4, 11, 3, 2, -1],
    [7, 4, 11, 11, 4, 2, 2, 4, 9, 2, 9, 1, -1],
    [4, 8, 7, 2, 1, 10, -1],
    [7, 4, 3, 3, 4, 0, 10, 2, 1, -1],
    [10, 2, 9, 9, 2, 0, 7, 4, 8, -1],
    [10, 2, 3, 10, 3, 4, 3, 7, 4, 9, 10, 4, -1],
    [1, 10, 3, 3, 10, 11, 4, 8, 7, -1],
    [10, 11, 1, 11, 7, 4, 1, 11, 4, 1, 4, 0, -1],
    [7, 4, 8, 9, 3, 0, 9, 11, 3, 9, 10, 11, -1],
    [7, 4, 11, 4, 9, 11, 9, 10, 11, -1],
    [9, 4, 5, -1],
    [9, 4, 5, 8, 0, 3, -1],
    [4, 5, 0, 0, 5, 1, -1],
    [5, 8, 4, 5, 3, 8, 5, 1, 3, -1],
    [9, 4, 5, 11, 3, 2, -1],
    [2, 11, 0, 0, 11, 8, 5, 9, 4, -1],
    [4, 5, 0, 0, 5, 1, 11, 3, 2, -1],
    [5, 1, 4, 1, 2, 11, 4, 1, 11, 4, 11, 8, -1],
    [1, 10, 2, 5, 9, 4, -1],
    [9, 4, 5, 0, 3, 8, 2, 1, 10, -1],
    [2, 5, 10, 2, 4, 5, 2, 0, 4, -1],
    [10, 2, 5, 5, 2, 4, 4, 2, 3, 4, 3, 8, -1],
    [11, 3, 10, 10, 3, 1, 4, 5, 9, -1],
    [4, 5, 9, 10, 0, 1, 10, 8, 0, 10, 11, 8, -1],
    [11, 3, 0, 11, 0, 5, 0, 4, 5, 10, 11, 5, -1],
    [4, 5, 8, 5, 10, 8, 10, 11, 8, -1],
    [8, 7, 9, 9, 7, 5, -1],
    [3, 9, 0, 3, 5, 9, 3, 7, 5, -1],
    [7, 0, 8, 7, 1, 0, 7, 5, 1, -1],
    [7, 5, 3, 3, 5, 1, -1],
    [5, 9, 7, 7, 9, 8, 2, 11, 3, -1],
    [2, 11, 7, 2, 7, 9, 7, 5, 9, 0, 2, 9, -1],
    [2, 11, 3, 7, 0, 8, 7, 1, 0, 7, 5, 1, -1],
    [2, 11, 1, 11, 7, 1, 7, 5, 1, -1],
    [8, 7, 9, 9, 7, 5, 2, 1, 10, -1],
    [10, 2, 1, 3, 9, 0, 3, 5, 9, 3, 7, 5, -1],
    [7, 5, 8, 5, 10, 2, 8, 5, 2, 8, 2, 0, -1],
    [10, 2, 5, 2, 3, 5, 3, 7, 5, -1],
    [8, 7, 5, 8, 5, 9, 11, 3, 10, 3, 1, 10, -1],
    [5, 11, 7, 10, 11, 5, 1, 9, 0, -1],
    [11, 5, 10, 7, 5, 11, 8, 3, 0, -1],
    [5, 11, 7, 10, 11, 5, -1],
    [6, 7, 11, -1],
    [7, 11, 6, 3, 8, 0, -1],
    [6, 7, 11, 0, 9, 1, -1],
    [9, 1, 8, 8, 1, 3, 6, 7, 11, -1],
    [3, 2, 7, 7, 2, 6, -1],
    [0, 7, 8, 0, 6, 7, 0, 2, 6, -1],
    [6, 7, 2, 2, 7, 3, 9, 1, 0, -1],
    [6, 7, 8, 6, 8, 1, 8, 9, 1, 2, 6, 1, -1],
    [11, 6, 7, 10, 2, 1, -1],
    [3, 8, 0, 11, 6, 7, 10, 2, 1, -1],
    [0, 9, 2, 2, 9, 10, 7, 11, 6, -1],
    [6, 7, 11, 8, 2, 3, 8, 10, 2, 8, 9, 10, -1],
    [7, 10, 6, 7, 1, 10, 7, 3, 1, -1],
    [8, 0, 7, 7, 0, 6, 6, 0, 1, 6, 1, 10, -1],
    [7, 3, 6, 3, 0, 9, 6, 3, 9, 6, 9, 10, -1],
    [6, 7, 10, 7, 8, 10, 8, 9, 10, -1],
    [11, 6, 8, 8, 6, 4, -1],
    [6, 3, 11, 6, 0, 3, 6, 4, 0, -1],
    [11, 6, 8, 8, 6, 4, 1, 0, 9, -1],
    [1, 3, 9, 3, 11, 6, 9, 3, 6, 9, 6, 4, -1],
    [2, 8, 3, 2, 4, 8, 2, 6, 4, -1],
    [4, 0, 6, 6, 0, 2, -1],
    [9, 1, 0, 2, 8, 3, 2, 4, 8, 2, 6, 4, -1],
    [9, 1, 4, 1, 2, 4, 2, 6, 4, -1],
    [4, 8, 6, 6, 8, 11, 1, 10, 2, -1],
    [1, 10, 2, 6, 3, 11, 6, 0, 3, 6, 4, 0, -1],
    [11, 6, 4, 11, 4, 8, 10, 2, 9, 2, 0, 9, -1],
    [10, 4, 9, 6, 4, 10, 11, 2, 3, -1],
    [4, 8, 3, 4, 3, 10, 3, 1, 10, 6, 4, 10, -1],
    [1, 10, 0, 10, 6, 0, 6, 4, 0, -1],
    [4, 10, 6, 9, 10, 4, 0, 8, 3, -1],
    [4, 10, 6, 9, 10, 4, -1],
    [6, 7, 11, 4, 5, 9, -1],
    [4, 5, 9, 7, 11, 6, 3, 8, 0, -1],
    [1, 0, 5, 5, 0, 4, 11, 6, 7, -1],
    [11, 6, 7, 5, 8, 4, 5, 3, 8, 5, 1, 3, -1],
    [3, 2, 7, 7, 2, 6, 9, 4, 5, -1],
    [5, 9, 4, 0, 7, 8, 0, 6, 7, 0, 2, 6, -1],
    [3, 2, 6, 3, 6, 7, 1, 0, 5, 0, 4, 5, -1],
    [6, 1, 2, 5, 1, 6, 4, 7, 8, -1],
    [10, 2, 1, 6, 7, 11, 4, 5, 9, -1],
    [0, 3, 8, 4, 5, 9, 11, 6, 7, 10, 2, 1, -1],
    [7, 11, 6, 2, 5, 10, 2, 4, 5, 2, 0, 4, -1],
    [8, 4, 7, 5, 10, 6, 3, 11, 2, -1],
    [9, 4, 5, 7, 10, 6, 7, 1, 10, 7, 3, 1, -1],
    [10, 6, 5, 7, 8, 4, 1, 9, 0, -1],
    [4, 3, 0, 7, 3, 4, 6, 5, 10, -1],
    [10, 6, 5, 8, 4, 7, -1],
    [9, 6, 5, 9, 11, 6, 9, 8, 11, -1],
    [11, 6, 3, 3, 6, 0, 0, 6, 5, 0, 5, 9, -1],
    [11, 6, 5, 11, 5, 0, 5, 1, 0, 8, 11, 0, -1],
    [11, 6, 3, 6, 5, 3, 5, 1, 3, -1],
    [9, 8, 5, 8, 3, 2, 5, 8, 2, 5, 2, 6, -1],
    [5, 9, 6, 9, 0, 6, 0, 2, 6, -1],
    [1, 6, 5, 2, 6, 1, 3, 0, 8, -1],
    [1, 6, 5, 2, 6, 1, -1],
    [2, 1, 10, 9, 6, 5, 9, 11, 6, 9, 8, 11, -1],
    [9, 0, 1, 3, 11, 2, 5, 10, 6, -1],
    [11, 0, 8, 2, 0, 11, 10, 6, 5, -1],
    [3, 11, 2, 5, 10, 6, -1],
    [1, 8, 3, 9, 8, 1, 5, 10, 6, -1],
    [6, 5, 10, 0, 1, 9, -1],
    [8, 3, 0, 5, 10, 6, -1],
    [6, 5, 10, -1],
    [10, 5, 6, -1],
    [0, 3, 8, 6, 10, 5, -1],
    [10, 5, 6, 9, 1, 0, -1],
    [3, 8, 1, 1, 8, 9, 6, 10, 5, -1],
    [2, 11, 3, 6, 10, 5, -1],
    [8, 0, 11, 11, 0, 2, 5, 6, 10, -1],
    [1, 0, 9, 2, 11, 3, 6, 10, 5, -1],
    [5, 6, 10, 11, 1, 2, 11, 9, 1, 11, 8, 9, -1],
    [5, 6, 1, 1, 6, 2, -1],
    [5, 6, 1, 1, 6, 2, 8, 0, 3, -1],
    [6, 9, 5, 6, 0, 9, 6, 2, 0, -1],
    [6, 2, 5, 2, 3, 8, 5, 2, 8, 5, 8, 9, -1],
    [3, 6, 11, 3, 5, 6, 3, 1, 5, -1],
    [8, 0, 1, 8, 1, 6, 1, 5, 6, 11, 8, 6, -1],
    [11, 3, 6, 6, 3, 5, 5, 3, 0, 5, 0, 9, -1],
    [5, 6, 9, 6, 11, 9, 11, 8, 9, -1],
    [5, 6, 10, 7, 4, 8, -1],
    [0, 3, 4, 4, 3, 7, 10, 5, 6, -1],
    [5, 6, 10, 4, 8, 7, 0, 9, 1, -1],
    [6, 10, 5, 1, 4, 9, 1, 7, 4, 1, 3, 7, -1],
    [7, 4, 8, 6, 10, 5, 2, 11, 3, -1],
    [10, 5, 6, 4, 11, 7, 4, 2, 11, 4, 0, 2, -1],
    [4, 8, 7, 6, 10, 5, 3, 2, 11, 1, 0, 9, -1],
    [1, 2, 10, 11, 7, 6, 9, 5, 4, -1],
    [2, 1, 6, 6, 1, 5, 8, 7, 4, -1],
    [0, 3, 7, 0, 7, 4, 2, 1, 6, 1, 5, 6, -1],
    [8, 7, 4, 6, 9, 5, 6, 0, 9, 6, 2, 0, -1],
    [7, 2, 3, 6, 2, 7, 5, 4, 9, -1],
    [4, 8, 7, 3, 6, 11, 3, 5, 6, 3, 1, 5, -1],
    [5, 0, 1, 4, 0, 5, 7, 6, 11, -1],
    [9, 5, 4, 6, 11, 7, 0, 8, 3, -1],
    [11, 7, 6, 9, 5, 4, -1],
    [6, 10, 4, 4, 10, 9, -1],
    [6, 10, 4, 4, 10, 9, 3, 8, 0, -1],
    [0, 10, 1, 0, 6, 10, 0, 4, 6, -1],
    [6, 10, 1, 6, 1, 8, 1, 3, 8, 4, 6, 8, -1],
    [9, 4, 10, 10, 4, 6, 3, 2, 11, -1],
    [2, 11, 8, 2, 8, 0, 6, 10, 4, 10, 9, 4, -1],
    [11, 3, 2, 0, 10, 1, 0, 6, 10, 0, 4, 6, -1],
    [6, 8, 4, 11, 8, 6, 2, 10, 1, -1],
    [4, 1, 9, 4, 2, 1, 4, 6, 2, -1],
    [3, 8, 0, 4, 1, 9, 4, 2, 1, 4, 6, 2, -1],
    [6, 2, 4, 4, 2, 0, -1],
    [3, 8, 2, 8, 4, 2, 4, 6, 2, -1],
    [4, 6, 9, 6, 11, 3, 9, 6, 3, 9, 3, 1, -1],
    [8, 6, 11, 4, 6, 8, 9, 0, 1, -1],
    [11, 3, 6, 3, 0, 6, 0, 4, 6, -1],
    [8, 6, 11, 4, 6, 8, -1],
    [10, 7, 6, 10, 8, 7, 10, 9, 8, -1],
    [3, 7, 0, 7, 6, 10, 0, 7, 10, 0, 10, 9, -1],
    [6, 10, 7, 7, 10, 8, 8, 10, 1, 8, 1, 0, -1],
    [6, 10, 7, 10, 1, 7, 1, 3, 7, -1],
    [3, 2, 11, 10, 7, 6, 10, 8, 7, 10, 9, 8, -1],
    [2, 9, 0, 10, 9, 2, 6, 11, 7, -1],
    [0, 8, 3, 7, 6, 11, 1, 2, 10, -1],
    [7, 6, 11, 1, 2, 10, -1],
    [2, 1, 9, 2, 9, 7, 9, 8, 7, 6, 2, 7, -1],
    [2, 7, 6, 3, 7, 2, 0, 1, 9, -1],
    [8, 7, 0, 7, 6, 0, 6, 2, 0, -1],
    [7, 2, 3, 6, 2, 7, -1],
    [8, 1, 9, 3, 1, 8, 11, 7, 6, -1],
    [11, 7, 6, 1, 9, 0, -1],
    [6, 11, 7, 0, 8, 3, -1],
    [11, 7, 6, -1],
    [7, 11, 5, 5, 11, 10, -1],
    [10, 5, 11, 11, 5, 7, 0, 3, 8, -1],
    [7, 11, 5, 5, 11, 10, 0, 9, 1, -1],
    [7, 11, 10, 7, 10, 5, 3, 8, 1, 8, 9, 1, -1],
    [5, 2, 10, 5, 3, 2, 5, 7, 3, -1],
    [5, 7, 10, 7, 8, 0, 10, 7, 0, 10, 0, 2, -1],
    [0, 9, 1, 5, 2, 10, 5, 3, 2, 5, 7, 3, -1],
    [9, 7, 8, 5, 7, 9, 10, 1, 2, -1],
    [1, 11, 2, 1, 7, 11, 1, 5, 7, -1],
    [8, 0, 3, 1, 11, 2, 1, 7, 11, 1, 5, 7, -1],
    [7, 11, 2, 7, 2, 9, 2, 0, 9, 5, 7, 9, -1],
    [7, 9, 5, 8, 9, 7, 3, 11, 2, -1],
    [3, 1, 7, 7, 1, 5, -1],
    [8, 0, 7, 0, 1, 7, 1, 5, 7, -1],
    [0, 9, 3, 9, 5, 3, 5, 7, 3, -1],
    [9, 7, 8, 5, 7, 9, -1],
    [8, 5, 4, 8, 10, 5, 8, 11, 10, -1],
    [0, 3, 11, 0, 11, 5, 11, 10, 5, 4, 0, 5, -1],
    [1, 0, 9, 8, 5, 4, 8, 10, 5, 8, 11, 10, -1],
    [10, 3, 11, 1, 3, 10, 9, 5, 4, -1],
    [3, 2, 8, 8, 2, 4, 4, 2, 10, 4, 10, 5, -1],
    [10, 5, 2, 5, 4, 2, 4, 0, 2, -1],
    [5, 4, 9, 8, 3, 0, 10, 1, 2, -1],
    [2, 10, 1, 4, 9, 5, -1],
    [8, 11, 4, 11, 2, 1, 4, 11, 1, 4, 1, 5, -1],
    [0, 5, 4, 1, 5, 0, 2, 3, 11, -1],
    [0, 11, 2, 8, 11, 0, 4, 9, 5, -1],
    [5, 4, 9, 2, 3, 11, -1],
    [4, 8, 5, 8, 3, 5, 3, 1, 5, -1],
    [0, 5, 4, 1, 5, 0, -1],
    [5, 4, 9, 3, 0, 8, -1],
    [5, 4, 9, -1],
    [11, 4, 7, 11, 9, 4, 11, 10, 9, -1],
    [0, 3, 8, 11, 4, 7, 11, 9, 4, 11, 10, 9, -1],
    [11, 10, 7, 10, 1, 0, 7, 10, 0, 7, 0, 4, -1],
    [3, 10, 1, 11, 10, 3, 7, 8, 4, -1],
    [3, 2, 10, 3, 10, 4, 10, 9, 4, 7, 3, 4, -1],
    [9, 2, 10, 0, 2, 9, 8, 4, 7, -1],
    [3, 4, 7, 0, 4, 3, 1, 2, 10, -1],
    [7, 8, 4, 10, 1, 2, -1],
    [7, 11, 4, 4, 11, 9, 9, 11, 2, 9, 2, 1, -1],
    [1, 9, 0, 4, 7, 8, 2, 3, 11, -1],
    [7, 11, 4, 11, 2, 4, 2, 0, 4, -1],
    [4, 7, 8, 2, 3, 11, -1],
    [9, 4, 1, 4, 7, 1, 7, 3, 1, -1],
    [7, 8, 4, 1, 9, 0, -1],
    [3, 4, 7, 0, 4, 3, -1],
    [7, 8, 4, -1],
    [11, 10, 8, 8, 10, 9, -1],
    [0, 3, 9, 3, 11, 9, 11, 10, 9, -1],
    [1, 0, 10, 0, 8, 10, 8, 11, 10, -1],
    [10, 3, 11, 1, 3, 10, -1],
    [3, 2, 8, 2, 10, 8, 10, 9, 8, -1],
    [9, 2, 10, 0, 2, 9, -1],
    [8, 3, 0, 10, 1, 2, -1],
    [2, 10, 1, -1],
    [2, 1, 11, 1, 9, 11, 9, 8, 11, -1],
    [11, 2, 3, 9, 0, 1, -1],
    [11, 0, 8, 2, 0, 11, -1],
    [3, 11, 2, -1],
    [1, 8, 3, 9, 8, 1, -1],
    [1, 9, 0, -1],
    [8, 3, 0, -1],
    [-1],
]


# ============================================================================
# Marching Cubes Algorithm
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