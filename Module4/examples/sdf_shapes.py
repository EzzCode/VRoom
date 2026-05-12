"""
SDF shape prototypes - examples of how to build voxel grids for testing.

Run any section by uncommenting it, then call run_marching_cubes on the result.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch
import matplotlib.pyplot as plt
from marching_cubes import run_marching_cubes, export_obj

N = 64

# ============================================================================
# Sphere example
# ============================================================================
voxel_grid_numpy = np.zeros((N, N, N))
center = N / 2
radius = 20

for i in range(N):
    for j in range(N):
        for k in range(N):
            distance = np.sqrt((i - center)**2 + (j - center)**2 + (k - center)**2)
            voxel_grid_numpy[i, j, k] = radius - distance

############################################

# Torus example
# coords = np.linspace(-1, 1, N)
# x, y, z = np.meshgrid(coords, coords, coords, indexing='ij')
# R_major = 0.5  # Distance from center of hole to center of tube
# r_minor = 0.2  # Radius of the tube itself
# distance = np.sqrt((np.sqrt(x**2 + y**2) - R_major)**2 + z**2)
# voxel_grid_numpy = r_minor - distance

############################################

# Box example
# coords = np.linspace(-1, 1, N)
# x, y, z = np.meshgrid(coords, coords, coords, indexing='ij')
# box_size = 0.5
# distance = np.maximum(np.maximum(np.abs(x), np.abs(y)), np.abs(z))
# voxel_grid_numpy = box_size - distance

############################################

# Merging objects (union of two spheres)
# coords = np.linspace(-1, 1, N)
# x, y, z = np.meshgrid(coords, coords, coords, indexing='ij')
# dist1 = np.sqrt((x + 0.3)**2 + y**2 + z**2)
# dist2 = np.sqrt((x - 0.3)**2 + y**2 + z**2)
# combined_distance = np.minimum(dist1, dist2)
# voxel_grid_numpy = 0.35 - combined_distance

############################################

# Convert NumPy to PyTorch if needed
voxel_grid_torch = torch.from_numpy(voxel_grid_numpy).float()
print(f"Voxel grid shape: {voxel_grid_torch.shape}")

# Visualize a 2D slice
middle_slice = voxel_grid_numpy[:, :, int(N/2)]

plt.figure(figsize=(6, 6))
plt.title("2D Slice of the 3D Voxel Grid")
plt.imshow(middle_slice, cmap='RdBu', origin='lower')
plt.colorbar(label="SDF Value (Distance to surface)")
plt.contour(middle_slice, levels=[0], colors='black', linewidths=2)
plt.show()

# Run Marching Cubes and export
vertices, triangles, _ = run_marching_cubes(voxel_grid_numpy, N)
print(f"Marching Cubes complete: {len(vertices)} vertices, {len(triangles)} triangles")

output_path = os.path.join(os.path.dirname(__file__), "output_mesh.obj")
export_obj(vertices, triangles, output_path)
