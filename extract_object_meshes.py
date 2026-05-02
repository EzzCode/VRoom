"""
Extract single-object meshes using semantic masks.

For each object label found in the semantic maps:
1. Mask depth maps (zero out pixels that don't belong to this object)
2. Unproject remaining pixels to 3D to find the object's bounding box
3. Run TSDF fusion with a tight grid around just that object
4. Export individual colored mesh

Output: objects/ folder with one OBJ per object label.
"""

import numpy as np
import json
import os
import time
from PIL import Image
from generate_sdf import fuse_tsdf
from marching_cubes import run_marching_cubes
from export_ply import export_ply_binary
from utils import remove_small_components, compute_depth_trunc, unproject_to_3d

# ============================================================================
# 1. Load Camera Data
# ============================================================================
input_dir = os.path.join(os.path.dirname(__file__), "inputs")
# Create the objects folder in the same directory as inputs folder
output_dir = os.path.join(os.path.dirname(__file__), "objects")
os.makedirs(output_dir, exist_ok=True)

# Load camera.json to get camera intrinsics and extrinsics
with open(os.path.join(input_dir, "cameras.json"), "r") as f:
    cameras = json.load(f)

# Number of cameras is the same as the number of depth maps
num_depth_files = len(os.listdir(os.path.join(input_dir, "raw_depth")))
cameras = cameras[:num_depth_files]
num_cams = len(cameras)
print(f"Loaded {num_cams} cameras")

# ============================================================================
# 2. Load All Data Once (shared across all objects)
# ============================================================================
print("Loading depth maps, RGB images, and semantic masks...")

depth_maps_raw = []
color_images_raw = []
semantic_maps = []
intrinsics_list = []
extrinsics_list = []

for i, cam in enumerate(cameras):
    # Intrinsics
    fx, fy = cam["fx"], cam["fy"]
    W, H = cam["width"], cam["height"]
    cx, cy = W / 2.0, H / 2.0 # principal point (center of the image)
    # Intrinsics matrix K
    # K = [fx, 0, cx]
    #     [0, fy, cy]
    #     [0, 0,  1 ]
    intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    # Extrinsics, rotation in cameras.json is camera-to-world (C2W)
    R_c2w = np.array(cam["rotation"])
    R_w2c = R_c2w.T  # transpose to get world-to-camera
    pos = np.array(cam["position"])
    extrinsics = np.eye(4) # create 4x4 identity matrix
    extrinsics[:3, :3] = R_w2c # set the top-left 3x3 matrix to the rotation matrix
    extrinsics[:3, 3] = -R_w2c @ pos # set the last column to the translation vector

    # Depth
    depth_maps_raw.append(np.load(os.path.join(input_dir, "raw_depth", f"{i:05d}.npy")))

    # RGB
    rgb = np.array(Image.open(os.path.join(input_dir, "renders", f"{i:05d}.png")))
    # Drop alpha channel, normalize to [0, 1] and append
    color_images_raw.append(rgb[:, :, :3].astype(np.float64) / 255.0)

    # Semantic
    sem = np.array(Image.open(os.path.join(input_dir, "semantic", f"{i:05d}.png")))
    semantic_maps.append(sem)

    intrinsics_list.append(intrinsics)
    extrinsics_list.append(extrinsics)

print("All data loaded.")

# ============================================================================
# 3. Find All Object Labels (across all views)
# ============================================================================
all_labels = set()
for sem in semantic_maps:
    all_labels.update(np.unique(sem)) # discover unique objects
all_labels = sorted(all_labels)
print(f"\nFound {len(all_labels)} unique labels: {all_labels}")

# Count number of pixels for each object across all views (semantic maps is a list
# containing images from each view)
label_counts = {}
for label_id in all_labels:
    total_pixels = sum(np.sum(sem == label_id) for sem in semantic_maps)
    label_counts[label_id] = total_pixels

# Sort by pixel count (process largest objects first)
sorted_labels = sorted(label_counts.keys(), key=lambda l: -label_counts[l])
print("\nLabel pixel counts (across all views):")
for label_id in sorted_labels:
    print(f"  Label {label_id:3d}: {label_counts[label_id]:,} pixels")

# ============================================================================
# 5. Process Each Object
# ============================================================================
# Skip labels with too few pixels (noise) or the dominant background label
MIN_PIXELS = 50000  # minimum total pixels across all views to process
# BBOX_DEPTH_TRUNC is now computed automatically per object (see compute_depth_trunc)
SKIP_LABELS = []

print(f"\n{'='*60}")
print(f"Processing objects (min {MIN_PIXELS:,} pixels)...")
print(f"{'='*60}")
special_label = [70]
for label_id in sorted_labels:
    if label_id in SKIP_LABELS:
        print(f"\n--- Skipping label {label_id} (in skip list) ---")
        continue
    if label_counts[label_id] < MIN_PIXELS:
        print(f"\n--- Skipping label {label_id} ({label_counts[label_id]:,} pixels < {MIN_PIXELS:,} min) ---")
        continue

    print(f"\n{'='*60}")
    print(f"Object label {label_id} ({label_counts[label_id]:,} pixels)")
    print(f"{'='*60}")

    # --- Auto-compute depth_trunc for this object ---
    BBOX_DEPTH_TRUNC = compute_depth_trunc(depth_maps_raw, semantic_maps, label_id)
    print(f"  Auto depth_trunc: {BBOX_DEPTH_TRUNC:.2f} m")

    # --- Step A: Create masked depth maps for this object ---
    masked_depths = []
    masked_colors = []
    for i in range(num_cams):
        d = depth_maps_raw[i].copy()
        mask = (semantic_maps[i] == label_id) & (d > 0) & (d < BBOX_DEPTH_TRUNC)
        d[~mask] = 0  # zero out non-object pixels and far depths
        masked_depths.append(d)
        masked_colors.append(color_images_raw[i])  # color stays full, only depth gates the TSDF

    # --- Step B: Find object's 3D bounding box ---
    all_world_pts = []
    for i in range(num_cams):
        # Only use pixels within depth_trunc for bbox computation
        mask = (semantic_maps[i] == label_id) & (depth_maps_raw[i] > 0) & (depth_maps_raw[i] < BBOX_DEPTH_TRUNC)
        pts = unproject_to_3d(depth_maps_raw[i], mask, intrinsics_list[i], extrinsics_list[i])
        if len(pts) > 0:
            # Subsample for speed (don't need all points for bbox)
            if len(pts) > 5000:
                idx = np.random.choice(len(pts), 5000, replace=False)
                pts = pts[idx]
            all_world_pts.append(pts)

    if len(all_world_pts) == 0:
        print(f"  No 3D points found, skipping.")
        continue

    all_world_pts = np.vstack(all_world_pts)
    # Use percentiles instead of min/max to ignore outliers from noisy masks
    obj_min = np.percentile(all_world_pts, 2, axis=0)
    obj_max = np.percentile(all_world_pts, 98, axis=0)
    obj_size = obj_max - obj_min

    print(f"  3D bbox: min={obj_min.round(3)}, max={obj_max.round(3)}")
    print(f"  3D size: {(obj_size * 100).round(1)} cm")

    # --- Step C: Set grid parameters ---
    padding = 0.05  # 5cm padding instead of 10cm for a tighter resolution mesh
    grid_min = obj_min - padding
    grid_max = obj_max + padding
    grid_size = grid_max - grid_min

    N = 128  # resolution per object (enough for single objects)
    voxel_size = grid_size.max() / N
    trunc_margin = voxel_size * 5
    depth_trunc = BBOX_DEPTH_TRUNC

    print(f"  Grid: {N}^3, voxel={voxel_size:.4f}m, trunc={trunc_margin:.4f}m")

    # --- Step D: Run TSDF fusion ---
    print(f"  Fusing...")
    t0 = time.time()
    fused_grid, fused_colors, obs_count = fuse_tsdf(
        masked_depths, intrinsics_list, extrinsics_list,
        grid_shape=(N, N, N),
        voxel_size=voxel_size,
        trunc_margin=trunc_margin,
        color_images=masked_colors,
        grid_origin=grid_min,
        depth_trunc=depth_trunc
    )
    t1 = time.time()
    print(f"  TSDF fusion: {t1 - t0:.2f}s")

    # --- Step D2: Filter low-confidence voxels (removes flying pixels) ---
    # Voxels seen by fewer than min_obs cameras are unreliable — set to +1 (outside)
    # so Marching Cubes won't generate surfaces there.
    min_obs = 2
    low_conf_mask = obs_count < min_obs
    n_removed = np.sum((fused_grid < 0) & low_conf_mask)
    fused_grid[low_conf_mask] = 1.0
    if fused_colors is not None:
        fused_colors[low_conf_mask] = 0.0
    print(f"  Confidence filter: removed {n_removed} low-confidence surface voxels (min_obs={min_obs})")

    # --- Step E: Run Marching Cubes ---
    print(f"  Marching Cubes...")
    t2 = time.time()
    vertices, triangles, vertex_colors = run_marching_cubes(fused_grid, N, color_grid=fused_colors)
    t3 = time.time()
    print(f"  Result: {len(vertices)} vertices, {len(triangles)} triangles")
    print(f"  Marching Cubes: {t3 - t2:.2f}s")

    if len(triangles) == 0:
        print(f"  No mesh generated, skipping.")
        continue

    # --- Step E2: Remove flying fragments ---
    vertices, triangles, vertex_colors = remove_small_components(
        vertices, triangles, vertex_colors, min_ratio=0.05
    )
    print(f"  After cleanup: {len(vertices)} vertices, {len(triangles)} triangles")

    # Scale to world coordinates (vectorized)
    verts_arr = np.array(vertices) * voxel_size + grid_min
    scaled_vertices = [tuple(v) for v in verts_arr]

    # --- Step F: Export ---
    t4 = time.time()
    ply_path = os.path.join(output_dir, f"object_{label_id:03d}.ply")
    export_ply_binary(scaled_vertices, triangles, ply_path, vertex_colors=vertex_colors)
    t5 = time.time()
    print(f"  PLY export: {t5 - t4:.2f}s")
    print(f"  Saved: {ply_path}")

print(f"\n{'='*60}")
print(f"Done! Check the 'objects/' folder for individual meshes.")
print(f"{'='*60}")
