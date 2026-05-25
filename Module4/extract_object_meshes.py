import argparse
import json
import os
import time

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
import torch

from export_ply import export_ply_binary
from generate_sdf import fuse_tsdf
from marching_cubes import run_marching_cubes
from utils import remove_small_components, compute_depth_trunc, unproject_to_3d

# 1. Parse arguments

parser = argparse.ArgumentParser(description="Main script for extracting meshes")

parser.add_argument("--min_pixels", type=int, default=10000,
                     help="Minimum total numbers of pixels for an object across all scenes" \
                     "to process the object (default: 10000)")
parser.add_argument("--bbox_clip",    type=float, default=2.0,
                     help="Clip outer N%% of 3D points on each side when calculating" \
                     " bounding box (default: 2.0). Lower = tighter bbox," \
                     " higher = more outlier tolerance" \
                     " Padding is done after clipping to re-add any removed valid points.")
parser.add_argument("--padding", type=float, default=0.22,
                     help="Bounding box padding as percentage of object size (default: 22%)")
parser.add_argument("--resolution",   type=int,   default=128,
                     help="TSDF grid resolution N (N^3 voxels, default: 128)." \
                       " Higher = finer mesh but slower.")
parser.add_argument("--trunc_factor", type=float, default=5.0,
                     help="TSDF truncation margin = voxel_size * trunc_factor"
                       " (default: 5.0)")
parser.add_argument("--min_obs",      type=int,   default=3,
                     help="Min cameras that must observe a voxel to keep it (default: 3).")
parser.add_argument("--min_component",type=float, default=0.05,
                     help="Remove mesh fragments smaller than this fraction of" \
                     " the largest component (default: 0.05)")
parser.add_argument("--smooth_sigma", type=float, default=0.8,
                     help="Gaussian smoothing sigma applied to TSDF grid before marching cubes"
                     " (default: 0.8). Try 0.5-2.0 in voxel units.")
parser.add_argument("--depth_margin", type=float, default=1.1,
                     help="Depth truncation margin multiplier (default: 1.1).")
parser.add_argument("--depth_percentile", type=float, default=99.0,
                     help="Percentile of valid object depth values used to compute depth truncation (default: 99.0).")
parser.add_argument("--label",        type=int,   default=None,
                     help="Process only this label ID (default: None (all labels))")
args = parser.parse_args()

# Set parameters from arguments
MIN_PIXELS    = args.min_pixels
BBOX_CLIP     = args.bbox_clip
PADDING       = args.padding
N             = args.resolution
TRUNC_FACTOR  = args.trunc_factor
MIN_OBS       = args.min_obs
MIN_COMPONENT = args.min_component
SMOOTH_SIGMA  = args.smooth_sigma
DEPTH_MARGIN      = args.depth_margin
DEPTH_PERCENTILE  = args.depth_percentile
LABEL_FILTER      = args.label

print ("Loaded Parameters:")
print(f"MIN_PIXELS: {MIN_PIXELS}")
print(f"BBOX_CLIP: {BBOX_CLIP}")
print(f"PADDING: {PADDING}")
print(f"N: {N}")
print(f"TRUNC_FACTOR: {TRUNC_FACTOR}")
print(f"MIN_OBS: {MIN_OBS}")
print(f"MIN_COMPONENT: {MIN_COMPONENT}")
print(f"SMOOTH_SIGMA: {SMOOTH_SIGMA}")
print(f"DEPTH_MARGIN: {DEPTH_MARGIN}")
print(f"DEPTH_PERCENTILE: {DEPTH_PERCENTILE}")
print(f"LABEL_FILTER: {LABEL_FILTER}\n")

start_time = time.time() # keep track of total runtime

# 2. Load camera data

# Set up input and output paths
curr_directory = os.path.dirname(__file__)
input_dir = os.path.join(curr_directory, "inputs")
output_dir = os.path.join(curr_directory, "objects")
os.makedirs(output_dir, exist_ok=True) # create output directory if it doesn't exist

# Load cameras
with open(os.path.join(input_dir, "cameras.json"), 'r') as f:
    cameras = json.load(f)

num_depth_files = len(os.listdir(os.path.join(input_dir, "raw_depth")))
cameras = cameras[:num_depth_files] # make sure number of cameras is same as depths
num_cams = len(cameras)
print(f"Loaded {num_cams} cameras\n")

# 3. Load all data
print("Loading depth maps, RGB images, and semantic masks...")

# Set up lists
depth_maps_raw = []
color_images_raw = []
semantic_maps = []
intrinsics_list = []
extrinsics_list = []


for i, cam in enumerate(cameras):
    # Load depth, color and semantic maps for each view

    depth_maps_raw.append(np.load(os.path.join(input_dir, "raw_depth", f"{i:05d}.npy")))
    rgba = np.array(Image.open(os.path.join(input_dir, "renders", f"{i:05d}.png")))
    # Drop alpha channel and convert to [0,1] range
    color_images_raw.append(rgba[:, :, :3].astype(np.float64) / 255.0)
    semantic_maps.append(np.array(Image.open(os.path.join(input_dir, "semantic", f"{i:05d}.png"))))

    # Load intrinsics and extrinsics

    # Intrinsics
    fx = cam["fx"]
    fy = cam["fy"]
    W = cam["width"]
    H = cam["height"]
    # Principal point
    cx = W / 2.0
    cy = H / 2.
    # Intrinsics matrix K
    # K = [fx, 0, cx]
    #     [0, fy, cy]
    #     [0, 0,  1 ]
    intrinsics_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    intrinsics_list.append(intrinsics_matrix)

    # Extrinsics
    R_c2w = np.array(cam["rotation"]) # rotation from camera to world
    R_w2c = R_c2w.T # rotation from world to camera is transpose of camera to world
    pos = np.array(cam["position"])
    translation_vector = -R_w2c @ pos
    # Extrinsics matrix E
    # E = [R_w2c | t]
    #     [0 0 0 | 1]
    extrinsics_matrix = np.eye(4) # 4x4 identity matrix
    extrinsics_matrix[:3, :3] = R_w2c
    extrinsics_matrix[:3, 3] = translation_vector
    extrinsics_list.append(extrinsics_matrix)

print("Finished loading data.\n")

# 4. Load unique labels from semantic maps
all_labels = set()
for semantic_map in semantic_maps:
    all_labels.update(np.unique(semantic_map))
all_labels = sorted(all_labels)
print(f"Found {len(all_labels)} unique labels across all semantic maps: {all_labels}")

# Count pixels for each label across all views
label_pixel_counts = {}
for label in all_labels:
    total_pixels = 0
    for semantic_map in semantic_maps:
        total_pixels += np.sum(semantic_map == label)
    label_pixel_counts[label] = total_pixels
# Sort labels by pixel count in descending order to process large objects first
sorted_labels = sorted(label_pixel_counts.keys(), key=lambda l: -label_pixel_counts[l])
print("Pixel counts for each label across all views:")
for label in sorted_labels:
    print(f"Label {label}: {label_pixel_counts[label]} pixels")
print()  # blank line after pixel count list

# 5. Process each object

for label_id in sorted_labels:
    torch.cuda.empty_cache() # clear GPU cache to avoid fragmentation issues
    if LABEL_FILTER is not None and label_id != LABEL_FILTER:
        continue
    if label_pixel_counts[label_id] < MIN_PIXELS:
        print("\nSkipping label", label_id, "(", label_pixel_counts[label_id], "pixels <", MIN_PIXELS, "min)")
        continue

    print("\nLabel", label_id, "(", label_pixel_counts[label_id], "pixels)")

    # 5.1: Compute per-object depth truncation
    # Exclude depth pixels that are too far from the camera for the object
    # , as they are likely noise and outliers that can break the mesh extraction.
    BBOX_DEPTH_TRUNC = compute_depth_trunc(depth_maps_raw,
                                           semantic_maps, label_id,
                                           percentile=DEPTH_PERCENTILE, margin=DEPTH_MARGIN)
    print("Depth truncation:", round(BBOX_DEPTH_TRUNC, 2))

    # 5.2: Mask depth maps and collect data for cameras that see this object
    masked_depths = []
    masked_colors = []
    active_intrinsics = [] # keep intrinsics for views with valid pixels for object
    active_extrinsics = [] # keep extrinsics for views with valid pixels for object
    all_world_pts = []
    for i in range(num_cams):
        depth_map = depth_maps_raw[i].copy()
        valid_pixels_mask = ((semantic_maps[i] == label_id) & (depth_map > 0) & 
                            (depth_map < BBOX_DEPTH_TRUNC))
        
        if valid_pixels_mask.sum() == 0: # if no valid pixels for view, skip
            continue
        depth_map[~valid_pixels_mask] = 0 # set invalid pixels to 0
        masked_depths.append(depth_map)
        # Masking depth is enough
        masked_colors.append(color_images_raw[i])
        active_intrinsics.append(intrinsics_list[i])
        active_extrinsics.append(extrinsics_list[i])
        
        # Find 3D pts for bounding box calculation
        pts = unproject_to_3d(depth_map, valid_pixels_mask, 
                              intrinsics_list[i], extrinsics_list[i])
        if len(pts) > 0:
            if len(pts) > 5000: # if too many points, randomly sample to speed up
                # We don't need all points for bounding box calculation
                sampled_indices = np.random.choice(len(pts), size=5000, replace=False)
                pts = pts[sampled_indices]
            all_world_pts.append(pts)

    if len(all_world_pts) == 0:
        print("No 3D points found for label", label_id, ", skipping.")
        continue
    all_world_pts = np.vstack(all_world_pts) # stack all pts for bbox calculation

    # 5.3: Calculate 3D bounding box from unprojected points

    # Use percentiles instead of min/max to ignore outliers from noisy masks
    # so if there are outliers in depth values of an object, we ignore them
    # and get a tighter bounding box. If the object extends beyond the calculated box,
    # padding will keep it included.
    obj_min = np.percentile(all_world_pts, BBOX_CLIP, axis=0)
    obj_max = np.percentile(all_world_pts, 100 - BBOX_CLIP, axis=0)
    obj_size = obj_max - obj_min # size of object in world units
    print("Bbox: min =", obj_min.round(3), "max =", obj_max.round(3), "size =", obj_size.round(3))

    # 5.4: Compute TSDF grid parameters (pad bbox then calculate voxel size)
    padding_world_units = obj_size.max() * PADDING
    grid_min = obj_min - padding_world_units
    grid_max = obj_max + padding_world_units
    grid_size = grid_max - grid_min # width, height, depth of grid in world units

    voxel_size = grid_size.max() / N # max to make sure object fits in grid
    trunc_margin = voxel_size * TRUNC_FACTOR # tsdf truncation margin
    depth_trunc = BBOX_DEPTH_TRUNC
    print("Grid:", str(N) + "^3, voxel =", round(voxel_size, 4), "trunc =", round(trunc_margin, 4))

    # 5.5: Run TSDF fusion
    print("Fusing", len(masked_depths), "cameras (skipped", num_cams - len(masked_depths), ")...")
    t_fusion = time.time()
    fused_grid, fused_colors, obs_count = fuse_tsdf(
        masked_depths, active_intrinsics, active_extrinsics,
        grid_shape=(N, N, N),
        voxel_size=voxel_size,
        trunc_margin=trunc_margin,
        color_images=masked_colors,
        grid_origin=grid_min,
        depth_trunc=depth_trunc
    )
    print("TSDF fusion:", round(time.time() - t_fusion, 2), "s")

    # 5.6: Filter low-confidence voxels
    low_confidence_mask = obs_count < MIN_OBS # voxels seen by < MIN_OBS cameras

    # Count how many voxels were removed
    num_voxels_removed = np.sum(low_confidence_mask)

    fused_grid[low_confidence_mask] = 1.0 # set their sdf values to +1 (outside object)
    if fused_colors is not None:
        # To be safe, set colors of low-confidence voxels to black
        fused_colors[low_confidence_mask] = 0.0
    print("Confidence filter:", num_voxels_removed, "voxels removed (min_obs =", MIN_OBS, ")")

    # 5.7: Apply Gaussian smoothing to the TSDF grid for smoother meshes
    if SMOOTH_SIGMA > 0:
        fused_grid = gaussian_filter(fused_grid.astype(np.float32), sigma=SMOOTH_SIGMA)
        print("Gaussian smoothing applied (sigma =", SMOOTH_SIGMA, ")")

    # 5.8: Run marching cubes to extract mesh from TSDF grid
    t_mc = time.time()
    vertices, triangles, vertex_colors = run_marching_cubes(fused_grid, N, color_grid=fused_colors)
    print("Marching cubes:", len(vertices), "vertices,", len(triangles), "triangles in", round(time.time() - t_mc, 2), "s")

    if len(triangles) == 0:
        print("No mesh generated for label", label_id, ", skipping.")
        continue

    # 5.9: Remove small disconnected components from the mesh
    vertices, triangles, vertex_colors = remove_small_components(
        vertices, triangles, vertex_colors, min_ratio=MIN_COMPONENT)
    print("After cleanup:", len(vertices), "vertices,", len(triangles), "triangles")
        
    # 5.10: Scale and translate vertices from grid coordinates back to world coordinates
    verts_arr = np.array(vertices) * voxel_size + grid_min # vectorized scaling

    # Turn the array of vertices back into a list of tuples to be used in export
    # Example: if verts_arr[i] = [x, y, z], then scaled_vertices[i] = (x, y, z)
    scaled_vertices = [tuple(v) for v in verts_arr]

    # 5.11: Export mesh to PLY file
    ply_path = os.path.join(output_dir, f"object_{label_id:03d}.ply")
    export_ply_binary(scaled_vertices, triangles, ply_path, vertex_colors=vertex_colors)
    print("Saved", ply_path)

total_end_time = time.time()
total_time = total_end_time - start_time
print("\nPipeline complete. Total time:", round(total_time, 2), "s")