import os
import argparse
from pathlib import Path
import open3d as o3d
import numpy as np
from tqdm import tqdm

def clean_mesh(mesh_path, out_path, cluster_keep=1, smooth_iters=50, min_triangles=100):
    """
    Cleans a triangle mesh by removing disconnected floaters and smoothing the surface.
    """
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if not mesh.has_triangles():
        return False
        
    # 1. Connected Component Filtering (Isolate the main object)
    triangle_clusters, cluster_triangles, _ = mesh.cluster_connected_triangles()
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_triangles = np.asarray(cluster_triangles)
    
    if cluster_triangles.size > 0:
        actual_keep = min(cluster_keep, cluster_triangles.size)
        threshold1 = np.sort(cluster_triangles)[-actual_keep]
        threshold = max(int(threshold1), int(min_triangles))
        
        remove_mask = cluster_triangles[triangle_clusters] < threshold
        mesh.remove_triangles_by_mask(remove_mask)
        mesh.remove_unreferenced_vertices()
        mesh.remove_degenerate_triangles()
        
    # 2. Taubin Smoothing (removes jagged artifacts without shrinking the volume)
    if smooth_iters > 0 and len(mesh.vertices) > 0:
        mesh = mesh.filter_smooth_taubin(number_of_iterations=smooth_iters)
        mesh.compute_vertex_normals()
        
    if len(mesh.vertices) > 0:
        o3d.io.write_triangle_mesh(str(out_path), mesh)
        return True
    return False

def main():
    parser = argparse.ArgumentParser(description="Clean extracted GS meshes")
    parser.add_argument("--meshes_dir", type=str, required=True, help="Directory containing label_* folders")
    parser.add_argument("--cluster_keep", type=int, default=1, help="Keep only the N largest chunks (1 = absolute cleanest)")
    parser.add_argument("--smooth_iters", type=int, default=50, help="Taubin smoothing iterations (0 to disable)")
    args = parser.parse_args()

    base_dir = Path(args.meshes_dir)
    if not base_dir.exists():
        print(f"Error: {base_dir} does not exist.")
        return

    label_dirs = [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("label_")]
    
    print(f"Found {len(label_dirs)} label directories. Starting cleanup...")
    print(f"Settings: Keep Top {args.cluster_keep} Clusters | Taubin Smoothing: {args.smooth_iters} iters\n")

    for label_dir in tqdm(label_dirs, desc="Cleaning Meshes"):
        raw_path = label_dir / "raw.ply"
        if not raw_path.exists():
            continue
            
        # We process the raw mesh so we don't double-filter the already filtered one
        # Save the result as "super_clean.ply"
        out_path = label_dir / "super_clean.ply"
        
        success = clean_mesh(
            mesh_path=raw_path, 
            out_path=out_path, 
            cluster_keep=args.cluster_keep,
            smooth_iters=args.smooth_iters
        )
        
    print("\nCleanup Complete! All super_clean.ply files have been generated.")

if __name__ == "__main__":
    main()
