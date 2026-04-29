import os
import sys
import argparse
import numpy as np
import torch
import cv2
from pathlib import Path

# Add project root and ObjectGS to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
objectgs_path = project_root / "temp_deps" / "ObjectGS"
sys.path.append(str(objectgs_path))

from target_replenishment.core.objectgs_bridge import (
    load_gaussians, get_anchor_positions, create_virtual_camera, render_view
)
from target_replenishment.core import diagnostics as diag

def look_at(camera_pos, target_pos, up_vector):
    forward = target_pos - camera_pos
    forward = forward / np.linalg.norm(forward)
    
    right = np.cross(up_vector, forward)
    right = right / np.linalg.norm(right)
    
    up = np.cross(forward, right)
    up = up / np.linalg.norm(up)
    
    # R represents the rotation from camera space to world space?
    # Actually, in Colmap/3DGS:
    # cam_pts = R @ world_pts + T
    # So R transforms world to cam.
    # The rows of R are Right, Down, Forward for Colmap (X right, Y down, Z forward).
    R = np.vstack((right, -up, forward))
    T = -R @ camera_pos
    return R, T

def generate_360_video(model_path, output_mp4, object_id=8, n_frames=90):
    print(f"Loading model {model_path}...")
    gaussians, pipe_config = load_gaussians(model_path, -1)
    
    # Load cameras to find a good starting viewpoint and "up" vector
    cameras_json = Path(model_path) / "cameras.json"
    if not cameras_json.exists():
        # Fallback to parent dir if searching in replenished model
        cameras_json = Path(model_path).parent.parent / "cameras.json"
        
    import json
    with open(cameras_json, 'r') as f:
        cam_data = json.load(f)
    
    cam_centers = diag.camera_centers_from_cameras_json(cam_data)
    up_vector = diag.estimate_scene_up_from_cameras(cam_data)
    
    anchor_xyz = get_anchor_positions(gaussians)
    if object_id is not None:
        labels = gaussians.label_ids.squeeze(-1).cpu().numpy()
        anchor_xyz = anchor_xyz[labels == object_id]
        
    if len(anchor_xyz) == 0:
        print("No anchors found for object!")
        return
        
    center = anchor_xyz.mean(axis=0)
    
    # Find average distance of cameras from the object
    dists = np.linalg.norm(cam_centers - center, axis=1)
    avg_dist = np.median(dists)
    base = diag.orbit_base_direction_from_cameras(cam_centers, center, up_vector)
    side = np.cross(up_vector, base)
    side = side / max(np.linalg.norm(side), 1e-8)
    
    # Use the K from the first camera
    c0 = cam_data[0]
    width, height = c0['width'], c0['height']
    fx = c0['fx']
    fy = c0['fy']
    K = np.array([[fx, 0, width/2], [0, fy, height/2], [0, 0, 1]])

    # Render frames
    frames = []
    bg_color = torch.ones(3, dtype=torch.float32, device="cuda")
    
    print("Rendering 360 trajectory...")
    for i in range(n_frames):
        angle = 2 * np.pi * i / n_frames
        
        radial = np.cos(angle) * base + np.sin(angle) * side
        cam_pos = center + radial * avg_dist * 0.5 + up_vector * avg_dist * 0.2
        
        R, T = look_at(cam_pos, center, up_vector)
        
        virt_cam = create_virtual_camera(R, T, K, width, height)
        # Render isolated object
        res = render_view(gaussians, virt_cam, pipe_config, bg_color, object_label_id=object_id)
        
        rgb = res['rgb'].permute(1, 2, 0).cpu().numpy()
        rgb = (rgb * 255).astype(np.uint8)
        frames.append(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        
        if i % 10 == 0:
            print(f"Rendered {i}/{n_frames} frames")
            
    print("Saving video...")
    out = cv2.VideoWriter(output_mp4, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))
    for f in frames:
        out.write(f)
    out.release()
    print(f"Saved {output_mp4}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output", default="output_360.mp4")
    parser.add_argument("--object_id", type=int, default=8)
    args = parser.parse_args()
    
    generate_360_video(args.model_path, args.output, args.object_id)
