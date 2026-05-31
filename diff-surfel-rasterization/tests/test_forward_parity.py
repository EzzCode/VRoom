import torch
import torchvision.utils as vutils
import os

from diff_surfel_rasterization import rasterization_2dgs_inria_wrapper
from cuda_rasterizer_rewrite import rasterize_2dgs

def test_forward_parity():
    # Ensure gradients and randomness don't interfere
    torch.manual_seed(42)
    device = torch.device("cuda:0")
    
    print("Setting up dummy 2DGS data...")
    # 1. Camera Setup
    img_W = 512
    img_H = 512
    fx = 256.0
    fy = 256.0
    cx = 256.0
    cy = 256.0
    
    cam_intrinsics = torch.tensor([
        [[fx, 0.0, cx],
         [0.0, fy, cy],
         [0.0, 0.0, 1.0]]
    ], device=device) # [1, 3, 3]
    
    # Identity matrix for world-to-camera (camera is at origin)
    w2cam_mats = torch.eye(4, device=device).unsqueeze(0) # [1, 4, 4]
    
    # 2. Dummy 2DGS Parameters
    N = 5
    # Place surfels nicely in front of the camera at Z=5.0
    points_world_space = torch.tensor([
        [ 0.0,  0.0, 5.0],  # Center
        [ 1.0,  1.0, 5.0],  # Bottom Right
        [-1.0,  1.0, 5.0],  # Bottom Left
        [ 1.0, -1.0, 5.0],  # Top Right
        [-1.0, -1.0, 5.0],  # Top Left
    ], device=device)
    
    # Identity quaternion (w, x, y, z)
    quats = torch.tensor([
        [1.0, 0.0, 0.0, 0.0]
    ] * N, device=device)
    
    # 2D Scales (surfels use 2 scales, 3rd is ignored but we provide 3 for API compat)
    scale_vecs = torch.tensor([
        [0.5, 0.5, 1.0] # scaled down to fit on screen
    ] * N, device=device)
    
    # Opacities
    opacities = torch.tensor([0.9] * N, device=device)
    
    # Colors (RGB)
    colors_feat = torch.tensor([
        [1.0, 0.0, 0.0], # Red center
        [0.0, 1.0, 0.0], # Green
        [0.0, 0.0, 1.0], # Blue
        [1.0, 1.0, 0.0], # Yellow
        [1.0, 0.0, 1.0], # Magenta
    ], device=device)
    
    backgrounds = torch.tensor([
        [0.1, 0.1, 0.1]
    ], device=device) # Dark gray bg
    
    print("\nExecuting Original Rasterizer...")
    (render_colors_orig, render_alphas_orig), meta_orig = rasterization_2dgs_inria_wrapper(
        means=points_world_space,
        quats=quats,
        scales=scale_vecs,
        opacities=opacities,
        colors=colors_feat,
        viewmats=w2cam_mats,
        Ks=cam_intrinsics,
        width=img_W,
        height=img_H,
        near_plane=0.1,
        far_plane=100.0,
        backgrounds=backgrounds,
        depth_ratio=0
    )
    
    print("Executing Rewritten Rasterizer...")
    (render_colors_new, render_alphas_new), meta_new = rasterize_2dgs(
        points_world_space=points_world_space,
        quats=quats,
        scale_vecs=scale_vecs,
        opacities=opacities,
        colors_feat=colors_feat,
        w2cam_mats=w2cam_mats,
        cam_intrinsics=cam_intrinsics,
        img_W=img_W,
        img_H=img_H,
        near_plane=0.1,
        far_plane=100.0,
        backgrounds=backgrounds,
        depth_ratio=0
    )
    
    # 3. Mathematical Proof
    # Render colors are concatenated with depth in the last channel: [C, H, W, D+1]
    color_orig = render_colors_orig[..., :-1] 
    color_new  = render_colors_new[..., :-1]
    
    depth_orig = render_colors_orig[..., -1:]
    depth_new  = render_colors_new[..., -1:]
    
    max_color_diff = torch.max(torch.abs(color_orig - color_new)).item()
    max_alpha_diff = torch.max(torch.abs(render_alphas_orig - render_alphas_new)).item()
    max_depth_diff = torch.max(torch.abs(depth_orig - depth_new)).item()
    
    print("\n==============================")
    print("       PARITY RESULTS")
    print("==============================")
    print(f"Max Color Diff: {max_color_diff:.8e}")
    print(f"Max Alpha Diff: {max_alpha_diff:.8e}")
    print(f"Max Depth Diff: {max_depth_diff:.8e}")
    
    print("\n--- Meta Comparisons ---")
    meta_keys = ["normals_rend", "normals_surf", "render_distloss", "means2d", "radii"]
    all_meta_match = True
    for key in meta_keys:
        if key in meta_orig and key in meta_new:
            t_orig = meta_orig[key]
            t_new = meta_new[key]
            if isinstance(t_orig, torch.Tensor) and isinstance(t_new, torch.Tensor):
                if key == "means2d" and t_orig.shape != t_new.shape:
                    t_orig = t_orig[..., :2]
                max_diff = torch.max(torch.abs(t_orig - t_new)).item()
                print(f"Max {key} Diff: {max_diff:.8e}")
                if max_diff > 1e-4:
                    all_meta_match = False
    
    # Small tolerance for floating point rounding diffs between compilers
    if max_color_diff < 1e-4 and max_alpha_diff < 1e-4 and all_meta_match:
        print("\n✅ SUCCESS: Outputs match perfectly!")
    else:
        print("\n❌ FAILED: Outputs diverge!")
        
    # 4. Visual Proof
    # Permute to [C, H, W] for torchvision
    img_orig = color_orig[0].permute(2, 0, 1)
    img_new  = color_new[0].permute(2, 0, 1)
    
    out_dir = os.path.dirname(os.path.abspath(__file__)) + "/fwd_parity"
    os.makedirs(out_dir, exist_ok=True)
    out_orig = os.path.join(out_dir, "fwd_parity_orig.png")
    out_new  = os.path.join(out_dir, "fwd_parity_new.png")
    
    vutils.save_image(img_orig, out_orig)
    vutils.save_image(img_new, out_new)
    
    print(f"\nVisual proofs saved to:")
    print(f"- {out_orig}")
    print(f"- {out_new}")

if __name__ == "__main__":
    test_forward_parity()
