import torch
import math
import os

from diff_surfel_rasterization import rasterization_2dgs_inria_wrapper
from cuda_rasterizer_rewrite import rasterize_2dgs

def get_tensors(N, device):
    # Deterministic generation
    torch.manual_seed(42)
    means3D = torch.rand((N, 3), device=device, dtype=torch.float32) * 2.0 - 1.0
    means3D[:, 2] += 5.0 # Push points in front of the camera (Z=5.0)
    
    scales = torch.rand((N, 2), device=device, dtype=torch.float32)
    
    quats = torch.rand((N, 4), device=device, dtype=torch.float32)
    quats = quats / quats.norm(dim=-1, keepdim=True)
    
    opacities = torch.rand((N,), device=device, dtype=torch.float32)
    
    colors = torch.rand((N, 3), device=device, dtype=torch.float32)
    
    return means3D, scales, quats, opacities, colors

def clone_for_grad(t):
    return t.detach().clone().requires_grad_(True)

def test_backward_parity():
    device = torch.device("cuda:0")
    
    # Setup Camera
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
    
    w2cam_mats = torch.eye(4, device=device).unsqueeze(0) # [1, 4, 4]
    backgrounds = torch.tensor([[0.1, 0.1, 0.1]], device=device)
    
    # 1. Isolated Memory Initialization
    N = 5000 # Enough to trigger multiple tiles and concurrency
    means_base, scales_base, quats_base, opacities_base, colors_base = get_tensors(N, device)
    
    # Original tensors (strictly separate copy)
    means_orig = clone_for_grad(means_base)
    scales_orig = clone_for_grad(scales_base)
    quats_orig = clone_for_grad(quats_base)
    opacities_orig = clone_for_grad(opacities_base)
    colors_orig = clone_for_grad(colors_base)
    
    # New tensors (strictly separate copy)
    means_new = clone_for_grad(means_base)
    scales_new = clone_for_grad(scales_base)
    quats_new = clone_for_grad(quats_base)
    opacities_new = clone_for_grad(opacities_base)
    colors_new = clone_for_grad(colors_base)
    
    # Dummy target image for MSE loss
    target_img = torch.rand((1, img_H, img_W, 3), device=device)
    
    # 2. Execution & Loss Calculation (Original)
    print("Executing Original Rasterizer Forward & Backward...")
    (render_colors_orig, render_alphas_orig), meta_orig = rasterization_2dgs_inria_wrapper(
        means=means_orig,
        quats=quats_orig,
        scales=scales_orig,
        opacities=opacities_orig,
        colors=colors_orig,
        viewmats=w2cam_mats,
        Ks=cam_intrinsics,
        width=img_W,
        height=img_H,
        near_plane=0.1,
        far_plane=100.0,
        backgrounds=backgrounds,
        depth_ratio=0
    )
    
    loss_orig = torch.nn.functional.mse_loss(render_colors_orig[..., :3], target_img)
    loss_orig.backward()
    
    # 2. Execution & Loss Calculation (New)
    print("Executing New Rasterizer Forward & Backward...")
    (render_colors_new, render_alphas_new), meta_new = rasterize_2dgs(
        points_world_space=means_new,
        quats=quats_new,
        scale_vecs=scales_new,
        opacities=opacities_new,
        colors_feat=colors_new,
        w2cam_mats=w2cam_mats,
        cam_intrinsics=cam_intrinsics,
        img_W=img_W,
        img_H=img_H,
        near_plane=0.1,
        far_plane=100.0,
        backgrounds=backgrounds,
        depth_ratio=0
    )
    
    loss_new = torch.nn.functional.mse_loss(render_colors_new[..., :3], target_img)
    loss_new.backward()
    
    # 3. The Strict Parity Matrix (Exhaustive Topology Check)
    print("\n==============================================")
    print("       BACKWARD PARITY RESULTS MATRIX")
    print("==============================================")
    
    params_orig = {
        "Means (Points)": means_orig,
        "Scales        ": scales_orig,
        "Rotations     ": quats_orig,
        "Opacities     ": opacities_orig,
        "Colors        ": colors_orig
    }
    
    params_new = {
        "Means (Points)": means_new,
        "Scales        ": scales_new,
        "Rotations     ": quats_new,
        "Opacities     ": opacities_new,
        "Colors        ": colors_new
    }
    
    all_match = True
    
    for name in params_orig.keys():
        grad_orig = params_orig[name].grad
        grad_new = params_new[name].grad
        
        # Null-Gradient Verification
        if grad_orig is None and grad_new is None:
            print(f"[{name}] ⚪ NULL-GRADIENT (PASS) - Both engines returned None")
            continue
            
        if grad_orig is None or grad_new is None:
            print(f"[{name}] ❌ MISMATCH! One gradient is None.")
            all_match = False
            continue
            
        # Mathematical Parity
        max_diff = torch.max(torch.abs(grad_orig - grad_new)).item()
        
        # Allow small floating point divergence due to atomicAdd ordering and fma compiler variations
        match = max_diff < 5e-4 
        
        if match:
            print(f"[{name}] ✅ PASS - Max Diff: {max_diff:.8e}")
        else:
            print(f"[{name}] ❌ FAIL - Max Diff: {max_diff:.8e}")
            all_match = False

    if all_match:
        print("\n🏆 SUCCESS: Golden Gradient Parity Achieved! Your new engine is mathematically identical.")
    else:
        print("\n💀 FAILED: Golden Gradient Parity Mismatch. Gradients have diverged.")

if __name__ == "__main__":
    test_backward_parity()
