import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys, json, torch, gc, math
import argparse
import numpy as np
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
import gsplat 
import shutil

REPO_ROOT = Path(__file__).resolve().parent.parent 
sys.path.insert(0, str(REPO_ROOT))

from gstrain.vroom_core.core.model.anchor_field import AnchorCloud
from gstrain.vroom_core.utilities.gaussian_decoder import GaussianDecoder
from gstrain.vroom_core.utilities.utils import CheckpointManager

# ==========================================
# 🎯 RENDERER (RGB + DEPTH + SEMANTICS VIA TWO-PASS)
# ==========================================
def render_3d(viewpoint_camera, decoded_output, gaussian_positions, normalized_rotations, background_color, semantics=None):
    xyz = gaussian_positions
    color = decoded_output["color"]
    opacity = decoded_output["opacity"]
    scaling = decoded_output["scaling"]
    rot = normalized_rotations

    render_device = xyz.device
    background_color = background_color.to(render_device)
    K = torch.tensor(
        [[viewpoint_camera.fx, 0, viewpoint_camera.cx],
         [0, viewpoint_camera.fy, viewpoint_camera.cy],
         [0, 0, 1]], dtype=torch.float32, device=render_device,
    )
    viewmat = viewpoint_camera.world_view_transform.transpose(0, 1).to(render_device).float()

    # 🎬 PASS 1: RGB and Depth
    results = gsplat.rasterization(
        means=xyz, quats=rot, scales=scaling, opacities=opacity.squeeze(-1), colors=color,
        viewmats=viewmat[None], Ks=K[None], width=int(viewpoint_camera.image_width), height=int(viewpoint_camera.image_height),
        backgrounds=background_color[None], packed=False, render_mode="RGB+ED"
    )
    
    render_colors = results[0] 
    colors, depths = render_colors[..., 0:3], render_colors[..., 3:4]
    
    out = {
        "render": colors[0].permute(2, 0, 1),
        "depth": depths[0].permute(2, 0, 1)
    }
    
    # 🎬 PASS 2: Semantics (Tricking the rasterizer)
    if semantics is not None:
        sem_bg = torch.zeros(semantics.shape[-1], dtype=torch.float32, device=render_device)
        
        sem_results = gsplat.rasterization(
            means=xyz, quats=rot, scales=scaling, opacities=opacity.squeeze(-1), colors=semantics,
            viewmats=viewmat[None], Ks=K[None], width=int(viewpoint_camera.image_width), height=int(viewpoint_camera.image_height),
            backgrounds=sem_bg[None], packed=False, render_mode="RGB"
        )
        
        render_semantics = sem_results[0]
        out["semantics"] = render_semantics[0].argmax(dim=-1)
        
    return out

# ==========================================
# 🎯 CAMERA MATH
# ==========================================
def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))
    P = torch.zeros(4, 4)
    P[0, 0] = 1 / tanHalfFovX
    P[1, 1] = 1 / tanHalfFovY
    P[2, 2] = (zfar + znear) / (zfar - znear)
    P[2, 3] = -(2 * zfar * znear) / (zfar - znear)
    P[3, 2] = 1.0
    return P

class DummyView:
    def __init__(self, cam_dict, device="cuda", scale=1.0):
        self.image_width = int(cam_dict["width"] * scale)
        self.image_height = int(cam_dict["height"] * scale)
        self.fx = float(cam_dict.get("fx", self.image_width)) * scale
        self.fy = float(cam_dict.get("fy", self.image_height)) * scale
        self.cx = self.image_width / 2.0
        self.cy = self.image_height / 2.0
        
        self.FoVx = 2 * math.atan(self.image_width / (2 * self.fx))
        self.FoVy = 2 * math.atan(self.image_height / (2 * self.fy))
        
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = np.array(cam_dict["rotation"])
        c2w[:3, 3] = np.array(cam_dict["position"])
        w2c = np.linalg.inv(c2w)
        
        self.world_view_transform = torch.tensor(w2c, dtype=torch.float32).transpose(0, 1).to(device)
        self.projection_matrix = getProjectionMatrix(0.01, 100.0, self.FoVx, self.FoVy).transpose(0,1).to(device)
        self.camera_center = torch.tensor(cam_dict["position"], dtype=torch.float32).to(device)

# ==========================================
# 🎯 MAIN PIPELINE
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description="Extract RGB, Depth, and Semantics from a trained VRoom model for mesh generation.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model directory")
    parser.add_argument("--output_dir", type=str, default="./mesh_generation/inputs", help="Directory to save the extracted inputs")
    parser.add_argument("--iteration", type=int, default=-1, help="Specific iteration checkpoint to load (default: latest)")
    parser.add_argument("--limit_frames", type=int, default=None, help="Process only the first N frames")
    return parser.parse_args()

def main():
    args = parse_args()
    model_dir = Path(args.model_path)
    exp_dir = Path(args.output_dir)
    
    print(f"🚀 Running VRoom Mesh Extraction Pipeline...")
    
    with open(model_dir / "config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)
        
    feat_dim = cfg["model"]["feat_dim"]
    gs_per_anchor = cfg["model"]["gs_per_anchor"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    anchor_cloud = AnchorCloud(gaussians_per_anchor=gs_per_anchor, feature_dim=feat_dim, device=device)
    decoder = GaussianDecoder(feature_dim=feat_dim, anchor_cloud=anchor_cloud).to(device)
    chkp_manager = CheckpointManager(anchor_cloud, decoder)
    
    chkpt_dir = model_dir / "checkpoints"
    if args.iteration != -1:
        ply_path = chkpt_dir / f"iter_{args.iteration}" / "anchor_cloud.ply"
    else:
        iters = [d for d in chkpt_dir.iterdir() if d.is_dir() and d.name.startswith("iter_")]
        latest_iter = sorted(iters, key=lambda x: int(x.name.split("_")[1]))[-1]
        ply_path = latest_iter / "anchor_cloud.ply"
        
    cam_path = model_dir / "cameras.json"
    
    print(f"💾 Loading checkpoint: {ply_path.parent.name}...")
    payload = chkp_manager.load_anchor_field(str(ply_path))
    
    anchor_cloud.anchors_positions = torch.nn.Parameter(payload["anchor"].to(device))
    anchor_cloud.gaussians_offsets = torch.nn.Parameter(payload["offset"].to(device))
    anchor_cloud.anchor_features = torch.nn.Parameter(payload["feature"].to(device))
    anchor_cloud.anchors_log_scales = torch.nn.Parameter(payload["log_scaling"].to(device))
    anchor_cloud.anchors_rotations = torch.nn.Parameter(payload["rotation"].to(device))
    
    if payload.get("labels") is not None:
        anchor_cloud.semantic_labels = payload["labels"].to(device)
        num_classes = int(anchor_cloud.semantic_labels.max().item()) + 1
        print(f"🏷️ Found Semantics! Detected {num_classes} distinct classes.")
    else:
        anchor_cloud.semantic_labels = None
        num_classes = 0
    
    chkp_manager.load_decoder(str(ply_path.parent))
        
    with open(cam_path, "r") as f: 
        camera_data = json.load(f)
    if isinstance(camera_data, dict): 
        camera_data = list(camera_data.values())
        
    if args.limit_frames:
        camera_data = camera_data[:args.limit_frames]
        
    print(f"📸 Loaded {len(camera_data)} cameras.")
    
    print("🧹 Preparing clean output directories...")
    # 🟢 Cleaned folder name changed here
    for subfolder in ["renders", "raw_depth", "semantic"]:
        target = exp_dir / subfolder
        if target.exists():
            shutil.rmtree(target)

    os.makedirs(exp_dir / "renders", exist_ok=True)
    os.makedirs(exp_dir / "raw_depth", exist_ok=True)      
    # 🟢 Creation folder name changed here
    os.makedirs(exp_dir / "semantic", exist_ok=True)
    
    with open(exp_dir / "cameras.json", "w") as f:
        json.dump(camera_data, f, indent=4)
        
    bg = torch.ones(3, dtype=torch.float32, device=device) 

    print("🎬 Processing frames...")
    with torch.no_grad():
        for idx, cam_dict in enumerate(camera_data): 
            view = DummyView(cam_dict, device=device, scale=1.0) 
            visible_mask = torch.ones(anchor_cloud.anchors_positions.shape[0], dtype=torch.bool, device=device)
            
            decoded_dict = decoder.forward_pass(anchor_cloud, visible_mask, view)
            
            xyz_positions = anchor_cloud.instantiate_gaussian_positions(
                visible_mask=visible_mask,
                negative_opacity_filter=decoded_dict["negative_opacity_filter"]
            )
            
            normalized_rots = torch.nn.functional.normalize(decoded_dict["rotations"], dim=-1)
            
            gaussian_semantics = None
            if anchor_cloud.semantic_labels is not None:
                visible_labels = anchor_cloud.semantic_labels[visible_mask].squeeze(-1).long()
                one_hot = F.one_hot(visible_labels, num_classes=num_classes).float()
                expanded_one_hot = one_hot.unsqueeze(1).expand(-1, decoder.number_gaussians_per_anchor, -1)
                gaussian_semantics = expanded_one_hot[decoded_dict["negative_opacity_filter"]]
            
            pkg = render_3d(
                viewpoint_camera=view,
                decoded_output={"color": decoded_dict["color"], "opacity": decoded_dict["opacity"], "scaling": decoded_dict["scaling"]},
                gaussian_positions=xyz_positions,
                normalized_rotations=normalized_rots,
                background_color=bg,
                semantics=gaussian_semantics
            )
            
            # ==========================================
            # 🎯 ZERO-COPY SAVE (BYPASSING BROKEN .NUMPY)
            # ==========================================
            # Save RGB 
            render_tensor = pkg["render"].clone().clamp(0, 1).detach().cpu().contiguous()
            rgb = np.from_dlpack(render_tensor).transpose(1, 2, 0)
            Image.fromarray((rgb * 255).astype(np.uint8)).save(exp_dir / "renders" / f"{idx:05d}.png")
            
            # Save Raw Depth
            depth_tensor = pkg["depth"].clone().detach().cpu().squeeze(0).contiguous()
            depth = np.from_dlpack(depth_tensor).copy()
            np.save(exp_dir / "raw_depth" / f"{idx:05d}.npy", depth)
            
            # Save Raw Semantics
            if "semantics" in pkg:
                sem_tensor = pkg["semantics"].clone().detach().cpu().to(torch.uint8).contiguous()
                sem_map = np.from_dlpack(sem_tensor).copy()
                # 🟢 Save folder name changed here
                Image.fromarray(sem_map).save(exp_dir / "semantic" / f"{idx:05d}.png")
            # ==========================================
            
            if idx % 10 == 0 or idx == len(camera_data) - 1:
                print(f"Processed frame {idx + 1}/{len(camera_data)}")
                
            del pkg, decoded_dict, gaussian_semantics
            gc.collect(); torch.cuda.empty_cache()
            
    print(f"🎉 Extraction Complete! The suite is waiting in {exp_dir.absolute()}")

if __name__ == "__main__":
    main()