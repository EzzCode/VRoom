import os
import json
import torch
import yaml
import numpy as np
from PIL import Image
from pathlib import Path

from vroom_core.models.facade import GaussianModel
from vroom_core.data.scene_pipeline import TrainingScene
from gaussian_renderer import render

def extract_mesh_inputs(model_path, output_dir, iteration=30000):
    """
    Loads a finished 3DGS model and extracts the 2D render, depth, 
    and semantic arrays required for mesh generation.
    """
    model_path = Path(model_path)
    exp_dir = Path(output_dir)
    
    print(f"Starting extraction for model at: {model_path}")
    
    os.makedirs(exp_dir / "raw_depth", exist_ok=True)
    os.makedirs(exp_dir / "semantic", exist_ok=True)
    os.makedirs(exp_dir / "renders", exist_ok=True)

    # Load the model config to get dataset and training parameters
    config_path = model_path / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Could not find config.yaml at {model_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    
    model_params = cfg.get("model_params", {})
    m_kwargs = model_params.get("model_config", {}).get("kwargs", {})
    
    # Dynamically pull the source dataset path from the config 
    source_path = model_params.get("source_path", "")
    if not source_path:
        raise ValueError("source_path not found in config.yaml!")
    
    gaussians = GaussianModel(
        n_offsets=m_kwargs.get("n_offsets", 5), feat_dim=m_kwargs.get("feat_dim", 32),
        view_dim=m_kwargs.get("view_dim", 3), appearance_dim=m_kwargs.get("appearance_dim", 0),
        voxel_size=m_kwargs.get("voxel_size", -1.0), gs_attr=m_kwargs.get("gs_attr", "3D"),
        render_mode=m_kwargs.get("render_mode", "RGB+ED"), tile_size_2dgs=m_kwargs.get("tile_size_2dgs", 8)
    )
    
    # Convert the dataset parameters into a simple object for command-line access
    class DatasetArgs:
        def __init__(self, **kwargs): self.__dict__.update(kwargs)
    
    # Create a dataset args object to initialize the training scene
    d_args = DatasetArgs(
        source_path=source_path, model_path=str(model_path), resolution=model_params.get("resolution", -1),
        resolution_scales=model_params.get("resolution_scales", [1.0]), images=model_params.get("images", "images"),
        depths=model_params.get("depths", "depths"), masks=model_params.get("masks", "masks"),
        data_device=model_params.get("data_device", "cuda"), data_format=model_params.get("data_format", "colmap"),
        eval=model_params.get("eval", False), llffhold=model_params.get("llffhold", 8),
        add_mask=model_params.get("add_mask", False), add_depth=model_params.get("add_depth", False),
        white_background=model_params.get("white_background", False), random_background=False,
        ratio=model_params.get("ratio", 1), global_appearance=model_params.get("global_appearance", False),
        pretrained_checkpoint="", center=model_params.get("center", [0.0, 0.0, 0.0]),
        scale=model_params.get("scale", 1.0), dataset_name=model_params.get("dataset_name", ""),
        exp_name=model_params.get("exp_name", "")
    )

    # Initialize the training scene to load the dataset and model parameters
    scene = TrainingScene(d_args, gaussians, load_iteration=iteration, shuffle=False)
    bg = torch.ones(3, dtype=torch.float32, device=gaussians.device)
    
    class PipeArgs:
        def __init__(self):
            self.compute_cov3D_python = False
            self.convert_SHs_python = False
            self.debug = False
    pipe = PipeArgs()

    print(f"Extracting 2D arrays to {exp_dir}...")
    cam_list = []
    
    for idx, view in enumerate(scene.getTrainCameras()):
        pkg = render(view, gaussians, pipe, bg)
        
        if "render" in pkg:
            rgb = pkg["render"].clone().clamp(0, 1).detach().cpu().numpy().transpose(1, 2, 0)
            Image.fromarray((rgb * 255).astype(np.uint8)).save(exp_dir / "renders" / f"{idx:05d}.png")
        
        # Depth maps are saved as .npy files because they are 2D arrays and need to store
        # depth values with high precision
        if "render_depth" in pkg:
            np.save(exp_dir / "raw_depth" / f"{idx:05d}.npy", pkg["render_depth"].squeeze().detach().cpu().numpy())
            
        if "render_semantics" in pkg:
            s = pkg["render_semantics"].detach().squeeze()
            if s.dim() == 3:
                s = torch.argmax(s, dim=-1) if s.shape[-1] < s.shape[0] else torch.argmax(s, dim=0)
            s_np = s.cpu().numpy().astype(np.uint8)
            Image.fromarray(s_np).save(exp_dir / "semantic" / f"{idx:05d}.png")
            
        w2c = view.world_view_transform.cpu().numpy().T
        c2w = np.linalg.inv(w2c)
        W, H = view.image_width, view.image_height
        fx = float(view.foVx_to_focal(W) if hasattr(view, 'foVx_to_focal') else W / (2 * np.tan(view.FoVx / 2)))
        fy = float(view.foVy_to_focal(H) if hasattr(view, 'foVy_to_focal') else H / (2 * np.tan(view.FoVy / 2)))
        
        cam_list.append({"id": idx, "width": W, "height": H, "fx": fx, "fy": fy, "position": c2w[:3, 3].tolist(), "rotation": c2w[:3, :3].tolist()})
        print(f"Processed frame {idx:05d}")

    with open(exp_dir / "cameras.json", "w") as f: 
        json.dump(cam_list, f, indent=4)
        
    print(f"\n All mesh inputs extracted to: {exp_dir}")
    return exp_dir

if __name__ == "__main__":
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="Extract 2D arrays from trained GS model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save the extracted arrays")
    parser.add_argument("--iteration", type=int, default=30000, help="Iteration number to load")
    
    args = parser.parse_args()
    
    try:
        extract_mesh_inputs(args.model_path, args.output_dir, args.iteration)
    except Exception as e:
        print(f"Error extracting mesh inputs: {e}", file=sys.stderr)
        sys.exit(1)
