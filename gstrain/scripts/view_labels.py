import sys
import numpy as np
import torch
from pathlib import Path
from argparse import ArgumentParser
from plyfile import PlyData

# Add the project root to sys.path to allow imports from any directory
root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

try:
    from gstrain.vroom_core.models.facade import GaussianModel
except ImportError:
    print("Error: Could not import vroom_core. Make sure you are running this from the gstrain directory.")
    sys.exit(1)

def main():
    parser = ArgumentParser(description="Inspect labels and anchor counts in a VRoom PLY checkpoint")
    parser.add_argument("ply_path", type=str, help="Path to the point_cloud.ply file")
    args = parser.parse_all_args() if hasattr(parser, "parse_all_args") else parser.parse_args()

    ply_path = Path(args.ply_path)
    if not ply_path.exists():
        print(f"Error: Path not found: {ply_path}")
        sys.exit(1)

    print(f"Loading checkpoint: {ply_path.name}...")
    
    # Initialize a dummy model to use its PLY loading logic
    # We use n_offsets=10 as a default, but it doesn't affect label counting
    model = GaussianModel(
        n_offsets=10, 
        feat_dim=32, 
        view_dim=3, 
        appearance_dim=0, 
        voxel_size=0.001, 
        gs_attr="3D", 
        render_mode="RGB+ED"
    )

    try:
        model.load_ply(str(ply_path))
    except Exception as e:
        print(f"Error loading PLY: {e}")
        sys.exit(1)

    print("-" * 40)
    print(f"Total Anchors: {model.field.anchor.shape[0]:,}")
    
    if model.field.label_ids is not None:
        label_ids = model.field.label_ids.view(-1)
        unique_labels = torch.unique(label_ids).tolist()
        
        print(f"Unique Labels Found ({len(unique_labels)}):")
        print(unique_labels)
        print("-" * 40)
        print(f"{'Label ID':<10} | {'Anchor Count':<15}")
        print("-" * 40)
        
        # Count occurrences
        counts = {}
        for label in unique_labels:
            count = (label_ids == label).sum().item()
            counts[label] = count
            print(f"{label:<10} | {count:<15,}")
            
        print("-" * 40)
        
        # Check if codec exists
        if model.field.codec:
            print("Codec Info: Detected existing label mapping.")
    else:
        print("No labels found in this PLY.")

if __name__ == "__main__":
    main()
