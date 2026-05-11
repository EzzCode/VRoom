import os
import subprocess
import sys
import time
import torch
from pathlib import Path
from argparse import ArgumentParser

# Add the project root to sys.path to allow imports from any directory
# Assuming the script is in gstrain/scripts/
root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

try:
    from gstrain.vroom_core.models.facade import GaussianModel
except ImportError:
    print("Error: Could not import vroom_core. Make sure you are running this from the gstrain directory.")
    sys.exit(1)

def get_labels(model_path, iteration):
    ply_path = Path(model_path) / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
    if not ply_path.exists():
        print(f"Error: PLY file not found at {ply_path}")
        return []

    print(f"Loading labels from {ply_path}...")
    
    # Initialize a dummy model to use its PLY loading logic
    # We use default params consistent with 2DGS office-scene config
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
        return []

    if model.field.label_ids is not None:
        label_ids = model.field.label_ids.view(-1)
        unique_labels = torch.unique(label_ids).tolist()
        # Sort labels and typically exclude 0 if it's "unlabeled", 
        # but the user might want it. Based on previous runs, we'll keep all found.
        return sorted(unique_labels)
    else:
        print("No labels found in this PLY.")
        return []

def run_label_export(model_path, source_path, label_id, iteration):
    # Use sys.executable to ensure we use the same environment
    cmd = [
        sys.executable, "scripts/export_object_meshes.py",
        "--model_path", model_path,
        "--source_path", source_path,
        "--label_id", str(label_id),
        "--iteration", str(iteration),
        "--white_background"
    ]
    
    print(f"\n" + "="*60)
    print(f"STARTING EXPORT FOR LABEL: {label_id}")
    print(f"Command: {' '.join(cmd)}")
    print("="*60 + "\n")
    
    start_time = time.time()
    try:
        # Run and stream output
        process = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)
        process.wait()
        
        if process.returncode == 0:
            duration = time.time() - start_time
            print(f"\nSUCCESS: Label {label_id} finished in {duration:.2f} seconds.")
        else:
            print(f"\nERROR: Label {label_id} failed with return code {process.returncode}")
            
    except Exception as e:
        print(f"\nEXCEPTION during label {label_id}: {e}")

if __name__ == "__main__":
    parser = ArgumentParser(description="Batch export meshes from a 2DGS/VRoom checkpoint")
    parser.add_argument("--model_path", type=str, default="/home/hussein_essam/gs-workspace/VRoom-Integration/output/office-scene/2dgs_office_scene/2026-04-18_18-15-23", help="Path to the model output directory")
    parser.add_argument("--source_path", type=str, default="/home/hussein_essam/gs-workspace/VRoom-Integration/office_scratch/office-scene", help="Path to the source dataset")
    parser.add_argument("--iteration", type=int, default=30000, help="Iteration number")
    parser.add_argument("--skip_zero", action="store_true", help="Skip label 0 (background)")
    
    args = parser.parse_args()
    
    MODEL_PATH = args.model_path
    SOURCE_PATH = args.source_path
    ITERATION = args.iteration
    
    labels = get_labels(MODEL_PATH, ITERATION)
    
    if args.skip_zero:
        labels = [l for l in labels if l != 0]
        
    if not labels:
        print("No labels found to export. Exiting.")
        sys.exit(0)
        
    print(f"Batch Export started for {len(labels)} labels: {labels}")
    print(f"Model: {MODEL_PATH}")
    
    for i, label in enumerate(labels):
        print(f"\nProgress: {i+1}/{len(labels)}")
        run_label_export(MODEL_PATH, SOURCE_PATH, label, ITERATION)
        
    print("\n" + "#"*60)
    print("ALL BATCH EXPORTS COMPLETED")
    print("#"*60)
