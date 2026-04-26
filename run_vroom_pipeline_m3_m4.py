# This script runs training, extracts mesh inputs, then generates meshes
# The following commented code is code that should exist in the main run_vroom_pipeline.py file
# Since I'm not sure if Module1-Module3 integration is finished, the following is a separate script that 
# integrates Module3 and Module 4.

# # ... existing training code ...
#     # stream_command(
#     #     py_u + [str(trainer_script), "--config", str(new_config_path)],
#     #     "GS Optimization",
#     #     total=30000,
#     #     pattern=r"Iter (\d+)"
#     # )

#     # ==========================================
#     # STEP 4: DYNAMIC EXTRACTION
#     # ==========================================
#     print("\n[4/4] STAGE: VRoom Mesh Extraction")
    
#     # 1. Auto-detect the model that just finished training
#     # Since we don't want hardcoded scene names, we search the output 
#     # folder and dynamically grab the one with the newest timestamp.
#     output_base = workspace_root / "output"
#     config_files = list(output_base.rglob("config.yaml"))
    
#     if not config_files:
#         print("Error: Could not find any trained model in output/ directory!")
#         sys.exit(1)
        
#     latest_config = sorted(config_files, key=lambda x: x.stat().st_mtime)[-1]
#     fresh_model_dir = latest_config.parent
#     mesh_inputs_dir = fresh_model_dir / "mesh_inputs"
    
#     print(f" Found fresh model at: {fresh_model_dir.relative_to(workspace_root)}")

#     # 2. Run the Extractor
#     # Since your extractor script already prints "Processed frame X", 
#     # we can use your awesome stream_command to give it a progress bar too!
#     extractor_script = workspace_root / "vroom_core" / "export" / "mesh_inputs_extractor.py"
    
#     # NOTE: For this stream_command to work, make sure your mesh_inputs_extractor.py 
#     # has an argparse or sys.argv setup at the bottom to accept these two paths!
#     # (Or you can just 'import' the function directly here to bypass subprocess)
#     stream_command(
#         py_u + [str(extractor_script), "--model_path", str(fresh_model_dir), "--output_dir", str(mesh_inputs_dir)],
#         "Extracting 2D Arrays",
#         total=total_images, 
#         pattern=r"Processed frame (\d+)" 
#     )

#     # ==========================================
#     # STEP 5: MESH GENERATION (Optional)
#     # ==========================================
#     print("\n[5/5] STAGE: Meshing")
#     mesher_script = workspace_root / "vroom_core" / "export" / "extract_object_meshes.py"
    
#     # Run your mesher using the newly generated inputs
#     stream_command(
#         py_u + [str(mesher_script), "--inputs", str(mesh_inputs_dir)],
#         "Generating 3D Mesh",
#         total=None # No progress bar needed unless your mesher prints percentages
#     )

#     print("\n" + "=" * 60)
#     print(" VROOM PIPELINE COMPLETE! ")
#     print(f" Outputs saved in: {fresh_model_dir}")
#     print("=" * 60)

import argparse
import subprocess
import sys
import os
import re
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import yaml

def stream_command(cmd, desc, total=None, pattern=None, log_file=None):
    """Run a command and update a progress bar based on output parsing."""
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    log_handle = None
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = open(log_path, "a", encoding="utf-8")
        log_handle.write("\n" + "=" * 80 + "\n")
        log_handle.write(f"[{datetime.now().isoformat(timespec='seconds')}] {desc}\n")
        log_handle.write(f"CMD: {' '.join(cmd)}\n")
        log_handle.write("=" * 80 + "\n")
        log_handle.flush()
    
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, universal_newlines=True, env=env
    )
    
    pbar = tqdm(total=total, desc=desc, unit="step", leave=True, dynamic_ncols=True)
    
    try:
        for line in process.stdout:
            if log_handle:
                log_handle.write(line)
                log_handle.flush()

            if pattern:
                match = re.search(pattern, line)
                if match:
                    try:
                        current = int(match.group(1))
                        if hasattr(pbar, 'last_val'):
                            diff = current - pbar.last_val
                            if diff > 0:
                                pbar.update(diff)
                                pbar.last_val = current
                        else:
                            pbar.update(current)
                            pbar.last_val = current
                    except (ValueError, IndexError):
                        pass
            pbar.refresh()
    finally:
        pbar.close()
        if log_handle:
            log_handle.close()
    
    process.wait()
    if process.returncode != 0:
        print(f"\nError: {desc} failed with exit code {process.returncode}")
        sys.exit(process.returncode)

def main():
    if not os.environ.get("PYTHONUNBUFFERED"):
        os.environ["PYTHONUNBUFFERED"] = "1"
        os.execv(sys.executable, [sys.executable] + sys.argv)

    # FIX 1: Updated the arguments to match what Step 0 actually needs
    parser = argparse.ArgumentParser(description="VRoom Module 3 -> 4 Integration")
    parser.add_argument("--base_config", required=True, help="Path to your gs-train config YAML")
    parser.add_argument("--data_path", required=True, help="Path to the Kaggle dataset")
    parser.add_argument("--scene_name", required=True, help="Name of the specific scene (e.g., bed)")
    args = parser.parse_args()

    workspace_root = Path(__file__).parent.resolve()
    py_u = [sys.executable, "-u"]

    print("\n" + "=" * 60)
    print(f" VROOM M3->M4 PIPELINE | Scene: {args.scene_name}")
    print("=" * 60)

    # ==========================================
    # STEP 0: DYNAMIC CONFIG GENERATION
    # ==========================================
    data_path = Path(args.data_path).resolve()
    base_config = Path(args.base_config).resolve()

    with open(base_config, "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    # Inject the Kaggle dataset path dynamically!
    config.setdefault("model_params", {})["source_path"] = str(data_path)
    
    # FIX 2: Save the generated config safely in the workspace, NOT in the read-only input folder
    new_config_path = workspace_root / f"auto_{args.scene_name}_config.yaml"
    with open(new_config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)
        
    print(f"\n[0/3] DYNAMIC CONFIG: Saved temporary config to {new_config_path.name}")

    # ==========================================
    # STEP 1: TRAINING (Module 3)
    # ==========================================
    print("\n[1/3] STAGE: 3D Gaussian Training")
    trainer_script = workspace_root / "gs-train" / "trainer.py"
    
    # FIX 3: Wrapped new_config_path in str()
    stream_command(
        py_u + [str(trainer_script), "--config", str(new_config_path), "--scene_name", args.scene_name],
        "GS Optimization",
        total=30000,
        pattern=r"(\d+)/30000"
    )

    # ==========================================
    # STEP 2: DYNAMIC EXTRACTION (Module 4)
    # ==========================================
    print("\n[2/3] STAGE: VRoom Mesh Generation")
    
    output_base = workspace_root / "output"
    config_files = list(output_base.rglob("config.yaml"))
    
    if not config_files:
        print("Error: Could not find any trained model in output/ directory!")
        sys.exit(1)
        
    latest_config = sorted(config_files, key=lambda x: x.stat().st_mtime)[-1]
    fresh_model_dir = latest_config.parent
    mesh_inputs_dir = fresh_model_dir / "mesh_inputs"
    
    print(f" Found fresh model at: {fresh_model_dir.relative_to(workspace_root)}")

    extractor_script = workspace_root / "gs-train" / "vroom_core" / "export" / "mesh_inputs_extractor.py"

    stream_command(
        py_u + [str(extractor_script), "--model_path", str(fresh_model_dir), "--output_dir", str(mesh_inputs_dir)],
        "Extracting 2D Arrays",
        total=None,
        pattern=r"Processed frame (\d+)" 
    )

    # ==========================================
    # STEP 3: MESH GENERATION (Module 4)
    # ==========================================
    print("\n[3/3] STAGE: Meshing")
    mesher_script = workspace_root / "gs-train" / "vroom_core" / "export" / "extract_object_meshes.py"
    
    stream_command(
        py_u + [str(mesher_script), "--inputs", str(mesh_inputs_dir)],
        "Generating 3D Mesh",
        total=None
    )

    print("\n" + "=" * 60)
    print(" M3->M4 PIPELINE COMPLETE!")
    print(f" Mesh generated in: {fresh_model_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()