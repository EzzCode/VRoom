import argparse
import subprocess
import sys
import yaml
import re
import os
from pathlib import Path
from tqdm import tqdm

def stream_command(cmd, desc, total=None, pattern=None):
    """Run a command and update a progress bar based on output parsing."""
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
        env=env
    )
    
    pbar = tqdm(total=total, desc=desc, unit="img", leave=True, dynamic_ncols=True)
    
    try:
        for line in process.stdout:
            # Check for generic "Frame XXXX" or "Frame [XXXX/YYYY]" patterns
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
            
            # Specific handling for COLMAP matching: "Matching image [X/Y]"
            colmap_match = re.search(r"Matching image \[(\d+)/(\d+)\]", line)
            if colmap_match:
                curr, tot = int(colmap_match.group(1)), int(colmap_match.group(2))
                if pbar.total != tot:
                    pbar.total = tot
                    pbar.refresh()
                if hasattr(pbar, 'last_colmap'):
                    diff = curr - pbar.last_colmap
                    if diff > 0:
                        pbar.update(diff)
                        pbar.last_colmap = curr
                else:
                    pbar.update(curr)
                    pbar.last_colmap = curr

            # Mandatory flush to ensure the tqdm bar actually moves in the parent terminal
            pbar.refresh()
                
        process.wait()
    finally:
        pbar.close()
    
    if process.returncode != 0:
        print(f"\nError: {desc} failed with exit code {process.returncode}")
        sys.exit(process.returncode)

def main():
    # Force python to be unbuffered for this script itself
    if not os.environ.get("PYTHONUNBUFFERED"):
        os.environ["PYTHONUNBUFFERED"] = "1"
        os.execv(sys.executable, [sys.executable] + sys.argv)

    parser = argparse.ArgumentParser(description="Enhanced E2E VRoom Pipeline Runner")
    parser.add_argument("--data_path", required=True, help="Path to your scene dataset")
    parser.add_argument("--base_config", required=True, help="Path to a base gs-train config YAML")
    args = parser.parse_args()

    data_path = Path(args.data_path).resolve()
    base_config = Path(args.base_config).resolve()

    if not data_path.exists():
        print(f"Error: Data path does not exist: {data_path}")
        sys.exit(1)

    image_dir = data_path / "images"
    total_images = len(list(image_dir.glob("*"))) if image_dir.exists() else 0

    print("\n" + "=" * 60)
    print(f" VROOM PIPELINE ENGINE | Dataset: {data_path.name}")
    print(f" Total Images Found: {total_images}")
    print("=" * 60)
    
    workspace_root = Path(__file__).parent.resolve()
    py = sys.executable

    # Use 'python -u' for sub-commands as well
    py_u = [py, "-u"]

    # STEP 1: COLMAP
    print(f"\n[1/3] STAGE: Structure-from-Motion (COLMAP)")
    colmap_runner = workspace_root / "Module-1" / "colmap_runner.py"
    stream_command(py_u + [str(colmap_runner), "--data_path", str(data_path)], "SfM Matching", total=total_images)

    # STEP 2: SAM3 & Tracking
    print(f"\n[2/3] STAGE: Semantic Tracking (SAM3 + Tracker)")
    module1_runner = workspace_root / "Module-1" / "module1_runner.py"
    
    stream_command(
        py_u + [str(module1_runner), "--data_path", str(data_path), "--skip_colmap", "--skip_tracking", "--skip_voting"],
        "SAM3 Masking",
        total=total_images,
        pattern=r"Frame (\d+)"
    )
    
    stream_command(
        py_u + [str(module1_runner), "--data_path", str(data_path), "--skip_colmap", "--skip_masks", "--skip_voting"],
        "Object Tracking",
        total=total_images,
        pattern=r"Frame (\d+)"
    )

    stream_command(
        py_u + [str(module1_runner), "--data_path", str(data_path), "--skip_colmap", "--skip_masks", "--skip_tracking"],
        "3D Voting",
        total=None
    )

    # DYNAMIC CONFIG GENERATION
    print(f"\n[3/3] STAGE: VRoom Neural Training")
    with open(base_config, "r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    config.setdefault("model_params", {})["source_path"] = str(data_path)
    config["model_params"]["masks"] = "tracked/id_maps"
    config["model_params"]["add_mask"] = True
    
    new_config_path = data_path / "auto_vroom_config.yaml"
    with open(new_config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    # STEP 3: Training
    trainer_script = workspace_root / "gs-train" / "trainer.py"
    stream_command(
        py_u + [str(trainer_script), "--config", str(new_config_path)],
        "GS Optimization",
        total=30000,
        pattern=r"Iter (\d+)"
    )

    print("\n" + "=" * 60)
    print(" PIPELINE COMPLETE! 🎉")
    print("=" * 60)

if __name__ == "__main__":
    main()
