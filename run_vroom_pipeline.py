import argparse
import subprocess
import sys
import yaml
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="End-to-End VRoom Pipeline Runner")
    parser.add_argument("--data_path", required=True, help="Path to your scene dataset")
    parser.add_argument("--base_config", required=True, help="Path to a base gs-train config YAML")
    args = parser.parse_args()

    data_path = Path(args.data_path).resolve()
    base_config = Path(args.base_config).resolve()

    if not data_path.exists():
        print(f"Error: Data path does not exist: {data_path}")
        sys.exit(1)

    if not base_config.exists():
        print(f"Error: Base config does not exist: {base_config}")
        sys.exit(1)

    print("=" * 60)
    print("STEP 1: Running Module-1 (Data Preparation & Mask Tracking)")
    print("=" * 60)
    
    # We assume this script is run from the workspace root
    workspace_root = Path(__file__).parent.resolve()
    module1_script = workspace_root / "Module-1" / "module1_runner.py"
    if not module1_script.exists():
        print(f"Error: Could not find {module1_script}.")
        sys.exit(1)

    try:
        subprocess.run([sys.executable, str(module1_script), "--data_path", str(data_path)], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Module-1 failed with error code {e.returncode}")
        sys.exit(e.returncode)

    print("\n" + "=" * 60)
    print("STEP 2: Generating Dynamic Training Configuration")
    print("=" * 60)

    try:
        with open(base_config, "r", encoding="utf-8") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
    except Exception as e:
        print(f"Error loading YAML config: {e}")
        sys.exit(1)

    # Inject dynamic pipeline parameters
    if "model_params" not in config:
        config["model_params"] = {}
    config["model_params"]["source_path"] = str(data_path)
    config["model_params"]["masks"] = "tracked/id_maps"
    config["model_params"]["add_mask"] = True

    # Save the updated config in the dataset directory
    new_config_path = data_path / "auto_vroom_config.yaml"
    try:
        with open(new_config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, sort_keys=False)
        print(f"Successfully generated dynamic config at {new_config_path}")
    except Exception as e:
        print(f"Error writing new YAML config: {e}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("STEP 3: Running gs-train (VRoom Training)")
    print("=" * 60)

    trainer_script = workspace_root / "gs-train" / "trainer.py"
    if not trainer_script.exists():
        print(f"Error: Could not find {trainer_script}.")
        sys.exit(1)

    try:
        subprocess.run([sys.executable, str(trainer_script), "--config", str(new_config_path)], check=True)
    except subprocess.CalledProcessError as e:
        print(f"gs-train failed with error code {e.returncode}")
        sys.exit(e.returncode)

    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
