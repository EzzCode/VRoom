import os
import yaml
import json
from pathlib import Path

def map_to_new_json(cfg: dict) -> dict:
    model_params = cfg.get("model_params", {})
    optim_params = cfg.get("optim_params", {})
    pipeline_params = cfg.get("pipeline_params", {})
    
    # 1. experiment
    experiment = {}
    for key, val in model_params.items():
        if key != "model_config":
            experiment[key] = val
            
    # 2. model
    model_config = model_params.get("model_config", {})
    model = {
        "name": model_config.get("name", "GaussianModel")
    }
    for key, val in model_config.get("kwargs", {}).items():
        model[key] = val
        
    # 3. pipeline
    pipeline = dict(pipeline_params)
    
    # 4. optimization
    learning_rates = {}
    loss_weights = {}
    densification = {}
    depth_loss = {}
    iterations = optim_params.get("iterations", 30000)
    
    for key, val in optim_params.items():
        if key == "iterations":
            continue
            
        # learning rates
        if key.endswith("_lr"):
            lr_name = key[:-3]
            learning_rates[lr_name] = val
        elif "_lr_" in key:
            lr_name = key.replace("_lr_", "_")
            learning_rates[lr_name] = val
            
        # loss weights
        elif key.startswith("lambda_") or key == "normal_start_iter" or key == "dist_start_iter":
            loss_weights[key] = val
            
        # depth loss
        elif key == "start_depth" or key.startswith("depth_"):
            if key == "start_depth":
                depth_loss["start_depth"] = val
            else:
                depth_loss[key.replace("depth_", "")] = val
                
        # densification
        elif key in ["start_stat", "update_from", "update_interval", "update_until", "overlap", "growing_type", "pruning_type", "min_opacity", "success_threshold", "update_ratio", "extra_ratio", "extra_up"]:
            densification[key] = val
        elif key == "densify_grad_threshold":
            densification["grad_threshold"] = val
        elif key == "densification":
            densification["enabled"] = val
        else:
            # Catch-all for any other optim_params keys
            loss_weights[key] = val
            
    return {
        "experiment": experiment,
        "model": model,
        "pipeline": pipeline,
        "optimization": {
            "iterations": iterations,
            "learning_rates": learning_rates,
            "loss_weights": loss_weights,
            "densification": densification,
            "depth_loss": depth_loss
        }
    }

def main():
    config_dir = Path(__file__).resolve().parent.parent / "config"
    print(f"Scanning for YAML config files in {config_dir}...")
    
    yaml_files = list(config_dir.rglob("*.yaml")) + list(config_dir.rglob("*.yml"))
    if not yaml_files:
        print("No YAML files found.")
        return
        
    for yaml_path in yaml_files:
        print(f"Converting: {yaml_path}")
        with open(yaml_path, "r", encoding="utf-8") as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
            
        new_cfg = map_to_new_json(cfg)
        json_path = yaml_path.with_suffix(".json")
        
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(new_cfg, f, indent=4)
            
        print(f"Saved JSON: {json_path}")
        os.remove(yaml_path)
        print(f"Deleted YAML: {yaml_path}")

if __name__ == "__main__":
    main()
