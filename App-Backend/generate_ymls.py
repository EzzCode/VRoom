import os
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
ROOT_DIR = BACKEND_DIR.parent

def clean_yml(input_file, output_file, excludes, extra_index, replacements=None):
    if replacements is None:
        replacements = {}
    
    with open(input_file, 'r') as f:
        lines = f.readlines()
        
    new_lines = []
    for line in lines:
        # Check excludes
        if any(f"- {x}" in line for x in excludes):
            continue
            
        # Drop dev-only type hint packages which cause conflicts
        if line.strip().startswith("- types-"):
            continue
        
        # Replace defaults channel
        if "- defaults" in line:
            line = line.replace("- defaults", "- conda-forge")
            
        # Apply specific replacements (like clip)
        for old, new in replacements.items():
            if old in line:
                line = line.replace(old, new)
                
        # Inject index URL
        if "- pip:" in line:
            new_lines.append(line)
            new_lines.append(f"    - --extra-index-url {extra_index}\n")
            continue
            
        new_lines.append(line)
        
    with open(output_file, 'w') as f:
        f.writelines(new_lines)

# Generate pipeline
clean_yml(
    str(ROOT_DIR / "environments" / "environment_pipeline.yml"), 
    str(BACKEND_DIR / "modal_pipeline.yml"),
    excludes=['ucrt', 'vc', 'vc14_runtime', 'vs2015_runtime', 'pywin32', 'vroom', 'custom-differentiable-rasterizer', 'diff-surfel-rasterization'],
    extra_index="https://download.pytorch.org/whl/cu118"
)

# Generate masks
clean_yml(
    str(ROOT_DIR / "environments" / "environment_masks.yml"),
    str(BACKEND_DIR / "modal_masks.yml"),
    excludes=['ucrt', 'vc', 'vc14_runtime', 'vs2015_runtime', 'pywin32'],
    extra_index="https://download.pytorch.org/whl/cu126",
    replacements={"clip==1.0": "git+https://github.com/ultralytics/CLIP.git"}
)

print("Generated modal_pipeline.yml and modal_masks.yml inside App-Backend/ successfully.")
