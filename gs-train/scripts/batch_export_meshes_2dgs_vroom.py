import os
import subprocess
import sys
import time

def run_label_export(model_path, source_path, label_id, iteration):
    cmd = [
        "python", "export_object_meshes.py",
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
    # Points to the 2DGS VRoom Output
    MODEL_PATH = "/home/hussein_essam/gs-workspace/VRoom/outputs/office2dgsVroom"
    SOURCE_PATH = "/home/hussein_essam/gs-workspace/VRoom/gs-train/datasets/replica/office_0"
    ITERATION = 30000
    
    # 49 Unique Labels actually found in this specific checkpoint
    LABELS = [
        2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 19, 20, 21, 22, 23, 
        24, 26, 27, 28, 30, 32, 33, 34, 35, 36, 37, 38, 39, 40, 42, 44, 46, 
        48, 49, 50, 51, 54, 55, 56, 57, 58, 61, 63, 64, 66
    ]
    
    print(f"Batch Export started for {len(LABELS)} labels.")
    print(f"Model: {MODEL_PATH}")
    
    for i, label in enumerate(LABELS):
        print(f"\nProgress: {i+1}/{len(LABELS)}")
        run_label_export(MODEL_PATH, SOURCE_PATH, label, ITERATION)
        
    print("\n" + "#"*60)
    print("ALL BATCH EXPORTS COMPLETED")
    print("#"*60)
