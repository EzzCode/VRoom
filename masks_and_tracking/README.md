# VRoom Masks and Tracking Module

This module processes raw images to perform open-vocabulary object segmentation, multi-modal 2D tracking, and 3D labeling via consensus voting.

## Contents

- [sam_inference.py](file:///d:/Engineering/CUFE/GP2/VRoom/masks_and_tracking/sam_inference.py): Loads the Ultralytics SAM 3 model and extracts raw instance masks based on text prompts (e.g. "furniture").
- [mask_processor.py](file:///d:/Engineering/CUFE/GP2/VRoom/masks_and_tracking/mask_processor.py): Applies spatial rules to filter out background segments, merges overlapping or close-proximity masks, and splits disconnected components.
- [object_tracker.py](file:///d:/Engineering/CUFE/GP2/VRoom/masks_and_tracking/object_tracker.py): Tracks segmented masks across frames using Kalman filtering and Hungarian matching. Utilizes color and texture (LBP) cues, centroid priors, and global camera-motion compensation.
- [vote.py](file:///d:/Engineering/CUFE/GP2/VRoom/masks_and_tracking/vote.py): Projects sparse 3D points from COLMAP onto 2D tracked ID maps, labels points using temporal consensus/majority voting, and merges fragmented tracker IDs (alias merging).
- [runner.py](file:///d:/Engineering/CUFE/GP2/VRoom/masks_and_tracking/runner.py): Package-level orchestrator that runs the segmentation, tracking, and voting sequentially.
- [opencv_scratch.py](file:///d:/Engineering/CUFE/GP2/VRoom/masks_and_tracking/opencv_scratch.py): A custom, from-scratch NumPy/PIL-based replacement for OpenCV algorithms (Kalman filter, Lucas-Kanade optical flow, affine estimation, morphological dilation) to eliminate strict `cv2` runtime dependencies when requested.

## Prerequisites & Model Weights

1. **Python Dependencies**:
   ```bash
   pip install numpy opencv-python scipy plyfile ultralytics
   ```

2. **SAM 3 Checkpoint**:
   Place the Ultralytics SAM3 weights (e.g., `sam3.pt`) in the `models/` sub-directory:
   ```text
   masks_and_tracking/
   ├── models/
   │   └── sam3.pt  <-- Place checkpoint here
   ```

## Usage

You can run this stage on pre-reconstructed scenes using:

```bash
python masks_and_tracking/runner.py --data_path data/room_scene
```

### Key Parameters:
- `--sam_ckpt`: Path to SAM 3 checkpoint (defaults to `masks_and_tracking/models/sam3.pt`).
- `--text_prompts`: Target objects to segment (e.g. `furniture`).
- `--min_mask_area`: Minimum pixel count to keep a mask (default `120`).
- `--use_opencv`: Force standard `cv2` for performance instead of the custom `opencv_scratch` module.
