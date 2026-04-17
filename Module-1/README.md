# VRoom Module-1: Feature Extraction & Object Tracking

This module processes raw images to generate sparse 3D point clouds and per-object tracking masks. It combines **COLMAP** for Structure-from-Motion, **SAM 2** for high-quality object segmentation, and a **Multi-Modal Tracker** to maintain object IDs across frames. Finally, it projects masks back onto the 3D points for object labeling.

## Overview

The pipeline consists of four main scripts:
1. **`colmap_runner.py`**: Automates COLMAP feature extraction and sparse reconstruction.
2. **`mask_processor.py`**: Uses SAM 2 to generate per-frame segmentation masks and filters backgrounds.
3. **`object_tracker.py`**: Tracks segmented objects across frames using Optical Flow, Color, and Shape features, generating consistent 16-bit ID maps.
4. **`vote.py`**: Projects the 3D points from COLMAP onto the 2D ID maps to assign labels to the point cloud through multi-view voting.

---

## Prerequisites & Installation

```markdown
### 1. Python Dependencies
Ensure you have the required Python packages installed.
```bash
pip install numpy opencv-python scipy plyfile scikit-learn
pip install "git+https://github.com/facebookresearch/sam2.git"
```
```

### 2. COLMAP
COLMAP is required for extracting camera poses and generating the sparse point cloud.
- **Download**: [COLMAP Releases](https://colmap.github.io/install.html) (Windows binaries are available).
- **Setup**: Extract the COLMAP folder and add its path to your system's `PATH` environment variable.
- **Verify**: Open a new terminal and run `colmap help`. It should print the COLMAP help text.

### 3. SAM 2 Checkpoints
The `mask_processor.py` script requires a pre-trained Segment Anything Model 2 (SAM 2) checkpoint.
- **Download**: You need the `sam2.1_hiera_large.pt` checkpoint.
  - [Download sam2.1_hiera_large.pt natively from Meta](https://github.com/facebookresearch/sam2#download-checkpoints)
- **Placement**: Place the downloaded `.pt` file inside the `Module-1/models/` directory. Create the `models` folder if it doesn't exist.
```text
VRoom/
└── Module-1/
    ├── models/
    │   └── sam2.1_hiera_large.pt  <-- Place here
    ├── mask_processor.py
    └── ...
```

---

## Pipeline Usage

Assuming your raw images are in `data/room_scene/images`, run the following steps in order from the `VRoom` system root or `Module-1` directory (paths in examples assume running from the root containing `Module-1`).

### One-Command Runner (Recommended)

Run the full Module-1 pipeline end-to-end:

```bash
python Module-1/module1_runner.py --data_path data/room_scene
```

Useful options:

```bash
# Balanced robust defaults (recommended)
python Module-1/module1_runner.py --data_path data/room_scene --profile balanced

# Conservative profile: avoid accidental object merges
python Module-1/module1_runner.py --data_path data/room_scene --profile conservative

# Recall profile: keep more candidate masks (for missing objects)
python Module-1/module1_runner.py --data_path data/room_scene --profile recall

# Rebuild COLMAP output
python Module-1/module1_runner.py --data_path data/room_scene --force_colmap

# Skip expensive earlier stages and only run voting
python Module-1/module1_runner.py --data_path data/room_scene --skip_colmap --skip_masks --skip_tracking

# Use CPU for SAM (slower, but works without CUDA)
python Module-1/module1_runner.py --data_path data/room_scene --device cpu

# Tighten voting confidence
python Module-1/module1_runner.py --data_path data/room_scene --min_confidence 0.45 --min_support 4
```

### Step 1: 3D Reconstruction (COLMAP)
Extract camera poses and build a sparse point cloud.

```bash
python Module-1/colmap_runner.py --data_path data/room_scene
```
**Output**: Creates a COLMAP database and a `sparse/0/` folder containing `cameras.bin`, `images.bin`, and `points3D.bin`.

### Step 2: Mask Generation & Filtering (SAM 2)
Generate foreground object masks for every image.

```bash
python Module-1/mask_processor.py \
    --input_dir data/room_scene/images \
    --output_dir data/room_scene/sam_output
```
**Output**: Creates `masks/` (`.npz` archives keyed by the image stem) and `visible_masks/` (debug PNG overlays).

### Step 3: Object Tracking
Link the SAM masks across frames to assign consistent IDs.

```bash
python Module-1/object_tracker.py \
    --input_dir data/room_scene/images \
    --mask_dir data/room_scene/sam_output/masks \
    --output_dir data/room_scene/tracked
```
*Note: This generates 16-bit PNG ID maps, where pixel values correspond to object IDs. A value of 0 is background.*
**Output**: Creates `id_maps/` (16-bit PNGs whose filenames exactly match the source image stem) and `tracked_vis/` (debug tracking overlays).

### Step 4: 3D Point Cloud Voting
Project the 3D points onto the tracked 2D ID maps to label the scene in 3D space.

```bash
python Module-1/vote.py \
    --data_path data/room_scene \
    --sparse_dir sparse/0 \
    --mask_dir tracked/id_maps \
    --output_dir labeled_output \
    --algorithm majority
```
**Output**: Creates labeled PLY point clouds in `data/room_scene/labeled_output/`, including separate clouds per tracked object.

### Naming Contract

Module-1 now expects a strict one-to-one filename contract:
- image `train_rgb_0000.png` -> tracker ID map `train_rgb_0000.png`
- COLMAP image stem -> voter mask filename with the same stem plus `.png`

The pipeline fails fast if a mask is missing instead of guessing an alternate frame name.
