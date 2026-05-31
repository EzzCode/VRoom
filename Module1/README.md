# VRoom Module1: Object Segmentation, Tracking & 3D Labeling

This module processes raw images to generate labeled 3D point clouds with per-object identities. It combines **COLMAP** for Structure-from-Motion, **SAM 3** (Ultralytics) for open-vocabulary object segmentation, a **multi-modal tracker** with temporal consensus to maintain IDs across frames, and a **3D voting** stage with alias merging to label the sparse point cloud.

## Architecture

The pipeline consists of five source files orchestrated by a single runner:

| File | Role |
|---|---|
| `colmap_runner.py` | Automates COLMAP feature extraction and sparse reconstruction |
| `sam_inference.py` | Loads the Ultralytics SAM3 model and produces raw boolean masks |
| `mask_processor.py` | Applies rule-based spatial post-processing (background filter, overlap/proximity merge, component split) |
| `object_tracker.py` | Tracks segmented objects across frames using Kalman + Hungarian matching with temporal consensus |
| `vote.py` | Projects 3D points onto tracked 2D ID maps, resolves labels via majority voting, and merges fragmented tracker IDs in 3D (alias merging) |

Supporting files:
- `colmap_loader.py` — Vendored COLMAP binary/text reader (Inria/GRAPHDECO)
- `module1_runner.py` — End-to-end pipeline orchestrator
- `models/` — Directory for model checkpoints (e.g. `sam3.pt`)

---

## Prerequisites & Installation

### 1. Python Dependencies
```bash
pip install numpy opencv-python scipy plyfile ultralytics
```

### 2. COLMAP
COLMAP is required for extracting camera poses and generating the sparse point cloud.
- **Download**: [COLMAP Releases](https://colmap.github.io/install.html) (Windows binaries are available).
- **Setup**: Extract the COLMAP folder and add its path to your system's `PATH` environment variable.
- **Verify**: Open a new terminal and run `colmap help`. It should print the COLMAP help text.

### 3. SAM 3 Checkpoint
The segmentation stage requires an Ultralytics SAM3 checkpoint.
- **Placement**: Place the downloaded `.pt` file inside the `Module1/models/` directory.
```text
VRoom/
└── Module1/
    ├── models/
    │   └── sam3.pt  <-- Place here
    ├── sam_inference.py
    ├── mask_processor.py
    └── ...
```

---

## Pipeline Usage

Assuming your raw images are in `data/room_scene/images`, run the following from the `VRoom` root directory.

### One-Command Runner (Recommended)

```bash
python Module1/module1_runner.py --data_path data/room_scene
```

Useful options:

```bash
# Rebuild COLMAP output
python Module1/module1_runner.py --data_path data/room_scene --force_colmap

# Skip expensive earlier stages and only re-run voting
python Module1/module1_runner.py --data_path data/room_scene --skip_colmap --skip_masks --skip_tracking

# Use CPU for SAM (slower, but works without CUDA)
python Module1/module1_runner.py --data_path data/room_scene --device cpu
```

### Step 1: 3D Reconstruction (COLMAP)
Extract camera poses and build a sparse point cloud.

```bash
python Module1/colmap_runner.py --data_path data/room_scene
```
**Output**: Creates a COLMAP database and a `sparse/0/` folder containing `cameras.bin`, `images.bin`, and `points3D.bin`.

### Step 2: Mask Generation & Filtering (SAM 3)
Generate foreground object masks for every image.

```bash
python Module1/mask_processor.py \
    --input_dir data/room_scene/images \
    --output_dir data/room_scene/sam_output
```
**Output**: Creates `masks/` (`.npz` archives) and `visible_masks/` (debug PNG overlays).

### Step 3: Object Tracking
Link the SAM masks across frames to assign consistent IDs.

```bash
python Module1/object_tracker.py \
    --input_dir data/room_scene/images \
    --mask_dir data/room_scene/sam_output/masks \
    --output_dir data/room_scene/tracked
```
The tracker emits stable object IDs only after a candidate has persisted for the confirmation window, which filters short-lived SAM fragments before 3D voting.
**Output**: Creates `id_maps/` (16-bit PNGs where pixel values = object IDs, 0 = background) and `tracked_vis/` (debug tracking overlays).

### Step 4: 3D Point Cloud Voting
Project the 3D points onto the tracked 2D ID maps to label the scene in 3D space.

```bash
python Module1/vote.py \
    --data_path data/room_scene \
    --sparse_dir sparse/0 \
    --mask_dir tracked/id_maps \
    --output_dir labeled_output \
    --algorithm majority
```
**Output**: Creates labeled PLY point clouds in `labeled_output/`, including per-object clouds and an `alias_merge_map.json` documenting how fragmented tracker IDs were consolidated.

### Naming Contract

Module1 expects a strict one-to-one filename contract:
- image `train_rgb_0000.png` → tracker ID map `train_rgb_0000.png`
- COLMAP image stem → voter mask filename with the same stem plus `.png`

The pipeline fails fast if a mask is missing instead of guessing an alternate frame name.
