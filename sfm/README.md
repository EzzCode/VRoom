# VRoom SfM Module: 3D Reconstruction & Camera Alignment

This module automates the Structure-from-Motion (SfM) pipeline using **COLMAP** to recover camera poses and generate a sparse 3D point cloud of the scene.

## Contents

- [colmap_runner.py](file:///d:/Engineering/CUFE/GP2/VRoom/sfm/colmap_runner.py): Orchestrates feature extraction, matching, and incremental triangulation.
- [colmap_loader.py](file:///d:/Engineering/CUFE/GP2/VRoom/sfm/colmap_loader.py): A custom reader (binary/text parser) for loading COLMAP camera parameters, image/keypoint data, and 3D point coordinates.

## Prerequisites

1. **COLMAP Installation**: 
   - Download the latest binary from [COLMAP Releases](https://colmap.github.io/install.html).
   - Extract and add the directory containing the `colmap` executable to your system's `PATH`.
   - Verify it is available by running `colmap -h` in a new terminal window.

## Usage

You can run the reconstruction directly on a dataset directory:

```bash
python sfm/colmap_runner.py --data_path data/room_scene
```

### Options
- `--force_colmap`: Force re-running reconstruction even if the output directory already exists.
- `--colmap_path`: Explicitly path to the COLMAP executable (defaults to looking up `colmap` in your environment PATH).

### Output
The runner saves the outputs in the dataset directory under `colmap_output/`:
- `sparse/0/` (or `sparse/`): Contains binary files describing the reconstruction:
  - `cameras.bin`: Intrinsics (focal length, principal point, distortion).
  - `images.bin`: Extrinsics (rotation and translation matrices) and keypoint correspondences.
  - `points3D.bin`: Coordinate data and RGB colors for the sparse 3D points.
- `database.db`: SQLite database containing keypoints and match statistics.
