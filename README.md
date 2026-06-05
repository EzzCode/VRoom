# VRoom

VRoom is a comprehensive 3D reconstruction and rendering pipeline that generates virtual rooms from scenes. The pipeline processes raw scene images through Structure-from-Motion (SfM), semantic segmentation and tracking, Gaussian Splatting training, and final mesh generation. The repository also includes a companion mobile application.

## Features & Pipeline Stages

The full pipeline is coordinated via `full_pipeline_runner.py`, which executes the following stages:

1. **SfM (Structure-from-Motion)**: Camera pose estimation and sparse point cloud generation via COLMAP.
2. **Masks & Tracking**: 2D semantic segmentation (using SAM3), object tracking across frames, and 3D voting for label consensus.
3. **Gaussian Splatting Training (`gstrain`)**: 3D Gaussian Splatting training on the reconstructed scene.
4. **Mesh Generation**: Extraction of RGB, depth, and semantics to generate final 3D meshes for objects.

## Repository Structure

- `full_pipeline_runner.py`: The main script to run the entire end-to-end pipeline.
- `sfm_label_runner.py`: A combined runner for only the SfM and Semantic Labeling stages.
- `sfm/`: Structure-from-Motion utilities using COLMAP.
- `masks_and_tracking/`: Segmentation and object tracking scripts.
- `gstrain/`: Gaussian Splatting training module.
- `mesh_generation/`: Utilities for extracting meshes from the trained model.
- `diff-surfel-rasterization/`: CUDA rasterizer backend.
- `Mobile-APP/`: React Native (Expo) companion mobile application.

## Setup & Installation

### 1. Clone the Repository

If you cloned the repository without `--recursive`, pull the necessary submodules first:

```bash
git submodule update --init --recursive
```

### 2. External Dependencies (CUDA & COLMAP)

Before building and running the pipeline, ensure you have the following installed and configured on your system:

- **CUDA Toolkit**: Required for the CUDA rasterizer and GPU acceleration. Download and install the appropriate CUDA Toolkit version for your system. Ensure that the CUDA binary directory (e.g., `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin` on Windows or `/usr/local/cuda/bin` on Linux) is added to your system's `PATH` environment variable.
- **COLMAP**: Required for the Structure-from-Motion (SfM) stage. Download the COLMAP binaries or build from source. Ensure the directory containing the `colmap` executable is added to your system's `PATH` so it can be invoked from the command line.

### 3. Build the CUDA Rasterizer

To build the custom CUDA rasterizer required for Gaussian Splatting, the method depends on your operating system.

**For Linux:**
```bash
make clean build
```

**For Windows:**
```bash
cd diff-surfel-rasterization
python -m pip install -e . --no-build-isolation
```

### 4. Conda Environments

The pipeline requires specific Conda environments to run different stages:
- **`GP`**: Used for the Masks & Tracking pipeline.
- **`objectgs`**: Used for SfM, Gaussian Splatting training, and Mesh Generation.

You can create these environments using the provided configuration files in the repository:

```bash
# 1. Create the GP environment
conda env create -f environment_gp.yml

# 2. Create the objectgs environment
conda env create -f environment_objectgs.yml
```

### 5. Install VRoom Package

To allow the runner scripts to properly import the core modules (`gstrain`, `masks_and_tracking`, etc.) across the pipeline, install the repository as an editable package in your Conda environments. 

Activate your environment and run from the repository root:

```bash
pip install -e .
```
*(This uses the `pyproject.toml` file to manage package imports and dependencies).*

## Running the Pipeline

### Full Pipeline

You can run the entire pipeline end-to-end using the `full_pipeline_runner.py` script:

```bash
python full_pipeline_runner.py --data_path /path/to/scene_folder
```

**Key Arguments:**
- `--data_path`: Path to the scene folder containing an `images/` subdirectory.
- `--out_base_dir`: (Optional) Specify a unified directory to save all pipeline outputs.
- `--skip_colmap`, `--skip_masks`, `--skip_training`, `--skip_mesh_gen`: Flags to skip specific stages of the pipeline.
- `--dry_run`: Print the commands that would be executed without actually running them.
- `--small_run`: Limit training iterations and frames for a quick test run.

For a full list of configuration options, run:
```bash
python full_pipeline_runner.py --help
```

### SfM & Labeling Only

If you only want to run the COLMAP reconstruction and semantic labeling, use:

```bash
python sfm_label_runner.py --data_path /path/to/scene_folder
```

### Manual Training (3DOVS)

If there is a specific make target for training, you can use:
```bash
make run_train DATASET=3dovs
```
*(Note: Ensure you have the `objectgs` environment activated when running standalone training commands).*

## Mobile App

The repository includes a mobile application located in the `Mobile-APP/` directory. It is built using React Native and Expo. To run it, navigate to the directory and install dependencies:

```bash
cd Mobile-APP
npm install
npx expo start
```