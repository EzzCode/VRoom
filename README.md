# VRoom

VRoom is a comprehensive 3D reconstruction and rendering pipeline that generates virtual rooms from scenes. The pipeline processes raw scene images through Structure-from-Motion (SfM), semantic segmentation and tracking, Gaussian Splatting training, and final mesh generation. The repository also includes a companion mobile application.

## Features & Pipeline Stages

The full pipeline is coordinated via `full_pipeline_runner.py`, which executes the following stages:

1. **SfM (Structure-from-Motion)**: Camera pose estimation and sparse point cloud generation via COLMAP.
2. **Masks & Tracking**: 2D semantic segmentation (using SAM3), object tracking across frames, and 3D voting for label consensus.
3. **Gaussian Splatting Training (`gstrain`)**: 3D Gaussian Splatting training on the reconstructed scene.
4. **Object Refiner**: Optional Per-object quality enhancement by selecting the best real observation, generating a novel-view orbit with SV3D, and training a dedicated per-object Gaussian model from the combined real and synthetic views.
5. **Mesh Generation**: Extraction of RGB, depth, and semantics to generate final 3D meshes for objects.

## Repository Structure

- `full_pipeline_runner.py`: The main script to run the entire end-to-end pipeline.
- `sfm_label_runner.py`: A combined runner for only the SfM and Semantic Labeling stages.
- `sfm/`: Structure-from-Motion utilities using COLMAP.
- `masks_and_tracking/`: Segmentation and object tracking scripts.
- `gstrain/`: Gaussian Splatting training module.
- `mesh_generation/`: Utilities for extracting meshes from the trained model.
- `object_refiner/`: Per-object Gaussian refinement using SV3D novel-view synthesis.
- `diff-surfel-rasterization/`: CUDA rasterizer backend.
- `Mobile-APP/`: React Native (Expo) companion mobile application.

## Setup & Installation

### 1. Clone the Repository

```bash
git clone --recursive https://github.com/EzzCode/VRoom.git
```

If you cloned the repository without `--recursive`, pull the necessary submodules first:

```bash
git submodule update --init --recursive
```

The required submodules are defined in .gitmodules

### 2. External Dependencies (CUDA & COLMAP)

Before building and running the pipeline, ensure you have the following installed and configured on your system:

- **CUDA Toolkit**: Required for the CUDA rasterizer and GPU acceleration. Download and install the appropriate CUDA Toolkit version for your system. Ensure that the CUDA binary directory (e.g., `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin` on Windows or `/usr/local/cuda/bin` on Linux) is added to your system's `PATH` environment variable.
- **COLMAP**: Required for the Structure-from-Motion (SfM) stage. Ensure the directory containing the `colmap` executable is added to your system's `PATH`.
  - **Windows**: Download the [COLMAP binaries](https://colmap.github.io/install.html).
  - **Linux / WSL**: We provide a helper script to build COLMAP 3.10 from source with CUDA enabled. Run `./environments/install_colmap_linux.sh` from the repository root.
- **SAM3 / Ultralytics**: Required for the Masks & Tracking stage. The `./sam3.pt` segmentation model weights are downloaded automatically by Ultralytics on first use (it will be saved to your current working directory). Set the `--ultralytics_home` argument (or the `ULTRALYTICS_HOME` environment variable) to a directory with sufficient storage space for the checkpoint cache. A CUDA-capable GPU is strongly recommended.
- **SV3D (sv3d-diffusers)**: Required for the Object Refiner stage. Clone the `chenguolin/sv3d-diffusers` repository into `external_deps/sv3d-diffusers/` inside the workspace root:
  ```bash
  git clone https://github.com/chenguolin/sv3d-diffusers external_deps/sv3d-diffusers
  ```
  Model weights are downloaded from Hugging Face on first use. Set the `HF_HOME` environment variable (or update `HF_CACHE_DIR` in `object_refiner/constants.py`) to point to a directory with sufficient storage space.

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
- **`masks`**: Used for the Masks & Tracking pipeline.
- **`pipeline`**: Used for SfM, Gaussian Splatting training, and Mesh Generation.

You can create these environments using the provided configuration files in the repository:

**For Windows:**
```bash
# 1. Create the masks environment
conda env create -f environments/environment_masks.yml

# 2. Create the pipeline environment
conda env create -f environments/environment_pipeline.yml
```

**For Linux / WSL:**
```bash
# 1. Create the masks environment
conda env create -f environments/linux_environment_masks.yml

# 2. Create the pipeline environment
conda env create -f environments/linux_environment_pipeline.yml
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

### Object Refiner

After full-scene Gaussian training, run the object refiner to produce per-object high-quality Gaussian models. The refiner requires the trained scene model path and the scene output directory:

```bash
python refine_all_objects.py \
  --model_path /path/to/training/gs_model/<run_timestamp> \
  --scene_dir /path/to/scene_output \
  --iterations 20000
```

**Key Arguments:**
- `--model_path`: Path to the trained scene Gaussian model directory.
- `--scene_dir`: Root output directory of the pipeline run (contains `labeled_output/`, `tracking_output/`, etc.).
- `--object_ids`: (Optional) Space-separated list of object IDs to refine. Refines all objects if omitted.
- `--iterations`: Number of training iterations per object.
- `--reuse_sv3d`: Skip SV3D generation if outputs already exist on disk.
- `--debug`: Enable visual debugging artifacts.

For a full list of options, run:
```bash
python refine_all_objects.py --help
```

### Manual Training (3DOVS)

If there is a specific make target for training, you can use:
```bash
make run_train DATASET=3dovs
```
*(Note: Ensure you have the `pipeline` environment activated when running standalone training commands).*

## Backend Deployment

The VRoom 2DGS pipeline can be deployed as a serverless backend using [Modal](https://modal.com/). This allows you to offload the heavy CUDA-accelerated processing (COLMAP, SAM3, and Gaussian Splatting) to an A10G cloud GPU while exposing a clean REST API.

Detailed instructions on how to configure, deploy, and interact with the backend API are **not** included in this file. Please refer to the dedicated backend guide:

👉 **[Vroom_BE.md](App-Backend.md)**

## Mobile App

The repository includes a mobile application located in the `Mobile-APP/` directory. It is built using React Native and Expo. To run it, navigate to the directory and install dependencies:

```bash
cd Mobile-APP
npm install
npx expo run:android
```