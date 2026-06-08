"""
Modal.com deployment for the VRoom 2DGS Pipeline API.

This script provisions a high-performance Linux container on Modal, installs
all dependencies, compiles the custom CUDA rasterizers, and hosts the FastAPI server.

GPU Selection:
- Uses A10G (24 GB VRAM, 600 GB/s bandwidth, 164 KB shared mem/SM) for optimal 
  price/performance with tile-based rasterization and gradient accumulation.

NOTE on Environments:
Your local `environment_pipeline.yml` contains Windows-specific dependencies 
(e.g., `ucrt`, `vc`, `pywin32`) and cannot be used in Modal's Linux containers. 
Instead, we build a unified Linux-compatible environment here that contains PyTorch 
for CUDA 11.8, COLMAP, SAM2, and compiles your local CUDA extensions directly 
into the image.
"""

from __future__ import annotations

import modal

# ── Modal App ────────────────────────────────────────────────────────────
app = modal.App("vroom-2dgs-pipeline")

# ── Container Image ──────────────────────────────────────────────────────
# We use the official NVIDIA CUDA 11.8 development image as the base so we
# have nvcc and the full CUDA toolkit available to compile the rasterizers.
vroom_image = (
    modal.Image.from_registry(
        "nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04", add_python="3.10"
    )
    .apt_install(
        "git",
        "wget",
        "build-essential",
        "cmake",
        "ninja-build",
        "libgl1-mesa-glx",
        "libglib2.0-0",
        "xvfb",    # Virtual framebuffer for headless COLMAP OpenGL
        "libboost-program-options-dev",
        "libboost-filesystem-dev",
        "libboost-graph-dev",
        "libboost-system-dev",
        "libboost-test-dev",
        "libsuitesparse-dev",
        "libfreeimage-dev",
        "libgoogle-glog-dev",
        "libgflags-dev",
        "libglew-dev",
        "freeglut3-dev",
        "libxmu-dev",
        "libxi-dev",
        "libatlas-base-dev",
        "libcgal-dev",
        "lz4",
        "libceres-dev",
        "libeigen3-dev",
        "libmetis-dev",
        "libflann-dev",
        "libsqlite3-dev",
    )
    # 1. Compile COLMAP 3.10 from source with CUDA enabled and GUI disabled
    .run_commands(
        "git clone --branch 3.10 --depth 1 https://github.com/colmap/colmap.git /opt/colmap-source",
        "cd /opt/colmap-source && mkdir build && cd build && cmake .. -GNinja -DCUDA_ENABLED=ON -DGUI_ENABLED=OFF -DCMAKE_CUDA_ARCHITECTURES=\"80;86\" -DCMAKE_BUILD_TYPE=Release && ninja install"
    )
    # 2. Install Miniconda directly onto the NVIDIA devel image
    .run_commands(
        "wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh",
        "bash miniconda.sh -b -p /opt/conda",
        "rm miniconda.sh",
        "ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh"
    )
    # Force PyTorch to compile extensions for A10G architecture (Compute 8.6), add conda to PATH, and add PyTorch indexes
    .env({
        "PATH": "/opt/conda/bin:$PATH", 
        "TORCH_CUDA_ARCH_LIST": "8.0;8.6",
        "PIP_EXTRA_INDEX_URL": "https://download.pytorch.org/whl/cu118 https://download.pytorch.org/whl/cu126",
        "PIP_FIND_LINKS": "https://data.pyg.org/whl/torch-2.1.2+cu118.html",
        "MAX_JOBS": "4"
    })
    # 2. Add local environment files
    .add_local_file("modal_pipeline.yml", remote_path="/app/modal_pipeline.yml", copy=True)
    .add_local_file("modal_masks.yml", remote_path="/app/modal_masks.yml", copy=True)
    # 3. Build Conda envs
    .run_commands(
        "conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main",
        "conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r",
        "conda env create -f /app/modal_pipeline.yml",
        "conda env create -f /app/modal_masks.yml"
    )
    # 4. Copy and compile local CUDA extensions specifically into the `pipeline` conda environment
    .add_local_dir(
        "../diff-surfel-rasterization",
        remote_path="/build/diff-surfel-rasterization",
        copy=True,
    )
    .run_commands(
        "conda run -n pipeline pip install --no-build-isolation /build/diff-surfel-rasterization"
    )
    # 5. Install FastAPI backend dependencies into the global environment to serve web requests
    .pip_install(
        "fastapi>=0.115.0",
        "uvicorn[standard]>=0.32.0",
        "python-multipart>=0.0.18",
        "boto3>=1.35.0",
        "pydantic-settings>=2.7.0",
        "pyyaml==6.0.3",
        "huggingface_hub>=0.20.0",
    )
)


def ignore_vroom_paths(path):
    import os
    # Pad with slashes to ensure exact directory matching works for relative paths
    path_str = "/" + str(path).replace(os.sep, "/").strip("/") + "/"
    return any(
        excluded in path_str
        for excluded in [
            "/.git/",
            "/__pycache__/",
            "/node_modules/",
            "/Mobile-APP/",
            "/.vscode/",
            "/jobs_data/",
            "/brain/",
            "/output/",
            "/gstrain/datasets/",
            "/datasets/",
            # Exclude the folders we already built into the image
            "/diff-surfel-rasterization/",
            "/gsplat-object/",
        ]
    )


vroom_image = vroom_image.add_local_dir(
    local_path="..",
    remote_path="/app",
    ignore=ignore_vroom_paths,
)

# ── Shared Jobs Volume ───────────────────────────────────────────────────
jobs_volume = modal.Volume.from_name("vroom-jobs-data", create_if_missing=True)

# ── GPU Pipeline Function ───────────────────────────────────────────────
@app.function(
    image=vroom_image,
    gpu="A100",
    timeout=10800,  # 3 hours max per job
    secrets=[modal.Secret.from_dotenv()],
    volumes={"/app/jobs_data": jobs_volume},
)
def run_pipeline_on_gpu(cli_args: list[str], work_dir: str) -> dict:
    """Execute the VRoom pipeline as a subprocess on an A10G GPU container."""
    import subprocess
    from pathlib import Path

    log_file = Path(work_dir) / "pipeline.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    import os
    env = os.environ.copy()
    env["PYTHONPATH"] = "/app"

    env["PYTHONUNBUFFERED"] = "1"

    import pty
    log_content = []

    master_fd, slave_fd = pty.openpty()

    process = subprocess.Popen(
        cli_args,
        stdout=slave_fd,
        stderr=subprocess.STDOUT,
        cwd=work_dir,
        env=env,
    )
    os.close(slave_fd)

    with open(log_file, "w") as lf:
        while True:
            try:
                data = os.read(master_fd, 4096)
            except OSError:
                break
            if not data:
                break
            text = data.decode("utf-8", errors="replace")
            print(text, end="", flush=True)
            lf.write(text)
            lf.flush()
            
            log_content.append(text)
            if len(log_content) > 1000:
                log_content = log_content[-1000:]

    returncode = process.wait()
    os.close(master_fd)

    # VERY IMPORTANT: Modal Volumes do not automatically sync changes made
    # inside a container to the cloud. We must explicitly commit the volume
    # before exiting so the FastAPI container can read the logs and meshes!
    jobs_volume.commit()

    return {
        "returncode": returncode,
        "log_tail": "".join(log_content),
    }


# ── Pipeline Orchestrator (Decoupled Background Task) ────────────────────
@app.function(
    image=vroom_image,
    secrets=[modal.Secret.from_dotenv()],
    timeout=10800,  # 3 hours
    volumes={"/app/jobs_data": jobs_volume},
)
def orchestrate_pipeline_modal(job_id: str, cli_args: list[str], work_dir: str):
    """
    Runs completely detached from the 60-second FastAPI web container.
    This safely downloads SAM3, spawns the GPU container, and handles S3 uploads!
    """
    import sys
    import os
    sys.path.insert(0, "/app/App-Backend")
    os.environ["RUNNING_IN_MODAL"] = "1"
    
    from app.services.pipeline_worker import run_pipeline_modal_logic
    run_pipeline_modal_logic(job_id, cli_args, work_dir)



# ── FastAPI Web Endpoint ─────────────────────────────────────────────────
@app.function(
    image=vroom_image,
    secrets=[modal.Secret.from_dotenv()],
    min_containers=1,  # Keep one container warm for low-latency API responses
    volumes={"/app/jobs_data": jobs_volume},
)
@modal.concurrent(max_inputs=10)  # Handle multiple API requests concurrently
@modal.asgi_app()
def fastapi_entrypoint():
    """Serve the FastAPI app via Modal's ASGI integration."""
    import sys
    import os

    # Set an environment variable so the backend knows it's running in Modal
    os.environ["RUNNING_IN_MODAL"] = "1"

    sys.path.insert(0, "/app/App-Backend")

    from app.main import app

    return app
