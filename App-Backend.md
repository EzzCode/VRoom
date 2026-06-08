# VRoom 2DGS Pipeline Backend Guide

This document details the exact requirements, setup procedures, and API references for running the VRoom 2DGS (2D Gaussian Splatting) Reconstruction and Semantic Meshing Pipeline backend on [Modal](https://modal.com/).

> [!IMPORTANT]
> The backend is designed as a highly scalable, serverless architecture that decouples low-latency FastAPI endpoints from heavy, long-running A10G GPU tasks. Modal handles container scaling, volume synchronization, and caching automatically.

## 1. Prerequisites

Before deploying or running the backend, ensure you have the following prerequisites configured:

- **Modal Account:** You must have a Modal account and the Modal CLI authenticated locally.
- **HuggingFace Account:** Required to download the `sam3.pt` weights.
- **AWS S3 Bucket:** Required for storing input images and trained output meshes (e.g., `s3://scene-recon-assets-be`).
- **Conda (Miniconda/Anaconda):** Required to manage your local deployment environment.

## 2. Environment Configuration

### The `.env` File
Modal injects secrets securely into the containers. You must create a `.env` file in the `App-Backend` directory containing your credentials and configuration.

```env
# Modal API Authentication
API_KEY_SECRET=...

# HuggingFace (For SAM3 weights)
HF_TOKEN=hf_your_huggingface_token

# AWS S3 Configuration (For data ingestion and result uploads)
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-1
S3_ENABLED=true
```

### Local Deployment Environment
You need a local python environment with the `modal` package installed to deploy the application to the cloud.
```bash
conda create -n modal_env python=3.10 -y
conda activate modal_env
pip install modal
modal setup  # Authenticate with your Modal account
```

## 3. Deployment

### Generating Clean Linux Dependencies
Before deploying, you **must** ensure your Conda environment YAML files are stripped of any Windows-specific system packages and local pip dependencies. We provide a utility script to do this automatically:
```bash
# Run this from the project root
python App-Backend/generate_ymls.py
```
This script will parse your raw exports, remove incompatible packages (like `ucrt`, `vc`, `pywin32`), ensure PyTorch has the correct CUDA indices, and output `modal_pipeline.yml` and `modal_masks.yml`.

### Deploying to Modal
Once your environment variables are set and the local conda environment is activated, you can build and deploy the backend to Modal's servers.

```bash
cd /home/galal/VRoom/App-Backend
conda activate modal_env

# Deploy the pipeline to Modal
modal deploy modal_app.py
```

> [!TIP]
> The first deployment takes ~15-20 minutes because Modal builds a custom container from scratch, downloading CUDA toolkits and compiling COLMAP 3.10 natively with `-DCMAKE_CUDA_ARCHITECTURES=86`. All subsequent deployments will be cached and finish in seconds!

## 4. API Reference and Commands

The backend exposes a secure FastAPI interface. All routes are mounted under the `/api/v1` prefix.

### Endpoint 1: Start a Reconstruction Job
Submits a job to the background orchestrator. The API immediately returns an HTTP 202 and a `job_id` while spinning up the A10G GPU container in the background.

```bash
curl -s -X POST "https://galalmohamed2003--vroom-2dgs-pipeline-fastapi-entrypoint.modal.run/api/v1/jobs/start-recon-2dgs/json" \
     -H "Authorization: Bearer <API_KEY_SECRET>" \
     -H "Content-Type: application/json" \
     -d '{
           "data_source": {
             "s3_uri": "s3://scene-recon-assets-be/images/"
           }
         }'
```
**Response:**
```json
{"job_id": "eb7685212907", "status": "pending"}
```

### Endpoint 2: Poll Job Status
Use the `job_id` to check the progress of the pipeline.

```bash
curl -s -X GET "https://galalmohamed2003--vroom-2dgs-pipeline-fastapi-entrypoint.modal.run/api/v1/jobs/eb7685212907" \
     -H "Authorization: Bearer <API_KEY_SECRET>"
```
**Response (`processing`):**
```json
{
  "job_id": "eb7685212907",
  "status": "processing",
  "current_stage": "running_gpu",
  "error_message": null,
  "created_at": "2026-06-06T09:33:43.090237+00:00",
  "updated_at": "2026-06-06T09:34:37.201709+00:00"
}
```

### Endpoint 3: Download the Trained 2DGS Model
Once the job `status` is `"completed"`, retrieve the pre-signed S3 URL for the `.ply` splat file.

```bash
curl -s -X GET "https://galalmohamed2003--vroom-2dgs-pipeline-fastapi-entrypoint.modal.run/api/v1/jobs/eb7685212907/download/splat" \
     -H "Authorization: Bearer <API_KEY_SECRET>"
```

### Endpoint 4: Download the Extracted Meshes
Retrieve the pre-signed S3 URL for a bulk `.zip` containing all the individual semantic mesh files.

```bash
curl -s -X GET "https://galalmohamed2003--vroom-2dgs-pipeline-fastapi-entrypoint.modal.run/api/v1/jobs/eb7685212907/download/meshes/bulk" \
     -H "Authorization: Bearer <API_KEY_SECRET>"
```

## 5. Debugging & Maintenance

### Real-Time Logs
Logs automatically stream to the Modal dashboard. To view them locally via CLI:
```bash
modal app logs vroom-2dgs-pipeline
```
For a constant stream use:
```bash
modal app logs -f vroom-2dgs-pipeline
```

### Debugging Active Containers
If you need to enter an actively running GPU container to inspect files or processes:
```bash
# 1. List active containers
modal container list

# 2. Exec into the container (Replace Container ID)
modal container exec ta-01KTE4HKY2J1JXE6S6KRKH5TA6 -- /bin/bash
```

> [!NOTE]
> The backend automatically cleans up raw images from the Modal volume after processing to conserve space, but preserves the generated outputs to ensure that your S3 download URLs can be dynamically generated.
