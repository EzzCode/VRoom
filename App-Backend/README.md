# VRoom Backend API

Production-ready FastAPI backend for the VRoom 2D Gaussian Splatting pipeline, deployed on [Modal.com](https://modal.com).

## Architecture

```
App-Backend/
├── app/
│   ├── main.py              # FastAPI app factory, middleware, lifespan
│   ├── config.py            # Settings from env vars (.env supported)
│   ├── dependencies.py      # GPU lock, job store singletons
│   ├── security.py          # API key auth
│   ├── models/
│   │   ├── job.py           # Job lifecycle schemas
│   │   └── pipeline.py      # Pipeline parameter schemas
│   ├── routers/
│   │   ├── uploads.py       # GET /upload-url (S3 presigned POST)
│   │   ├── jobs.py          # POST /jobs/start-recon-2dgs, GET /jobs/{id}
│   │   └── downloads.py     # Download endpoints (splat, meshes, bulk zip)
│   └── services/
│       ├── job_store.py     # In-memory job state store
│       ├── s3.py            # S3 helpers with local fallback
│       └── pipeline_worker.py  # Background GPU task
├── modal_app.py             # Modal.com deployment wrapper
├── requirements.txt
└── README.md
```

## Configuration

All settings are loaded from environment variables. Create a `.env` file:

```bash
# Required
VROOM_API_KEY=your-secret-api-key

# S3 (optional — falls back to local disk if omitted)
AWS_S3_BUCKET=vroom-pipeline-outputs
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...

# Conda environments (must match server setup)
PIPELINE_CONDA_ENV=pipeline
MASKS_CONDA_ENV=masks
```

## Local Development

```bash
cd App-Backend
pip install -r requirements.txt
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

## Modal Deployment

```bash
# 1. Create secrets in Modal dashboard
modal secret create vroom-secrets \
  VROOM_API_KEY=your-key \
  AWS_S3_BUCKET=your-bucket \
  AWS_REGION=us-east-1 \
  AWS_ACCESS_KEY_ID=AKIA... \
  AWS_SECRET_ACCESS_KEY=...

# 2. Deploy
cd App-Backend
modal deploy modal_app.py

# 3. For local dev with Modal
modal serve modal_app.py
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/api/v1/uploads/upload-url` | S3 presigned POST URL |
| `POST` | `/api/v1/jobs/start-recon-2dgs` | Submit job (multipart) |
| `POST` | `/api/v1/jobs/start-recon-2dgs/json` | Submit job (JSON body) |
| `GET` | `/api/v1/jobs/{job_id}` | Poll job status |
| `GET` | `/api/v1/jobs/{job_id}/download/splat` | Download trained .ply |
| `GET` | `/api/v1/jobs/{job_id}/download/meshes` | Download mesh files |
| `GET` | `/api/v1/jobs/{job_id}/download/meshes/bulk` | Download meshes .zip |

## GPU Selection

The pipeline runs on an **NVIDIA A10G** (24 GB VRAM, Ampere):
- 164 KB shared memory per SM → near-zero spills in tile-based rasterization
- 600 GB/s memory bandwidth → fast grad accumulation in backward render
- 31.2 FP32 TFLOPS → efficient matrix math

Expected speedup: **1.2 sec/it → 0.15–0.3 sec/it (3–7+ it/sec)**
