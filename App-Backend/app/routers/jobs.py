"""
Job management endpoints:

- ``POST /jobs/start-recon-2dgs`` — submit a new pipeline job
- ``GET  /jobs/{job_id}``         — poll job status
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
    status,
)

from app.config import settings
from app.dependencies import gpu_lock, job_store
from app.models.job import (
    JobStatus,
    JobStatusResponse,
    StartJobRequest,
    StartJobResponse,
)
from app.security import require_api_key
from app.services import s3 as s3_service
from app.services.pipeline_worker import run_pipeline

router = APIRouter(prefix="/jobs", tags=["jobs"])

# Maximum individual file size (10 MB per image)
_MAX_IMAGE_BYTES = 10 * 1024 * 1024


# ── POST /jobs/start-recon-2dgs ──────────────────────────────────────────
@router.post(
    "/start-recon-2dgs",
    response_model=StartJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(require_api_key)],
    summary="Submit a new 2DGS reconstruction job.",
)
async def start_job(
    # JSON body (used when data_source is S3 URI or local path)
    params: Optional[str] = Form(None, description="JSON-encoded StartJobRequest"),
    # Multipart image uploads
    files: Optional[List[UploadFile]] = File(None, description="Batch of image frames"),
) -> StartJobResponse:
    """Accept a new pipeline job.

    Three ingestion modes:
    1. **JSON body** with ``data_source.s3_uri`` or ``data_source.local_path``
       → pass ``params`` as form field, no files.
    2. **Multipart upload** → pass ``files`` + ``params`` as form fields.
    3. **Pure JSON** (no form) → the client can POST JSON directly when not
       uploading files (handled by the overloaded endpoint below).
    """
    # ── Parse params ─────────────────────────────────────────────────────
    if params:
        try:
            request = StartJobRequest.model_validate_json(params)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid params JSON: {exc}",
            )
    else:
        request = StartJobRequest()

    # ── Enforce queue depth (max 2: 1 running + 1 queued) ────────────────
    if job_store.active_or_pending_count() >= settings.max_queued_jobs:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="GPU queue is full. Maximum 2 jobs allowed "
            "(1 running + 1 queued). Try again later.",
        )

    # ── Create working directory ─────────────────────────────────────────
    job_id = uuid.uuid4().hex[:12]
    work_dir = settings.jobs_data_dir / job_id
    work_dir.mkdir(parents=True, exist_ok=True)
    
    job_entry = job_store.create_job(
        work_dir=work_dir,
        params=request,
        job_id=job_id,
    )

    images_dir = work_dir / "images"

    # ── Resolve data source ──────────────────────────────────────────────
    has_files = files is not None and len(files) > 0
    has_s3 = request.data_source and request.data_source.s3_uri
    has_local = request.data_source and request.data_source.local_path

    if not has_files and not has_s3 and not has_local:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No data source provided. Supply files, s3_uri, or local_path.",
        )

    if has_files:
        # Save uploaded images to work_dir/images/
        images_dir.mkdir(parents=True, exist_ok=True)
        for upload_file in files:  # type: ignore[union-attr]
            # Validate extension
            ext = os.path.splitext(upload_file.filename or "")[1].lower()
            if ext not in settings.allowed_image_extensions:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"File '{upload_file.filename}' has disallowed "
                    f"extension '{ext}'. Allowed: "
                    f"{settings.allowed_image_extensions}",
                )
            # Validate size
            contents = await upload_file.read()
            if len(contents) > _MAX_IMAGE_BYTES:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"File '{upload_file.filename}' exceeds 10 MB limit.",
                )
            # Rename to UUID to prevent path traversal
            safe_name = f"{uuid.uuid4().hex}{ext}"
            dest = images_dir / safe_name
            dest.write_bytes(contents)

    elif has_s3:
        # Download images from S3 to work_dir/images/
        images_dir.mkdir(parents=True, exist_ok=True)
        s3_uri: str = request.data_source.s3_uri  # type: ignore[union-attr]
        # Parse s3://bucket/prefix format
        if s3_uri.startswith("s3://"):
            parts = s3_uri[5:].split("/", 1)
            s3_prefix = parts[1] if len(parts) > 1 else ""
        else:
            s3_prefix = s3_uri
        count = s3_service.download_directory_from_s3(s3_prefix, images_dir)
        if count == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"No files found at S3 URI: {s3_uri}",
            )

    elif has_local:
        # Validate and symlink local path
        local_src = Path(request.data_source.local_path)  # type: ignore[union-attr]
        # Security: ensure the path exists and is a directory
        if not local_src.is_dir():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Local path does not exist or is not a directory: "
                f"{local_src}",
            )
        # Symlink images/ into work_dir
        # If the source already has an images/ subdirectory, link that;
        # otherwise link the whole directory as images/
        src_images = local_src / "images"
        if src_images.is_dir():
            # The source has the expected structure — symlink the parent
            # so that work_dir mirrors the data_path layout
            if not images_dir.exists():
                os.symlink(src_images, images_dir)
        else:
            if not images_dir.exists():
                os.symlink(local_src, images_dir)

    # ── Spawn background task ────────────────────────────────────────────
    asyncio.create_task(run_pipeline(job_entry.job_id, job_store, gpu_lock))

    return StartJobResponse(
        job_id=job_entry.job_id,
        status=JobStatus.PENDING,
    )


# ── Overloaded endpoint for pure JSON body (no form data) ────────────────
@router.post(
    "/start-recon-2dgs/json",
    response_model=StartJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(require_api_key)],
    summary="Submit a new job with a pure JSON body (no file uploads).",
)
async def start_job_json(request: StartJobRequest) -> StartJobResponse:
    """Convenience endpoint when no files are uploaded."""
    # Reuse the form-based endpoint logic by encoding params as JSON
    return await start_job(
        params=request.model_dump_json(),
        files=None,
    )


# ── GET /jobs/{job_id} ───────────────────────────────────────────────────
@router.get(
    "/{job_id}",
    response_model=JobStatusResponse,
    dependencies=[Depends(require_api_key)],
    summary="Poll job status.",
)
def get_job_status(job_id: str) -> JobStatusResponse:
    entry = job_store.get_job(job_id)
    if entry is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' not found.",
        )
    return JobStatusResponse(
        job_id=entry.job_id,
        status=entry.status,
        current_stage=entry.current_stage,
        error_message=entry.error_message,
        created_at=entry.created_at,
        updated_at=entry.updated_at,
    )
