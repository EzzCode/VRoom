"""
Download endpoints — serve pipeline outputs via S3 presigned GET URLs
or local file paths when S3 is disabled.

- ``GET /jobs/{job_id}/download/splat``       → trained 2DGS .ply
- ``GET /jobs/{job_id}/download/meshes``       → individual mesh .ply files
- ``GET /jobs/{job_id}/download/meshes/bulk``  → zipped archive of all meshes
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse

from app.config import settings
from app.dependencies import job_store
from app.models.job import (
    DownloadUrlResponse,
    JobStatus,
    MeshDownloadResponse,
)
from app.security import require_api_key
from app.services import s3 as s3_service

router = APIRouter(prefix="/jobs", tags=["downloads"])


def _require_completed_job(job_id: str) -> Path:
    """Validate job exists and is completed; return work_dir."""
    entry = job_store.get_job(job_id)
    if entry is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' not found.",
        )
    if entry.status != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Job '{job_id}' is not completed (status: {entry.status.value}).",
        )

    # If running in Modal, reload the volume so we can see the newly written outputs
    import os
    if os.environ.get("RUNNING_IN_MODAL") == "1":
        try:
            from modal_app import jobs_volume
            jobs_volume.reload()
        except ImportError:
            pass

    return entry.work_dir


def _make_download_response(
    job_id: str,
    local_path: Path,
    s3_key: str,
) -> DownloadUrlResponse:
    """Generate a presigned S3 URL or fall back to a local file path hint."""
    if settings.s3_enabled:
        url = s3_service.generate_presigned_get(s3_key)
        if url:
            return DownloadUrlResponse(
                url=url,
                expires_in_seconds=settings.presigned_url_expiry_seconds,
            )

    # Fallback: return a local API path that the client can fetch
    return DownloadUrlResponse(
        url=f"/api/v1/jobs/{job_id}/download/local/{local_path.name}",
        expires_in_seconds=0,  # no expiry for local files
    )


# ── Helper: find the latest .ply in the training output ──────────────────
def _find_splat_ply(work_dir: Path) -> Path | None:
    """Locate the trained anchor_cloud.ply in the latest checkpoint."""
    training_dir = work_dir / "output" / "training" / "gs_model"
    if not training_dir.exists():
        return None

    # Find the latest timestamped directory
    subdirs = [d for d in training_dir.iterdir() if d.is_dir()]
    if not subdirs:
        return None
    latest = max(subdirs, key=lambda d: d.stat().st_ctime)

    # Look for checkpoints/iter_*/anchor_cloud.ply
    checkpoints = latest / "checkpoints"
    if not checkpoints.exists():
        return None
    iters = sorted(
        [d for d in checkpoints.iterdir() if d.is_dir() and d.name.startswith("iter_")],
        key=lambda d: int(d.name.split("_")[1]),
    )
    if not iters:
        return None
    ply = iters[-1] / "anchor_cloud.ply"
    return ply if ply.exists() else None


# ── GET /jobs/{job_id}/download/splat ────────────────────────────────────
@router.get(
    "/{job_id}/download/splat",
    response_model=DownloadUrlResponse,
    dependencies=[Depends(require_api_key)],
    summary="Get download URL for the trained 2DGS .ply splat file.",
)
async def download_splat(job_id: str) -> DownloadUrlResponse:
    work_dir = _require_completed_job(job_id)
    ply_path = _find_splat_ply(work_dir)

    if ply_path is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Trained splat .ply file not found in job output.",
        )

    s3_key = f"jobs/{job_id}/training/{ply_path.relative_to(work_dir / 'output' / 'training').as_posix()}"
    return _make_download_response(job_id, ply_path, s3_key)


# ── GET /jobs/{job_id}/download/meshes ───────────────────────────────────
@router.get(
    "/{job_id}/download/meshes",
    response_model=MeshDownloadResponse,
    dependencies=[Depends(require_api_key)],
    summary="Get download URLs for individual mesh .ply files.",
)
def download_meshes(job_id: str) -> MeshDownloadResponse:
    work_dir = _require_completed_job(job_id)
    mesh_dir = work_dir / "output" / "mesh_objects"

    if not mesh_dir.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Mesh output directory not found.",
        )

    mesh_files = sorted(mesh_dir.glob("*.ply"))
    if not mesh_files:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No mesh .ply files found in output.",
        )

    urls: List[DownloadUrlResponse] = []
    for mf in mesh_files:
        s3_key = f"jobs/{job_id}/meshes/{mf.name}"
        urls.append(_make_download_response(job_id, mf, s3_key))

    return MeshDownloadResponse(meshes=urls)


# ── GET /jobs/{job_id}/download/meshes/bulk ──────────────────────────────
@router.get(
    "/{job_id}/download/meshes/bulk",
    response_model=DownloadUrlResponse,
    dependencies=[Depends(require_api_key)],
    summary="Get download URL for a .zip of all meshes.",
)
def download_meshes_bulk(job_id: str) -> DownloadUrlResponse:
    work_dir = _require_completed_job(job_id)
    zip_path = work_dir / "output" / "meshes.zip"

    if not zip_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Bulk mesh .zip not found. The pipeline may not have "
            "completed mesh generation.",
        )

    s3_key = f"jobs/{job_id}/meshes.zip"
    return _make_download_response(job_id, zip_path, s3_key)


# ── Local file serving fallback ──────────────────────────────────────────
@router.get(
    "/{job_id}/download/local/{filename}",
    dependencies=[Depends(require_api_key)],
    summary="Serve a local file directly (fallback when S3 is disabled).",
)
def download_local_file(job_id: str, filename: str) -> FileResponse:
    """Serve a file from the job output directory.

    The filename is validated against the actual output directory to
    prevent path traversal.
    """
    work_dir = _require_completed_job(job_id)

    # Sanitize filename — strip any path components
    safe_name = os.path.basename(filename)
    if safe_name != filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid filename.",
        )

    # Search in known output locations
    candidates = [
        work_dir / "output" / "mesh_objects" / safe_name,
        work_dir / "output" / "meshes.zip",
    ]

    # Also search in training checkpoints
    training_dir = work_dir / "output" / "training" / "gs_model"
    if training_dir.exists():
        for ply in training_dir.rglob(safe_name):
            candidates.append(ply)

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            # Verify the resolved path is still within work_dir
            resolved = candidate.resolve()
            if not str(resolved).startswith(str(work_dir.resolve()) + os.sep):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid file path.",
                )
            return FileResponse(
                path=str(resolved),
                filename=safe_name,
                headers={
                    "Content-Disposition": f'attachment; filename="{safe_name}"',
                    "X-Content-Type-Options": "nosniff",
                },
            )

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"File '{safe_name}' not found in job output.",
    )
