"""
Pydantic schemas for the Job lifecycle (requests, responses, status).
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field

from .pipeline import (
    ColmapParams,
    MaskTrackingParams,
    MeshParams,
    SkipStages,
    TrainingParams,
)


# ── Enums ────────────────────────────────────────────────────────────────
class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# ── Data source ──────────────────────────────────────────────────────────
class DataSource(BaseModel):
    """Exactly *one* of these should be provided.  If neither is set the
    caller is expected to upload frames via multipart form data instead."""

    s3_uri: Optional[str] = Field(
        None,
        description="S3 URI pointing to an images directory, e.g. "
        "s3://bucket/scenes/my_scene/images/",
    )
    local_path: Optional[str] = Field(
        None,
        description="Absolute path on the server filesystem, e.g. "
        "/data/scenes/my_scene/",
    )


# ── Job start ────────────────────────────────────────────────────────────
class StartJobRequest(BaseModel):
    """JSON body for ``POST /jobs/start-recon-2dgs``.

    When using multipart upload the request params are sent as a JSON
    string in the ``params`` form field instead of the request body.
    """

    data_source: Optional[DataSource] = None
    sam_prompt: Optional[str] = Field(
        None,
        description="Comma-separated SAM text prompts, e.g. "
        "'chair, table, sofa'. If empty or omitted, uses the default "
        "furniture prompts.",
    )
    colmap: ColmapParams = Field(default_factory=ColmapParams)
    masks_tracking: MaskTrackingParams = Field(default_factory=MaskTrackingParams)
    training: TrainingParams = Field(default_factory=TrainingParams)
    mesh: MeshParams = Field(default_factory=MeshParams)
    skip: SkipStages = Field(default_factory=SkipStages)


class StartJobResponse(BaseModel):
    job_id: str
    status: JobStatus


# ── Job status ───────────────────────────────────────────────────────────
class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    current_stage: Optional[str] = None
    error_message: Optional[str] = None
    created_at: str
    updated_at: str


# ── Downloads ────────────────────────────────────────────────────────────
class DownloadUrlResponse(BaseModel):
    url: str
    expires_in_seconds: int


class MeshDownloadResponse(BaseModel):
    meshes: List[DownloadUrlResponse]


class PresignedPostResponse(BaseModel):
    """Returned by ``GET /upload-url``."""

    url: str
    fields: dict
    object_key: str


class BulkUploadRequest(BaseModel):
    """JSON body for ``POST /uploads/upload-urls/bulk``."""
    filenames: List[str] = Field(max_length=1000)


class BulkUploadResponse(BaseModel):
    """Returned by ``POST /uploads/upload-urls/bulk``."""
    urls: List[PresignedPostResponse]
    s3_uri: str
