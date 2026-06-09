"""
``GET /upload-url`` — generate an S3 presigned POST URL for direct upload.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.config import settings
from app.models.job import PresignedPostResponse, BulkUploadRequest, BulkUploadResponse
from app.security import require_api_key
from app.services import s3 as s3_service

router = APIRouter(prefix="/uploads", tags=["uploads"])


@router.get(
    "/upload-url",
    response_model=PresignedPostResponse,
    dependencies=[Depends(require_api_key)],
    summary="Generate an S3 presigned POST URL for client-side upload.",
)
async def get_upload_url(
    filename: str = Query(
        ...,
        description="Original filename (used only for extension validation).",
    ),
) -> PresignedPostResponse:
    if not settings.s3_enabled:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="S3 is not configured. Use multipart upload or local path instead.",
        )

    # Validate extension against allow-list
    import os

    ext = os.path.splitext(filename)[1].lower()
    if ext not in settings.allowed_image_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File extension '{ext}' not allowed. "
            f"Allowed: {settings.allowed_image_extensions}",
        )

    # Generate a safe, unique object key (never trust user-supplied filenames)
    safe_key = f"uploads/{uuid.uuid4().hex}{ext}"

    result = s3_service.generate_presigned_post(
        object_key=safe_key,
        content_type="image/jpeg" if ext in (".jpg", ".jpeg") else "image/png",
        max_size_mb=settings.max_upload_size_mb,
    )
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate presigned URL.",
        )

    return PresignedPostResponse(**result)


@router.post(
    "/upload-urls/bulk",
    response_model=BulkUploadResponse,
    dependencies=[Depends(require_api_key)],
    summary="Generate multiple S3 presigned POST URLs for client-side upload.",
)
async def get_bulk_upload_urls(
    request: BulkUploadRequest,
) -> BulkUploadResponse:
    if not settings.s3_enabled:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="S3 is not configured. Use multipart upload or local path instead.",
        )

    import os
    responses = []
    
    session_id = uuid.uuid4().hex
    
    for filename in request.filenames:
        ext = os.path.splitext(filename)[1].lower()
        if ext not in settings.allowed_image_extensions:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File extension '{ext}' in filename '{filename}' not allowed. "
                f"Allowed: {settings.allowed_image_extensions}",
            )

        safe_key = f"uploads/{session_id}/{uuid.uuid4().hex}{ext}"
        result = s3_service.generate_presigned_post(
            object_key=safe_key,
            content_type="image/jpeg" if ext in (".jpg", ".jpeg") else "image/png",
            max_size_mb=settings.max_upload_size_mb,
        )
        
        if result is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to generate presigned URL for {filename}.",
            )
            
        responses.append(PresignedPostResponse(**result))

    return BulkUploadResponse(
        urls=responses,
        s3_uri=f"s3://{settings.aws_s3_bucket}/uploads/{session_id}/"
    )
