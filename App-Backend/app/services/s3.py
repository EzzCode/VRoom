"""
S3 helper utilities with automatic local-filesystem fallback.

When ``settings.s3_enabled`` is ``False``, presigned URLs are replaced
by local file paths, and upload/download functions operate on disk.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError

from app.config import settings

logger = logging.getLogger(__name__)

# ── S3 client singleton ─────────────────────────────────────────────────
_s3_client = None


def _get_s3_client():  # type: ignore[no-untyped-def]
    global _s3_client
    if _s3_client is None and settings.s3_enabled:
        _s3_client = boto3.client(
            "s3",
            region_name=settings.aws_region,
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
            config=BotoConfig(signature_version="s3v4"),
        )
    return _s3_client


# ── Presigned POST (for client-side uploads) ─────────────────────────────
def generate_presigned_post(
    object_key: str,
    content_type: str = "image/jpeg",
    max_size_mb: int = 50,
) -> Optional[Dict[str, Any]]:
    """Return ``{url, fields, object_key}`` or ``None`` if S3 is disabled."""
    client = _get_s3_client()
    if client is None:
        return None
    try:
        response = client.generate_presigned_post(
            Bucket=settings.aws_s3_bucket,
            Key=object_key,
            Fields={"Content-Type": content_type},
            Conditions=[
                {"Content-Type": content_type},
                ["content-length-range", 1, max_size_mb * 1024 * 1024],
            ],
            ExpiresIn=settings.presigned_url_expiry_seconds,
        )
        return {
            "url": response["url"],
            "fields": response["fields"],
            "object_key": object_key,
        }
    except ClientError:
        logger.exception("Failed to generate presigned POST for %s", object_key)
        return None


# ── Presigned GET (for client-side downloads) ────────────────────────────
def generate_presigned_get(object_key: str) -> Optional[str]:
    """Return a presigned GET URL or ``None`` if S3 is disabled."""
    client = _get_s3_client()
    if client is None:
        return None
    try:
        return client.generate_presigned_url(
            "get_object",
            Params={"Bucket": settings.aws_s3_bucket, "Key": object_key},
            ExpiresIn=settings.presigned_url_expiry_seconds,
        )
    except ClientError:
        logger.exception("Failed to generate presigned GET for %s", object_key)
        return None


# ── Upload local file to S3 ─────────────────────────────────────────────
def upload_file_to_s3(local_path: Path, object_key: str) -> bool:
    """Upload a single file.  Returns ``True`` on success."""
    client = _get_s3_client()
    if client is None:
        logger.info("S3 disabled — skipping upload of %s", local_path)
        return False
    try:
        client.upload_file(str(local_path), settings.aws_s3_bucket, object_key)
        return True
    except ClientError:
        logger.exception("Failed to upload %s to s3://%s/%s", local_path, settings.aws_s3_bucket, object_key)
        return False


def upload_directory_to_s3(local_dir: Path, s3_prefix: str) -> int:
    """Recursively upload a directory.  Returns count of files uploaded."""
    count = 0
    for root, _dirs, files in os.walk(local_dir):
        for filename in files:
            file_path = Path(root) / filename
            relative = file_path.relative_to(local_dir)
            object_key = f"{s3_prefix}/{relative.as_posix()}"
            if upload_file_to_s3(file_path, object_key):
                count += 1
    return count


# ── Download S3 prefix to local directory ────────────────────────────────
def download_directory_from_s3(s3_prefix: str, local_dir: Path) -> int:
    """Download all objects under *s3_prefix*.  Returns file count."""
    client = _get_s3_client()
    if client is None:
        logger.info("S3 disabled — skipping download of %s", s3_prefix)
        return 0
    count = 0
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=settings.aws_s3_bucket, Prefix=s3_prefix):
        for obj in page.get("Contents", []):
            key: str = obj["Key"]
            relative = key[len(s3_prefix) :].lstrip("/")
            if not relative:
                continue
            dest = local_dir / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            client.download_file(settings.aws_s3_bucket, key, str(dest))
            count += 1
    return count


# ── List S3 keys under a prefix ─────────────────────────────────────────
def list_s3_keys(prefix: str, suffix: str = "") -> List[str]:
    """Return all object keys matching *prefix* and optionally *suffix*."""
    client = _get_s3_client()
    if client is None:
        return []
    keys: List[str] = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=settings.aws_s3_bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key: str = obj["Key"]
            if suffix and not key.endswith(suffix):
                continue
            keys.append(key)
    return keys
