"""
Application configuration loaded exclusively from environment variables.

Put your S3 credentials and API key in a `.env` file or export them:

    export VROOM_API_KEY="your-secret-api-key"
    export AWS_S3_BUCKET="vroom-pipeline-outputs"
    export AWS_REGION="us-east-1"
    export AWS_ACCESS_KEY_ID="AKIA..."
    export AWS_SECRET_ACCESS_KEY="..."
"""

from __future__ import annotations

import logging
import os
import secrets
from pathlib import Path
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """All settings are resolved from environment variables.  No secrets are
    ever hard-coded.  For local development without S3, leave the AWS fields
    blank and the app will fall back to serving files from disk."""

    # ── Project paths ────────────────────────────────────────────────────
    vroom_project_root: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent.parent.parent,
        description="Absolute path to the VRoom workspace root.",
    )
    jobs_data_dir: Path = Field(
        default=Path(""),
        description="Where job working directories are created. "
        "Defaults to {vroom_project_root}/jobs_data.",
    )

    # ── AWS / S3 ─────────────────────────────────────────────────────────
    aws_s3_bucket: str = Field(default="", description="S3 bucket name.")
    aws_region: str = Field(default="us-east-1")
    aws_access_key_id: str = Field(default="")
    aws_secret_access_key: str = Field(default="")
    presigned_url_expiry_seconds: int = Field(
        default=3600, description="TTL for presigned S3 URLs."
    )

    # ── Conda environments ───────────────────────────────────────────────
    pipeline_conda_env: str = Field(
        default="pipeline",
        description="Conda env for SfM, Training, and Mesh Generation.",
    )
    masks_conda_env: str = Field(
        default="masks",
        description="Conda env for Masks & Tracking.",
    )

    # ── Auth ─────────────────────────────────────────────────────────────
    vroom_api_key: str = Field(default="", description="Bearer token for API auth.")

    # ── Upload limits ────────────────────────────────────────────────────
    max_upload_size_mb: int = Field(default=2048)
    allowed_image_extensions: List[str] = Field(
        default=[".jpg", ".jpeg", ".png"],
    )

    # ── Queue ────────────────────────────────────────────────────────────
    max_queued_jobs: int = Field(
        default=2,
        description="Maximum number of jobs allowed (1 running + 1 queued).",
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    def model_post_init(self, __context: object) -> None:
        # Default jobs_data_dir to {project_root}/jobs_data
        if not self.jobs_data_dir or str(self.jobs_data_dir) == ".":
            self.jobs_data_dir = self.vroom_project_root / "jobs_data"
        self.jobs_data_dir.mkdir(parents=True, exist_ok=True)

        # Warn if no API key is set and generate an ephemeral one
        if not self.vroom_api_key:
            self.vroom_api_key = secrets.token_hex(32)
            logger.warning(
                "VROOM_API_KEY not set. Generated ephemeral key: %s  "
                "This key is instance-isolated and will change on restart.",
                self.vroom_api_key,
            )
            
        if os.environ.get("RUNNING_IN_MODAL") == "1":
            self.pipeline_conda_env = "pipeline"
            self.masks_conda_env = "masks"

    @property
    def s3_enabled(self) -> bool:
        """Return True only if all required S3 config is present."""
        return bool(
            self.aws_s3_bucket
            and self.aws_access_key_id
            and self.aws_secret_access_key
        )


settings = Settings()
