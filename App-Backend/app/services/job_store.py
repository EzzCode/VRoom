"""
Thread-safe, in-memory job store.

TODO(scaling): Replace with Redis or PostgreSQL for horizontal scaling.
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional
from uuid import uuid4

from app.models.job import JobStatus, StartJobRequest


class JobEntry:
    """Mutable record for a single pipeline job."""

    __slots__ = (
        "job_id",
        "status",
        "current_stage",
        "error_message",
        "created_at",
        "updated_at",
        "work_dir",
        "params",
    )

    def __init__(
        self,
        job_id: str,
        work_dir: Path,
        params: StartJobRequest,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self.job_id = job_id
        self.status = JobStatus.PENDING
        self.current_stage: Optional[str] = None
        self.error_message: Optional[str] = None
        self.created_at = now
        self.updated_at = now
        self.work_dir = work_dir
        self.params = params


class JobStore:
    """Simple dict-backed store guarded by a threading lock."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: Dict[str, JobEntry] = {}
        self._modal_dict = None

    def _get_dict(self):
        import os
        if os.environ.get("RUNNING_IN_MODAL") == "1":
            if self._modal_dict is None:
                import modal
                self._modal_dict = modal.Dict.from_name("vroom-jobs-dict", create_if_missing=True)
            return self._modal_dict
        return None

    # ── CRUD ─────────────────────────────────────────────────────────────
    def create_job(self, work_dir: Path, params: StartJobRequest, job_id: Optional[str] = None) -> JobEntry:
        if job_id is None:
            job_id = uuid4().hex[:12]
        entry = JobEntry(job_id=job_id, work_dir=work_dir, params=params)
        with self._lock:
            d = self._get_dict()
            if d is not None:
                d[job_id] = entry
            else:
                self._jobs[job_id] = entry
        return entry

    def get_job(self, job_id: str) -> Optional[JobEntry]:
        with self._lock:
            d = self._get_dict()
            if d is not None:
                return d.get(job_id)
            return self._jobs.get(job_id)

    def update_job(
        self,
        job_id: str,
        *,
        status: Optional[JobStatus] = None,
        current_stage: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> None:
        with self._lock:
            d = self._get_dict()
            entry = d.get(job_id) if d is not None else self._jobs.get(job_id)
            if entry is None:
                return
            if status is not None:
                entry.status = status
            if current_stage is not None:
                entry.current_stage = current_stage
            if error_message is not None:
                entry.error_message = error_message
            entry.updated_at = datetime.now(timezone.utc).isoformat()
            
            if d is not None:
                d[job_id] = entry
            else:
                self._jobs[job_id] = entry

    # ── Queue depth ──────────────────────────────────────────────────────
    def active_or_pending_count(self) -> int:
        """Return the number of jobs that are PENDING or PROCESSING."""
        with self._lock:
            d = self._get_dict()
            jobs = d.values() if d is not None else self._jobs.values()
            return sum(
                1
                for j in jobs
                if j.status in (JobStatus.PENDING, JobStatus.PROCESSING)
            )
