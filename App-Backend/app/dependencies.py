"""
Shared FastAPI dependencies — GPU lock and job store singletons.
"""

from __future__ import annotations

import asyncio

from app.services.job_store import JobStore

# ── Singletons (created once, shared across the app lifespan) ────────────
gpu_lock = asyncio.Lock()
job_store = JobStore()
