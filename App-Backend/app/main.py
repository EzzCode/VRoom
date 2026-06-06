"""
FastAPI application factory.

- Lifespan context manager for startup/shutdown.
- Security headers middleware.
- Strict CORS (no wildcard origins).
- All routers mounted under ``/api/v1/``.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.routers import downloads, jobs, uploads

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


# ── Lifespan ─────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[no-untyped-def]
    logger.info("VRoom Backend starting up.")
    logger.info("  Project root : %s", settings.vroom_project_root)
    logger.info("  Jobs data dir: %s", settings.jobs_data_dir)
    logger.info("  S3 enabled   : %s", settings.s3_enabled)
    logger.info("  GPU queue cap : %d", settings.max_queued_jobs)
    yield
    logger.info("VRoom Backend shutting down.")


# ── App ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="VRoom 2DGS Pipeline API",
    description="Production backend for the VRoom 2D Gaussian Splatting "
    "reconstruction and semantic meshing pipeline.",
    version="1.0.0",
    lifespan=lifespan,
)


# ── CORS — strict origin list, no wildcard ───────────────────────────────
# TODO(security): Replace with actual production origins before deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)


# ── Security headers middleware ──────────────────────────────────────────
@app.middleware("http")
async def security_headers(request: Request, call_next):  # type: ignore[no-untyped-def]
    response: Response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Content-Security-Policy"] = "default-src 'none'; frame-ancestors 'none'"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    return response


# ── Mount routers ────────────────────────────────────────────────────────
app.include_router(uploads.router, prefix="/api/v1")
app.include_router(jobs.router, prefix="/api/v1")
app.include_router(downloads.router, prefix="/api/v1")


# ── Health check ─────────────────────────────────────────────────────────
@app.get("/health", tags=["health"])
async def health_check() -> dict:
    return {"status": "ok"}
