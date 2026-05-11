"""Canonical filesystem layout for object-isolation outputs.

This module is the single source of truth for the directory structure produced
under ``<output_root>/obj_<id>/``. All pipeline stages and debug tools import
these names so adding/renaming an artifact only happens in one place.

Layout::

    <output_root>/obj_<id>/
        00_scope_debug/                — object scope discovery debug
        01_extraction/                 — extracted RGB + masks
        01_extraction_debug/           — triptychs, contact sheet
        02_frame_scoring/              — conditioning-view scores
        02_frame_scoring_debug/        — bar/scatter/strip plots
        03_novel_views/                — SV3D hallucinations
        03_novel_views_debug/          — IoU strip, coverage overlay
        04_supervision_manifest.json   — train/val supervision plan
        04_supervision_audit/          — per-frame supervision audit
        04_supervision_debug/          — supervision debug renders
        05_training_summary.json       — training stage outputs
        05_renders/                    — train-time renders
        06_model/                      — final per-object Gaussian model
        07_scene/                      — scene snapshot used for training
        99_pipeline_summary.json       — per-object end-to-end summary
"""

from __future__ import annotations

from pathlib import Path


# ── Per-stage directory / file names ─────────────────────────────────────────

SCOPE_DEBUG_DIR = "00_scope_debug"

EXTRACTION_DIR = "01_extraction"
EXTRACTION_DEBUG_DIR = "01_extraction_debug"

FRAME_SCORING_DIR = "02_frame_scoring"
FRAME_SCORING_DEBUG_DIR = "02_frame_scoring_debug"

NOVEL_VIEWS_DIR = "03_novel_views"
NOVEL_VIEWS_DEBUG_DIR = "03_novel_views_debug"

SUPERVISION_MANIFEST_FILE = "04_supervision_manifest.json"
SUPERVISION_AUDIT_DIR = "04_supervision_audit"
SUPERVISION_DEBUG_DIR = "04_supervision_debug"

TRAINING_SUMMARY_FILE = "05_training_summary.json"
RENDERS_DIR = "05_renders"
MODEL_DIR = "06_model"
SCENE_DIR = "07_scene"

OBJECT_SUMMARY_FILE = "99_pipeline_summary.json"
BATCH_SUMMARY_FILE = "99_batch_summary.json"


# ── Helpers ──────────────────────────────────────────────────────────────────

def object_dir(output_root: str | Path, object_id: int) -> Path:
    """Return the canonical ``<output_root>/obj_<id>`` directory path."""
    return Path(output_root) / f"obj_{int(object_id)}"