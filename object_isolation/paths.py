from __future__ import annotations

from pathlib import Path


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


def object_dir(output_root: str | Path, object_id: int) -> Path:
    return Path(output_root) / f"obj_{int(object_id)}"