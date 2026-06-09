"""
Pydantic models for every parameter accepted by ``full_pipeline_runner.py``.

The default values are kept **exactly** in sync with the argparse definitions
in the runner and ``tracker_defaults.py`` so that a bare ``{}`` request body
produces the same behaviour as running the CLI with no flags.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


# ── COLMAP / SfM ─────────────────────────────────────────────────────────
class ColmapParams(BaseModel):
    camera_model: Literal["PINHOLE", "OPENCV", "SIMPLE_RADIAL", "RADIAL"] = "OPENCV"
    matcher_type: Literal["exhaustive", "sequential", "spatial"] = "sequential"
    force_colmap: bool = False


# ── Masks, Tracking & Voting ─────────────────────────────────────────────
class MaskTrackingParams(BaseModel):
    # SAM / mask params
    text_prompts: List[str] = Field(
        default=["chair", "table", "sofa", "bed", "desk", "cabinet"],
    )
    min_mask_area: int = Field(120, ge=1)
    max_area_ratio: float = Field(0.50, ge=0.0, le=1.0)
    border_threshold: float = Field(0.35, ge=0.0, le=1.0)
    merge_thresh: float = Field(0.78, ge=0.0, le=1.0)
    proximity_gap: int = Field(20, ge=0)
    proximity_color_thresh: float = Field(0.32, ge=0.0, le=1.0)
    no_split_disconnected: bool = False

    # Tracker weights (from tracker_defaults.py)
    iou_w: float = 0.75
    color_w: float = 0.25
    texture_w: float = 0.15
    bbox_w: float = 0.20
    match_threshold: float = 0.70
    patience: int = 28
    smoothing_factor: float = 0.40
    reid_threshold: float = 0.60
    disable_motion_comp: bool = False
    consensus_window: int = 8
    consensus_tie_margin: float = 0.05
    use_opencv: bool = False

    # Voting
    min_points: int = Field(10, ge=1)
    disable_alias_merge: bool = False
    alias_iou_thresh: float = Field(0.40, ge=0.0, le=1.0)
    alias_min_covisibility: int = Field(15, ge=1)


# ── Gaussian Splatting Training ──────────────────────────────────────────
class TrainingParams(BaseModel):
    num_iterations: Optional[int] = Field(None, ge=100)
    small_run: bool = False


# ── Mesh Generation ──────────────────────────────────────────────────────
class MeshParams(BaseModel):
    """Defaults match ``extract_object_meshes.py`` argparse."""
    pass


# ── Skip Stages ──────────────────────────────────────────────────────────
class SkipStages(BaseModel):
    skip_colmap: bool = False
    skip_masks: bool = False
    skip_tracking: bool = False
    skip_voting: bool = False
    skip_training: bool = False
    skip_mesh_gen: bool = False
