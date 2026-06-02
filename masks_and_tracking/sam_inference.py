"""SAM 3 Inference Module

Encapsulates all interactions with the Ultralytics SAM3 model. This module
owns model loading, checkpoint resolution, and raw mask extraction — fully
decoupled from the rule-based spatial post-processing in mask_processor.py.

Usage (as a library):
    from sam_inference import SAM3TextSegmenter
    segmenter = SAM3TextSegmenter(
        checkpoint="sam3.pt",
        device="cuda",
        text_prompts=["furniture"],
    )
    masks = segmenter.predict_raw_masks(frame_bgr)
"""

import sys
import os
import logging
from pathlib import Path
from typing import Any, List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ── Ultralytics environment setup ────────────────────────────────────────────

def _resolve_ultralytics_home(requested_home: Optional[str]) -> Path:
    """Resolve and create the Ultralytics cache/checkpoint directory.

    If no explicit path is given, defaults to a `.ultralytics` folder beside
    the active Python environment root (e.g. `d:/conda_envs/GP/.ultralytics`).
    This keeps downloaded weights close to the env and avoids polluting the
    user's home directory.
    """
    if requested_home:
        home = Path(requested_home).expanduser().resolve()
    else:
        env_root = Path(sys.executable).resolve().parent.parent
        home = (env_root / ".ultralytics").resolve()
    home.mkdir(parents=True, exist_ok=True)
    os.environ["ULTRALYTICS_HOME"] = str(home)
    return home


# ── SAM3 Segmenter ───────────────────────────────────────────────────────────

class SAM3TextSegmenter:
    """Thin wrapper around Ultralytics SAM3SemanticPredictor.

    Handles model loading with FP16, text-prompted inference, and conversion
    of the Ultralytics results object into a plain list of boolean numpy masks.
    """

    def __init__(
        self,
        checkpoint: str,
        device: str,
        text_prompts: Optional[List[str]] = None,
        min_mask_area: int = 120,
        ultralytics_home: Optional[str] = None,
    ):
        self.ultralytics_home = _resolve_ultralytics_home(ultralytics_home)
        self.checkpoint = checkpoint
        self.device = device
        self.text_prompts = text_prompts or ["furniture"]
        self.min_mask_area = min_mask_area
        self.semantic_predictor = self._load_model()

    def _load_model(self):
        """Create and configure a SAM3 semantic predictor."""
        from ultralytics.models.sam import SAM3SemanticPredictor  # type: ignore

        logger.info(f"Loading Ultralytics SAM3 model: {self.checkpoint}")
        logger.info(f"Ultralytics home: {self.ultralytics_home}")

        return SAM3SemanticPredictor(
            overrides=dict(
                conf=0.25,
                task="segment",
                mode="predict",
                model=self.checkpoint,
                half=True,
                device=self.device,
                verbose=False,
                save=False,  # Prevent Ultralytics from dumping runs/segment/predict/
            )
        )

    def _extract_masks_from_results(self, results: Any) -> List[np.ndarray]:
        """Convert Ultralytics results to boolean masks with min-area filtering."""
        masks: List[np.ndarray] = []
        if results is None:
            return masks

        items = results if isinstance(results, list) else [results]
        for item in items:
            if getattr(item, "masks", None) is None:
                continue
            data = item.masks.data
            if data is None:
                continue
            arr = data.detach().cpu().numpy()
            arr_bool = arr > 0.5
            areas = arr_bool.reshape(arr_bool.shape[0], -1).sum(axis=1)
            keep_idx = np.flatnonzero(areas >= self.min_mask_area)
            for i in keep_idx:
                masks.append(arr_bool[int(i)])

        return masks

    def predict_raw_masks(
        self,
        frame_bgr: np.ndarray,
        text_prompts: Optional[List[str]] = None,
    ) -> List[np.ndarray]:
        """Generate raw SAM3 masks without any rule-based post-processing.

        Args:
            frame_bgr: Input image in BGR (OpenCV) format.
            text_prompts: Optional text prompts to override instance defaults.

        Returns:
            List of boolean numpy masks, each with shape (H, W).
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        prompts = text_prompts or self.text_prompts

        results = self.semantic_predictor(source=frame_rgb, text=prompts, verbose=False)
        return self._extract_masks_from_results(results)
