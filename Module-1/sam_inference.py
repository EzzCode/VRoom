"""SAM 3 Inference Module

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
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

def _resolve_ultralytics_home(requested_home: Optional[str]) -> Path:
    """Resolve and create the Ultralytics cache/checkpoint directory."""
    if requested_home:
        home = Path(requested_home).expanduser().resolve()
    else:
        env_root = Path(sys.executable).resolve().parent.parent
        home = (env_root / ".ultralytics").resolve()
    home.mkdir(parents=True, exist_ok=True)
    os.environ["ULTRALYTICS_HOME"] = str(home)
    return home


class SAM3TextSegmenter:
    """Thin SAM3 semantic segmenter wrapper for text-only inference."""
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
        self.min_mask_area = int(min_mask_area)
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
                save=False,
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

    def predict_raw_masks(self, frame_bgr: np.ndarray, text_prompts: Optional[List[str]] = None) -> List[np.ndarray]:
        """Generate raw SAM3 masks for the frame without any rule-based post-processing.
        
        Args:
            frame_bgr: Input BGR frame.
            text_prompts: Optional prompts to override the defaults. 
        Returns:
            List of boolean masks.
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        prompts = text_prompts or self.text_prompts
        
        results = self.semantic_predictor(source=frame_rgb, text=prompts, verbose=False)
        return self._extract_masks_from_results(results)
