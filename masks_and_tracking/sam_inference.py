import os
import logging
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ── Ultralytics environment setup ────────────────────────────────────────────

def _resolve_ultralytics_home(requested_home: Optional[str]) -> Path:
    """Resolve and create the Ultralytics cache directory."""
    if requested_home:
        home = Path(requested_home).expanduser().resolve()
    else:
        # parent.parent of the executable is the conda env root
        import sys
        home = (Path(sys.executable).resolve().parent.parent / ".ultralytics").resolve()
    home.mkdir(parents=True, exist_ok=True)
    os.environ["ULTRALYTICS_HOME"] = str(home)  # must be set before any ultralytics import
    return home


# ── SAM3 Segmenter ───────────────────────────────────────────────────────────

class SAM3TextSegmenter:
    """Thin wrapper around Ultralytics SAM3SemanticPredictor."""

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
        self.predictor = self._load_model()

    def _load_model(self):
        """Build and return a SAM3 semantic predictor."""
        from ultralytics.models.sam import SAM3SemanticPredictor  # type: ignore

        logger.info("Loading SAM3 model: %s (home=%s)", self.checkpoint, self.ultralytics_home)
        return SAM3SemanticPredictor(
            overrides=dict(
                conf=0.25,
                task="segment",
                mode="predict",
                model=self.checkpoint,
                half=True,       # FP16 for speed
                device=self.device,
                verbose=False,
                save=False,      # don't write runs/segment/predict/
            )
        )

    def _extract_masks(self, results) -> List[np.ndarray]:
        """Convert Ultralytics results to boolean masks, filtering by min area."""
        masks: List[np.ndarray] = []
        if results is None:
            return masks
        for item in (results if isinstance(results, list) else [results]):
            if getattr(item, "masks", None) is None or item.masks.data is None:
                continue
            arr = item.masks.data.detach().cpu().numpy()
            bool_masks = arr > 0.5
            areas = bool_masks.reshape(bool_masks.shape[0], -1).sum(axis=1)
            for i in np.flatnonzero(areas >= self.min_mask_area):
                masks.append(bool_masks[int(i)])
        return masks

    def predict_raw_masks(
        self,
        frame_bgr: np.ndarray,
        text_prompts: Optional[List[str]] = None,
    ) -> List[np.ndarray]:
        """Run SAM3 and return raw boolean masks (H, W) without post-processing."""
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)  # SAM3 expects RGB
        prompts = text_prompts or self.text_prompts
        results = self.predictor(source=frame_rgb, text=prompts, verbose=False)
        return self._extract_masks(results)
