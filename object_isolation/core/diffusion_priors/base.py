"""
Diffusion-prior backend interface.

A backend takes one square RGB conditioning image and produces a series of
novel-view RGB renders, plus a per-view azimuth/elevation in the SV3D
V-frame (azimuth 0 at +Z_V, +X_V → +90°, elevation in [-90, +90]).

Concrete backends so far:
    sv3d.SV3DBackend  — Stable Video 3D (Stability AI), unconditional variant
                        (sv3d_u): 21 frames, equispaced azimuths starting from
                        the conditioning view, fixed elevation = cond elevation.

Backends MUST NOT touch the world frame; they live entirely in V-space.
The caller (hallucination.py) is responsible for V→W pose mapping using
the coordinate-frame math.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class HallucinatedView:
    """A single novel view emitted by a diffusion-prior backend."""
    rgb: np.ndarray              # (H, W, 3) uint8, output resolution
    azimuth_V_deg: float         # SV3D V-frame azimuth, in (-180, 180]
    elevation_V_deg: float       # SV3D V-frame elevation, in [-90, 90]
    is_conditioning: bool = False  # True for the regen of input frame (i=0 in sv3d_u)


class DiffusionPriorBackend(ABC):
    """Abstract base class for any 3D-aware image-to-multiview prior."""

    name: str = "abstract"
    native_resolution: int = 576  # square HxH the backend was trained at

    @property
    @abstractmethod
    def output_count(self) -> int:
        """Number of frames produced per call (e.g. 21 for sv3d_u)."""

    @abstractmethod
    def hallucinate(
        self,
        conditioning_rgb_uint8: np.ndarray,
        cond_elevation_deg: float,
        cond_azimuth_deg: float = 0.0,
        seed: Optional[int] = None,
    ) -> List[HallucinatedView]:
        """
        Run the diffusion prior on a single square conditioning image.

        Args
        ----
        conditioning_rgb_uint8: (H, H, 3) uint8 RGB. The backend may resize.
                                Should already be center-padded onto a neutral
                                background — the prior typically expects the
                                object to fill ~70% of the frame.
        cond_elevation_deg: V-frame elevation of the conditioning view.
                            sv3d_u keeps elevation constant; sv3d_p uses this
                            as the per-view elevation if no override.
        cond_azimuth_deg: V-frame azimuth of the conditioning view. Output
                          azimuths are offset by this (so frame 0 → cond view).
        seed: RNG seed for deterministic generation (optional).

        Returns
        -------
        list[HallucinatedView] of length `self.output_count`.
        """

    def unload(self) -> None:
        """Release GPU memory. Default is no-op; override if needed."""
