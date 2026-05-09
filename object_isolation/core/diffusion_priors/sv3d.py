"""
Stable Video 3D backend (sv3d_p variant, pose-conditioned).

Why sv3d_p over sv3d_u
----------------------
* sv3d_u was trained on circular orbits at 0° elevation only.
* sv3d_p is trained with per-frame (polar, azimuth) conditioning, so the same
  weights can render orbits at arbitrary elevations and (in principle)
  arbitrary per-frame poses. We need this because our conditioning frame is
  rarely at exactly 0° elevation in the V (SV3D) frame.

Why we vendor `chenguolin/sv3d-diffusers`
-----------------------------------------
The official Stability checkpoint at `stabilityai/sv3d` ships only raw
`.safetensors` files (no `model_index.json`), and stock diffusers
`StableVideoDiffusionPipeline` does not expose `polars_rad` / `azimuths_rad`.

`chenguolin/sv3d-diffusers` is a community port that re-packages SV3D-p in
diffusers convention and provides `SV3DUNetSpatioTemporalConditionModel` +
`StableVideo3DDiffusionPipeline` with proper per-frame pose inputs.

Expected to be cloned at:  `<workspace>/temp_deps/sv3d-diffusers/`
(auto-injected into `sys.path` below).

Low-VRAM (≈8 GB)
----------------
* fp16 weights, sequential CPU offload, VAE slicing+tiling,
  `decode_chunk_size=1`. `safe_mode=True` drops to 14 frames @ 512².
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch
from PIL import Image

from .base import DiffusionPriorBackend, HallucinatedView

logger = logging.getLogger(__name__)




def _ensure_hf_cache_env(cache_dir: Optional[str] = None) -> str:
    target = cache_dir or os.environ.get("HF_HOME") or r"A:\\hf_cache"
    target = str(Path(target).expanduser().resolve())
    Path(target).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", target)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(target, "hub"))
    return target


def _ensure_vendored_on_path() -> Path:
    """Locate `temp_deps/sv3d-diffusers` and prepend to sys.path."""
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        cand = parent / "temp_deps" / "sv3d-diffusers"
        if cand.exists() and (cand / "diffusers_sv3d").exists():
            if str(cand) not in sys.path:
                sys.path.insert(0, str(cand))
            return cand
    raise RuntimeError(
        "Could not locate `temp_deps/sv3d-diffusers/diffusers_sv3d/`. "
        "Clone https://github.com/chenguolin/sv3d-diffusers into "
        "`<workspace>/temp_deps/sv3d-diffusers/`."
    )


_SV3D_DIFFUSERS_REPO = "chenguolin/sv3d-diffusers"


class SV3DBackend(DiffusionPriorBackend):
    """Stable Video 3D-p backend driven by chenguolin's diffusers port."""

    name = "sv3d_p"
    native_resolution = 576

    def __init__(
        self,
        num_frames: int = 21,
        decode_chunk_size: int = 4,
        dtype: torch.dtype = torch.float16,
        offload_strategy: str = "sequential",  # "sequential" | "model" | "none"
        safe_mode: bool = False,
        num_inference_steps: int = 25,
        device: str = "cuda",
        hf_cache_dir: Optional[str] = None,
    ):
        if not torch.cuda.is_available() and device == "cuda":
            raise RuntimeError("SV3D requires CUDA; no GPU detected.")
        self.device = device
        self.dtype = dtype
        self._num_frames = 14 if safe_mode else num_frames
        self._resolution = 512 if safe_mode else 576
        self._decode_chunk_size = decode_chunk_size
        self._offload = offload_strategy
        self._num_inference_steps = num_inference_steps
        self._hf_cache_dir = _ensure_hf_cache_env(hf_cache_dir)
        self._pipe = None  # lazy-load

    @property
    def output_count(self) -> int:
        return self._num_frames

    # ────────────────────────────────────────────────────────────────────
    def _ensure_loaded(self):
        if self._pipe is not None:
            return

        _ensure_vendored_on_path()
        from diffusers_sv3d import (
            SV3DUNetSpatioTemporalConditionModel,
            StableVideo3DDiffusionPipeline,
        )
        from diffusers import AutoencoderKL, EulerDiscreteScheduler
        from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

        logger.info(
            "Loading SV3D-p (%s, dtype=%s, cache=%s)...",
            _SV3D_DIFFUSERS_REPO, self.dtype, self._hf_cache_dir,
        )

        unet = SV3DUNetSpatioTemporalConditionModel.from_pretrained(
            _SV3D_DIFFUSERS_REPO, subfolder="unet", torch_dtype=self.dtype,
        )
        vae = AutoencoderKL.from_pretrained(
            _SV3D_DIFFUSERS_REPO, subfolder="vae", torch_dtype=self.dtype,
        )
        scheduler = EulerDiscreteScheduler.from_pretrained(
            _SV3D_DIFFUSERS_REPO, subfolder="scheduler",
        )
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            _SV3D_DIFFUSERS_REPO, subfolder="image_encoder", torch_dtype=self.dtype,
        )
        feature_extractor = CLIPImageProcessor.from_pretrained(
            _SV3D_DIFFUSERS_REPO, subfolder="feature_extractor",
        )

        pipe = StableVideo3DDiffusionPipeline(
            image_encoder=image_encoder,
            feature_extractor=feature_extractor,
            unet=unet,
            vae=vae,
            scheduler=scheduler,
        )

        try:
            pipe.vae.enable_slicing()
        except Exception:
            pass
        try:
            pipe.vae.enable_tiling()
        except Exception:
            pass

        if self._offload == "sequential":
            try:
                pipe.enable_sequential_cpu_offload()
            except Exception as e:
                logger.warning("sequential offload failed (%s); falling back to .to(device).", e)
                pipe = pipe.to(self.device)
        elif self._offload == "model":
            try:
                pipe.enable_model_cpu_offload()
            except Exception as e:
                logger.warning("model offload failed (%s); falling back to .to(device).", e)
                pipe = pipe.to(self.device)
        else:
            pipe = pipe.to(self.device)

        self._pipe = pipe
        logger.info(
            "SV3D-p loaded. num_frames=%d, res=%d, offload=%s",
            self._num_frames, self._resolution, self._offload,
        )

    def unload(self):
        if self._pipe is not None:
            del self._pipe
            self._pipe = None
            torch.cuda.empty_cache()

    # ────────────────────────────────────────────────────────────────────
    @staticmethod
    def _build_orbit_poses(num_frames: int, cond_elevation_deg: float):
        """Default orbit: equi-azimuth at fixed conditioning elevation.

        Layout matches `chenguolin/sv3d-diffusers/infer.py`:
            azimuths_offset_deg = (i+1) * 360/n  for i in 0..n-1, last=0
            polars_rad          = 90° - elevation, constant per frame.
        """
        az_off_deg = (np.arange(1, num_frames + 1) * 360.0 / num_frames) % 360.0
        elevations_deg = np.full(num_frames, float(cond_elevation_deg))
        polars_rad = [float(np.deg2rad(90.0 - e)) for e in elevations_deg]
        last = az_off_deg[-1]
        azimuths_rad = [float(np.deg2rad((a - last) % 360.0)) for a in az_off_deg]
        return az_off_deg, polars_rad, azimuths_rad

    # ────────────────────────────────────────────────────────────────────
    def hallucinate(
        self,
        conditioning_rgb_uint8: np.ndarray,
        cond_elevation_deg: float,
        cond_azimuth_deg: float = 0.0,
        seed: Optional[int] = None,
        target_azimuths_offset_deg: Optional[Sequence[float]] = None,
        target_elevations_deg: Optional[Sequence[float]] = None,
    ) -> List[HallucinatedView]:
        self._ensure_loaded()

        if conditioning_rgb_uint8.ndim != 3 or conditioning_rgb_uint8.shape[2] != 3:
            raise ValueError("conditioning_rgb_uint8 must be (H,W,3) uint8 RGB")
        H, W = conditioning_rgb_uint8.shape[:2]
        if H != W:
            raise ValueError(f"SV3D expects a square conditioning image, got {H}×{W}")
        pil = Image.fromarray(conditioning_rgb_uint8).resize(
            (self._resolution, self._resolution), Image.LANCZOS
        )

        if target_azimuths_offset_deg is not None and target_elevations_deg is not None:
            az_off_deg = np.asarray(target_azimuths_offset_deg, dtype=np.float64) % 360.0
            elevations_deg = np.asarray(target_elevations_deg, dtype=np.float64)
            if len(az_off_deg) != self._num_frames or len(elevations_deg) != self._num_frames:
                raise ValueError(
                    f"target_*_deg must have length num_frames={self._num_frames}"
                )
            polars_rad = [float(np.deg2rad(90.0 - e)) for e in elevations_deg]
            last = az_off_deg[-1]
            azimuths_rad = [float(np.deg2rad((a - last) % 360.0)) for a in az_off_deg]
        else:
            az_off_deg, polars_rad, azimuths_rad = self._build_orbit_poses(
                self._num_frames, cond_elevation_deg
            )
            elevations_deg = np.full(self._num_frames, float(cond_elevation_deg))

        gen = None
        if seed is not None:
            gen = torch.Generator(device="cpu").manual_seed(int(seed))

        logger.info(
            "SV3D-p generating %d frames (steps=%d, chunk=%d, cond_el=%.2f°)...",
            self._num_frames, self._num_inference_steps,
            self._decode_chunk_size, cond_elevation_deg,
        )

        autocast_dtype = torch.float16 if self.dtype == torch.float16 else torch.float32
        with torch.no_grad(), torch.autocast(
            "cuda", dtype=autocast_dtype, enabled=(self.dtype == torch.float16)
        ):
            out = self._pipe(
                pil,
                height=self._resolution,
                width=self._resolution,
                num_frames=self._num_frames,
                num_inference_steps=self._num_inference_steps,
                decode_chunk_size=self._decode_chunk_size,
                polars_rad=polars_rad,
                azimuths_rad=azimuths_rad,
                generator=gen,
            )
        frames_pil = out.frames[0]
        if len(frames_pil) != self._num_frames:
            logger.warning(
                "SV3D-p returned %d frames (expected %d).",
                len(frames_pil), self._num_frames,
            )

        views: List[HallucinatedView] = []
        for i, pim in enumerate(frames_pil):
            # SV3D's orbit and our V-frame both use the same convention:
            # az=0 at +Z, increasing CW from above (toward +X at 90°).
            # Verified numerically: 21/21 manifest azimuths match without
            # negation; with negation only 1/21 matched (the conditioning
            # frame at 360°≡0°).
            az_abs = cond_azimuth_deg + float(az_off_deg[i])
            az_abs = ((az_abs + 180.0) % 360.0) - 180.0
            rgb = np.asarray(pim.convert("RGB"))
            views.append(HallucinatedView(
                rgb=rgb,
                azimuth_V_deg=float(az_abs),
                elevation_V_deg=float(elevations_deg[i]),
                is_conditioning=False,
            ))
        return views
