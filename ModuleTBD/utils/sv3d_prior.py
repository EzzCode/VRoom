import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

_SV3D_REPO = "chenguolin/sv3d-diffusers"
_hf_cache_dir = r"A:\hf_cache"

@dataclass
class HallucinatedView:
    rgb: np.ndarray        # (H, W, 3) uint8
    azimuth_deg: float     # V-frame azimuth in (-180, 180]
    elevation_deg: float   # V-frame elevation in [-90, 90]
    is_conditioning: bool = False # True for the input view, False for hallucinated views


def _setup_hf_cache(cache_dir=None):
    target = cache_dir or os.environ.get("HF_HOME") or _hf_cache_dir
    target = str(Path(target).expanduser().resolve())
    Path(target).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", target)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(target, "hub"))
    return target


def _add_sv3d_to_path():
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "temp_deps" / "sv3d-diffusers"
        if candidate.exists() and (candidate / "diffusers_sv3d").exists():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            return
    raise RuntimeError(
        "Could not find temp_deps/sv3d-diffusers/diffusers_sv3d/. "
        "Clone https://github.com/chenguolin/sv3d-diffusers into "
        "<workspace>/temp_deps/sv3d-diffusers/."
    )


def _build_orbit_poses(num_frames, cond_elevation_deg):
    """Equi-azimuth orbit at fixed elevation. Matches chenguolin/sv3d-diffusers/infer.py."""
    az_off_deg = (np.arange(1, num_frames + 1) * 360.0 / num_frames) % 360.0
    elevations_deg = np.full(num_frames, float(cond_elevation_deg))
    last = az_off_deg[-1]
    polars_rad = [float(np.deg2rad(90.0 - e)) for e in elevations_deg]
    azimuths_rad = [float(np.deg2rad((a - last) % 360.0)) for a in az_off_deg]
    return az_off_deg, polars_rad, azimuths_rad


class SV3DBackend:
    native_resolution = 576

    def __init__(self, num_frames=21, decode_chunk_size=4, dtype=torch.float16,
                 offload_strategy="sequential", safe_mode=False,
                 num_inference_steps=25, device="cuda", hf_cache_dir=None):
        if not torch.cuda.is_available() and device == "cuda":
            raise RuntimeError("SV3D requires CUDA; no GPU detected.")
        self.device = device
        self.dtype = dtype
        self._num_frames = 14 if safe_mode else num_frames
        self._resolution = 512 if safe_mode else 576
        self._decode_chunk_size = decode_chunk_size
        self._offload = offload_strategy
        self._num_inference_steps = num_inference_steps
        self._hf_cache_dir = _setup_hf_cache(hf_cache_dir)
        self._pipe = None

    def _load(self):
        if self._pipe is not None:
            return
        _add_sv3d_to_path()
        from diffusers_sv3d import SV3DUNetSpatioTemporalConditionModel, StableVideo3DDiffusionPipeline # type: ignore
        from diffusers import AutoencoderKL, EulerDiscreteScheduler
        from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

        logger.info("Loading SV3D-p from %s (dtype=%s)...", _SV3D_REPO, self.dtype)

        pipe = StableVideo3DDiffusionPipeline(
            unet=SV3DUNetSpatioTemporalConditionModel.from_pretrained(
                _SV3D_REPO, subfolder="unet", torch_dtype=self.dtype),
            vae=AutoencoderKL.from_pretrained(
                _SV3D_REPO, subfolder="vae", torch_dtype=self.dtype),
            scheduler=EulerDiscreteScheduler.from_pretrained(
                _SV3D_REPO, subfolder="scheduler"),
            image_encoder=CLIPVisionModelWithProjection.from_pretrained(
                _SV3D_REPO, subfolder="image_encoder", torch_dtype=self.dtype),
            feature_extractor=CLIPImageProcessor.from_pretrained(
                _SV3D_REPO, subfolder="feature_extractor"),
        )

        for method in (pipe.vae.enable_slicing, pipe.vae.enable_tiling):
            try:
                method()
            except Exception:
                pass

        if self._offload == "sequential":
            try:
                pipe.enable_sequential_cpu_offload()
            except Exception as e:
                logger.warning("sequential offload failed (%s); using .to(device).", e)
                pipe = pipe.to(self.device)
        elif self._offload == "model":
            try:
                pipe.enable_model_cpu_offload()
            except Exception as e:
                logger.warning("model offload failed (%s); using .to(device).", e)
                pipe = pipe.to(self.device)
        else:
            pipe = pipe.to(self.device)

        self._pipe = pipe
        logger.info("SV3D-p ready. frames=%d, res=%d, offload=%s",
                    self._num_frames, self._resolution, self._offload)

    def unload(self):
        if self._pipe is not None:
            del self._pipe
            self._pipe = None
            torch.cuda.empty_cache()

    def hallucinate(self, conditioning_rgb, cond_elevation_deg, cond_azimuth_deg=0.0, seed=None):
        """Run SV3D-p on a square uint8 RGB conditioning image. Returns list[HallucinatedView]."""
        self._load()

        if conditioning_rgb.ndim != 3 or conditioning_rgb.shape[2] != 3:
            raise ValueError("conditioning_rgb must be (H, W, 3) uint8 RGB")
        H, W = conditioning_rgb.shape[:2]
        if H != W:
            raise ValueError(f"SV3D expects a square image, got {H}x{W}")

        pil = Image.fromarray(conditioning_rgb).resize(
            (self._resolution, self._resolution), Image.LANCZOS)
        az_off_deg, polars_rad, azimuths_rad = _build_orbit_poses(
            self._num_frames, cond_elevation_deg)
        elevations_deg = np.full(self._num_frames, float(cond_elevation_deg))

        gen = torch.Generator(device="cpu").manual_seed(int(seed)) if seed is not None else None

        logger.info("SV3D-p generating %d frames (steps=%d, cond_el=%.1f°)...",
                    self._num_frames, self._num_inference_steps, cond_elevation_deg)

        autocast_dtype = torch.float16 if self.dtype == torch.float16 else torch.float32
        with torch.no_grad(), torch.autocast("cuda", dtype=autocast_dtype,
                                              enabled=(self.dtype == torch.float16)):
            out = self._pipe(
                pil,
                height=self._resolution, width=self._resolution,
                num_frames=self._num_frames,
                num_inference_steps=self._num_inference_steps,
                decode_chunk_size=self._decode_chunk_size,
                polars_rad=polars_rad,
                azimuths_rad=azimuths_rad,
                generator=gen,
            )

        views = []
        for i, frame in enumerate(out.frames[0]):
            az_abs = ((cond_azimuth_deg + float(az_off_deg[i]) + 180.0) % 360.0) - 180.0
            views.append(HallucinatedView(
                rgb=np.asarray(frame.convert("RGB")),
                azimuth_deg=float(az_abs),
                elevation_deg=float(elevations_deg[i]),
            ))
        return views
