"""
Visual debug for Phase 5 (SV3D hallucination).

Outputs:
    conditioning_panel.png   — input image + mapped V-pose info
    sv3d_grid.png            — all SV3D outputs in a grid, az labelled
    iou_strip.png            — per-frame side-by-side: SV3D | ObjectGS render | IoU
    coverage_overlay.png     — polar plot of real vs hallucinated azimuths (V-frame)
    summary.json
"""
from __future__ import annotations

import json
import logging
import math
import sys
from pathlib import Path
from typing import List

import cv2
import numpy as np

_VROOM_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_VROOM_ROOT) not in sys.path:
    sys.path.insert(0, str(_VROOM_ROOT))

from object_isolation.core.scope import discover_object_scope
from object_isolation.core.hallucination import run_hallucination
from object_isolation.core.diffusion_priors.sv3d import SV3DBackend

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _putlbl(img, text, org, fg=(255, 255, 255), bg=(0, 0, 0), scale=0.5, thick=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, bg, thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, fg, thick, cv2.LINE_AA)


def _imread(p):
    img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    return img


def _to_bgr(img):
    if img is None:
        return None
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


# ─────────────────────────────────────────────────────────────────────────────
# Panel 1: conditioning view
# ─────────────────────────────────────────────────────────────────────────────

def make_conditioning_panel(manifest: dict, debug_dir: Path):
    cond = manifest["conditioning"]
    img = _imread(cond["image_path"])
    if img is None:
        return None
    img = _to_bgr(img)
    H, W = img.shape[:2]
    canvas = np.full((H + 80, W, 3), 245, dtype=np.uint8)
    canvas[80:, :] = img
    _putlbl(canvas, "Conditioning view (SV3D input)", (10, 26),
            fg=(20, 20, 20), bg=(255, 255, 255), scale=0.6)
    _putlbl(canvas, f"cam={cond['cam_index']}  img={cond['img_name']}", (10, 50),
            fg=(60, 60, 60), bg=(255, 255, 255), scale=0.5)
    _putlbl(canvas, f"az_V={cond['azimuth_V_deg']:.1f}  el_V={cond['elevation_V_deg']:.1f}  score={cond['score']:.3f}",
            (10, 72), fg=(60, 60, 60), bg=(255, 255, 255), scale=0.5)
    out = debug_dir / "conditioning_panel.png"
    cv2.imwrite(str(out), canvas)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Panel 2: SV3D output grid
# ─────────────────────────────────────────────────────────────────────────────

def make_sv3d_grid(manifest: dict, debug_dir: Path, cols: int = 7, tile: int = 220):
    frames = manifest["frames"]
    if not frames:
        return None
    rows = (len(frames) + cols - 1) // cols
    canvas = np.full((rows * (tile + 30) + 40, cols * tile, 3), 240, dtype=np.uint8)
    _putlbl(canvas, "SV3D outputs (azimuth labelled, V-frame)", (10, 28),
            fg=(20, 20, 20), bg=(255, 255, 255), scale=0.6)

    for k, fr in enumerate(frames):
        r, c = divmod(k, cols)
        x0 = c * tile
        y0 = 40 + r * (tile + 30)
        img = _to_bgr(_imread(fr["sv3d_raw_path"]))
        if img is not None:
            img = cv2.resize(img, (tile, tile), interpolation=cv2.INTER_AREA)
            canvas[y0:y0 + tile, x0:x0 + tile] = img
        # Border green=accepted, red=rejected.
        col = (60, 180, 60) if fr["accepted"] else (60, 60, 200)
        cv2.rectangle(canvas, (x0 + 1, y0 + 1), (x0 + tile - 2, y0 + tile - 2), col, 2)
        # Label.
        lbl_y = y0 + tile + 22
        _putlbl(canvas, f"#{fr['index']:02d} az={fr['azimuth_V_deg']:+.0f}",
                (x0 + 6, lbl_y), fg=(20, 20, 20), bg=(255, 255, 255), scale=0.45)
        _putlbl(canvas, f"IoU={fr['iou_with_objgs']:.2f}",
                (x0 + tile - 90, lbl_y), fg=col, bg=(255, 255, 255), scale=0.45)

    out = debug_dir / "sv3d_grid.png"
    cv2.imwrite(str(out), canvas)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Panel 3: IoU strip (SV3D | ObjectGS | overlay)
# ─────────────────────────────────────────────────────────────────────────────

def make_iou_strip(manifest: dict, debug_dir: Path, tile: int = 200, max_rows: int = 21):
    frames = manifest["frames"][:max_rows]
    if not frames:
        return None
    cols = 3
    row_h = tile + 36
    canvas = np.full((row_h * len(frames) + 40, cols * tile + 200, 3), 245, dtype=np.uint8)
    _putlbl(canvas, "Per-frame: SV3D | ObjectGS render | overlap", (10, 28),
            fg=(20, 20, 20), bg=(255, 255, 255), scale=0.55)

    for r, fr in enumerate(frames):
        y0 = 40 + r * row_h
        sv3d = _to_bgr(_imread(fr["sv3d_raw_path"]))
        ref = _to_bgr(_imread(fr["objgs_ref_path"]))
        if sv3d is None or ref is None:
            continue
        sv3d = cv2.resize(sv3d, (tile, tile), interpolation=cv2.INTER_AREA)
        ref = cv2.resize(ref, (tile, tile), interpolation=cv2.INTER_AREA)
        # Overlap visualization: red = sv3d-only, green = ref-only, yellow = both.
        from object_isolation.core.hallucination import _alpha_from_white_bg
        m_sv3d = _alpha_from_white_bg(cv2.cvtColor(sv3d, cv2.COLOR_BGR2RGB))
        m_ref = _alpha_from_white_bg(cv2.cvtColor(ref, cv2.COLOR_BGR2RGB))
        ov = np.full((tile, tile, 3), 245, dtype=np.uint8)
        only_sv3d = m_sv3d & ~m_ref
        only_ref = m_ref & ~m_sv3d
        both = m_sv3d & m_ref
        ov[only_sv3d] = (60, 60, 220)   # red
        ov[only_ref] = (60, 200, 60)    # green
        ov[both] = (60, 220, 220)       # yellow

        canvas[y0:y0 + tile, 0:tile] = sv3d
        canvas[y0:y0 + tile, tile:2*tile] = ref
        canvas[y0:y0 + tile, 2*tile:3*tile] = ov

        # Side info.
        info_x = 3 * tile + 10
        col = (60, 180, 60) if fr["accepted"] else (60, 60, 200)
        _putlbl(canvas, f"#{fr['index']:02d}", (info_x, y0 + 24),
                fg=(20, 20, 20), bg=(255, 255, 255), scale=0.55)
        _putlbl(canvas, f"az {fr['azimuth_V_deg']:+.1f}", (info_x, y0 + 48),
                fg=(60, 60, 60), bg=(255, 255, 255), scale=0.45)
        _putlbl(canvas, f"IoU {fr['iou_with_objgs']:.2f}", (info_x, y0 + 72),
                fg=col, bg=(255, 255, 255), scale=0.5)
        status = "KEEP" if fr["accepted"] else f"DROP: {fr['reject_reason']}"
        _putlbl(canvas, status, (info_x, y0 + 96),
                fg=col, bg=(255, 255, 255), scale=0.45)

        # Header.
        if r == 0:
            _putlbl(canvas, "SV3D", (8, 38),
                    fg=(40, 40, 40), bg=(255, 255, 255), scale=0.5)
            _putlbl(canvas, "ObjectGS", (tile + 8, 38),
                    fg=(40, 40, 40), bg=(255, 255, 255), scale=0.5)
            _putlbl(canvas, "Overlap", (2 * tile + 8, 38),
                    fg=(40, 40, 40), bg=(255, 255, 255), scale=0.5)

    out = debug_dir / "iou_strip.png"
    cv2.imwrite(str(out), canvas)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Panel 4: polar coverage plot (real cams vs hallucinated)
# ─────────────────────────────────────────────────────────────────────────────

def make_coverage_overlay(manifest: dict, scope_cameras: list, debug_dir: Path,
                          size: int = 600):
    canvas = np.full((size, size, 3), 250, dtype=np.uint8)
    cx, cy = size // 2, size // 2
    R = int(size * 0.42)
    cv2.circle(canvas, (cx, cy), R, (180, 180, 180), 1)
    # Cardinal markers.
    for az_deg, lbl in [(0, "az=0\n(front)"), (90, "+90"), (180, "180"), (-90, "-90")]:
        rad = math.radians(az_deg)
        x = int(cx + R * math.sin(rad))
        y = int(cy - R * math.cos(rad))
        cv2.line(canvas, (cx, cy), (x, y), (220, 220, 220), 1)
        _putlbl(canvas, lbl.split("\n")[0], (x - 12, y - 4),
                fg=(120, 120, 120), bg=(255, 255, 255), scale=0.4)

    _putlbl(canvas, "Azimuth coverage (V-frame)", (10, 28),
            fg=(20, 20, 20), bg=(255, 255, 255), scale=0.55)
    _putlbl(canvas, "gray = real cams,  green = SV3D kept,  red = SV3D dropped",
            (10, 50), fg=(60, 60, 60), bg=(255, 255, 255), scale=0.42)

    # Real cameras.
    for cam in scope_cameras:
        az = cam.get("azimuth_V_deg")
        if az is None or not math.isfinite(az):
            continue
        rad = math.radians(az)
        x = int(cx + R * math.sin(rad))
        y = int(cy - R * math.cos(rad))
        cv2.circle(canvas, (x, y), 4, (140, 140, 140), -1)

    # Hallucinated views, slight inner radius.
    R2 = int(R * 0.85)
    for fr in manifest["frames"]:
        az = fr["azimuth_V_deg"]
        if not math.isfinite(az):
            continue
        rad = math.radians(az)
        x = int(cx + R2 * math.sin(rad))
        y = int(cy - R2 * math.cos(rad))
        col = (60, 180, 60) if fr["accepted"] else (60, 60, 200)
        cv2.circle(canvas, (x, y), 6, col, 2)

    # Conditioning marker.
    cond_az = manifest["conditioning"]["azimuth_V_deg"]
    rad = math.radians(cond_az)
    x = int(cx + R * math.sin(rad))
    y = int(cy - R * math.cos(rad))
    cv2.drawMarker(canvas, (x, y), (200, 80, 200), cv2.MARKER_STAR, 18, 2)
    _putlbl(canvas, "cond", (x + 8, y), fg=(140, 40, 140),
            bg=(255, 255, 255), scale=0.45)

    out = debug_dir / "coverage_overlay.png"
    cv2.imwrite(str(out), canvas)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration
# ─────────────────────────────────────────────────────────────────────────────

def run_debug(model_path: str, object_id: int, output_root: str,
              iou_threshold: float = 0.20, fov_y_deg: float = 50.0,
              num_inference_steps: int = 25, safe_mode: bool = False,
              seed: int = 0, reuse_sv3d: bool = False):
    out_dir = Path(output_root) / f"obj_{object_id}"
    phase4_scores = out_dir / "phase4" / "scores.json"
    phase5_dir = out_dir / "phase5"
    debug_dir = out_dir / "debug_phase05"
    debug_dir.mkdir(parents=True, exist_ok=True)
    if not phase4_scores.exists():
        raise FileNotFoundError(f"Run Phase 4 first: missing {phase4_scores}")

    # Need scope, local_sv3d, gaussians/pipe_config (for ref renders).
    scope, _world_local, local_sv3d, gaussians, pipe_config = discover_object_scope(
        model_path=model_path, object_label_id=object_id,
    )

    backend = SV3DBackend(num_inference_steps=num_inference_steps,
                          safe_mode=safe_mode)
    manifest = run_hallucination(
        scope=scope, local_sv3d=local_sv3d,
        gaussians=gaussians, pipe_config=pipe_config,
        scores_json_path=phase4_scores, output_dir=phase5_dir,
        object_label_id=object_id, backend=backend,
        iou_threshold=iou_threshold, fov_y_deg=fov_y_deg, seed=seed,
        reuse_sv3d=reuse_sv3d,
    )

    make_conditioning_panel(manifest, debug_dir)
    make_sv3d_grid(manifest, debug_dir)
    make_iou_strip(manifest, debug_dir)
    make_coverage_overlay(manifest, scope.cameras, debug_dir)

    summary = {
        "manifest_path": str(phase5_dir / "hallucination_index.json"),
        "n_views": manifest["n_views"],
        "n_kept": manifest["n_kept"],
        "iou_threshold": iou_threshold,
        "conditioning": manifest["conditioning"],
    }
    with open(debug_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Phase 5 debug saved to: %s", debug_dir)
    return summary


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Phase 5 SV3D hallucination visual debug.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--object_id", required=True, type=int)
    parser.add_argument("--output_root", default="object_isolation/outputs")
    parser.add_argument("--iou_threshold", type=float, default=0.20)
    parser.add_argument("--fov_y_deg", type=float, default=50.0)
    parser.add_argument("--num_inference_steps", type=int, default=25)
    parser.add_argument("--safe_mode", action="store_true",
                        help="Reduce num_frames to 14 and resolution to 512 if VRAM-tight.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reuse_sv3d", action="store_true",
                        help="Skip diffusion; reload sv3d_raw/*.png from a prior run.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
    run_debug(args.model_path, args.object_id, args.output_root,
              iou_threshold=args.iou_threshold, fov_y_deg=args.fov_y_deg,
              num_inference_steps=args.num_inference_steps,
              safe_mode=args.safe_mode, seed=args.seed,
              reuse_sv3d=args.reuse_sv3d)


if __name__ == "__main__":
    main()
