"""Generate deterministic before/after comparison renders for one object.

The script renders two models using the exact same camera poses and saves:
  - before_view_{i}.png
  - after_view_{i}.png
  - compare_view_{i}.png
  - diff_view_{i}.png

It also writes camera metadata JSON for reproducibility.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


def _ensure_import_paths():
    root = Path(__file__).resolve().parent.parent
    objectgs_path = root / "temp_deps" / "ObjectGS"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    if str(objectgs_path) not in sys.path:
        sys.path.insert(0, str(objectgs_path))


_ensure_import_paths()

from target_replenishment.core.objectgs_bridge import (  # noqa: E402
    create_virtual_camera,
    get_anchor_positions,
    load_gaussians,
    render_view,
)
from target_replenishment.core import diagnostics as diag  # noqa: E402


def _ensure_model_side_files(model_before: Path, model_after: Path):
    """Copy side files if missing in the after-model folder."""
    for name in ("cameras.json", "config.yaml"):
        src = model_before / name
        dst = model_after / name
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)


def _load_intrinsics_from_cameras_json(model_path: Path):
    with open(model_path / "cameras.json", "r", encoding="utf-8") as f:
        cams = json.load(f)
    if not cams:
        raise ValueError(f"No cameras found in {model_path / 'cameras.json'}")
    c0 = cams[0]
    width = int(c0["width"])
    height = int(c0["height"])
    k = np.array(
        [[c0["fx"], 0, width / 2.0], [0, c0["fy"], height / 2.0], [0, 0, 1]],
        dtype=np.float32,
    )
    return k, width, height, cams


def _build_orbit_cameras(center, radius, k, width, height, n_views, cam_data):
    up = diag.estimate_scene_up_from_cameras(cam_data)
    cam_centers = diag.camera_centers_from_cameras_json(cam_data)
    base = diag.orbit_base_direction_from_cameras(cam_centers, center, up)
    side = np.cross(up, base)
    side = side / max(np.linalg.norm(side), 1e-8)

    # Keep camera outside object with safer framing than the old tight orbit.
    dist = max(radius * 2.5, 0.5)
    angular = 2.0 * np.arctan(radius / max(dist, 1e-6))
    fov = np.clip(angular / 0.55, np.radians(30.0), np.radians(100.0))
    fx = (width / 2.0) / np.tan(fov / 2.0)
    fy = (height / 2.0) / np.tan(fov / 2.0)
    k_safe = np.array([[fx, 0.0, width / 2.0], [0.0, fy, height / 2.0], [0.0, 0.0, 1.0]], dtype=np.float32)

    up_offset = np.clip(radius * 0.2, 0.05, 0.3 * dist)

    cameras = []
    for i in range(n_views):
        angle = 2.0 * np.pi * i / n_views
        radial = np.cos(angle) * base + np.sin(angle) * side
        cam_pos = center + radial.astype(np.float32) * dist + up * up_offset
        r, t = diag._look_at(cam_pos.astype(np.float32), center.astype(np.float32), up)
        cameras.append(
            {
                "index": i,
                "azimuth_deg": float(np.degrees(angle)),
                "cam_pos": cam_pos,
                "up": up,
                "R": r,
                "T": t,
                "K": k_safe.copy(),
                "width": width,
                "height": height,
            }
        )
    return cameras


def _save_camera_metadata(path: Path, object_id: int, center, radius, cameras):
    data = {
        "object_id": int(object_id),
        "object_center": np.asarray(center, dtype=np.float32).tolist(),
        "object_radius": float(radius),
        "n_views": len(cameras),
        "scene_up": np.asarray(cameras[0]["up"], dtype=np.float32).tolist() if cameras else None,
        "cameras": [
            {
                "index": c["index"],
                "azimuth_deg": c["azimuth_deg"],
                "cam_pos": np.asarray(c["cam_pos"], dtype=np.float32).tolist(),
                "up": np.asarray(c["up"], dtype=np.float32).tolist(),
                "R": np.asarray(c["R"], dtype=np.float32).tolist(),
                "T": np.asarray(c["T"], dtype=np.float32).tolist(),
                "K": np.asarray(c["K"], dtype=np.float32).tolist(),
                "width": int(c["width"]),
                "height": int(c["height"]),
            }
            for c in cameras
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _load_camera_metadata(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    cameras = []
    for c in data.get("cameras", []):
        cameras.append(
            {
                "index": int(c["index"]),
                "azimuth_deg": float(c["azimuth_deg"]),
                "cam_pos": np.asarray(c["cam_pos"], dtype=np.float32),
                "R": np.asarray(c["R"], dtype=np.float32),
                "T": np.asarray(c["T"], dtype=np.float32),
                "K": np.asarray(c["K"], dtype=np.float32),
                "width": int(c["width"]),
                "height": int(c["height"]),
            }
        )
    center = np.asarray(data["object_center"], dtype=np.float32)
    radius = float(data["object_radius"])
    object_id = int(data["object_id"])
    return object_id, center, radius, cameras


def _render_with_cameras(gaussians, pipe_config, cameras, object_id):
    bg = torch.ones(3, dtype=torch.float32, device="cuda")
    outputs = []
    for c in cameras:
        cam = create_virtual_camera(c["R"], c["T"], c["K"], c["width"], c["height"])
        res = render_view(gaussians, cam, pipe_config, bg, object_label_id=object_id)
        rgb = (res["rgb"].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        outputs.append(rgb)
    return outputs


def _save_rgb(path: Path, rgb: np.ndarray):
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def _build_diff_heatmap(before_rgb: np.ndarray, after_rgb: np.ndarray):
    diff = np.abs(after_rgb.astype(np.int16) - before_rgb.astype(np.int16)).astype(np.uint8)
    diff_gray = np.mean(diff, axis=2).astype(np.uint8)
    boosted = np.clip(diff_gray.astype(np.float32) * 4.0, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(boosted, cv2.COLORMAP_JET)


def main():
    parser = argparse.ArgumentParser(description="Fixed-pose before/after object comparison")
    parser.add_argument("--model_before", required=True, help="Path to original ObjectGS model")
    parser.add_argument("--model_after", required=True, help="Path to replenished/fine-tuned model")
    parser.add_argument("--output_dir", required=True, help="Directory for output images")
    parser.add_argument("--object_id", type=int, required=True, help="Object label ID to compare")
    parser.add_argument("--n_views", type=int, default=8, help="Number of orbit views")
    parser.add_argument("--iteration", type=int, default=-1, help="Iteration to load")
    parser.add_argument(
        "--camera_metadata",
        default=None,
        help="Optional camera metadata JSON path. If present, reuse exact poses.",
    )
    args = parser.parse_args()

    model_before = Path(args.model_before)
    model_after = Path(args.model_after)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Allow passing the replenishment root directory directly.
    if not (model_before / "point_cloud").exists() and (model_before / "final_model").exists():
        model_before = model_before / "final_model"
    if not (model_after / "point_cloud").exists() and (model_after / "final_model").exists():
        model_after = model_after / "final_model"

    if not model_before.exists():
        raise FileNotFoundError(f"Missing --model_before: {model_before}")
    if not model_after.exists():
        raise FileNotFoundError(f"Missing --model_after: {model_after}")

    _ensure_model_side_files(model_before, model_after)

    print("Loading before model...")
    g_before, pp_before = load_gaussians(str(model_before), args.iteration)

    metadata_path = Path(args.camera_metadata) if args.camera_metadata else out_dir / "camera_metadata.json"
    if metadata_path.exists():
        object_id_meta, center, radius, cameras = _load_camera_metadata(metadata_path)
        if object_id_meta != args.object_id:
            raise ValueError(
                f"camera_metadata object_id={object_id_meta} does not match requested {args.object_id}"
            )
        print(f"Loaded camera metadata from {metadata_path}")
    else:
        k, width, height, cam_data = _load_intrinsics_from_cameras_json(model_before)
        labels = g_before.label_ids.squeeze(-1).cpu().numpy()
        xyz = get_anchor_positions(g_before)
        obj_xyz = xyz[labels == args.object_id]
        if len(obj_xyz) == 0:
            raise ValueError(f"Object ID {args.object_id} not found in before model labels")
        center = np.median(obj_xyz, axis=0)
        radius = float(np.percentile(np.linalg.norm(obj_xyz - center.reshape(1, 3), axis=1), 90.0))
        cameras = _build_orbit_cameras(center, radius, k, width, height, args.n_views, cam_data)
        _save_camera_metadata(metadata_path, args.object_id, center, radius, cameras)
        print(f"Saved camera metadata to {metadata_path}")

    print("Rendering before views...")
    before_frames = _render_with_cameras(g_before, pp_before, cameras, args.object_id)

    print("Loading after model...")
    g_after, pp_after = load_gaussians(str(model_after), args.iteration)

    print("Rendering after views...")
    after_frames = _render_with_cameras(g_after, pp_after, cameras, args.object_id)

    print("Saving comparison outputs...")
    for i, (before_rgb, after_rgb) in enumerate(zip(before_frames, after_frames)):
        _save_rgb(out_dir / f"before_view_{i}.png", before_rgb)
        _save_rgb(out_dir / f"after_view_{i}.png", after_rgb)

        compare = np.hstack([before_rgb.copy(), after_rgb.copy()])
        cv2.putText(compare, "BEFORE", (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
        cv2.putText(compare, "AFTER", (before_rgb.shape[1] + 12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        _save_rgb(out_dir / f"compare_view_{i}.png", compare)

        diff_heat_bgr = _build_diff_heatmap(before_rgb, after_rgb)
        cv2.imwrite(str(out_dir / f"diff_view_{i}.png"), diff_heat_bgr)

    print(f"Saved {len(before_frames)} fixed-pose view comparisons to {out_dir}")


if __name__ == "__main__":
    main()
