"""IO and comparison-rendering helpers extracted from run_replenishment.py.

These helpers are orthogonal to orchestration logic — they handle disk IO
(images, JSON metadata) and auto-comparison camera setup/rendering.

Public API:
    save_image(img, path)
    save_coverage_plot(coverage, path)
    build_comparison_cameras(center, radius, orbit_radius, up_vector,
                             input_cam_position, width, height, n_views)
    render_object_with_cameras(gaussians, pipe_config, cameras, object_id)
    save_camera_metadata(path, object_id, center, radius, cameras)
    save_auto_comparison(before_frames, after_frames, out_dir)
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch


__all__ = [
    "save_image",
    "save_coverage_plot",
    "build_comparison_cameras",
    "render_object_with_cameras",
    "save_camera_metadata",
    "save_auto_comparison",
]


def save_image(img: np.ndarray, path: Path):
    """Save an image to disk."""
    if img.ndim == 3 and img.shape[2] == 3:
        cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    else:
        cv2.imwrite(str(path), img)


def save_coverage_plot(coverage, path: Path):
    """Save a simple coverage histogram visualization."""
    n_bins = len(coverage.coverage_map)
    H, W = 200, 400
    img = np.ones((H, W, 3), dtype=np.uint8) * 255

    bar_w = W // n_bins
    for i, val in enumerate(coverage.coverage_map):
        bar_h = int(val * (H - 20))
        x0 = i * bar_w
        x1 = x0 + bar_w - 1
        y0 = H - 10 - bar_h
        y1 = H - 10
        # Green = covered, red = gap
        color = (0, 180, 0) if val >= 0.1 else (0, 0, 200)
        cv2.rectangle(img, (x0, y0), (x1, y1), color, -1)

    # Mark input camera azimuth
    input_bin = int((coverage.input_azimuth + np.pi) / (2 * np.pi) * n_bins)
    input_bin = np.clip(input_bin, 0, n_bins - 1)
    cx = input_bin * bar_w + bar_w // 2
    cv2.circle(img, (cx, 5), 5, (255, 0, 0), -1)

    cv2.imwrite(str(path), img)


def build_comparison_cameras(center, radius, orbit_radius, up_vector,
                             input_cam_position, width, height, n_views):
    from target_replenishment.render_360 import look_at

    up = up_vector.astype(np.float32)

    # Keep comparison cameras outside the object and near training-view distance.
    dist = max(float(orbit_radius) * 0.9, float(radius) * 2.5, 0.5)

    # Recompute a wider focal length for comparison rendering so object framing is stable.
    angular = 2.0 * np.arctan(float(radius) / max(dist, 1e-6))
    fov = np.clip(angular / 0.55, np.radians(30.0), np.radians(100.0))
    fx = (width / 2.0) / np.tan(fov / 2.0)
    fy = (height / 2.0) / np.tan(fov / 2.0)
    k = np.array([[fx, 0.0, width / 2.0],
                  [0.0, fy, height / 2.0],
                  [0.0, 0.0, 1.0]], dtype=np.float32)

    # Use input camera direction as the orbit start to avoid random extreme angles.
    ref_vec = (input_cam_position.astype(np.float32) - center.astype(np.float32))
    vertical = float(np.dot(ref_vec, up))
    horizontal = ref_vec - vertical * up
    if np.linalg.norm(horizontal) < 1e-6:
        horizontal = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if abs(np.dot(horizontal, up)) > 0.9:
            horizontal = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    basis_h = horizontal / (np.linalg.norm(horizontal) + 1e-8)
    basis_v = np.cross(up, basis_h)
    basis_v = basis_v / (np.linalg.norm(basis_v) + 1e-8)

    # Preserve some elevation from the source camera while clamping extremes.
    z_offset = float(np.clip(vertical, -0.3 * dist, 0.3 * dist))

    cams = []
    for i in range(n_views):
        angle = 2.0 * np.pi * i / n_views
        cam_pos = (
            center
            + dist * np.cos(angle) * basis_h
            + dist * np.sin(angle) * basis_v
            + z_offset * up
        ).astype(np.float32)
        r, t = look_at(cam_pos.astype(np.float32), center.astype(np.float32), up)
        cams.append({
            'index': i,
            'azimuth_deg': float(np.degrees(angle)),
            'cam_pos': cam_pos,
            'R': r,
            'T': t,
            'K': k.copy(),
            'width': width,
            'height': height,
        })
    return cams


def render_object_with_cameras(gaussians, pipe_config, cameras, object_id):
    from target_replenishment.core.objectgs_bridge import create_virtual_camera, render_view

    bg = torch.ones(3, dtype=torch.float32, device='cuda')
    frames = []
    for cam_data in cameras:
        cam = create_virtual_camera(
            cam_data['R'],
            cam_data['T'],
            cam_data['K'],
            cam_data['width'],
            cam_data['height'],
        )
        res = render_view(gaussians, cam, pipe_config, bg, object_label_id=object_id)
        rgb = (res['rgb'].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        frames.append(rgb)
    return frames


def save_camera_metadata(path, object_id, center, radius, cameras):
    payload = {
        'object_id': int(object_id),
        'object_center': np.asarray(center, dtype=np.float32).tolist(),
        'object_radius': float(radius),
        'n_views': len(cameras),
        'cameras': [
            {
                'index': c['index'],
                'azimuth_deg': c['azimuth_deg'],
                'cam_pos': np.asarray(c['cam_pos'], dtype=np.float32).tolist(),
                'R': np.asarray(c['R'], dtype=np.float32).tolist(),
                'T': np.asarray(c['T'], dtype=np.float32).tolist(),
                'K': np.asarray(c['K'], dtype=np.float32).tolist(),
                'width': int(c['width']),
                'height': int(c['height']),
            }
            for c in cameras
        ],
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)


def save_auto_comparison(before_frames, after_frames, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    diffs = []
    for i, (before_rgb, after_rgb) in enumerate(zip(before_frames, after_frames)):
        save_image(before_rgb, out_dir / f"before_view_{i}.png")
        save_image(after_rgb, out_dir / f"after_view_{i}.png")

        compare = np.hstack([before_rgb.copy(), after_rgb.copy()])
        cv2.putText(compare, 'BEFORE', (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
        cv2.putText(compare, 'AFTER', (before_rgb.shape[1] + 12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        save_image(compare, out_dir / f"compare_view_{i}.png")

        abs_diff = np.abs(after_rgb.astype(np.int16) - before_rgb.astype(np.int16)).astype(np.uint8)
        diff_gray = np.mean(abs_diff, axis=2).astype(np.uint8)
        boosted = np.clip(diff_gray.astype(np.float32) * 4.0, 0, 255).astype(np.uint8)
        diff_heat = cv2.applyColorMap(boosted, cv2.COLORMAP_JET)
        cv2.imwrite(str(out_dir / f"diff_view_{i}.png"), diff_heat)

        diffs.append(float(diff_gray.mean()))

    return {
        'n_views': len(diffs),
        'mean_abs_diff': float(np.mean(diffs)) if diffs else 0.0,
        'max_abs_diff': float(np.max(diffs)) if diffs else 0.0,
    }
