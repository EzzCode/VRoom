"""Sim(3) alignment between COLMAP SfM and ARCore metric poses.

Computes the similarity transform (scale, rotation, translation) that maps
an arbitrary-scale COLMAP reconstruction to the metric coordinate frame
defined by ARCore poses.  The aligned model is written back to disk so that
all downstream stages (SAM, tracking, voting, GS training) operate in meters.

References
----------
Umeyama, S. (1991). Least-squares estimation of transformation parameters
between two point patterns. IEEE PAMI, 13(4), 376-380.
"""

from __future__ import annotations

import json
import logging
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from colmap_loader import (
    qvec2rotmat,
    read_extrinsics_binary,
    read_extrinsics_text,
    read_intrinsics_binary,
    read_intrinsics_text,
    read_points3D_binary,
    read_points3D_text,
    rotmat2qvec,
)

logger = logging.getLogger(__name__)

VALID_TRACKING_STATES = {"normal", "tracking", "tracking_ok", "ok"}
MIN_ALIGNMENT_CORRESPONDENCES = 5
MAX_ACCEPTABLE_RMSE_METERS = 0.50


# ## Data classes #############################################################

@dataclass(frozen=True)
class AlignmentResult:
    """Result of a Sim(3) alignment between two camera sets."""
    scale: float
    rotation: np.ndarray      # 3×3
    translation: np.ndarray   # 3
    rmse: float
    num_correspondences: int


# ## Umeyama Sim(3) alignment ################################################

def umeyama_alignment(
    source: np.ndarray,
    target: np.ndarray,
) -> AlignmentResult:
    """Compute the Sim(3) transform: target ≈ s · R @ source + t.

    Parameters
    ----------
    source : (N, 3) array – source points (COLMAP camera centres).
    target : (N, 3) array – target points (ARCore camera centres, in meters).

    Returns
    -------
    AlignmentResult with scale, rotation, translation, and RMSE.
    """
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError(f"source and target must be (N, 3); got {source.shape}, {target.shape}")
    n = source.shape[0]
    if n < MIN_ALIGNMENT_CORRESPONDENCES:
        raise ValueError(
            f"Need at least {MIN_ALIGNMENT_CORRESPONDENCES} correspondences, got {n}."
        )

    # Centroids
    mu_s = source.mean(axis=0)
    mu_t = target.mean(axis=0)

    # Centre
    src = source - mu_s
    tgt = target - mu_t

    # Covariance
    cov = tgt.T @ src / n

    # SVD
    U, D, Vt = np.linalg.svd(cov)

    # Correct for reflection
    S = np.eye(3, dtype=np.float64)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0

    R = U @ S @ Vt

    # Scale
    var_s = np.sum(src ** 2) / n
    scale = float(np.trace(np.diag(D) @ S) / var_s)

    # Translation
    t = mu_t - scale * R @ mu_s

    # RMSE
    aligned = scale * (source @ R.T) + t
    rmse = float(np.sqrt(np.mean(np.sum((target - aligned) ** 2, axis=1))))

    return AlignmentResult(
        scale=scale,
        rotation=R,
        translation=t,
        rmse=rmse,
        num_correspondences=n,
    )


# ## Pose loading helpers ####################################################

def _load_arcore_camera_centres(data_path: Path) -> dict[str, np.ndarray]:
    """Load ARCore camera centres (meters) keyed by frame_id.

    Only returns frames with a valid tracking state.
    """
    poses_path = data_path / "poses.json"
    tracking_path = data_path / "tracking.json"

    with open(poses_path, "r", encoding="utf-8") as f:
        poses_raw = json.load(f)
    with open(tracking_path, "r", encoding="utf-8") as f:
        tracking_raw = json.load(f)

    # Normalise list
    if isinstance(poses_raw, dict):
        poses_raw = poses_raw.get("frames") or poses_raw.get("records") or []
    if isinstance(tracking_raw, dict):
        tracking_raw = tracking_raw.get("frames") or tracking_raw.get("records") or []

    tracking_state_by_frame = {}
    for entry in tracking_raw:
        fid = str(entry["frame_id"])
        tracking_state_by_frame[fid] = str(entry["tracking_state"]).lower()

    centres: dict[str, np.ndarray] = {}
    for entry in poses_raw:
        fid = str(entry["frame_id"])
        state = tracking_state_by_frame.get(fid, str(entry.get("tracking_state", "")).lower())
        if state not in VALID_TRACKING_STATES:
            continue
        c2w = np.asarray(entry["camera_to_world"], dtype=np.float64)
        if c2w.shape != (4, 4):
            continue
        centres[fid] = c2w[:3, 3].copy()

    return centres


def _load_colmap_camera_centres(sparse_dir: Path) -> dict[str, np.ndarray]:
    """Load COLMAP SfM camera centres keyed by image stem (= frame_id).

    Returns the camera centre in COLMAP's arbitrary world frame.
    """
    images_bin = sparse_dir / "images.bin"
    images_txt = sparse_dir / "images.txt"
    if images_bin.exists():
        images = read_extrinsics_binary(str(images_bin))
    elif images_txt.exists():
        images = read_extrinsics_text(str(images_txt))
    else:
        raise FileNotFoundError(f"No COLMAP images file in {sparse_dir}")

    centres: dict[str, np.ndarray] = {}
    for img in images.values():
        R = qvec2rotmat(img.qvec)
        t = img.tvec
        centre = -R.T @ t  # camera centre in world
        stem = Path(img.name).stem
        centres[stem] = centre.astype(np.float64)
    return centres


# ## COLMAP model transform ##################################################

def _read_colmap_model(sparse_dir: Path):
    """Read the full COLMAP model (cameras, images, points3D).

    Returns (cameras_dict, images_dict, points3D_data, binary_flag).
    """
    is_binary = (sparse_dir / "images.bin").exists()

    if is_binary:
        images = read_extrinsics_binary(str(sparse_dir / "images.bin"))
        cameras = read_intrinsics_binary(str(sparse_dir / "cameras.bin"))
        xyzs, rgbs, errors = read_points3D_binary(str(sparse_dir / "points3D.bin"))
    else:
        images = read_extrinsics_text(str(sparse_dir / "images.txt"))
        cameras = read_intrinsics_text(str(sparse_dir / "cameras.txt"))
        xyzs, rgbs, errors = read_points3D_text(str(sparse_dir / "points3D.txt"))

    return cameras, images, (xyzs, rgbs, errors), is_binary


def _transform_colmap_image(img, R_sim3, scale, t_sim3):
    """Apply Sim(3) to a single COLMAP image's extrinsics.

    Given Sim(3):  p_new = s * R_sim3 @ p_old + t_sim3

    The new world-to-camera extrinsics are:
        R_cam_new = R_cam_old @ R_sim3^T
        t_cam_new = s * t_cam_old  -  R_cam_new @ t_sim3
    """
    R_cam_old = qvec2rotmat(img.qvec)
    t_cam_old = img.tvec

    R_cam_new = R_cam_old @ R_sim3.T
    t_cam_new = scale * t_cam_old - R_cam_new @ t_sim3

    qvec_new = rotmat2qvec(R_cam_new)
    return qvec_new, t_cam_new


def _write_colmap_model_text(
    sparse_dir: Path,
    cameras,
    images,
    points3D_data,
):
    """Write a COLMAP model in text format.

    Overwrites any existing files (binary or text) in sparse_dir.
    """
    xyzs, rgbs, errors = points3D_data

    # Remove binary files if present (we replace with text)
    for name in ["cameras.bin", "images.bin", "points3D.bin"]:
        p = sparse_dir / name
        if p.exists():
            p.unlink()

    # cameras.txt
    with open(sparse_dir / "cameras.txt", "w", encoding="utf-8") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        for cam in cameras.values():
            params_str = " ".join(f"{p:.12f}" for p in cam.params)
            f.write(f"{cam.id} {cam.model} {cam.width} {cam.height} {params_str}\n")

    # images.txt
    with open(sparse_dir / "images.txt", "w", encoding="utf-8") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for img in images.values():
            q = " ".join(f"{v:.17g}" for v in img.qvec)
            t = " ".join(f"{v:.17g}" for v in img.tvec)
            f.write(f"{img.id} {q} {t} {img.camera_id} {img.name}\n")
            # Write points2D line
            if img.xys is not None and len(img.xys) > 0:
                parts = []
                for j in range(len(img.xys)):
                    parts.append(f"{img.xys[j, 0]:.6f} {img.xys[j, 1]:.6f} {img.point3D_ids[j]}")
                f.write(" ".join(parts) + "\n")
            else:
                f.write("\n")

    # points3D.txt
    with open(sparse_dir / "points3D.txt", "w", encoding="utf-8") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
        for i in range(len(xyzs)):
            x, y, z = xyzs[i]
            r, g, b = int(rgbs[i, 0]), int(rgbs[i, 1]), int(rgbs[i, 2])
            err = float(errors[i])
            f.write(f"{i + 1} {x:.17g} {y:.17g} {z:.17g} {r} {g} {b} {err:.6f}\n")


def apply_sim3_to_colmap_model(
    sparse_dir: Path,
    result: AlignmentResult,
) -> None:
    """Transform an entire COLMAP model in-place using the given Sim(3).

    After this call the model is in the metric coordinate frame (meters).
    """
    cameras, images, points3D_data, _ = _read_colmap_model(sparse_dir)
    xyzs, rgbs, errors = points3D_data

    R = result.rotation
    s = result.scale
    t = result.translation

    # Transform 3D points:  p_new = s * R @ p_old + t
    if len(xyzs) > 0:
        xyzs_new = s * (xyzs @ R.T) + t
    else:
        xyzs_new = xyzs

    # Transform camera extrinsics
    from colmap_loader import Image as ColmapImage
    new_images = {}
    for img_id, img in images.items():
        qvec_new, tvec_new = _transform_colmap_image(img, R, s, t)
        new_images[img_id] = ColmapImage(
            id=img.id,
            qvec=qvec_new,
            tvec=tvec_new,
            camera_id=img.camera_id,
            name=img.name,
            xys=img.xys,
            point3D_ids=img.point3D_ids,
        )

    _write_colmap_model_text(sparse_dir, cameras, new_images, (xyzs_new, rgbs, errors))
    logger.info(
        "Applied Sim(3) to COLMAP model in %s (scale=%.6f, RMSE=%.4fm)",
        sparse_dir, s, result.rmse,
    )


# ## Main entry point ########################################################

def compute_metric_alignment(
    output_path: Path,
    data_path: Path,
) -> AlignmentResult | None:
    """Align a COLMAP SfM model to ARCore metric poses.

    1. Loads ARCore camera positions from ``data_path/poses.json``.
    2. Loads COLMAP camera positions from ``output_path/sparse/0/``.
    3. Matches cameras by frame_id.
    4. Runs Umeyama alignment → (scale, R, t).
    5. Transforms the COLMAP model in-place.
    6. Writes ``scene_transform.json`` (scale=1, units=meters).

    Returns the AlignmentResult, or None if alignment is not possible.
    """
    sparse_dir = output_path / "sparse" / "0"

    # Check prerequisites
    if not (data_path / "poses.json").exists():
        logger.info("No ARCore poses.json found; skipping metric alignment.")
        return None
    if not (data_path / "tracking.json").exists():
        logger.info("No ARCore tracking.json found; skipping metric alignment.")
        return None

    # Load camera centres
    arcore_centres = _load_arcore_camera_centres(data_path)
    colmap_centres = _load_colmap_camera_centres(sparse_dir)

    # Match by frame_id
    common = sorted(set(arcore_centres) & set(colmap_centres))
    logger.info(
        "Metric alignment: %d ARCore frames, %d COLMAP frames, %d common.",
        len(arcore_centres), len(colmap_centres), len(common),
    )

    if len(common) < MIN_ALIGNMENT_CORRESPONDENCES:
        logger.warning(
            "Only %d common frames (need %d); skipping alignment.",
            len(common), MIN_ALIGNMENT_CORRESPONDENCES,
        )
        return None

    src = np.array([colmap_centres[fid] for fid in common])
    tgt = np.array([arcore_centres[fid] for fid in common])

    result = umeyama_alignment(src, tgt)

    logger.info(
        "Sim(3) alignment: scale=%.6f, RMSE=%.4fm, correspondences=%d",
        result.scale, result.rmse, result.num_correspondences,
    )

    if result.rmse > MAX_ACCEPTABLE_RMSE_METERS:
        logger.warning(
            "Alignment RMSE %.4fm exceeds threshold %.2fm. "
            "Model will NOT be transformed; scene_transform will use scene_units.",
            result.rmse, MAX_ACCEPTABLE_RMSE_METERS,
        )
        return None

    # Apply the Sim(3) to the COLMAP model in-place
    apply_sim3_to_colmap_model(sparse_dir, result)

    # Write scene_transform.json — the model is now in meters,
    # so the transform is identity (scale=1, offset=0).
    scene_transform = {
        "offset": [0.0, 0.0, 0.0],
        "scale": 1.0,
        "units": "meters",
        "up_axis": "y",
        "handedness": "right",
    }
    with open(output_path / "scene_transform.json", "w", encoding="utf-8") as f:
        json.dump(scene_transform, f, indent=2)

    # Write alignment diagnostics
    alignment_info = {
        "method": "umeyama_sim3",
        "scale": result.scale,
        "rotation": result.rotation.tolist(),
        "translation": result.translation.tolist(),
        "rmse_meters": result.rmse,
        "num_correspondences": result.num_correspondences,
        "matched_frame_ids": common,
    }
    with open(output_path / "metric_alignment.json", "w", encoding="utf-8") as f:
        json.dump(alignment_info, f, indent=2)

    return result
