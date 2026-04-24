"""Helpers for metric mobile-scene bundle ingestion and known-pose COLMAP export."""

from __future__ import annotations

import json
import math
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


GS_TRAIN_ROOT = Path(__file__).resolve().parents[1] / "gs-train"
if str(GS_TRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(GS_TRAIN_ROOT))

from vroom_core.utils.geometry import arcore_camera_to_world_to_colmap_extrinsics, rotation_matrix_to_quaternion


REQUIRED_MANIFEST_FIELDS = {
    "scene_id",
    "capture_id",
    "platform",
    "device_model",
    "app_version",
    "frame_count",
    "image_width",
    "image_height",
    "frame_rate",
    "timebase",
    "world_up_axis",
    "units",
    "capture_mode",
    "capture_status",
}
REQUIRED_INTRINSICS_FIELDS = {"frame_id", "timestamp_ns", "fx", "fy", "cx", "cy", "width", "height"}
REQUIRED_POSE_FIELDS = {"frame_id", "timestamp_ns", "tracking_state", "camera_to_world", "pose_source"}
REQUIRED_TRACKING_FIELDS = {"frame_id", "timestamp_ns", "tracking_state"}
REQUIRED_SAM_MANIFEST_FIELDS = {"model_name", "model_version", "input_space", "mask_resolution", "preprocess_transform", "frame_join_key"}
VALID_TRACKING_STATES = {"normal", "tracking", "tracking_ok", "ok"}
MAX_POSITION_JUMP_METERS = 1.5
MAX_ROTATION_JUMP_DEGREES = 75.0


@dataclass(frozen=True)
class FramePoseRecord:
    frame_id: str
    timestamp_ns: int
    tracking_state: str
    camera_to_world: np.ndarray
    pose_source: str


@dataclass(frozen=True)
class FrameIntrinsicRecord:
    frame_id: str
    timestamp_ns: int
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    distortion: list[float]


@dataclass(frozen=True)
class TrackingRecord:
    frame_id: str
    timestamp_ns: int
    tracking_state: str


@dataclass(frozen=True)
class MetricBundle:
    root: Path
    manifest: dict
    image_root: Path
    intrinsics_by_frame: dict[str, FrameIntrinsicRecord]
    poses_by_frame: dict[str, FramePoseRecord]
    tracking_by_frame: dict[str, TrackingRecord]
    sam_manifest: dict | None
    valid_frame_ids: list[str]
    rejected_frame_ids: list[str]
    warnings: list[str]


def _load_json(path: Path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _require_fields(payload: dict, required: Iterable[str], context: str) -> None:
    missing = [field for field in required if field not in payload]
    if missing:
        raise ValueError(f"{context} is missing required fields: {missing}")


def _normalize_records(payload, context: str) -> list[dict]:
    if isinstance(payload, dict):
        if "frames" in payload and isinstance(payload["frames"], list):
            payload = payload["frames"]
        elif "records" in payload and isinstance(payload["records"], list):
            payload = payload["records"]
        else:
            raise ValueError(f"{context} must be a list or contain a 'frames'/'records' list.")
    if not isinstance(payload, list):
        raise ValueError(f"{context} must resolve to a list of records.")
    return payload


def _as_matrix4(value, context: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (4, 4):
        raise ValueError(f"{context} must be a 4x4 matrix, got shape {array.shape}.")
    return array


def _rotation_angle_degrees(a: np.ndarray, b: np.ndarray) -> float:
    relative = a.T @ b
    trace = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(math.acos(trace)))


def _serialize_colmap_camera(camera_id: int, intr: FrameIntrinsicRecord, model: str = "PINHOLE") -> str:
    return f"{camera_id} {model} {intr.width} {intr.height} {intr.fx:.12f} {intr.fy:.12f} {intr.cx:.12f} {intr.cy:.12f}\n"


def _serialize_colmap_image(image_id: int, qvec: np.ndarray, tvec: np.ndarray, camera_id: int, name: str) -> str:
    q = " ".join(f"{float(value):.17g}" for value in qvec.tolist())
    t = " ".join(f"{float(value):.17g}" for value in tvec.tolist())
    return f"{image_id} {q} {t} {camera_id} {name}\n\n"


def _symlink_or_copy(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        return
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def detect_bundle_image_root(bundle_root: Path) -> Path:
    for candidate in [bundle_root / "images", bundle_root / "frames"]:
        if candidate.exists() and candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"No images/ or frames/ directory found under {bundle_root}")


def load_metric_bundle(bundle_root: str | Path) -> MetricBundle:
    root = Path(bundle_root).resolve()
    manifest_path = root / "manifest.json"
    intrinsics_path = root / "intrinsics.json"
    poses_path = root / "poses.json"
    tracking_path = root / "tracking.json"
    sam_manifest_path = root / "sam_manifest.json"
    sam_root = root / "sam_masks"

    for required_path in [manifest_path, intrinsics_path, poses_path, tracking_path]:
        if not required_path.exists():
            raise FileNotFoundError(f"Missing required metric bundle artifact: {required_path}")

    image_root = detect_bundle_image_root(root)
    manifest = _load_json(manifest_path)
    _require_fields(manifest, REQUIRED_MANIFEST_FIELDS, "manifest.json")
    if str(manifest["units"]).lower() != "meters":
        raise ValueError("Metric bundle manifest must declare units='meters'.")
    if str(manifest["platform"]).lower() != "arcore":
        raise ValueError("Only ARCore metric bundles are supported in v1.")
    if str(manifest["capture_mode"]).lower() != "arcore_rgb":
        raise ValueError("Only capture_mode='arcore_rgb' is supported in v1.")
    if str(manifest["capture_status"]).lower() != "completed":
        raise ValueError("Capture bundle is not complete. Interrupted or aborted sessions must be rejected in v1.")

    intrinsics_records = {}
    for entry in _normalize_records(_load_json(intrinsics_path), "intrinsics.json"):
        _require_fields(entry, REQUIRED_INTRINSICS_FIELDS, "intrinsics.json record")
        frame_id = str(entry["frame_id"])
        intrinsics_records[frame_id] = FrameIntrinsicRecord(
            frame_id=frame_id,
            timestamp_ns=int(entry["timestamp_ns"]),
            fx=float(entry["fx"]),
            fy=float(entry["fy"]),
            cx=float(entry["cx"]),
            cy=float(entry["cy"]),
            width=int(entry["width"]),
            height=int(entry["height"]),
            distortion=[float(value) for value in entry.get("distortion", [])],
        )

    poses_records = {}
    for entry in _normalize_records(_load_json(poses_path), "poses.json"):
        _require_fields(entry, REQUIRED_POSE_FIELDS, "poses.json record")
        frame_id = str(entry["frame_id"])
        poses_records[frame_id] = FramePoseRecord(
            frame_id=frame_id,
            timestamp_ns=int(entry["timestamp_ns"]),
            tracking_state=str(entry["tracking_state"]).lower(),
            camera_to_world=_as_matrix4(entry["camera_to_world"], f"poses.json[{frame_id}] camera_to_world"),
            pose_source=str(entry["pose_source"]),
        )

    tracking_records = {}
    for entry in _normalize_records(_load_json(tracking_path), "tracking.json"):
        _require_fields(entry, REQUIRED_TRACKING_FIELDS, "tracking.json record")
        frame_id = str(entry["frame_id"])
        tracking_records[frame_id] = TrackingRecord(
            frame_id=frame_id,
            timestamp_ns=int(entry["timestamp_ns"]),
            tracking_state=str(entry["tracking_state"]).lower(),
        )

    sam_manifest = None
    if sam_manifest_path.exists():
        sam_manifest = _load_json(sam_manifest_path)
        _require_fields(sam_manifest, REQUIRED_SAM_MANIFEST_FIELDS, "sam_manifest.json")
        if sam_manifest.get("frame_join_key") != "frame_id":
            raise ValueError("sam_manifest.json must declare frame_join_key='frame_id'.")
        if not sam_root.exists():
            raise FileNotFoundError(f"Missing sam_masks directory: {sam_root}")

    frame_ids = sorted(poses_records.keys())
    if len(frame_ids) != int(manifest["frame_count"]):
        raise ValueError(
            f"manifest frame_count={manifest['frame_count']} does not match poses.json frame count={len(frame_ids)}."
        )

    warnings: list[str] = []
    rejected: list[str] = []
    valid: list[str] = []
    previous_pose: FramePoseRecord | None = None
    previous_dt: int | None = None

    for frame_id in frame_ids:
        if frame_id not in intrinsics_records:
            raise ValueError(f"Missing intrinsics.json entry for frame_id={frame_id}.")
        if frame_id not in tracking_records:
            raise ValueError(f"Missing tracking.json entry for frame_id={frame_id}.")

        image_path = _resolve_frame_asset(image_root, frame_id)
        if image_path is None:
            raise FileNotFoundError(f"No image asset found for frame_id={frame_id} under {image_root}.")
        _validate_frame_dimensions(image_path, intrinsics_records[frame_id])
        _validate_timestamp_match(frame_id, intrinsics_records[frame_id].timestamp_ns, poses_records[frame_id].timestamp_ns, "intrinsics/poses")
        _validate_timestamp_match(frame_id, tracking_records[frame_id].timestamp_ns, poses_records[frame_id].timestamp_ns, "tracking/poses")
        if sam_manifest is not None:
            _validate_mask_alignment(root, frame_id, sam_root)

        tracking_state = tracking_records[frame_id].tracking_state
        pose_tracking_state = poses_records[frame_id].tracking_state
        if tracking_state != pose_tracking_state:
            raise ValueError(
                f"Tracking state mismatch for frame_id={frame_id}: poses={pose_tracking_state}, tracking={tracking_state}"
            )

        reject_reason = None
        if tracking_state not in VALID_TRACKING_STATES:
            reject_reason = f"tracking_state={tracking_state}"
        elif previous_pose is not None:
            dt = poses_records[frame_id].timestamp_ns - previous_pose.timestamp_ns
            if dt <= 0:
                raise ValueError(f"Non-monotonic timestamps detected at frame_id={frame_id}.")
            position_jump = float(np.linalg.norm(
                poses_records[frame_id].camera_to_world[:3, 3] - previous_pose.camera_to_world[:3, 3]
            ))
            rotation_jump = _rotation_angle_degrees(
                previous_pose.camera_to_world[:3, :3],
                poses_records[frame_id].camera_to_world[:3, :3],
            )
            if previous_dt is not None and dt > previous_dt * 5:
                warnings.append(f"Large timestamp gap before frame_id={frame_id}: {dt} ns")
            if position_jump > MAX_POSITION_JUMP_METERS:
                reject_reason = f"pose_jump_m={position_jump:.3f}"
            elif rotation_jump > MAX_ROTATION_JUMP_DEGREES:
                reject_reason = f"pose_jump_deg={rotation_jump:.3f}"
            previous_dt = dt
        else:
            previous_dt = None

        if reject_reason is not None:
            warnings.append(f"Rejecting frame_id={frame_id}: {reject_reason}")
            rejected.append(frame_id)
            continue

        valid.append(frame_id)
        previous_pose = poses_records[frame_id]

    if not valid:
        raise ValueError("No valid frames remain after metric bundle validation.")

    return MetricBundle(
        root=root,
        manifest=manifest,
        image_root=image_root,
        intrinsics_by_frame=intrinsics_records,
        poses_by_frame=poses_records,
        tracking_by_frame=tracking_records,
        sam_manifest=sam_manifest,
        valid_frame_ids=valid,
        rejected_frame_ids=rejected,
        warnings=warnings,
    )


def _resolve_frame_asset(image_root: Path, frame_id: str) -> Path | None:
    for extension in [".png", ".jpg", ".jpeg", ".JPG", ".JPEG"]:
        candidate = image_root / f"{frame_id}{extension}"
        if candidate.exists():
            return candidate
    return None


def _validate_frame_dimensions(image_path: Path, intrinsics: FrameIntrinsicRecord) -> None:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required for metric bundle image validation.") from exc
    with Image.open(image_path) as image:
        width, height = image.size
    if width != intrinsics.width or height != intrinsics.height:
        raise ValueError(
            f"Image size mismatch for frame_id={intrinsics.frame_id}: image={width}x{height}, "
            f"intrinsics={intrinsics.width}x{intrinsics.height}"
        )


def _validate_timestamp_match(frame_id: str, left: int, right: int, context: str) -> None:
    if left != right:
        raise ValueError(f"Timestamp mismatch for frame_id={frame_id} in {context}: {left} != {right}")


def _validate_mask_alignment(bundle_root: Path, frame_id: str, sam_root: Path) -> None:
    manifest_entry = None
    sam_manifest = _load_json(bundle_root / "sam_manifest.json")
    if isinstance(sam_manifest, dict):
        entries = sam_manifest.get("frames") or sam_manifest.get("records") or []
        for entry in entries:
            if str(entry.get("frame_id")) == frame_id:
                manifest_entry = entry
                break
    if manifest_entry is None:
        mask_candidates = [sam_root / f"{frame_id}.png", sam_root / f"{frame_id}.npz"]
        if not any(path.exists() for path in mask_candidates):
            raise FileNotFoundError(f"No SAM mask found for frame_id={frame_id} under {sam_root}")
        return
    preprocess = manifest_entry.get("preprocess_transform")
    if preprocess is None:
        raise ValueError(f"SAM manifest entry for frame_id={frame_id} is missing preprocess_transform.")
    mask_path = bundle_root / str(manifest_entry.get("mask_path", ""))
    if not mask_path.exists():
        raise FileNotFoundError(f"SAM mask path for frame_id={frame_id} does not exist: {mask_path}")


def summarize_metric_bundle(bundle: MetricBundle) -> dict:
    valid_poses = [bundle.poses_by_frame[frame_id] for frame_id in bundle.valid_frame_ids]
    camera_path_length = 0.0
    heading_vectors = []
    for previous, current in zip(valid_poses, valid_poses[1:]):
        camera_path_length += float(
            np.linalg.norm(current.camera_to_world[:3, 3] - previous.camera_to_world[:3, 3])
        )
    for record in valid_poses:
        heading = record.camera_to_world[:3, :3] @ np.array([0.0, 0.0, -1.0], dtype=np.float64)
        heading_norm = np.linalg.norm(heading)
        if heading_norm > 1e-12:
            heading_vectors.append(heading / heading_norm)
    angular_coverage = 0.0
    if heading_vectors:
        heading_stack = np.stack(heading_vectors, axis=0)
        centroid = heading_stack.mean(axis=0)
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm > 1e-12:
            dots = np.clip(heading_stack @ (centroid / centroid_norm), -1.0, 1.0)
            angular_coverage = float(np.degrees(np.max(np.arccos(dots))))
    contiguous_tracking_runs = _count_tracking_runs(bundle)
    return {
        "scene_id": bundle.manifest["scene_id"],
        "capture_id": bundle.manifest["capture_id"],
        "platform": bundle.manifest["platform"],
        "reconstruction_mode": "known_pose_triangulation",
        "input_frame_count": int(bundle.manifest["frame_count"]),
        "valid_frame_count": len(bundle.valid_frame_ids),
        "rejected_frame_count": len(bundle.rejected_frame_ids),
        "rejected_frame_ratio": 0.0 if int(bundle.manifest["frame_count"]) == 0 else len(bundle.rejected_frame_ids) / float(bundle.manifest["frame_count"]),
        "rejected_frame_ids": bundle.rejected_frame_ids,
        "camera_path_length_m": camera_path_length,
        "angular_coverage_deg": angular_coverage,
        "contiguous_tracking_runs": contiguous_tracking_runs,
        "tracking_quality_summary": "stable" if not bundle.rejected_frame_ids else "degraded",
        "world_up_axis": bundle.manifest["world_up_axis"],
        "units": bundle.manifest["units"],
        "warnings": bundle.warnings,
    }


def _count_tracking_runs(bundle: MetricBundle) -> int:
    runs = 0
    active = False
    for frame_id in sorted(bundle.poses_by_frame, key=lambda value: bundle.poses_by_frame[value].timestamp_ns):
        is_valid = frame_id in set(bundle.valid_frame_ids)
        if is_valid and not active:
            runs += 1
            active = True
        elif not is_valid:
            active = False
    return runs


def export_known_pose_colmap_workspace(bundle: MetricBundle, data_root: str | Path) -> dict:
    data_root = Path(data_root).resolve()
    images_dir = data_root / "images"
    sparse_dir = data_root / "sparse" / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    camera_lines = ["# Camera list with one line of data per camera:\n", "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"]
    image_lines = ["# Image list with two lines of data per image:\n", "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n", "#   POINTS2D[] as (X, Y, POINT3D_ID)\n"]

    camera_key_to_id: dict[tuple, int] = {}
    camera_id_by_frame: dict[str, int] = {}
    next_camera_id = 1
    sorted_valid = sorted(bundle.valid_frame_ids, key=lambda value: bundle.poses_by_frame[value].timestamp_ns)

    for image_id, frame_id in enumerate(sorted_valid, start=1):
        intr = bundle.intrinsics_by_frame[frame_id]
        pose = bundle.poses_by_frame[frame_id]
        image_path = _resolve_frame_asset(bundle.image_root, frame_id)
        if image_path is None:
            raise FileNotFoundError(f"Missing image asset for frame_id={frame_id}")
        linked_path = images_dir / image_path.name
        _symlink_or_copy(image_path, linked_path)

        camera_key = ("PINHOLE", intr.width, intr.height, round(intr.fx, 9), round(intr.fy, 9), round(intr.cx, 9), round(intr.cy, 9))
        if camera_key not in camera_key_to_id:
            camera_key_to_id[camera_key] = next_camera_id
            camera_lines.append(_serialize_colmap_camera(next_camera_id, intr, model="PINHOLE"))
            next_camera_id += 1
        camera_id = camera_key_to_id[camera_key]
        camera_id_by_frame[frame_id] = camera_id

        rotation, translation = arcore_camera_to_world_to_colmap_extrinsics(pose.camera_to_world)
        qvec = rotation_matrix_to_quaternion(rotation)
        image_lines.append(_serialize_colmap_image(image_id, qvec, translation, camera_id, image_path.name))

    with open(sparse_dir / "cameras.txt", "w", encoding="utf-8") as handle:
        handle.writelines(camera_lines)
    with open(sparse_dir / "images.txt", "w", encoding="utf-8") as handle:
        handle.writelines(image_lines)
    with open(sparse_dir / "points3D.txt", "w", encoding="utf-8") as handle:
        handle.write("# Empty points file; populated by COLMAP point_triangulator.\n")

    summary = summarize_metric_bundle(bundle)
    summary.update(
        {
            "frames_root": str(bundle.image_root),
            "colmap_workspace": str(data_root),
            "valid_frame_ids": sorted_valid,
            "camera_count": len(camera_key_to_id),
        }
    )
    with open(data_root / "reconstruction_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    with open(data_root / "scene_transform.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "offset": [0.0, 0.0, 0.0],
                "scale": 1.0,
                "units": "meters",
                "up_axis": str(bundle.manifest.get("world_up_axis", "y")).lower(),
                "handedness": "right",
            },
            handle,
            indent=2,
        )
    return summary
