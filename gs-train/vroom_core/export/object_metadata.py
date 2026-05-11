"""Object measurement and metadata export helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

from vroom_core.utils.geometry import SceneTransform, invert_scene_transform, load_scene_transform


SCHEMA_VERSION = "1.0.0"


@dataclass(frozen=True)
class SceneExportContext:
    model_path: Path
    source_path: Path
    scene_transform: SceneTransform
    scene_id: str
    capture_id: str
    reconstruction_summary: dict
    capture_manifest: dict


def load_scene_export_context(model_path: str | Path, source_path: str | Path) -> SceneExportContext:
    model_root = Path(model_path).resolve()
    source_root = Path(source_path).resolve()
    scene_transform = _resolve_scene_transform(model_root, source_root)
    scene_manifest = _read_json_if_exists(model_root / "scene_manifest.json")
    capture_manifest = scene_manifest.get("capture_manifest") or _read_json_if_exists(source_root / "manifest.json")
    reconstruction_summary = scene_manifest.get("reconstruction_summary") or _read_json_if_exists(source_root / "reconstruction_summary.json")
    scene_id = str(capture_manifest.get("scene_id") or reconstruction_summary.get("scene_id") or source_root.name)
    capture_id = str(capture_manifest.get("capture_id") or reconstruction_summary.get("capture_id") or source_root.name)
    return SceneExportContext(
        model_path=model_root,
        source_path=source_root,
        scene_transform=scene_transform,
        scene_id=scene_id,
        capture_id=capture_id,
        reconstruction_summary=reconstruction_summary,
        capture_manifest=capture_manifest,
    )


def _resolve_scene_transform(model_root: Path, source_root: Path) -> SceneTransform:
    for candidate in [model_root / "scene_transform.json", source_root / "scene_transform.json"]:
        if candidate.exists():
            return load_scene_transform(candidate)
    return SceneTransform(offset=np.zeros(3, dtype=np.float32), scale=1.0, units="scene_units", up_axis="y", handedness="right")


def _read_json_if_exists(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def read_mesh_vertices(path: str | Path) -> tuple[np.ndarray, PlyData]:
    ply = PlyData.read(str(path))
    vertex = ply["vertex"]
    vertices = np.stack([np.asarray(vertex["x"]), np.asarray(vertex["y"]), np.asarray(vertex["z"])], axis=1).astype(np.float32)
    return vertices, ply


def write_mesh_with_vertices(path: str | Path, template: PlyData, vertices: np.ndarray) -> None:
    vertex = template["vertex"]
    vertex_dtype = vertex.data.dtype
    packed = np.array(vertex.data, copy=True)
    packed["x"] = vertices[:, 0]
    packed["y"] = vertices[:, 1]
    packed["z"] = vertices[:, 2]
    elements = [PlyElement.describe(packed.astype(vertex_dtype), "vertex")]
    for element in template.elements[1:]:
        elements.append(PlyElement.describe(np.array(element.data, copy=True), element.name))
    PlyData(elements, text=template.text).write(str(path))


def convert_mesh_to_metric_scene_space(mesh_path: str | Path, scene_transform: SceneTransform, output_path: str | Path | None = None) -> np.ndarray:
    vertices, template = read_mesh_vertices(mesh_path)
    metric_vertices = invert_scene_transform(vertices, scene_transform.offset, scene_transform.scale)
    if output_path is not None:
        write_mesh_with_vertices(output_path, template, metric_vertices)
    return metric_vertices


def localize_metric_mesh(mesh_path: str | Path, output_path: str | Path) -> dict:
    vertices, template = read_mesh_vertices(mesh_path)
    center, axes, extents = _compute_oriented_bounding_box(vertices)
    local_vertices = (vertices - center) @ axes
    write_mesh_with_vertices(output_path, template, local_vertices.astype(np.float32))
    return {
        "center": center,
        "axes": axes,
        "extents": extents,
    }


def build_measurement_record(context: SceneExportContext, object_id: str, label_id: int, points_metric: np.ndarray, artifacts: dict) -> dict:
    if points_metric.shape[0] == 0:
        raise ValueError(f"Cannot measure empty geometry for {object_id}")
    center, axes, extents = _compute_oriented_bounding_box(points_metric)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = axes
    transform[:3, 3] = center
    up_axis = str(context.scene_transform.up_axis or "y").lower()
    up_vector = _axis_vector(up_axis)
    plane_height = float(np.min(points_metric @ up_vector))
    centroid = np.mean(points_metric, axis=0)
    bottom_center = centroid - up_vector * (float(np.dot(centroid, up_vector)) - plane_height)
    confidence = _derive_confidence(context.reconstruction_summary, context.scene_transform)
    tracking_quality_summary = str(context.reconstruction_summary.get("tracking_quality_summary", "unknown"))
    record = {
        "object_id": object_id,
        "label_id": int(label_id),
        "scene_id": context.scene_id,
        "source_capture_id": context.capture_id,
        "schema_version": SCHEMA_VERSION,
        "coordinate_system": {
            "units": "meters" if context.scene_transform.units == "meters" else context.scene_transform.units,
            "up_axis": up_axis,
            "handedness": context.scene_transform.handedness,
        },
        "object_to_scene_transform": transform.tolist(),
        "dimensions_m": (extents * 2.0).astype(float).tolist(),
        "oriented_bounding_box": {
            "center_m": center.astype(float).tolist(),
            "axes": axes.astype(float).tolist(),
            "extents_m": extents.astype(float).tolist(),
        },
        "support_plane": {
            "normal": up_vector.astype(float).tolist(),
            "offset_m": -plane_height,
        },
        "bottom_center_m": bottom_center.astype(float).tolist(),
        "confidence": confidence,
        "tracking_quality_summary": tracking_quality_summary,
        "measurement_method": "obb_from_metric_scene_mesh",
        "reconstruction_version": str(context.reconstruction_summary.get("reconstruction_mode", "v1")),
        "artifacts": artifacts,
    }
    return record


def save_measurement_record(path: str | Path, payload: dict) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def save_scene_index(path: str | Path, context: SceneExportContext, objects: list[dict]) -> None:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "scene_id": context.scene_id,
        "source_capture_id": context.capture_id,
        "coordinate_system": {
            "units": "meters" if context.scene_transform.units == "meters" else context.scene_transform.units,
            "up_axis": context.scene_transform.up_axis,
            "handedness": context.scene_transform.handedness,
        },
        "reconstruction_summary": context.reconstruction_summary,
        "objects": objects,
    }
    save_measurement_record(path, payload)


def _axis_vector(axis_name: str) -> np.ndarray:
    mapping = {
        "x": np.array([1.0, 0.0, 0.0], dtype=np.float64),
        "y": np.array([0.0, 1.0, 0.0], dtype=np.float64),
        "z": np.array([0.0, 0.0, 1.0], dtype=np.float64),
    }
    return mapping.get(axis_name, mapping["y"])


def _compute_oriented_bounding_box(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    if points.shape[0] < 3 or np.linalg.norm(centered) < 1e-12:
        axes = np.eye(3, dtype=np.float64)
        mins = np.min(points, axis=0)
        maxs = np.max(points, axis=0)
        extents = (maxs - mins) * 0.5
        center = (mins + maxs) * 0.5
        return center.astype(np.float64), axes, extents.astype(np.float64)
    covariance = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    axes = eigenvectors[:, order]
    if np.linalg.det(axes) < 0:
        axes[:, -1] *= -1.0
    local_points = centered @ axes
    mins = np.min(local_points, axis=0)
    maxs = np.max(local_points, axis=0)
    extents = (maxs - mins) * 0.5
    local_center = (mins + maxs) * 0.5
    center = centroid + axes @ local_center
    return center.astype(np.float64), axes.astype(np.float64), extents.astype(np.float64)


def _derive_confidence(reconstruction_summary: dict, scene_transform: SceneTransform) -> float:
    confidence = 0.95 if scene_transform.units == "meters" else 0.75
    rejected_ratio = reconstruction_summary.get("rejected_frame_ratio")
    if rejected_ratio is not None:
        confidence -= min(float(rejected_ratio) * 0.4, 0.35)
    if str(reconstruction_summary.get("tracking_quality_summary", "")).lower() not in {"stable", "good", "normal"}:
        confidence -= 0.15
    triangulated_points = reconstruction_summary.get("triangulated_point_count")
    if triangulated_points is not None and int(triangulated_points) < 500:
        confidence -= 0.10
    return float(max(0.05, min(confidence, 0.99)))
