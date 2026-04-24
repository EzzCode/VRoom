"""Greenfield VRoom runtime."""

from .data.camera_system import FrameRecord, RenderCamera
from .data.colmap_io import (
    quaternion_to_rotation,
    read_extrinsics_binary,
    read_extrinsics_text,
    read_intrinsics_binary,
    read_intrinsics_text,
)
from .data.scene_pipeline import SceneBundle, TrainingScene, camera_to_json, compute_nerf_normalization, load_colmap_bundle

from .models.facade import GaussianModel
from .models.semantics import SemanticCodec

from .utils.geometry import (
    PointCloudSample,
    SceneTransform,
    apply_scene_transform,
    arcore_camera_to_world_to_colmap_extrinsics,
    focal_to_fov,
    fov_to_focal,
    invert_scene_transform,
    load_scene_transform,
    pil_image_to_tensor,
    projection_matrix,
    rotation_matrix_to_quaternion,
    save_scene_transform,
    world_to_view_matrix,
)
from .utils.runtime import ensure_directory, exponential_lr_schedule, seed_everything

from .training.loss_engine import compute_losses

from .export import (
    MeshExportResult,
    MeshFusionOptions,
    ObjectMeshExporter,
    SceneExportContext,
    build_measurement_record,
    convert_mesh_to_metric_scene_space,
    localize_metric_mesh,
    load_scene_export_context,
    save_measurement_record,
    save_scene_index,
)

from .viewer import viewer_protocol

__all__ = [
    "FrameRecord",
    "GaussianModel",
    "PointCloudSample",
    "SceneTransform",
    "RenderCamera",
    "SceneBundle",
    "SemanticCodec",
    "MeshExportResult",
    "MeshFusionOptions",
    "ObjectMeshExporter",
    "SceneExportContext",
    "TrainingScene",
    "build_measurement_record",
    "apply_scene_transform",
    "arcore_camera_to_world_to_colmap_extrinsics",
    "camera_to_json",
    "compute_losses",
    "compute_nerf_normalization",
    "ensure_directory",
    "exponential_lr_schedule",
    "focal_to_fov",
    "fov_to_focal",
    "invert_scene_transform",
    "load_scene_transform",
    "load_colmap_bundle",
    "load_scene_export_context",
    "localize_metric_mesh",
    "pil_image_to_tensor",
    "projection_matrix",
    "quaternion_to_rotation",
    "read_extrinsics_binary",
    "read_extrinsics_text",
    "read_intrinsics_binary",
    "read_intrinsics_text",
    "rotation_matrix_to_quaternion",
    "save_scene_transform",
    "save_measurement_record",
    "save_scene_index",
    "convert_mesh_to_metric_scene_space",
    "seed_everything",
    "world_to_view_matrix",
    "viewer_protocol",
]
