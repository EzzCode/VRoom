"""Export per-object meshes from a saved VRoom run."""

import os
import sys
from pathlib import Path

# Add the project root to sys.path to allow imports from any directory
root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

from argparse import ArgumentParser
import torch
import yaml

from vroom_core.data.scene_pipeline import TrainingScene
from vroom_core.export import (
    MeshFusionOptions,
    ObjectMeshExporter,
    build_measurement_record,
    convert_mesh_to_metric_scene_space,
    localize_metric_mesh,
    load_scene_export_context,
    save_measurement_record,
    save_scene_index,
)
from vroom_core.models.facade import GaussianModel
from typing import Optional, List, Dict, Tuple

def _load_config(model_path: Path, config_path: Optional[str]) -> Tuple[Dict, Path]:
    candidate = Path(config_path).resolve() if config_path is not None else model_path / "config.yaml"
    if not candidate.exists():
        raise FileNotFoundError(
            f"Could not find a config file at {candidate}. Pass --config or export from a run that contains config.yaml."
        )
    with open(candidate, "r", encoding="utf-8") as handle:
        return yaml.load(handle, Loader=yaml.FullLoader), candidate


def _resolve_source_path(model_params: dict, config_file: Path, source_override: Optional[str], scene_name: Optional[str]) -> str:
    source_path = source_override or model_params.get("source_path")
    if not source_path:
        raise ValueError("source_path is missing. Pass --source_path or provide a config with model_params.source_path.")
    if scene_name is not None:
        source_path = os.path.join(source_path, scene_name)
    if not os.path.isabs(source_path):
        source_path = os.path.abspath(os.path.join(Path(__file__).resolve().parent, source_path))
    return source_path


def _build_dataset_args(model_params: dict, source_path: str, model_path: str):
    class DatasetArgs:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    return DatasetArgs(
        source_path=source_path,
        model_path=model_path,
        resolution=model_params.get("resolution", -1),
        resolution_scales=model_params.get("resolution_scales", [1.0]),
        images=model_params.get("images", "images"),
        depths=model_params.get("depths", "depths"),
        masks=model_params.get("masks", "masks"),
        data_device=model_params.get("data_device", "cuda"),
        data_format=model_params.get("data_format", "colmap"),
        eval=model_params.get("eval", False),
        llffhold=model_params.get("llffhold", 8),
        add_mask=model_params.get("add_mask", False),
        add_depth=model_params.get("add_depth", False),
        white_background=model_params.get("white_background", False),
        random_background=False,
        ratio=model_params.get("ratio", 1),
        global_appearance=model_params.get("global_appearance", False),
        pretrained_checkpoint="",
        center=model_params.get("center", [0.0, 0.0, 0.0]),
        scale=model_params.get("scale", 1.0),
        dataset_name=model_params.get("dataset_name", ""),
        exp_name=model_params.get("exp_name", ""),
    )


def _resolve_iteration(model_path: Path, requested: int) -> int:
    if requested >= 0:
        return requested
    point_cloud_root = model_path / "point_cloud"
    iterations = sorted(
        [path for path in point_cloud_root.iterdir() if path.is_dir() and path.name.startswith("iteration_")],
        key=lambda path: int(path.name.split("_")[-1]),
    )
    if not iterations:
        raise FileNotFoundError(f"No iteration_* directories found under {point_cloud_root}")
    return int(iterations[-1].name.split("_")[-1])


def main():
    parser = ArgumentParser(description="Export object meshes from a saved VRoom run")
    parser.add_argument("--model_path", type=str, required=True, help="Run directory containing point_cloud/")
    parser.add_argument("--config", type=str, default=None, help="Config used for training. Defaults to <model_path>/config.yaml")
    parser.add_argument("--source_path", type=str, default=None, help="Override dataset scene path for camera loading")
    parser.add_argument("--scene_name", type=str, default=None, help="Append a scene name onto config source_path when needed")
    parser.add_argument("--iteration", type=int, default=-1, help="Checkpoint iteration to export, or -1 for latest")
    parser.add_argument("--label_id", type=int, action="append", default=None, help="Semantic label id to export. Can be used multiple times.")
    parser.add_argument("--all_labels", action="store_true", help="Export one mesh per non-zero label in the checkpoint")
    parser.add_argument("--voxel_size", type=float, default=-1.0)
    parser.add_argument("--depth_trunc", type=float, default=-1.0)
    parser.add_argument("--sdf_trunc", type=float, default=-1.0)
    parser.add_argument("--mesh_res", type=int, default=256)
    parser.add_argument("--cluster_keep", type=int, default=10)
    parser.add_argument("--gpu", type=str, default="-1")
    parser.add_argument("--no_prefilter", action="store_true")
    parser.add_argument("--white_background", action="store_true")
    parser.add_argument("--skip_metadata", action="store_true", help="Skip metric metadata export")
    args = parser.parse_args()

    model_path = Path(args.model_path).resolve()
    cfg, config_file = _load_config(model_path, args.config)
    model_params = cfg.get("model_params", {})
    model_kwargs = model_params.get("model_config", {}).get("kwargs", {})
    source_path = _resolve_source_path(model_params, config_file, args.source_path, args.scene_name)

    if args.gpu != "-1":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        print("Using GPU:", os.environ["CUDA_VISIBLE_DEVICES"])
        torch.cuda.set_device(int(args.gpu))

    iteration = _resolve_iteration(model_path, args.iteration)
    gaussians = GaussianModel(
        n_offsets=model_kwargs.get("n_offsets", 5),
        feat_dim=model_kwargs.get("feat_dim", 32),
        view_dim=model_kwargs.get("view_dim", 3),
        appearance_dim=model_kwargs.get("appearance_dim", 0),
        voxel_size=model_kwargs.get("voxel_size", -1.0),
        gs_attr=model_kwargs.get("gs_attr", "3D"),
        render_mode=model_kwargs.get("render_mode", "RGB+ED"),
        tile_size_2dgs=model_kwargs.get("tile_size_2dgs", 8),
    )
    dataset_args = _build_dataset_args(model_params, source_path, str(model_path))
    scene = TrainingScene(dataset_args, gaussians, load_iteration=iteration, shuffle=False)

    background = torch.ones(3, dtype=torch.float32, device=gaussians.device) if args.white_background else scene.background
    exporter = ObjectMeshExporter(gaussians, background, add_prefilter=not args.no_prefilter)
    all_labels = exporter.available_labels(skip_zero=False)
    labels = [label for label in all_labels if label != 0]
    if args.label_id:
        requested = sorted(set(int(label) for label in args.label_id))
        labels = [label for label in all_labels if label in requested]
    elif not args.all_labels:
        raise SystemExit("Pass --label_id <id> or --all_labels to choose which objects to export.")
    if not labels:
        if all_labels == [0]:
            raise SystemExit(
                "This checkpoint only contains label 0, so it has no per-object semantic labels. "
                "You can still export the whole scene with --label_id 0."
            )
        raise SystemExit("No matching labels were found in this checkpoint.")

    options = MeshFusionOptions(
        voxel_size=args.voxel_size,
        sdf_trunc=args.sdf_trunc,
        depth_trunc=args.depth_trunc,
        mesh_res=args.mesh_res,
        cluster_keep=args.cluster_keep,
        mask_background=False,
    )

    output_root = model_path / "meshes"
    metadata_records = []
    export_context = None if args.skip_metadata else load_scene_export_context(model_path, source_path)
    for label_id in labels:
        result = exporter.export_label_mesh(scene.getTrainCameras(), label_id, output_root, options)
        if export_context is not None:
            label_dir = output_root / f"label_{label_id}"
            metric_mesh_path = label_dir / "filtered_metric_scene.ply"
            metric_vertices = convert_mesh_to_metric_scene_space(
                result.filtered_path,
                export_context.scene_transform,
                output_path=metric_mesh_path,
            )
            local_mesh_path = label_dir / "filtered_local_object.ply"
            localize_metric_mesh(metric_mesh_path, local_mesh_path)
            object_id = f"label_{label_id:02d}"
            metadata_path = label_dir / "metadata.json"
            metadata = build_measurement_record(
                export_context,
                object_id=object_id,
                label_id=label_id,
                points_metric=metric_vertices,
                artifacts={
                    "raw_mesh_path": str(result.raw_path),
                    "filtered_mesh_path": str(result.filtered_path),
                    "metric_scene_mesh_path": str(metric_mesh_path),
                    "local_object_mesh_path": str(local_mesh_path),
                },
            )
            save_measurement_record(metadata_path, metadata)
            metadata_records.append(
                {
                    "object_id": object_id,
                    "label_id": label_id,
                    "metadata_path": str(metadata_path),
                    "mesh_path": str(metric_mesh_path),
                }
            )
        print(
            f"label {result.label_id}: raw={result.raw_path} ({result.num_vertices} verts), "
            f"filtered={result.filtered_path} ({result.num_vertices_filtered} verts)"
        )
    if export_context is not None:
        save_scene_index(output_root / "scene_objects_index.json", export_context, metadata_records)


if __name__ == "__main__":
    main()
