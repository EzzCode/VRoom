"""Regenerate metric measurement metadata for exported object meshes."""

from __future__ import annotations

import sys
from argparse import ArgumentParser
from pathlib import Path

root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

from vroom_core.export import (
    build_measurement_record,
    convert_mesh_to_metric_scene_space,
    localize_metric_mesh,
    load_scene_export_context,
    save_measurement_record,
    save_scene_index,
)


def main():
    parser = ArgumentParser(description="Measure exported object meshes in metric scene space")
    parser.add_argument("--model_path", required=True, help="Run directory containing meshes/")
    parser.add_argument("--source_path", default=None, help="Optional dataset scene root override")
    args = parser.parse_args()

    model_path = Path(args.model_path).resolve()
    source_path = Path(args.source_path).resolve() if args.source_path else model_path
    meshes_root = model_path / "meshes"
    if not meshes_root.exists():
        raise FileNotFoundError(f"No meshes directory found under {model_path}")

    context = load_scene_export_context(model_path, source_path)
    records = []
    for label_dir in sorted(path for path in meshes_root.iterdir() if path.is_dir() and path.name.startswith("label_")):
        filtered_path = label_dir / "filtered.ply"
        if not filtered_path.exists():
            continue
        label_id = int(label_dir.name.split("_")[-1])
        metric_path = label_dir / "filtered_metric_scene.ply"
        metric_vertices = convert_mesh_to_metric_scene_space(filtered_path, context.scene_transform, output_path=metric_path)
        local_path = label_dir / "filtered_local_object.ply"
        localize_metric_mesh(metric_path, local_path)
        metadata_path = label_dir / "metadata.json"
        object_id = f"label_{label_id:02d}"
        metadata = build_measurement_record(
            context,
            object_id=object_id,
            label_id=label_id,
            points_metric=metric_vertices,
            artifacts={
                "filtered_mesh_path": str(filtered_path),
                "metric_scene_mesh_path": str(metric_path),
                "local_object_mesh_path": str(local_path),
            },
        )
        save_measurement_record(metadata_path, metadata)
        records.append(
            {
                "object_id": object_id,
                "label_id": label_id,
                "metadata_path": str(metadata_path),
                "mesh_path": str(metric_path),
            }
        )
        print(f"Measured {object_id} -> {metadata_path}")

    save_scene_index(meshes_root / "scene_objects_index.json", context, records)


if __name__ == "__main__":
    main()
