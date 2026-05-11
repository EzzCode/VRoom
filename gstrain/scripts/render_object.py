"""Render individual objects from a saved VRoom checkpoint.

Isolates Gaussians by their semantic label ID and renders them against
a white background, producing per-object renders for visualization
and downstream mesh extraction.
"""

import json
import os
import sys
import time
from pathlib import Path

# Add the project root to sys.path to allow imports from any directory
root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

from argparse import ArgumentParser
from pathlib import Path

import imageio
import numpy as np
import torch
import yaml
from tqdm import tqdm

from gstrain.gaussian_renderer.render import prefilter_voxel, render
from gstrain.vroom_core.models.facade import GaussianModel
from gstrain.vroom_core.data.scene_pipeline import TrainingScene


from typing import Optional, List, Dict, Tuple

def _load_model(model_path: Path, cfg: dict, iteration: int, source_path: Optional[str], scene_name: Optional[str]):
    model_params = cfg.get("model_params", {})
    model_kwargs = model_params.get("model_config", {}).get("kwargs", {})

    if scene_name is not None:
        model_params["exp_name"] = os.path.join(model_params.get("exp_name", ""), scene_name)
        model_params["source_path"] = os.path.join(model_params["source_path"], scene_name)

    if source_path is not None:
        model_params["source_path"] = source_path

    resolved_source = model_params.get("source_path", "")
    if not os.path.isabs(resolved_source):
        resolved_source = os.path.abspath(os.path.join(Path(__file__).resolve().parent, resolved_source))

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

    class DatasetArgs:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    dataset_args = DatasetArgs(
        source_path=resolved_source,
        model_path=str(model_path),
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

    scene = TrainingScene(dataset_args, gaussians, load_iteration=iteration, shuffle=False)
    gaussians.set_eval()
    return gaussians, scene


def _resolve_iteration(model_path: Path, requested: int) -> int:
    if requested >= 0:
        return requested
    pc_root = model_path / "point_cloud"
    iterations = sorted(
        [p for p in pc_root.iterdir() if p.is_dir() and p.name.startswith("iteration_")],
        key=lambda p: int(p.name.split("_")[-1]),
    )
    if not iterations:
        raise FileNotFoundError(f"No iteration directories found under {pc_root}")
    return int(iterations[-1].name.split("_")[-1])


def _save_rgba(tensor: torch.Tensor, path: str):
    """Save a [C,H,W] tensor (3 or 4 channels, 0-1 range) as PNG."""
    clamped = torch.clamp(tensor, 0.0, 1.0)
    array = (clamped.permute(1, 2, 0).detach().cpu().numpy() * 255.0).astype(np.uint8)
    imageio.imwrite(path, array)


class _RenderPipe:
    def __init__(self, add_prefilter: bool):
        self.add_prefilter = add_prefilter


@torch.no_grad()
def render_object_set(
    model_path: str,
    split_name: str,
    iteration: int,
    views,
    gaussians: GaussianModel,
    pipe: _RenderPipe,
    background: torch.Tensor,
    query_label_id: int,
):
    """Render only the Gaussians belonging to `query_label_id`."""
    render_dir = os.path.join(model_path, split_name, f"id_{query_label_id}", "renders")
    gt_dir = os.path.join(model_path, split_name, f"id_{query_label_id}", "gt")
    os.makedirs(render_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    # Build per-anchor object mask: only anchors with this label
    object_mask = (gaussians.label_ids.squeeze() == query_label_id)

    visible_count_list = []
    per_view_dict = {}

    for idx, view in enumerate(tqdm(views, desc=f"Rendering label {query_label_id} — {split_name}")):
        gaussians.set_anchor_mask(view.camera_center, view.resolution_scale)
        visible_mask = (
            prefilter_voxel(view, gaussians).squeeze()
            if pipe.add_prefilter
            else gaussians._anchor_mask
        )

        torch.cuda.synchronize()
        t_start = time.time()
        render_pkg = render(
            view, gaussians, pipe, background,
            visible_mask=visible_mask,
            training=False,
            object_mask=object_mask,
        )
        torch.cuda.synchronize()
        t_end = time.time()

        rendering = torch.clamp(render_pkg["render"], 0.0, 1.0)
        visible_count = render_pkg["visibility_filter"].sum()

        gt = view.original_image.cuda()
        alpha_mask = view.alpha_mask.cuda()
        rendering_rgba = torch.cat([rendering, alpha_mask], dim=0)
        gt_rgba = torch.cat([gt, alpha_mask], dim=0)

        _save_rgba(rendering_rgba, os.path.join(render_dir, f"{idx:05d}.png"))
        _save_rgba(gt_rgba, os.path.join(gt_dir, f"{idx:05d}.png"))

        visible_count_list.append(visible_count)
        per_view_dict[f"{idx:05d}.png"] = visible_count.item()

    with open(
        os.path.join(model_path, split_name, f"id_{query_label_id}", "per_view_count.json"),
        "w",
        encoding="utf-8",
    ) as fp:
        json.dump(per_view_dict, fp, indent=2)

    return visible_count_list


def main():
    parser = ArgumentParser(description="Render individual objects from a saved VRoom checkpoint")
    parser.add_argument("-m", "--model_path", type=str, required=True, help="Run directory")
    parser.add_argument("--source_path", type=str, default=None, help="Override dataset source path")
    parser.add_argument("--scene_name", type=str, default=None)
    parser.add_argument("--iteration", type=int, default=-1, help="Checkpoint iteration (-1 = latest)")
    parser.add_argument(
        "--query_label_id", type=int, default=-1,
        help="Label ID to render. Use -1 to render ALL non-zero labels.",
    )
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--no_prefilter", action="store_true")
    parser.add_argument("--gpu", type=str, default="-1")
    args = parser.parse_args()

    model_path = Path(args.model_path).resolve()
    config_path = model_path / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"No config.yaml found at {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    if args.gpu != "-1":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        torch.cuda.set_device(int(args.gpu))

    iteration = _resolve_iteration(model_path, args.iteration)
    print(f"Rendering objects from {model_path} — iteration {iteration}")

    gaussians, scene = _load_model(model_path, cfg, iteration, args.source_path, args.scene_name)
    pipe = _RenderPipe(add_prefilter=not args.no_prefilter)

    # White background for cleaner object isolation
    background = torch.ones(3, dtype=torch.float32, device=gaussians.device)

    # Determine which labels to render
    if gaussians.label_ids is None:
        raise SystemExit("This checkpoint has no semantic labels — cannot render individual objects.")

    all_labels = torch.unique(gaussians.label_ids.view(-1)).cpu().tolist()
    all_labels = sorted(int(l) for l in all_labels)
    print(f"Available labels in checkpoint: {all_labels}")

    if args.query_label_id == -1:
        labels_to_render = [l for l in all_labels if l != 0]
        print(f"Rendering all non-zero labels: {labels_to_render}")
    else:
        if args.query_label_id not in all_labels:
            raise SystemExit(f"Label {args.query_label_id} not found. Available: {all_labels}")
        labels_to_render = [args.query_label_id]

    for label_id in labels_to_render:
        print(f"\n--- Rendering label {label_id} ---")
        if not args.skip_train:
            render_object_set(
                str(model_path), "train", iteration,
                scene.getTrainCameras(), gaussians, pipe, background, label_id,
            )
        if not args.skip_test:
            render_object_set(
                str(model_path), "test", iteration,
                scene.getTestCameras(), gaussians, pipe, background, label_id,
            )

    print("\nObject rendering complete.")


if __name__ == "__main__":
    main()
