"""Render all train/test views from a saved VRoom checkpoint.

Saves per-view: rendered RGB, ground truth, semantic segmentation maps,
depth maps, and per-view Gaussian visible counts.
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
import torchvision
import yaml
from tqdm import tqdm

from gaussian_renderer.render import prefilter_voxel, render
from vroom_core.utils.checkpoints import CheckpointManager
from vroom_core.models.anchor_field import AnchorCloud
from vroom_core.models.decoder import GaussianDecoder
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

    anchor_cloud = AnchorCloud()
    decoder = GaussianDecoder(
        feature_dim=model_kwargs.get("feat_dim", 32),
        anchor_cloud=anchor_cloud,
    )
    decoder.gs_attr = model_kwargs.get("gs_attr", "3D")
    decoder.render_mode = model_kwargs.get("render_mode", "RGB+ED")
    decoder.tile_size_2dgs = model_kwargs.get("tile_size_2dgs", 8)

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

    scene = TrainingScene(dataset_args, anchor_cloud, decoder, load_iteration=iteration, shuffle=False)
    decoder.eval()
    return anchor_cloud, decoder, scene


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


def _save_mask(mask: torch.Tensor, path: str):
    """Save a 2D uint8 mask tensor as grayscale PNG."""
    array = mask.detach().cpu().numpy().astype(np.uint8)
    imageio.imwrite(path, array)


class _RenderPipe:
    def __init__(self, add_prefilter: bool):
        self.add_prefilter = add_prefilter


@torch.no_grad()
def render_set(
    model_path: str,
    split_name: str,
    iteration: int,
    views,
    anchor_cloud: AnchorCloud,
    decoder: GaussianDecoder,
    pipe: _RenderPipe,
    background: torch.Tensor,
):
    render_dir = os.path.join(model_path, split_name, f"ours_{iteration}", "renders")
    gt_dir = os.path.join(model_path, split_name, f"ours_{iteration}", "gt")
    semantic_dir = os.path.join(model_path, split_name, f"ours_{iteration}", "semantic")
    semantic_gt_dir = os.path.join(model_path, split_name, f"ours_{iteration}", "semantic_gt")
    os.makedirs(render_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(semantic_dir, exist_ok=True)
    os.makedirs(semantic_gt_dir, exist_ok=True)

    vis_depth = decoder.gs_attr == "2D"
    vis_normal = decoder.gs_attr == "2D"
    if vis_depth:
        depth_dir = os.path.join(model_path, split_name, f"ours_{iteration}", "depth")
        os.makedirs(depth_dir, exist_ok=True)
    if vis_normal:
        normal_dir = os.path.join(model_path, split_name, f"ours_{iteration}", "normal")
        os.makedirs(normal_dir, exist_ok=True)

    visible_count_list = []
    per_view_dict = {}

    for idx, view in enumerate(tqdm(views, desc=f"Rendering {split_name}")):
        visible_mask = (
            prefilter_voxel(view, anchor_cloud, decoder).squeeze()
            if pipe.add_prefilter
            else anchor_cloud.visibility_mask
        )

        torch.cuda.synchronize()
        t_start = time.time()
        decoded_output = decoder.forward_pass(
            anchor_cloud=anchor_cloud,
            visible_anchors_mask=visible_mask,
            camera=view,
        )
        render_pkg = render(
            viewpoint_camera=view,
            decoded_output=decoded_output,
            bg_color=background,
            gs_attr=decoder.gs_attr,
            render_mode=decoder.render_mode,
            tile_size_2dgs=decoder.tile_size_2dgs,
            semantics=None,
        )
        torch.cuda.synchronize()
        t_end = time.time()

        # --- RGB render ---
        rendering = torch.clamp(render_pkg["render"], 0.0, 1.0)
        visible_count = render_pkg["visibility_filter"].sum()
        gt = view.original_image.cuda()
        alpha_mask = view.alpha_mask.cuda()
        rendering_rgba = torch.cat([rendering * alpha_mask, alpha_mask], dim=0)
        gt_rgba = torch.cat([gt * alpha_mask, alpha_mask], dim=0)

        _save_rgba(rendering_rgba, os.path.join(render_dir, f"{idx:05d}.png"))
        _save_rgba(gt_rgba, os.path.join(gt_dir, f"{idx:05d}.png"))

        # --- Semantic segmentation ---
        semantic_map = render_pkg["render_semantics"]
        if anchor_cloud.semantic_manager is not None:
            object_ids = anchor_cloud.semantic_manager.one_hot_decode(semantic_map.unsqueeze(0), semantic_map.shape[0])
            imageio.imwrite(
                os.path.join(semantic_dir, f"{idx:05d}.png"),
                object_ids.squeeze().cpu().numpy().astype(np.uint8),
            )

        # --- Semantic GT (object mask) ---
        object_mask = view.object_mask.cuda()
        _save_mask(object_mask, os.path.join(semantic_gt_dir, f"{idx:05d}.png"))

        # --- Depth ---
        if vis_depth and render_pkg["render_depth"] is not None:
            depth_map = render_pkg["render_depth"]
            # Normalize depth for visualization
            valid = depth_map[depth_map > 0]
            if valid.numel() > 0:
                d_min, d_max = valid.min(), valid.max()
                vis_depth_map = (depth_map - d_min) / (d_max - d_min + 1e-8)
            else:
                vis_depth_map = torch.zeros_like(depth_map)
            vis_depth_map = vis_depth_map.repeat(3, 1, 1) if vis_depth_map.shape[0] == 1 else vis_depth_map
            vis_depth_rgba = torch.cat([vis_depth_map, alpha_mask], dim=0)
            _save_rgba(vis_depth_rgba, os.path.join(depth_dir, f"{idx:05d}.png"))

        # --- Normal ---
        if vis_normal and "render_normals" in render_pkg:
            normal_map = render_pkg["render_normals"][0]  # [H, W, 3]
            normal_vis = (normal_map * 0.5 + 0.5).clamp(0.0, 1.0)
            normal_vis = normal_vis.permute(2, 0, 1)  # [3, H, W]
            alpha_np = (alpha_mask * 255).byte().permute(1, 2, 0).cpu().numpy()
            normal_np = (normal_vis * 255).byte().permute(1, 2, 0).cpu().numpy()
            normal_rgba = np.concatenate([normal_np, alpha_np], axis=2)
            imageio.imwrite(os.path.join(normal_dir, f"{idx:05d}.png"), normal_rgba)

        visible_count_list.append(visible_count)
        per_view_dict[f"{idx:05d}.png"] = visible_count.item()

    with open(
        os.path.join(model_path, split_name, f"ours_{iteration}", "per_view_count.json"),
        "w",
        encoding="utf-8",
    ) as fp:
        json.dump(per_view_dict, fp, indent=2)

    return visible_count_list


def main():
    parser = ArgumentParser(description="Render all views from a saved VRoom checkpoint")
    parser.add_argument("-m", "--model_path", type=str, required=True, help="Run directory")
    parser.add_argument("--source_path", type=str, default=None, help="Override dataset source path")
    parser.add_argument("--scene_name", type=str, default=None)
    parser.add_argument("--iteration", type=int, default=-1, help="Checkpoint iteration (-1 = latest)")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--no_prefilter", action="store_true")
    parser.add_argument("--gpu", type=str, default="-1")
    args = parser.parse_args()

    model_path = Path(args.model_path).resolve()
    config_path = model_path / "config.json"
    if not config_path.exists():
        config_path = model_path / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"No config.json or config.yaml found at {model_path}")

    if config_path.suffix == ".json":
        import json
        from vroom_core.utils.config import parse_vroom_config
        with open(config_path, "r", encoding="utf-8") as f:
            raw_cfg = json.load(f)
        model_params, optim_params, pipeline_params = parse_vroom_config(raw_cfg)
        cfg = {
            "model_params": model_params,
            "optim_params": optim_params,
            "pipeline_params": pipeline_params
        }
    else:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)

    if args.gpu != "-1":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        torch.cuda.set_device(int(args.gpu))

    iteration = _resolve_iteration(model_path, args.iteration)
    print(f"Rendering {model_path} — iteration {iteration}")

    anchor_cloud, decoder, scene = _load_model(model_path, cfg, iteration, args.source_path, args.scene_name)
    pipe = _RenderPipe(add_prefilter=not args.no_prefilter)

    if not args.skip_train:
        print("Rendering train set...")
        render_set(str(model_path), "train", iteration, scene.getTrainCameras(), anchor_cloud, decoder, pipe, scene.background)

    if not args.skip_test:
        print("Rendering test set...")
        render_set(str(model_path), "test", iteration, scene.getTestCameras(), anchor_cloud, decoder, pipe, scene.background)

    print("Rendering complete.")


if __name__ == "__main__":
    main()
