"""VRoom training entrypoint."""

from __future__ import annotations

import logging
import os
from argparse import ArgumentParser
from dataclasses import fields
from datetime import datetime

import torch
import yaml

from vroom_core.models.facade import GaussianModel
from vroom_core.training.orchestration import PipeConfig, PipelineConfig, TrainingConfig, TrainingOrchestrator as Trainer
from vroom_core.utils.runtime import seed_everything
from vroom_core.data.scene_pipeline import TrainingScene


def _apply_overrides(config_object, values: dict) -> None:
    valid = {field.name for field in fields(config_object)}
    for key, value in values.items():
        if key in valid:
            setattr(config_object, key, value)


def _build_dataset_args(model_params: dict, source_path: str, model_path: str, pipeline_params: dict):
    class DatasetArgs:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    resolution = model_params.get("resolution", pipeline_params.get("resolution", -1))
    return DatasetArgs(
        source_path=source_path,
        model_path=model_path,
        resolution=resolution,
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
        random_background=model_params.get("random_background", False),
        ratio=model_params.get("ratio", 1),
        global_appearance=model_params.get("global_appearance", False),
        pretrained_checkpoint=model_params.get("pretrained_checkpoint", ""),
        center=model_params.get("center", [0.0, 0.0, 0.0]),
        scale=model_params.get("scale", 1.0),
        dataset_name=model_params.get("dataset_name", ""),
        exp_name=model_params.get("exp_name", ""),
    )


def _configure_logging() -> logging.Logger:
    logger = logging.getLogger("vroom")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s: %(message)s"))
    logger.addHandler(handler)
    return logger


def main():
    parser = ArgumentParser(description="VRoom training parameters")
    parser.add_argument("--config", type=str, required=True, help="Training config file path")
    parser.add_argument("--scene_name", type=str, default=None, help="Override scene name in config")
    parser.add_argument("--gpu", type=str, default="-1")
    parser.add_argument("--no_vis", action="store_true", default=False, help="Disable side-by-side visualization saving")
    parser.add_argument("--start_checkpoint", type=str, default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        cfg = yaml.load(handle, Loader=yaml.FullLoader)

    model_params = cfg.get("model_params", {})
    optim_params = cfg.get("optim_params", {})
    pipeline_params = cfg.get("pipeline_params", {})

    if args.scene_name is not None:
        model_params["exp_name"] = os.path.join(model_params.get("exp_name", "run"), args.scene_name)
        model_params["source_path"] = os.path.join(model_params["source_path"], args.scene_name)

    # --- Resume from checkpoint ---
    start_iter = 0
    if args.start_checkpoint:
        if not os.path.isdir(args.start_checkpoint):
            parser.error(f"--start_checkpoint path does not exist: {args.start_checkpoint}")
        dir_name = os.path.basename(args.start_checkpoint.rstrip("/"))
        if dir_name.startswith("iteration_"):
            start_iter = int(dir_name.replace("iteration_", ""))
        else:
            parser.error("--start_checkpoint must point to a directory named 'iteration_NNNN'")
        model_params["pretrained_checkpoint"] = args.start_checkpoint

    source_path = model_params.get("source_path")
    if source_path is None:
        parser.error("source_path is missing from model_params")
    if not os.path.isabs(source_path):
        source_path = os.path.abspath(os.path.join(os.path.dirname(__file__), source_path))

    if args.gpu != "-1":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        print("Using GPU:", os.environ["CUDA_VISIBLE_DEVICES"])
        torch.cuda.set_device(int(args.gpu))
    else:
        torch.cuda.set_device(0)

    seed_everything(quiet=False)

    dataset_name = model_params.get("dataset_name", "")
    exp_name = model_params.get("exp_name", os.path.basename(args.config).replace(".yaml", ""))
    model_path = os.path.join("output", dataset_name, exp_name, datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))
    os.makedirs(model_path, exist_ok=True)
    with open(os.path.join(model_path, "config.yaml"), "w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)

    train_config = TrainingConfig()
    pipeline_config = PipelineConfig()
    _apply_overrides(train_config, optim_params)
    _apply_overrides(pipeline_config, pipeline_params)
    pipeline_config.save_vis = not args.no_vis

    model_kwargs = model_params.get("model_config", {}).get("kwargs", {})
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

    dataset_args = _build_dataset_args(model_params, source_path, model_path, pipeline_params)
    logger = _configure_logging()
    logger.info("Optimizing " + model_path)
    scene = TrainingScene(dataset_args, gaussians, shuffle=pipeline_config.shuffle, logger=logger, weed_ratio=pipeline_config.weed_ratio)
    trainer = Trainer(train_config, pipeline_config, gaussians, scene, output_dir=model_path, logger=logger)
    trainer.run(first_iter=start_iter, resume_checkpoint_path=args.start_checkpoint)
    logger.info("\nTraining complete.")


if __name__ == "__main__":
    main()
