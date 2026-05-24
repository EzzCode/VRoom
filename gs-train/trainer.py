"""VRoom training entrypoint."""

from __future__ import annotations

import logging
import os
import json
from argparse import ArgumentParser
from datetime import datetime

from types import SimpleNamespace
import torch
from vroom_core.models.anchor_field import AnchorCloud
from vroom_core.models.decoder import GaussianDecoder
from vroom_core.training.orchestration import TrainingOrchestrator
from vroom_core.utils.runtime import seed_everything
from vroom_core.data.scene_pipeline import TrainingScene
from vroom_core.utils.config import load_vroom_config
from vroom_core.utils.checkpoints import CheckpointManager


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
    parser.add_argument("--no_vis", action="store_true", default=False, help="Disable visualization saving")
    parser.add_argument("--start_checkpoint", type=str, default=None)
    args = parser.parse_args()

    cfg, model_params, optim_params, pipeline_params = load_vroom_config(args.config)

    if args.scene_name is not None:
        model_params["exp_name"] = os.path.join(model_params.get("exp_name", "run"), args.scene_name)
        model_params["source_path"] = os.path.join(model_params["source_path"], args.scene_name)

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
    exp_name = model_params.get("exp_name", os.path.basename(args.config).replace(".json", ""))
    model_path = os.path.join(
        "output", dataset_name, exp_name, datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    os.makedirs(model_path, exist_ok=True)
    with open(os.path.join(model_path, "config.json"), "w", encoding="utf-8") as handle:
        json.dump(cfg, handle, indent=4)

    model_kwargs = model_params.get("model_config", {}).get("kwargs", {})
    anchor_cloud = AnchorCloud(
        voxel_size=model_kwargs.get("voxel_size", -1.0),
        gaussians_per_anchor=model_kwargs.get("gaussians_per_anchor", model_kwargs.get("n_offsets", 5)),
        feature_dim=model_kwargs.get("feat_dim", 32),
    )
    decoder = GaussianDecoder(
        feature_dim=model_kwargs.get("feat_dim", 32),
        anchor_cloud=anchor_cloud,
    ).to(anchor_cloud.device)
    decoder.gs_attr = model_kwargs.get("gs_attr", "3D")
    decoder.render_mode = model_kwargs.get("render_mode", "RGB+ED")
    decoder.tile_size_2dgs = model_kwargs.get("tile_size_2dgs", 8)

    logger = _configure_logging()
    logger.info("Optimizing " + model_path)

    dataset_args = _build_dataset_args(model_params, source_path, model_path)
    scene = TrainingScene(
        dataset_args,
        anchor_cloud,
        decoder,
        shuffle=pipeline_params.get("shuffle", True),
        logger=logger,
        weed_ratio=pipeline_params.get("weed_ratio", 0.0),
    )

    if args.no_vis:
        pipeline_params["save_vis"] = False

    configs = {
        "optimization": {
            **optim_params,
            "args": SimpleNamespace(**optim_params),
            "spatial_lr_scale": scene.cameras_extent,
            "anchor_cloud": anchor_cloud,
            "decoder": decoder,
        },
        "pipeline": {
            **pipeline_params,
            "output_dir": model_path,
            "bg_color": scene.background,
        },
        "densifier": cfg.get("densifier", {}),
    }

    orchestrator = TrainingOrchestrator(configs, scene=scene, logger=logger)

    os.makedirs(os.path.join(model_path, "visualization"), exist_ok=True)
    os.makedirs(os.path.join(model_path, "checkpoints"), exist_ok=True)

    orchestrator.train(scene.getTrainCameras())
    logger.info("\nTraining complete.")


if __name__ == "__main__":
    main()
