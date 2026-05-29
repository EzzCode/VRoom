"Training loop entrypoint"
from __future__ import annotations

import logging
import os
import json
from argparse import ArgumentParser
from datetime import datetime

from types import SimpleNamespace
import torch
from gstrain.vroom_core.core.model.anchor_field import AnchorCloud
from gstrain.vroom_core.core.model.decoder import GaussianDecoder
from gstrain.vroom_core.core.training.orchestration import TrainingOrchestrator
from gstrain.vroom_core.utilities.utils.runtime import seed_everything
from gstrain.vroom_core.utilities.data_utils.scene_pipeline import TrainingScene
from gstrain.vroom_core.utilities.utils.config import load_vroom_config
from gstrain.vroom_core.utilities.utils.checkpoints import CheckpointManager


def _build_dataset_args(model_params: dict, dataset_path: str, run_dir: str):
    class DatasetArgs:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    return DatasetArgs(
        dataset_path=dataset_path,
        model_path=run_dir,
        resolution=model_params.get("resolution", -1),
        resolution_scales=model_params.get("resolution_scales", [1.0]),
        frames=model_params.get("frames", "images"),
        depths=model_params.get("depths", "depths"),
        masks=model_params.get("masks", "masks"),
        dataset_storage_device=model_params.get("dataset_storage_device", "cuda"),
        data_format=model_params.get("data_format", "colmap"),
        eval=model_params.get("eval", False),
        llffhold=model_params.get("llffhold", 8),
        add_mask=model_params.get("add_mask", False),
        add_depth=model_params.get("add_depth", False),
        white_background=model_params.get("white_background", False),
        random_background=model_params.get("random_background", False),
        pc_downsampling_ratio=model_params.get("pc_downsampling_ratio", 1),
        global_appearance=model_params.get("global_appearance", False),
        pretrained_checkpoint=model_params.get("pretrained_checkpoint", ""),
        camera_center=model_params.get("camera_center", [0.0, 0.0, 0.0]),
        camera_scale=model_params.get("camera_scale", 1.0),
        dataset_name=model_params.get("dataset_name", ""),
        save_dir=model_params.get("save_dir", ""),
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
    parser.add_argument("--gpu", type=str, default="-1")
    parser.add_argument("--no_vis", action="store_true", default=False, help="Disable visualization saving")
    parser.add_argument("--start_checkpoint", type=str, default=None)
    args = parser.parse_args()

    cfg, model_params, optim_params, pipeline_params = load_vroom_config(args.config)

    source_path = model_params.get("dataset_path")
    if source_path is None:
        parser.error("dataset_path is missing from model_params")
    if not os.path.isabs(source_path):
        cwd_path = os.path.abspath(source_path)
        if os.path.exists(cwd_path):
            source_path = cwd_path
        else:
            source_path = os.path.abspath(os.path.join(os.path.dirname(__file__), source_path))


    if args.gpu != "-1":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        print("Using GPU:", os.environ["CUDA_VISIBLE_DEVICES"])
        torch.cuda.set_device(int(args.gpu))
    else:
        torch.cuda.set_device(0)

    seed_everything(quiet=False)

    dataset_name = model_params.get("dataset_name", "")
    exp_name = model_params.get("save_dir", os.path.basename(args.config).replace(".json", ""))
    run_dir = os.path.join(
        "output", dataset_name, exp_name, datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    os.makedirs(run_dir, exist_ok=True)
    # save a copy of the used configs in the run_dir
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as handle:
        json.dump(cfg, handle, indent=4)

    model_kwargs = model_params.get("model_config", {}).get("kwargs", {})
    anchor_cloud = AnchorCloud(
        voxel_size=model_kwargs.get("voxel_size", None),
        gaussians_per_anchor=model_kwargs.get("gs_per_anchor", 5),
        feature_dim=model_kwargs.get("feat_dim", 32),
    )
    decoder = GaussianDecoder(
        feature_dim=model_kwargs.get("feat_dim", 32),
        anchor_cloud=anchor_cloud,
    ).to(anchor_cloud.device)

    logger = _configure_logging()
    logger.info("Optimizing " + run_dir)

    dataset_args = _build_dataset_args(model_params, source_path, run_dir)
    scene = TrainingScene(
        dataset_args,
        anchor_cloud,
        decoder,
        shuffle=pipeline_params.get("shuffle", True),
        logger=logger,
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
            "output_dir": run_dir,
            "bg_color": scene.background,
        },
        "rendering": {
            "gaussian_type": model_kwargs.get("gaussian_type", "3D"),
            "render_mode": model_kwargs.get("render_mode", "RGB+ED"),
            "tile_size_2dgs": model_kwargs.get("tile_size_2dgs", 8),
        },
        "densifier": cfg.get("densifier", {}),
    }

    orchestrator = TrainingOrchestrator(configs, scene=scene, logger=logger)

    os.makedirs(os.path.join(run_dir, "visualization"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)

    orchestrator.train(scene.getTrainCameras())
    logger.info("\nTraining complete")


if __name__ == "__main__":
    main()
