from __future__ import annotations
import sys
from pathlib import Path

# Add the project root to sys.path to allow imports from any directory
root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

import torch

from vroom_core.utilities.gaussian_renderer.render import prefilter_voxel, render
from vroom_core.utilities.utils.checkpoints import CheckpointManager
from vroom_core.core.models.anchor_field import AnchorCloud, AnchorCloudData
from vroom_core.core.models.decoder import GaussianDecoder
from vroom_core.core.models.semantics import SemanticsManager
from vroom_core.utilities.viewer import viewer_protocol as network_gui


@dataclass
class ViewerPipe:
    add_prefilter: bool = True


def load_checkpoint(model_path: Path, iteration: int) -> tuple[AnchorCloud, GaussianDecoder, str]:
    checkpoints_dir = model_path / "checkpoints"
    iteration_dir = model_path / "point_cloud" / f"iteration_{iteration}"
    
    if checkpoints_dir.exists() and (checkpoints_dir / f"anchor_cloud_{iteration}.ply").exists():
        ply_path = checkpoints_dir / f"anchor_cloud_{iteration}.ply"
        decoder_dir = checkpoints_dir
    else:
        ply_path = iteration_dir / "point_cloud.ply"
        decoder_dir = iteration_dir

    dummy_cloud = AnchorCloud()
    dummy_decoder = GaussianDecoder(
        feature_dim=1,
        anchor_cloud=dummy_cloud,
    )
    manager = CheckpointManager(dummy_cloud, dummy_decoder)
    kwargs = manager.infer_bundle_kwargs(decoder_dir)
    
    gs_per_anchor = kwargs.get("n_offsets", 5)
    anchor_cloud = AnchorCloud(gaussians_per_anchor=gs_per_anchor)
    decoder = GaussianDecoder(
        feature_dim=kwargs.get("feat_dim", 32),
        anchor_cloud=anchor_cloud,
    )
    gaussian_type = kwargs.get("gaussian_type", kwargs.get("gs_attr", "3D"))
    decoder.render_mode = kwargs.get("render_mode", "RGB+ED")
    decoder.tile_size_2dgs = kwargs.get("tile_size_2dgs", 8)
    
    manager = CheckpointManager(anchor_cloud, decoder)
    payload = manager.load_anchor_field(str(ply_path))
    
    seeds = AnchorCloudData(
        anchors_positions=payload["anchor"],
        gaussians_offsets=payload["offset"],
        anchor_features=payload["feature"],
        anchors_log_scales=payload["log_scaling"],
        anchors_rotations=payload["rotation"],
        labels=payload["labels"],
        semantic_manager=None if payload["labels"] is None else SemanticsManager(torch.unique(payload["labels"].view(-1))),
        voxel_size=float(torch.exp(payload["log_scaling"][:, :3]).mean().item()) if payload["log_scaling"].numel() > 0 else 1.0,
    )
    anchor_cloud.set_anchors_cloud(seeds)
    manager.load_decoder(str(decoder_dir))
    decoder.eval()
    return anchor_cloud, decoder, gaussian_type


def serve_checkpoint(model_path: Path, iteration: int, source_path: Path, host: str, port: int, white_background: bool):
    background = torch.ones(3, dtype=torch.float32, device="cuda") if white_background else torch.zeros(3, dtype=torch.float32, device="cuda")
    pipe = ViewerPipe()
    anchor_cloud, decoder, gaussian_type = load_checkpoint(model_path, iteration)

    network_gui.init(host, port)
    print(f"Listening for GUI on {host}:{port}")
    print(f"Loaded checkpoint: {iteration_dir}")
    print(f"Advertising dataset path: {source_path}")

    while True:
        if network_gui.conn is None:
            network_gui.try_connect()
            continue
        try:
            payload = None
            custom_cam, _, pipe.add_prefilter, _ = network_gui.receive()
            if custom_cam is not None:
                visible = prefilter_voxel(custom_cam, anchor_cloud, gaussian_type).squeeze() if pipe.add_prefilter else anchor_cloud.visibility_mask
                decoded_output = decoder.forward_pass(
                    anchor_cloud=anchor_cloud,
                    visible_anchors_mask=visible,
                    camera=custom_cam,
                )
                from vroom_core.core.training.orchestration import prepare_gaussian_space_props
                gaussian_positions, normalized_rotations = prepare_gaussian_space_props(
                    anchor_cloud=anchor_cloud,
                    visible_anchors_mask=visible,
                    negative_opacity_filter=decoded_output["negative_opacity_filter"],
                    rotations_pred=decoded_output["rotations"],
                )

                image = render(
                    viewpoint_camera=custom_cam,
                    decoded_output=decoded_output,
                    gaussian_positions=gaussian_positions,
                    normalized_rotations=normalized_rotations,
                    bg_color=background,
                    gaussian_type=gaussian_type,
                    render_mode=decoder.render_mode,
                    tile_size_2dgs=decoder.tile_size_2dgs,
                    semantics=None,
                )["render"]
                payload = memoryview(
                    (torch.clamp(image, min=0.0, max=1.0) * 255)
                    .byte()
                    .permute(1, 2, 0)
                    .contiguous()
                    .cpu()
                    .numpy()
                )
            network_gui.send(payload, str(source_path))
        except KeyboardInterrupt:
            raise
        except Exception:
            network_gui.conn = None


def _resolve_iteration(model_path: Path, requested: int) -> int:
    if requested >= 0:
        return requested
    checkpoints_dir = model_path / "checkpoints"
    if checkpoints_dir.exists():
        iterations = []
        for path in checkpoints_dir.iterdir():
            if path.is_file() and path.name.startswith("anchor_cloud_") and path.name.endswith(".ply"):
                try:
                    it = int(path.stem.split("_")[-1])
                    iterations.append(it)
                except ValueError:
                    pass
        if iterations:
            return sorted(iterations)[-1]
    point_cloud_root = model_path / "point_cloud"
    iterations = sorted(
        [path for path in point_cloud_root.iterdir() if path.is_dir() and path.name.startswith("iteration_")],
        key=lambda path: int(path.name.split("_")[-1]),
    )
    if not iterations:
        raise FileNotFoundError(f"No checkpoint iterations found under {checkpoints_dir} or {point_cloud_root}")
    return int(iterations[-1].name.split("_")[-1])


def main():
    parser = ArgumentParser(description="View a saved VRoom checkpoint through the network GUI")
    parser.add_argument("--model_path", type=str, required=True, help="Run directory containing point_cloud/")
    parser.add_argument("--iteration", type=int, default=-1, help="Iteration number to load, or -1 for latest")
    parser.add_argument("--source_path", type=str, default=None, help="Original dataset root to advertise to the remote viewer")
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--white_background", action="store_true")
    args = parser.parse_args()

    model_path = Path(args.model_path).resolve()
    iteration = _resolve_iteration(model_path, args.iteration)
    source_path = Path(args.source_path).resolve() if args.source_path is not None else model_path
    serve_checkpoint(model_path, iteration, source_path, args.ip, args.port, args.white_background)


if __name__ == "__main__":
    main()
