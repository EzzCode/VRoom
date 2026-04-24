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

from gaussian_renderer.render import prefilter_voxel, render
from vroom_core.utils.checkpoints import CheckpointManager
from vroom_core.models.facade import GaussianModel
from vroom_core import viewer_protocol as network_gui


@dataclass
class ViewerPipe:
    add_prefilter: bool = True


def load_checkpoint(iteration_dir: Path) -> GaussianModel:
    probe = GaussianModel()
    manager = CheckpointManager(probe)
    kwargs = manager.infer_bundle_kwargs(iteration_dir)
    gaussians = GaussianModel(**kwargs)
    gaussians.load_ply(str(iteration_dir / "point_cloud.ply"))
    gaussians.load_mlp_checkpoints(str(iteration_dir))
    gaussians.set_eval()
    gaussians._anchor_mask = torch.ones(gaussians.get_anchor.shape[0], dtype=torch.bool, device=gaussians.device)
    return gaussians


def serve_checkpoint(iteration_dir: Path, source_path: Path, host: str, port: int, white_background: bool):
    background = torch.ones(3, dtype=torch.float32, device="cuda") if white_background else torch.zeros(3, dtype=torch.float32, device="cuda")
    pipe = ViewerPipe()
    gaussians = load_checkpoint(iteration_dir)

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
                gaussians.set_anchor_mask(custom_cam.camera_center, 1.0)
                visible = prefilter_voxel(custom_cam, gaussians).squeeze() if pipe.add_prefilter else gaussians._anchor_mask
                image = render(custom_cam, gaussians, pipe, background, visible, training=False)["render"]
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


def default_iteration_dir(model_path: Path) -> Path:
    point_cloud_root = model_path / "point_cloud"
    iterations = sorted(
        [path for path in point_cloud_root.iterdir() if path.is_dir() and path.name.startswith("iteration_")],
        key=lambda path: int(path.name.split("_")[-1]),
    )
    if not iterations:
        raise FileNotFoundError(f"No iteration_* directories found under {point_cloud_root}")
    return iterations[-1]


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
    iteration_dir = default_iteration_dir(model_path) if args.iteration == -1 else model_path / "point_cloud" / f"iteration_{args.iteration}"
    if not iteration_dir.exists():
        raise FileNotFoundError(iteration_dir)
    source_path = Path(args.source_path).resolve() if args.source_path is not None else model_path
    serve_checkpoint(iteration_dir, source_path, args.ip, args.port, args.white_background)


if __name__ == "__main__":
    main()
