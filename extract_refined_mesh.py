"""
Extract a mesh for a refined object produced by object_refiner.

The refined model is saved as:
  <refined_objects_dir>/obj_<ID>/06_model/
    point_cloud.ply   (anchor cloud, same format as VRoom checkpoints)
    opacity_mlp.pt
    cov_mlp.pt
    color_mlp.pt
    vroom_bundle.pt   (hyper-params dict)

This script:
  1. Wraps the 06_model/ folder into the layout expected by extract_mesh_inputs.py
     by creating a temporary checkpoints/iter_0/anchor_cloud.ply symlink (or copy).
  2. Renders all cameras from the scene's cameras.json through the refined model
     to produce depth + semantics + RGB renders.
  3. Runs extract_object_meshes.py on the renders using --label <object_id>.
"""

import argparse
import gc
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# ---------------------------------------------------------------------------
# Repo root on sys.path
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from gstrain.vroom_core.core.model.anchor_field import AnchorCloud
from gstrain.vroom_core.utilities.decoder.gaussian_decoder import GaussianDecoder
from gstrain.vroom_core.utilities.utils.utils import CheckpointManager
from gstrain.vroom_core.utilities.utils.render import render as vroom_render


# ---------------------------------------------------------------------------
# Camera helper (copied from extract_mesh_inputs.py)
# ---------------------------------------------------------------------------

def _get_projection_matrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan(fovY / 2)
    tanHalfFovX = math.tan(fovX / 2)
    P = torch.zeros(4, 4)
    P[0, 0] = 1 / tanHalfFovX
    P[1, 1] = 1 / tanHalfFovY
    P[2, 2] = (zfar + znear) / (zfar - znear)
    P[2, 3] = -(2 * zfar * znear) / (zfar - znear)
    P[3, 2] = 1.0
    return P


class DummyView:
    def __init__(self, cam_dict, device="cuda", scale=1.0):
        self.image_width = int(cam_dict["width"] * scale)
        self.image_height = int(cam_dict["height"] * scale)
        self.fx = float(cam_dict.get("fx", self.image_width)) * scale
        self.fy = float(cam_dict.get("fy", self.image_height)) * scale
        self.cx = self.image_width / 2.0
        self.cy = self.image_height / 2.0
        self.FoVx = 2 * math.atan(self.image_width / (2 * self.fx))
        self.FoVy = 2 * math.atan(self.image_height / (2 * self.fy))

        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = np.array(cam_dict["rotation"])
        c2w[:3, 3] = np.array(cam_dict["position"])
        w2c = np.linalg.inv(c2w)

        self.world_view_transform = torch.tensor(w2c, dtype=torch.float32).transpose(0, 1).to(device)
        self.projection_matrix = (
            _get_projection_matrix(0.01, 100.0, self.FoVx, self.FoVy).transpose(0, 1).to(device)
        )
        self.camera_center = torch.tensor(cam_dict["position"], dtype=torch.float32).to(device)


# ---------------------------------------------------------------------------
# Render helper
# ---------------------------------------------------------------------------

def render_view(anchor_cloud, decoder, view, num_classes, device):
    visible_mask = torch.ones(anchor_cloud.anchors_positions.shape[0], dtype=torch.bool, device=device)
    decoded_dict = decoder.forward_pass(anchor_cloud, visible_mask, view)

    xyz_positions = anchor_cloud.instantiate_gaussian_positions(
        visible_mask=visible_mask,
        negative_opacity_filter=decoded_dict["negative_opacity_filter"],
    )
    normalized_rots = F.normalize(decoded_dict["rotations"], dim=-1)

    gaussian_semantics = None
    if anchor_cloud.semantic_labels is not None and num_classes > 0:
        visible_labels = anchor_cloud.semantic_labels[visible_mask].squeeze(-1).long()
        one_hot = F.one_hot(visible_labels, num_classes=num_classes).float()
        expanded = one_hot.unsqueeze(1).expand(-1, decoder.number_gaussians_per_anchor, -1)
        gaussian_semantics = expanded[decoded_dict["negative_opacity_filter"]]

    bg = torch.ones(3, dtype=torch.float32, device=device)
    pkg = vroom_render(
        viewpoint_camera=view,
        decoded_output={
            "color": decoded_dict["color"],
            "opacity": decoded_dict["opacity"],
            "scaling": decoded_dict["scaling"],
        },
        gaussian_positions=xyz_positions,
        normalized_rotations=normalized_rots,
        background_color=bg,
        gaussian_type="2D",
        tile_Size=8,
        semantics=gaussian_semantics,
    )

    return {
        "render": pkg["render"],
        "depth": pkg["render_depth"],
        "semantics": pkg["render_semantics"].argmax(dim=0) if gaussian_semantics is not None else None,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract a mesh for a single refined object (object_refiner output)."
    )
    parser.add_argument(
        "--refined_model_dir",
        required=True,
        help="Path to the refined object's 06_model/ directory "
             "(e.g. test_30k_prototype_output/refined_objects/obj_1/06_model)",
    )
    parser.add_argument(
        "--cameras_json",
        required=True,
        help="Path to the scene cameras.json "
             "(e.g. test_30k_prototype_output/training/gs_model/<date>/cameras.json)",
    )
    parser.add_argument(
        "--object_id",
        type=int,
        required=True,
        help="Object label ID to extract the mesh for.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Root output directory for this object's mesh. "
             "Defaults to <refined_model_dir>/../../07_mesh",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=128,
        help="TSDF grid resolution (default: 128).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_dir = Path(args.refined_model_dir).resolve()
    cameras_json = Path(args.cameras_json).resolve()

    if args.output_dir is None:
        output_dir = model_dir.parent.parent / "07_mesh"
    else:
        output_dir = Path(args.output_dir).resolve()

    mesh_inputs_dir = output_dir / "mesh_inputs"
    mesh_objects_dir = output_dir / "mesh_objects"

    print(f"Refined model : {model_dir}")
    print(f"Cameras       : {cameras_json}")
    print(f"Output        : {output_dir}")

    # ------------------------------------------------------------------
    # 1. Load vroom_bundle.pt for hyper-params
    # ------------------------------------------------------------------
    bundle_path = model_dir / "vroom_bundle.pt"
    if not bundle_path.exists():
        raise FileNotFoundError(f"vroom_bundle.pt not found in {model_dir}")

    bundle = torch.load(str(bundle_path), map_location="cpu")
    n_offsets = int(bundle["n_offsets"])
    feature_dim = int(bundle["feature_dim"])
    tile_size = int(bundle.get("tile_Size", 8))
    knn_k = int(bundle.get("knn_k", 4))
    knn_chunk_size = int(bundle.get("knn_chunk_size", 2048))
    min_quantization_size = float(bundle.get("min_quantization_size", 1e-6))
    print(f"Bundle: n_offsets={n_offsets}, feature_dim={feature_dim}, tile_Size={tile_size}")

    # ------------------------------------------------------------------
    # 2. Build model
    # ------------------------------------------------------------------
    anchor_cloud = AnchorCloud(
        gaussians_per_anchor=n_offsets,
        feature_dim=feature_dim,
        knn_k=knn_k,
        knn_chunk_size=knn_chunk_size,
        min_quantization_size=min_quantization_size,
        device=device,
    )
    decoder = GaussianDecoder(feature_dim=feature_dim, anchor_cloud=anchor_cloud).to(device)
    chkp_manager = CheckpointManager(anchor_cloud, decoder)

    ply_path = model_dir / "point_cloud.ply"
    if not ply_path.exists():
        raise FileNotFoundError(f"point_cloud.ply not found in {model_dir}")

    print(f"Loading checkpoint from {ply_path} ...")
    payload = chkp_manager.load_anchor_field(str(ply_path))

    anchor_cloud.anchors_positions = torch.nn.Parameter(payload["anchor"].to(device))
    anchor_cloud.gaussians_offsets = torch.nn.Parameter(payload["offset"].to(device))
    anchor_cloud.anchor_features = torch.nn.Parameter(payload["feature"].to(device))
    anchor_cloud.anchors_log_scales = torch.nn.Parameter(payload["log_scaling"].to(device))
    anchor_cloud.anchors_rotations = torch.nn.Parameter(payload["rotation"].to(device))

    if payload.get("labels") is not None:
        anchor_cloud.semantic_labels = payload["labels"].to(device)
        num_classes = int(anchor_cloud.semantic_labels.max().item()) + 1
        print(f"Labels: {num_classes} classes")
    else:
        anchor_cloud.semantic_labels = None
        num_classes = 0

    chkp_manager.load_decoder(str(model_dir))
    print("Model loaded.")

    # ------------------------------------------------------------------
    # 3. Load cameras
    # ------------------------------------------------------------------
    with open(cameras_json, "r") as f:
        camera_data = json.load(f)
    if isinstance(camera_data, dict):
        camera_data = list(camera_data.values())
    print(f"Cameras: {len(camera_data)}")

    # ------------------------------------------------------------------
    # 4. Render all views → mesh_inputs_dir
    # ------------------------------------------------------------------
    for sub in ["renders", "raw_depth", "semantic"]:
        target = mesh_inputs_dir / sub
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True, exist_ok=True)

    with open(mesh_inputs_dir / "cameras.json", "w") as f:
        json.dump(camera_data, f, indent=4)

    print("Rendering frames...")
    t0 = time.time()
    with torch.no_grad():
        for idx, cam_dict in enumerate(camera_data):
            view = DummyView(cam_dict, device=device)
            pkg = render_view(anchor_cloud, decoder, view, num_classes, device)

            # RGB
            rgb_t = pkg["render"].clamp(0, 1).detach().cpu().contiguous()
            rgb = np.from_dlpack(rgb_t).transpose(1, 2, 0)
            Image.fromarray((rgb * 255).astype(np.uint8)).save(
                mesh_inputs_dir / "renders" / f"{idx:05d}.png"
            )

            # Depth
            depth_t = pkg["depth"].detach().cpu().squeeze(0).contiguous()
            depth = np.from_dlpack(depth_t).copy()
            np.save(mesh_inputs_dir / "raw_depth" / f"{idx:05d}.npy", depth)

            # Semantics
            if pkg["semantics"] is not None:
                sem_t = pkg["semantics"].detach().cpu().to(torch.uint8).contiguous()
                sem = np.from_dlpack(sem_t).copy()
                Image.fromarray(sem).save(mesh_inputs_dir / "semantic" / f"{idx:05d}.png")

            if idx % 20 == 0 or idx == len(camera_data) - 1:
                print(f"  Frame {idx + 1}/{len(camera_data)}")

            del pkg
            gc.collect()
            torch.cuda.empty_cache()

    print(f"Rendering done in {time.time() - t0:.1f}s")

    # ------------------------------------------------------------------
    # 5. Run mesh extraction
    # ------------------------------------------------------------------
    mesh_script = REPO_ROOT / "mesh_generation" / "extract_object_meshes.py"
    mesh_objects_dir.mkdir(parents=True, exist_ok=True)

    import subprocess
    cmd = [
        sys.executable, str(mesh_script),
        "--input_dir", str(mesh_inputs_dir),
        "--output_dir", str(mesh_objects_dir),
        "--label", str(args.object_id),
        "--resolution", str(args.resolution),
    ]
    print("\n=== Running mesh extraction ===")
    print(" ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"Mesh extraction failed (exit code {result.returncode})")

    print(f"\nMesh saved to: {mesh_objects_dir}")


if __name__ == "__main__":
    main()
