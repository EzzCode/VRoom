"""Checkpoint and PLY serialization for the VRoom core."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from plyfile import PlyData, PlyElement

from vroom_core.utils.runtime import ensure_directory


class CheckpointManager:
    def __init__(self, model) -> None:
        self.model = model

    def save_anchor_field(self, path: str) -> None:
        field = self.model.field
        ensure_directory(os.path.dirname(path))
        anchor = field.anchors_positions.detach().cpu().numpy()
        offsets = field.gaussians_offsets.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        features = field.anchor_features.detach().cpu().numpy()
        scales = field.anchors_log_scales.detach().cpu().numpy()
        rotations = field.anchors_rotations.detach().cpu().numpy()
        labels = field.semantic_labels.detach().cpu().numpy().reshape(-1, 1) if field.semantic_labels is not None else np.zeros((anchor.shape[0], 1), dtype=np.uint8)

        names = ["x", "y", "z"]
        names += [f"f_offset_{idx}" for idx in range(offsets.shape[1])]
        names += [f"f_anchor_feat_{idx}" for idx in range(features.shape[1])]
        names += [f"scale_{idx}" for idx in range(scales.shape[1])]
        names += [f"rot_{idx}" for idx in range(rotations.shape[1])]
        names.append("label")
        dtype = [(name, "f4") for name in names]
        rows = np.concatenate([anchor, offsets, features, scales, rotations, labels], axis=1)
        elements = np.empty(anchor.shape[0], dtype=dtype)
        elements[:] = list(map(tuple, rows))
        PlyData([PlyElement.describe(elements, "vertex")], obj_info=[f"num_anchor {anchor.shape[0]:.6f}"]).write(path)

    def load_anchor_field(self, path: str) -> dict:
        ply = PlyData.read(path).elements[0]
        anchor = np.stack([np.asarray(ply["x"]), np.asarray(ply["y"]), np.asarray(ply["z"])], axis=1).astype(np.float32)
        features = self._stack_prefixed(ply, "f_anchor_feat_")
        offset_flat = self._stack_prefixed(ply, "f_offset_")
        scales = self._stack_prefixed(ply, "scale_")
        rotations = self._stack_prefixed(ply, "rot_")
        labels = np.asarray(ply["label"])[..., None].astype(np.int64) if "label" in [prop.name for prop in ply.properties] else None
        return {
            "anchor": torch.tensor(anchor, dtype=torch.float32, device=self.model.device),
            "offset": torch.tensor(offset_flat.reshape(anchor.shape[0], 3, -1), dtype=torch.float32, device=self.model.device).transpose(1, 2).contiguous(),
            "feature": torch.tensor(features, dtype=torch.float32, device=self.model.device),
            "log_scaling": torch.tensor(scales, dtype=torch.float32, device=self.model.device),
            "rotation": torch.tensor(rotations, dtype=torch.float32, device=self.model.device),
            "labels": None if labels is None else torch.tensor(labels, dtype=torch.long, device=self.model.device),
        }

    def save_explicit(self, path: str) -> None:
        ensure_directory(os.path.dirname(path))
        explicit = self.model.materialize_explicit()
        dtype = [("x", "f4"), ("y", "f4"), ("z", "f4")]
        dtype += [(f"f_dc_{idx}", "f4") for idx in range(3)]
        dtype += [("opacity", "f4")]
        dtype += [(f"scale_{idx}", "f4") for idx in range(3)]
        dtype += [(f"rot_{idx}", "f4") for idx in range(4)]
        packed = np.concatenate(
            [
                explicit["xyz"],
                explicit["rgb"],
                explicit["opacity"],
                explicit["scaling"],
                explicit["rotation"],
            ],
            axis=1,
        )
        elements = np.empty(explicit["xyz"].shape[0], dtype=dtype)
        elements[:] = list(map(tuple, packed))
        PlyData([PlyElement.describe(elements, "vertex")]).write(path)

    def load_explicit(self, path: str) -> dict:
        ply = PlyData.read(path).elements[0]
        xyz = np.stack([np.asarray(ply["x"]), np.asarray(ply["y"]), np.asarray(ply["z"])], axis=1).astype(np.float32)
        rgb = np.stack([np.asarray(ply["f_dc_0"]), np.asarray(ply["f_dc_1"]), np.asarray(ply["f_dc_2"])], axis=1).astype(np.float32)
        opacity = np.asarray(ply["opacity"])[..., None].astype(np.float32)
        scaling = self._stack_prefixed(ply, "scale_").astype(np.float32)
        rotation = self._stack_prefixed(ply, "rot_").astype(np.float32)
        return {
            "xyz": torch.tensor(xyz, dtype=torch.float32, device=self.model.device),
            "rgb": torch.tensor(rgb, dtype=torch.float32, device=self.model.device),
            "opacity": torch.tensor(opacity, dtype=torch.float32, device=self.model.device),
            "scaling": torch.tensor(scaling, dtype=torch.float32, device=self.model.device),
            "rotation": torch.tensor(rotation, dtype=torch.float32, device=self.model.device),
        }

    def save_decoder(self, path: str) -> None:
        ensure_directory(path)
        self.model.set_eval()
        torch.jit.trace(self.model.decoder.opacity_head, torch.rand(1, self.model.feat_dim + self.model.view_dim, device=self.model.device)).save(
            os.path.join(path, "opacity_mlp.pt")
        )
        torch.jit.trace(self.model.decoder.covariance_head, torch.rand(1, self.model.feat_dim + self.model.view_dim, device=self.model.device)).save(
            os.path.join(path, "cov_mlp.pt")
        )
        torch.jit.trace(
            self.model.decoder.color_head,
            torch.rand(1, self.model.feat_dim + self.model.view_dim + self.model.appearance_dim, device=self.model.device),
        ).save(os.path.join(path, "color_mlp.pt"))
        if self.model.decoder.appearance is not None:
            torch.jit.trace(self.model.decoder.appearance, torch.zeros((1,), dtype=torch.long, device=self.model.device)).save(
                os.path.join(path, "embedding_appearance.pt")
            )
        torch.save(
            {
                "n_offsets": self.model.n_offsets,
                "feat_dim": self.model.feat_dim,
                "view_dim": self.model.view_dim,
                "appearance_dim": self.model.appearance_dim,
                "gs_attr": self.model.gs_attr,
                "render_mode": self.model.render_mode,
                "tile_size_2dgs": self.model.tile_size_2dgs,
            },
            os.path.join(path, "vroom_bundle.pt"),
        )
        self.model.set_train()

    def load_decoder(self, path: str) -> None:
        self.model.decoder.opacity_head = torch.jit.load(os.path.join(path, "opacity_mlp.pt")).to(self.model.device)
        self.model.decoder.covariance_head = torch.jit.load(os.path.join(path, "cov_mlp.pt")).to(self.model.device)
        self.model.decoder.color_head = torch.jit.load(os.path.join(path, "color_mlp.pt")).to(self.model.device)
        appearance_path = os.path.join(path, "embedding_appearance.pt")
        if self.model.appearance_dim > 0 and os.path.exists(appearance_path):
            self.model.decoder.appearance = torch.jit.load(appearance_path).to(self.model.device)

    def infer_bundle_kwargs(self, iteration_dir: Path) -> dict:
        bundle_path = iteration_dir / "vroom_bundle.pt"
        if bundle_path.exists():
            return torch.load(bundle_path, map_location="cpu")

        temp = self.load_anchor_field(str(iteration_dir / "point_cloud.ply"))
        opacity_mlp = torch.jit.load(str(iteration_dir / "opacity_mlp.pt"), map_location="cpu")
        color_mlp = torch.jit.load(str(iteration_dir / "color_mlp.pt"), map_location="cpu")
        opacity_params = dict(opacity_mlp.named_parameters())
        color_params = dict(color_mlp.named_parameters())
        first_opacity_weight = next(param for name, param in opacity_params.items() if name.endswith("weight"))
        last_opacity_weight = list(opacity_params.values())[-2]
        first_color_weight = next(param for name, param in color_params.items() if name.endswith("weight"))
        feat_dim = temp["feature"].shape[1]
        view_dim = first_opacity_weight.shape[1] - feat_dim
        appearance_dim = first_color_weight.shape[1] - feat_dim - view_dim
        return {
            "n_offsets": int(last_opacity_weight.shape[0]),
            "feat_dim": int(feat_dim),
            "view_dim": int(view_dim),
            "appearance_dim": int(appearance_dim),
            "gs_attr": "3D",
            "render_mode": "RGB+ED",
            "tile_size_2dgs": 8,
        }

    def _stack_prefixed(self, element, prefix: str) -> np.ndarray:
        names = sorted([prop.name for prop in element.properties if prop.name.startswith(prefix)], key=lambda name: int(name.split("_")[-1]))
        if not names:
            return np.zeros((len(element["x"]), 0), dtype=np.float32)
        return np.stack([np.asarray(element[name]) for name in names], axis=1).astype(np.float32)

