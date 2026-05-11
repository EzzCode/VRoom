"""Facade that exposes the renderer/trainer contract over the VRoom core."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from vroom_core.models.anchor_field import AnchorField, AnchorSeedBuilder, AnchorSeeds
from vroom_core.utils.checkpoints import CheckpointManager
from vroom_core.models.decoder import GaussianDecoder
from vroom_core.models.density import DensificationController
from vroom_core.models.semantics import SemanticCodec
from vroom_core.utils.runtime import exponential_lr_schedule



class GaussianModel(nn.Module):
    def __init__(
        self,
        n_offsets: int = 5,
        feat_dim: int = 32,
        view_dim: int = 3,
        appearance_dim: int = 0,
        voxel_size: float = -1.0,
        gs_attr: str = "3D",
        render_mode: str = "RGB+ED",
        tile_size_2dgs: int = 8,
        device: str = "cuda",
    ) -> None:
        super().__init__()
        if gs_attr not in {"2D", "3D"}:
            raise ValueError(f"gs_attr must be '2D' or '3D', got {gs_attr!r}")
        self.device = torch.device(device)
        self.n_offsets = n_offsets
        self.feat_dim = feat_dim
        self.view_dim = view_dim
        self.appearance_dim = appearance_dim
        self.voxel_size = float(voxel_size)
        self.gs_attr = gs_attr
        self.render_mode = render_mode
        self.tile_size_2dgs = tile_size_2dgs
        self.color_dim = 3
        self.weed_ratio = 0.0
        self.spatial_lr_scale = 1.0
        self._grad_clip_norm: Optional[float] = None
        self._training_args = None
        self._lr_schedulers = {}
        self.optimizer = None

        self.field = AnchorField(device=device)
        self.decoder = GaussianDecoder(feat_dim, view_dim, appearance_dim, n_offsets, device=device)
        self.density = DensificationController(n_offsets, device=device)
        self.checkpoints = CheckpointManager(self)

    @property
    def _anchor(self):
        return self.field.anchor

    @_anchor.setter
    def _anchor(self, value):
        self.field.anchor = value

    @property
    def _offset(self):
        return self.field.offset

    @_offset.setter
    def _offset(self, value):
        self.field.offset = value

    @property
    def _anchor_feat(self):
        return self.field.feature

    @_anchor_feat.setter
    def _anchor_feat(self, value):
        self.field.feature = value

    @property
    def _scaling(self):
        return self.field.log_scaling

    @_scaling.setter
    def _scaling(self, value):
        self.field.log_scaling = value

    @property
    def _rotation(self):
        return self.field.raw_rotation

    @_rotation.setter
    def _rotation(self, value):
        self.field.raw_rotation = value

    @property
    def _anchor_mask(self):
        return self.field.visible

    @_anchor_mask.setter
    def _anchor_mask(self, value):
        self.field.visible = value

    @property
    def label_ids(self):
        return self.field.label_ids

    @label_ids.setter
    def label_ids(self, value):
        self.field.label_ids = value

    @property
    def id_encoder(self):
        return self.field.codec

    @id_encoder.setter
    def id_encoder(self, value):
        self.field.codec = value

    @property
    def embedding_appearance(self):
        return self.decoder.appearance

    @embedding_appearance.setter
    def embedding_appearance(self, value):
        self.decoder.appearance = value

    @property
    def anchors(self) -> torch.Tensor:
        return self.field.anchor

    @property
    def get_anchor(self) -> torch.Tensor:
        return self.field.anchor

    @property
    def anchor_features(self) -> torch.Tensor:
        return self.field.feature

    @property
    def get_anchor_feat(self) -> torch.Tensor:
        return self.field.feature

    @property
    def offsets(self) -> torch.Tensor:
        return self.field.offset

    @property
    def get_offset(self) -> torch.Tensor:
        return self.field.offset

    @property
    def scaling(self) -> torch.Tensor:
        return torch.exp(self.field.log_scaling)

    @property
    def get_scaling(self) -> torch.Tensor:
        return self.scaling

    @property
    def rotation(self) -> torch.Tensor:
        return F.normalize(self.field.raw_rotation, dim=-1)

    @property
    def get_rotation(self) -> torch.Tensor:
        return self.rotation

    @property
    def semantics(self) -> torch.Tensor:
        if self.field.codec is None or self.field.label_ids is None:
            return torch.zeros((self.field.anchor.shape[0], 1), dtype=torch.float32, device=self.device)
        return self.field.codec.transform(self.field.label_ids.view(-1))

    @property
    def get_semantic(self) -> torch.Tensor:
        return self.semantics

    def initialize_anchors(self, *args, **kwargs):
        if not args:
            raise TypeError("initialize_anchors() requires at least one argument.")
        source = args[0]
        if hasattr(source, "points"):
            points = torch.from_numpy(source.points).float()
            labels = torch.from_numpy(source.label_ids).long() if getattr(source, "label_ids", None) is not None else None
            spatial_lr_scale = float(args[1]) if len(args) >= 2 else 1.0
            logger = args[3] if len(args) >= 4 else kwargs.get("logger", None)
        else:
            points = source
            labels = args[1] if len(args) >= 2 else None
            spatial_lr_scale = float(args[2]) if len(args) >= 3 else float(kwargs.get("spatial_lr_scale", 1.0))
            logger = args[3] if len(args) >= 4 else kwargs.get("logger", None)
        
        builder = AnchorSeedBuilder(self.n_offsets, self.feat_dim, self.voxel_size if self.voxel_size > 0 else kwargs.get("voxel_size", -1.0), device=str(self.device))
        seeds = builder.build(points, labels, logger)
        self.field.replace(seeds)
        self.spatial_lr_scale = spatial_lr_scale

    def configure_appearance(self, num_cameras: int):
        self.decoder.configure_appearance(num_cameras)

    set_appearance = configure_appearance

    def set_anchor_mask(self, *_args):
        self._anchor_mask = torch.ones(self.field.anchor.shape[0], dtype=torch.bool, device=self.device)

    def generate_neural_gaussians(self, viewpoint_camera, visible_mask: Optional[torch.Tensor] = None, training: bool = True):
        if visible_mask is None:
            visible_mask = torch.ones(self.field.anchor.shape[0], dtype=torch.bool, device=self.device)
        decoded = self.decoder.decode(self.field, viewpoint_camera, visible_mask, training)
        return (
            decoded.xyz,
            decoded.offsets,
            decoded.color,
            decoded.opacity,
            decoded.scaling,
            decoded.rotation,
            decoded.selection_mask,
            decoded.semantics,
        )

    def setup_training(self, training_args, grad_clip_norm: Optional[float] = None):
        self._training_args = training_args
        self._grad_clip_norm = grad_clip_norm
        self.rebuild_optimizer()
        self.density.reset(self.field.anchor.shape[0])

    training_setup = setup_training

    # ---------- parameter-group lookup helpers -------
    _FIELD_GROUP_NAMES = {"anchor", "offset", "feature", "scaling", "rotation"}

    def _is_field_group(self, name: str) -> bool:
        """True for optimizer groups that track per-anchor tensors (not MLPs)."""
        return name in self._FIELD_GROUP_NAMES

    # ---------- optimizer state preservation ----------

    def extend_optimizer_state(self, extension_dict: dict[str, torch.Tensor]) -> dict[str, nn.Parameter]:
        """Grow optimizer param-groups by concatenating *extension_dict* tensors.

        Preserves Adam momentum (exp_avg, exp_avg_sq) for existing entries and
        initializes zeros for newly appended entries.  MLP / embedding groups
        are left untouched.
        """
        updated: dict[str, nn.Parameter] = {}
        for group in self.optimizer.param_groups:
            if not self._is_field_group(group["name"]):
                continue
            assert len(group["params"]) == 1
            ext = extension_dict[group["name"]]
            old_param = group["params"][0]
            state = self.optimizer.state.get(old_param)
            if state is not None:
                state["exp_avg"] = torch.cat([state["exp_avg"], torch.zeros_like(ext)], dim=0)
                state["exp_avg_sq"] = torch.cat([state["exp_avg_sq"], torch.zeros_like(ext)], dim=0)
                del self.optimizer.state[old_param]
                new_param = nn.Parameter(torch.cat([old_param, ext], dim=0).requires_grad_(True))
                group["params"][0] = new_param
                self.optimizer.state[new_param] = state
            else:
                new_param = nn.Parameter(torch.cat([old_param, ext], dim=0).requires_grad_(True))
                group["params"][0] = new_param
            updated[group["name"]] = group["params"][0]
        return updated

    def prune_optimizer_state(self, keep_mask: torch.Tensor) -> dict[str, nn.Parameter]:
        """Shrink optimizer param-groups to only the entries where *keep_mask* is True.

        Preserves Adam momentum for surviving entries.  Clamps the upper-half of
        the 'scaling' group to 0.05 to avoid explosion after pruning.
        """
        updated: dict[str, nn.Parameter] = {}
        for group in self.optimizer.param_groups:
            if not self._is_field_group(group["name"]):
                continue
            old_param = group["params"][0]
            state = self.optimizer.state.get(old_param)
            trimmed = old_param[keep_mask].detach().clone()
            if group["name"] == "scaling":
                upper = trimmed[:, 3:]
                upper[upper > 0.05] = 0.05
                trimmed[:, 3:] = upper
            if state is not None:
                state["exp_avg"] = state["exp_avg"][keep_mask]
                state["exp_avg_sq"] = state["exp_avg_sq"][keep_mask]
                del self.optimizer.state[old_param]
                new_param = nn.Parameter(trimmed.requires_grad_(True))
                group["params"][0] = new_param
                self.optimizer.state[new_param] = state
            else:
                new_param = nn.Parameter(trimmed.requires_grad_(True))
                group["params"][0] = new_param
            updated[group["name"]] = group["params"][0]
        return updated

    def rebuild_optimizer(self):
        if self._training_args is None:
            return
        args = self._training_args
        groups = [
            {"params": [self.field.anchor], "lr": args.position_lr_init * self.spatial_lr_scale, "name": "anchor"},
            {"params": [self.field.offset], "lr": args.offset_lr_init * self.spatial_lr_scale, "name": "offset"},
            {"params": [self.field.feature], "lr": args.feature_lr, "name": "feature"},
            {"params": [self.field.log_scaling], "lr": args.scaling_lr, "name": "scaling"},
            {"params": [self.field.raw_rotation], "lr": args.rotation_lr, "name": "rotation"},
            {"params": self.decoder.opacity_head.parameters(), "lr": args.mlp_opacity_lr_init, "name": "opacity_head"},
            {"params": self.decoder.covariance_head.parameters(), "lr": args.mlp_cov_lr_init, "name": "covariance_head"},
            {"params": self.decoder.color_head.parameters(), "lr": args.mlp_color_lr_init, "name": "color_head"},
        ]
        if self.decoder.appearance is not None:
            groups.append({"params": self.decoder.appearance.parameters(), "lr": args.appearance_lr_init, "name": "appearance"})
        self.optimizer = torch.optim.Adam(groups, lr=0.0, eps=1e-15)
        self._lr_schedulers = {
            "anchor": exponential_lr_schedule(args.position_lr_init * self.spatial_lr_scale, args.position_lr_final * self.spatial_lr_scale, lr_delay_mult=args.position_lr_delay_mult, max_steps=args.position_lr_max_steps),
            "offset": exponential_lr_schedule(args.offset_lr_init * self.spatial_lr_scale, args.offset_lr_final * self.spatial_lr_scale, lr_delay_mult=args.offset_lr_delay_mult, max_steps=args.offset_lr_max_steps),
            "opacity_head": exponential_lr_schedule(args.mlp_opacity_lr_init, args.mlp_opacity_lr_final, lr_delay_mult=args.mlp_opacity_lr_delay_mult, max_steps=args.mlp_opacity_lr_max_steps),
            "covariance_head": exponential_lr_schedule(args.mlp_cov_lr_init, args.mlp_cov_lr_final, lr_delay_mult=args.mlp_cov_lr_delay_mult, max_steps=args.mlp_cov_lr_max_steps),
            "color_head": exponential_lr_schedule(args.mlp_color_lr_init, args.mlp_color_lr_final, lr_delay_mult=args.mlp_color_lr_delay_mult, max_steps=args.mlp_color_lr_max_steps),
        }
        if self.decoder.appearance is not None:
            self._lr_schedulers["appearance"] = exponential_lr_schedule(args.appearance_lr_init, args.appearance_lr_final, lr_delay_mult=args.appearance_lr_delay_mult, max_steps=args.appearance_lr_max_steps)

    def step_learning_rate(self, iteration: int):
        for group in self.optimizer.param_groups:
            scheduler = self._lr_schedulers.get(group["name"])
            if scheduler is not None:
                group["lr"] = scheduler(iteration)

    update_learning_rate = step_learning_rate

    def clip_gradients(self):
        if self._grad_clip_norm is not None and self._grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self._grad_clip_norm)

    def training_statis(self, opt, render_pkg: dict, width: int, height: int):
        self.density.accumulate(render_pkg, opt, width, height)

    def run_densify(self, opt, iteration: int):
        self.density.densify(self, opt, iteration)

    def prune_anchor(self, mask: torch.Tensor):
        keep = ~mask
        if self.optimizer is not None and self._training_args is not None:
            updated = self.prune_optimizer_state(keep)
            self.field.anchor = updated["anchor"]
            self.field.offset = updated["offset"]
            self.field.feature = updated["feature"]
            self.field.log_scaling = updated["scaling"]
            self.field.raw_rotation = updated["rotation"]
        else:
            self.field.prune(mask)
        if self.field.label_ids is not None:
            self.field.label_ids = self.field.label_ids[keep]
            if self.field.codec is not None:
                self.field.codec.fit(self.field.label_ids.view(-1))
        self.field.visible = torch.ones(self.field.anchor.shape[0], dtype=torch.bool, device=self.device)
        return mask

    def save_ply(self, path: str):
        self.checkpoints.save_anchor_field(path)

    def load_ply(self, path: str):
        payload = self.checkpoints.load_anchor_field(path)
        seeds = AnchorSeeds(
            anchors=payload["anchor"],
            offsets=payload["offset"],
            features=payload["feature"],
            log_scaling=payload["log_scaling"],
            rotations=payload["rotation"],
            labels=payload["labels"],
            codec=None if payload["labels"] is None else SemanticCodec.from_labels(payload["labels"].view(-1)),
            voxel_size=float(torch.exp(payload["log_scaling"][:, :3]).mean().item()) if payload["log_scaling"].numel() > 0 else 1.0,
        )
        self.field.replace(seeds)
    def save_mlp_checkpoints(self, path: str):
        self.checkpoints.save_decoder(path)

    def load_mlp_checkpoints(self, path: str):
        self.checkpoints.load_decoder(path)

    def set_eval(self):
        self.decoder.opacity_head.eval()
        self.decoder.covariance_head.eval()
        self.decoder.color_head.eval()
        if self.decoder.appearance is not None:
            self.decoder.appearance.eval()

    eval = set_eval

    def set_train(self):
        self.decoder.opacity_head.train()
        self.decoder.covariance_head.train()
        self.decoder.color_head.train()
        if self.decoder.appearance is not None:
            self.decoder.appearance.train()

    train_mode = set_train

    def clean(self):
        self.density.cleanup()
        torch.cuda.empty_cache()

    # ---------- training state snapshot (capture / restore) ----------

    def capture(self) -> dict:
        """Snapshot all training state for mid-training checkpoint resumption."""
        state = {
            "anchor": self.field.anchor.detach(),
            "offset": self.field.offset.detach(),
            "feature": self.field.feature.detach(),
            "log_scaling": self.field.log_scaling.detach(),
            "rotation": self.field.raw_rotation.detach(),
            "label_ids": self.field.label_ids,
            "spatial_lr_scale": self.spatial_lr_scale,
        }
        if self.optimizer is not None:
            state["optimizer"] = self.optimizer.state_dict()
        state["opacity_head"] = self.decoder.opacity_head.state_dict()
        state["covariance_head"] = self.decoder.covariance_head.state_dict()
        state["color_head"] = self.decoder.color_head.state_dict()
        if self.decoder.appearance is not None:
            state["appearance"] = self.decoder.appearance.state_dict()
        return state

    def restore(self, state: dict, training_args=None):
        """Restore training state from a previously captured snapshot."""
        seeds = AnchorSeeds(
            anchors=state["anchor"],
            offsets=state["offset"],
            features=state["feature"],
            log_scaling=state["log_scaling"],
            rotations=state["rotation"],
            labels=state.get("label_ids"),
            codec=None if state.get("label_ids") is None else SemanticCodec.from_labels(state["label_ids"].view(-1)),
            voxel_size=self.field.voxel_size,
        )
        self.field.replace(seeds)
        self.spatial_lr_scale = state.get("spatial_lr_scale", 1.0)
        self.decoder.opacity_head.load_state_dict(state["opacity_head"])
        self.decoder.covariance_head.load_state_dict(state["covariance_head"])
        self.decoder.color_head.load_state_dict(state["color_head"])
        if "appearance" in state and self.decoder.appearance is not None:
            self.decoder.appearance.load_state_dict(state["appearance"])
        if training_args is not None:
            self.setup_training(training_args)
        if "optimizer" in state and self.optimizer is not None:
            try:
                self.optimizer.load_state_dict(state["optimizer"])
            except Exception:
                pass  # shape mismatch after grow/prune - optimizer is already fresh
