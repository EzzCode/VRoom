"""Densification policy and training-state bookkeeping.

Supports both 'mean' and 'max' strategies for gradient accumulation
and opacity-based pruning.  Uses multi-scale hierarchical growing with
stochastic candidate selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from typing import Optional

import torch


@dataclass
class DensityState:
    anchor_opacity: torch.Tensor 
    anchor_visits: torch.Tensor # counts how many times the anchor was drawn on the screen
    offset_gradients: torch.Tensor
    offset_visits: torch.Tensor # counts how many times and offeset was drawn on the screen
    offset_opacity: torch.Tensor 
    max_radii: torch.Tensor

    @classmethod
    def allocate(cls, n_anchors: int, n_offsets: int, device) -> "DensityState":
        return cls(
            anchor_opacity=torch.zeros((n_anchors, 1), device=device),
            anchor_visits=torch.zeros((n_anchors, 1), device=device),
            offset_gradients=torch.zeros((n_anchors * n_offsets, 1), device=device),
            offset_visits=torch.zeros((n_anchors * n_offsets, 1), device=device),
            offset_opacity=torch.zeros((n_anchors * n_offsets, 1), device=device),
            max_radii=torch.zeros((n_anchors * n_offsets,), device=device),
        )


class DensificationController:
    def __init__(self, n_offsets: int, device: str = "cuda") -> None:
        self.n_offsets = n_offsets
        self.device = torch.device(device)
        self.state = DensityState.allocate(0, n_offsets, self.device)

    def reset(self, n_anchors: int) -> None:
        self.state = DensityState.allocate(n_anchors, self.n_offsets, self.device)

    # ------------------------------------------------------------------
    # Gradient / opacity accumulation
    # ------------------------------------------------------------------

    def accumulate(self, render_pkg: dict, opt, width: int, height: int) -> None:
        """Aggregate per-step statistics for the densification decision.

        Supports both ``pruning_type in {'mean', 'max'}`` and
        ``growing_type in {'mean', 'max'}``.
        """
        selection = render_pkg["selection_mask"] # a boolean mask indicating which child gaussians were selected for rendering. having opacity higher than a certain threshold. check decoder.py
        visible = render_pkg["visible_mask"] # a boolean mask indicating which anchors were visible in the current frame.
        opacity = render_pkg["opacity"] # children that survived the selection mask opacities
        grad_points = render_pkg["viewspace_points"]  # 2D (x,y) tensor representin the center of where the 3D gaussian landed on the 2D screen. these have the .grad attribute attached tgat will be populated to tell us how to fix its physical position.
        pixel_hits = render_pkg["visibility_filter"] # a boolean mask for children that survived after painting. after painting is completed some gaussian may end up not visible. or if its radii is small it doesn't cover a single pixel
        radii = render_pkg["radii"] 

        # if zero parent anchors are visible exit early
        if visible.numel() == 0:
            return
       
        n_visible = int(visible.sum().item()) # number of visible anchors

        # --- anchor-level opacity statistics ---
        per_offset_opacity = torch.zeros(
            (n_visible * self.n_offsets,), device=self.device
        )
        per_offset_opacity[selection] = opacity.detach().view(-1) # detach means we cut it from computation graph. we dont need to backprop through opacity
        opacity_matrix = per_offset_opacity.view(n_visible, self.n_offsets) # matrix with visible anchors as rows and offsets as columns

        pruning_mode = getattr(opt, "pruning_type", "mean") # default is mean mode
        if pruning_mode == "max": # if using max mode 
            batch_score = torch.abs(opacity_matrix.sum(dim=1, keepdim=True))
            self.state.anchor_opacity[visible] = torch.maximum(
                self.state.anchor_opacity[visible], batch_score
            )
        else: # if using mean mode
            active_counts = (
                selection.view(n_visible, self.n_offsets)
                .float()
                .sum(dim=1, keepdim=True)
                .clamp(min=1.0)
            )
            anchor_scores = opacity_matrix.sum(dim=1, keepdim=True) / active_counts
            self.state.anchor_opacity[visible] += anchor_scores

        self.state.anchor_visits[visible] += 1

        # --- offset-level gradient statistics ---
        per_offset_visible = (
            visible.unsqueeze(1).expand(-1, self.n_offsets).reshape(-1)
        )
        offset_flags = torch.zeros_like(
            self.state.offset_gradients, dtype=torch.bool
        ).squeeze(1) # repeat visible to the number of offsets 
        offset_flags[per_offset_visible] = selection
        temp = offset_flags.clone()
        offset_flags[temp] = pixel_hits

        gradients = grad_points.grad
        if gradients.dim() == 3:
            gradients = gradients.squeeze(0)
        gradients = gradients.clone()
        gradients[:, 0] *= width * 0.5
        gradients[:, 1] *= height * 0.5
        grad_norm = torch.norm(gradients[pixel_hits, :2], dim=-1, keepdim=True)

        growing_mode = getattr(opt, "growing_type", "mean")
        if growing_mode == "max":
            self.state.offset_gradients[offset_flags] = torch.maximum(
                self.state.offset_gradients[offset_flags], torch.abs(grad_norm)
            )
            self.state.max_radii[offset_flags] = torch.maximum(
                self.state.max_radii[offset_flags], radii[pixel_hits]
            )
            self.state.offset_opacity[offset_flags] += opacity.detach()[pixel_hits]
        else:
            self.state.offset_gradients[offset_flags] += grad_norm

        self.state.offset_visits[offset_flags] += 1

    # ------------------------------------------------------------------
    # Densification entry point
    # ------------------------------------------------------------------

    def densify(self, model, opt, iteration: int) -> None:
        if self.state.offset_visits.numel() == 0:
            return

        growing_mode = getattr(opt, "growing_type", "mean")
        if growing_mode == "max":
            raw_grads = self.state.offset_gradients.nan_to_num(0.0).view(-1)
            avg_offset_opacity = (
                self.state.offset_opacity / self.state.offset_visits.clamp(min=1.0)
            ).nan_to_num(0.0).view(-1)
            scores = raw_grads * self.state.max_radii * torch.pow(avg_offset_opacity, 0.2)
            commonly_visited_mask = (
                self.state.offset_visits.view(-1) > opt.update_interval * opt.success_threshold * 0.5
            )
            commonly_visited_mask = torch.logical_and(commonly_visited_mask, avg_offset_opacity > 0.15)
        else:
            scores = (
                self.state.offset_gradients / self.state.offset_visits.clamp(min=1.0)
            ).nan_to_num(0.0).view(-1)
            commonly_visited_mask = (
                self.state.offset_visits.view(-1) > opt.update_interval * opt.success_threshold * 0.5
            )

        # --- grow new anchors across multiple scales ---
        added = self._hierarchical_grow(model, opt, scores, commonly_visited_mask, iteration)

        # --- pad density state buffers after growing ---
        self._pad_offset_buffers(model, opt, commonly_visited_mask)

        # --- prune low-contribution anchors ---
        # Note: _hierarchical_grow alcommonly_visited_mask extended anchor_opacity/visits for new
        # anchors, so _compute_prune_mask alcommonly_visited_mask returns the correct total size.
        prune_mask = self._compute_prune_mask(opt)
        if prune_mask.any():
            model.prune_anchor(prune_mask)
            self._shrink_offset_buffers(prune_mask)

        # --- reset accumulation counters ---
        visited_anchors = (
            self.state.anchor_visits.view(-1) > opt.update_interval * opt.success_threshold
        )
        if visited_anchors.any():
            self.state.anchor_opacity[visited_anchors] = 0.0
            self.state.anchor_visits[visited_anchors] = 0.0

        remaining = ~prune_mask if prune_mask.any() else torch.ones(
            self.state.anchor_opacity.shape[0], dtype=torch.bool, device=self.device
        )
        self.state.anchor_opacity = self.state.anchor_opacity[remaining]
        self.state.anchor_visits = self.state.anchor_visits[remaining]
        self.state.max_radii = torch.zeros(
            model.field.anchor.shape[0] * self.n_offsets,
            dtype=torch.float,
            device=self.device,
        )

    def cleanup(self) -> None:
        self.reset(0)

    # ------------------------------------------------------------------
    # Multi-scale hierarchical growing
    # ------------------------------------------------------------------

    def _hierarchical_grow(
        self, model, opt, scores: torch.Tensor, commonly_visited_mask: torch.Tensor, iteration: int
    ) -> int:
        """Grow anchors at ``update_depth`` progressively coarser scales.

        Each scale level uses a higher gradient threshold and a stochastic
        mask that lets through fewer candidates — promoting diversity at fine
        scales and stability at coarse scales.
        """
        field = model.field
        total_added = 0
        baseline_n = field.anchor.shape[0] * self.n_offsets

        depth_levels = getattr(opt, "update_depth", 3)
        hierarchy_factor = max(getattr(opt, "update_hierachy_factor", 4), 1)
        init_factor = getattr(opt, "update_init_factor", 16)
        base_threshold = opt.densify_grad_threshold

        for level in range(depth_levels):
            # --- recompute anchor_extent every level (field may have grown) ---
            anchor_extent = torch.exp(field.log_scaling)[:, :3]

            # --- threshold grows exponentially per level ---
            level_threshold = base_threshold * ((hierarchy_factor // 2) ** level)
            above_threshold = (scores >= level_threshold) & commonly_visited_mask

            # --- pad candidate mask to match current (possibly larger) field ---
            current_n_offsets = field.anchor.shape[0] * self.n_offsets
            growth_delta = current_n_offsets - above_threshold.shape[0]
            if growth_delta > 0:
                above_threshold = torch.cat(
                    [above_threshold, torch.zeros(growth_delta, dtype=torch.bool, device=self.device)],
                    dim=0,
                )

            # --- stochastic thinning: survival rate *increases* with depth ---
            # rand > 0.5**(level+1)
            # level 0 → 50% survive, level 1 → 75%, level 2 → 87.5% …
            # Deeper levels compensate for their stricter gradient threshold by
            # keeping more of the candidates that did pass it.
            stochastic_filter = torch.rand_like(above_threshold.float()) > (0.5 ** (level + 1))
            candidates = above_threshold & stochastic_filter.to(self.device)

            if not candidates.any():
                continue

            child_xyz = (
                field.anchor.unsqueeze(1) + field.offset * anchor_extent.unsqueeze(1)
            )
            scale_divisor = max(hierarchy_factor ** level, 1)
            cell_size = field.voxel_size * (init_factor // scale_divisor)

            selected_xyz = child_xyz.reshape(-1, 3)[candidates]
            novel_anchors, novel_features, novel_labels = self._filter_novel_voxels(
                field, model, candidates, selected_xyz, cell_size, opt
            )

            if novel_anchors.shape[0] == 0:
                continue

            new_scaling = torch.log(
                torch.full((novel_anchors.shape[0], 6), float(cell_size), device=self.device).clamp(min=1e-6)
            )
            new_rotation = torch.zeros(
                (novel_anchors.shape[0], 4), device=self.device
            )
            new_rotation[:, 0] = 1.0
            new_offsets = torch.zeros(
                (novel_anchors.shape[0], model.n_offsets, 3), device=self.device
            )

            # --- extend field with optimizer state preservation ---
            extension = {
                "anchor": novel_anchors,
                "offset": new_offsets,
                "feature": novel_features,
                "scaling": new_scaling,
                "rotation": new_rotation,
            }
            if model.optimizer is not None:
                updated = model.extend_optimizer_state(extension)
                field.anchor = updated["anchor"]
                field.offset = updated["offset"]
                field.feature = updated["feature"]
                field.log_scaling = updated["scaling"]
                field.raw_rotation = updated["rotation"]
            else:
                field.append(
                    novel_anchors, new_offsets, novel_features,
                    new_scaling, new_rotation, novel_labels,
                )

            # --- update labels separately (not an optimizer group) ---
            if novel_labels is not None and field.label_ids is not None:
                field.label_ids = torch.cat([field.label_ids, novel_labels], dim=0)
            elif novel_labels is not None:
                field.label_ids = novel_labels
            if field.label_ids is not None and field.codec is not None:
                field.codec.fit(field.label_ids.view(-1))

            # --- extend density accumulators ---
            n_new = novel_anchors.shape[0]
            self.state.anchor_opacity = torch.cat([
                self.state.anchor_opacity,
                torch.zeros((n_new, 1), device=self.device),
            ], dim=0)
            self.state.anchor_visits = torch.cat([
                self.state.anchor_visits,
                torch.zeros((n_new, 1), device=self.device),
            ], dim=0)
            field.visible = torch.ones(
                field.anchor.shape[0], dtype=torch.bool, device=self.device
            )

            total_added += n_new
            torch.cuda.empty_cache()

        return total_added

    def _filter_novel_voxels(
        self,
        field,
        model,
        candidates: torch.Tensor,
        selected_xyz: torch.Tensor,
        cell_size: float,
        opt,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Deduplicate candidates and discard those overlapping existing anchors."""
        occupied_grid = torch.round(field.anchor / cell_size).to(torch.int64)
        candidate_grid = torch.round(selected_xyz / cell_size).to(torch.int64)
        unique_grid, inverse = torch.unique(candidate_grid, return_inverse=True, dim=0)

        if unique_grid.numel() == 0:
            return (
                torch.zeros((0, 3), device=self.device),
                torch.zeros((0, field.feature.shape[1]), device=self.device),
                None,
            )

        if getattr(opt, "overlap", False):
            novel = torch.ones(unique_grid.shape[0], dtype=torch.bool, device=self.device)
        elif unique_grid.shape[0] > 0 and occupied_grid.shape[0] > 0:
            novel = ~self._check_voxel_overlap(occupied_grid, unique_grid)
        else:
            novel = torch.ones(unique_grid.shape[0], dtype=torch.bool, device=self.device)

        if not novel.any():
            return (
                torch.zeros((0, 3), device=self.device),
                torch.zeros((0, field.feature.shape[1]), device=self.device),
                None,
            )

        novel_positions = unique_grid[novel].float() * cell_size

        # Pool features from contributing offsets
        repeated_feat = field.feature.repeat_interleave(
            model.n_offsets, dim=0
        )[candidates]
        pooled = torch.zeros(
            (unique_grid.shape[0], field.feature.shape[1]), device=self.device
        )
        counts = torch.zeros((unique_grid.shape[0], 1), device=self.device)
        pooled.index_add_(0, inverse, repeated_feat)
        counts.index_add_(
            0, inverse, torch.ones((inverse.shape[0], 1), device=self.device)
        )
        pooled = pooled / counts.clamp(min=1.0)
        novel_features = pooled[novel]

        # Pool labels from contributing offsets
        novel_labels = None
        if field.label_ids is not None:
            repeated_labels = field.label_ids.repeat_interleave(
                model.n_offsets, dim=0
            ).view(-1)[candidates]
            pooled_labels = torch.zeros(
                unique_grid.shape[0], dtype=repeated_labels.dtype, device=self.device
            )
            for idx in range(unique_grid.shape[0]):
                members = repeated_labels[inverse == idx]
                if members.numel() > 0:
                    pooled_labels[idx] = torch.bincount(members).argmax()
            novel_labels = pooled_labels[novel].view(-1, 1)

        return novel_positions, novel_features, novel_labels

    @staticmethod
    def _check_voxel_overlap(
        occupied: torch.Tensor, candidates: torch.Tensor, chunk_size: int = 4096
    ) -> torch.Tensor:
        """Check which *candidates* coincide with any *occupied* voxel.

        Uses a chunked comparison to avoid materializing a huge broadcast tensor.
        """
        n_chunks = (occupied.shape[0] + chunk_size - 1) // chunk_size
        overlap_parts = []
        for i in range(n_chunks):
            chunk = occupied[i * chunk_size : (i + 1) * chunk_size]
            matches = (candidates.unsqueeze(1) == chunk).all(-1).any(-1)
            overlap_parts.append(matches)
        return reduce(torch.logical_or, overlap_parts)

    # ------------------------------------------------------------------
    # Post-densification state management
    # ------------------------------------------------------------------

    def _pad_offset_buffers(self, model, opt, commonly_visited_mask: torch.Tensor) -> None:
        """Zero-out visited offset slots and pad buffers for newly grown anchors."""
        total_offsets = model.field.anchor.shape[0] * self.n_offsets

        commonly_visited_mask = commonly_visited_mask[: self.state.offset_visits.shape[0]]
        self.state.offset_visits[commonly_visited_mask] = 0
        self.state.offset_gradients[commonly_visited_mask] = 0
        self.state.offset_opacity[commonly_visited_mask] = 0

        deficit = total_offsets - self.state.offset_visits.shape[0]
        if deficit > 0:
            z1 = torch.zeros((deficit, 1), device=self.device)
            self.state.offset_visits = torch.cat(
                [self.state.offset_visits, z1], dim=0
            )
            self.state.offset_gradients = torch.cat(
                [self.state.offset_gradients, z1.clone()], dim=0
            )
            self.state.offset_opacity = torch.cat(
                [self.state.offset_opacity, z1.clone()], dim=0
            )

    def _shrink_offset_buffers(self, prune_mask: torch.Tensor) -> None:
        """Discard offset-level accumulators matching pruned anchors."""
        keep = ~prune_mask

        def _slice_offsets(buf: torch.Tensor) -> torch.Tensor:
            reshaped = buf.view(-1, self.n_offsets)[: keep.shape[0]][keep]
            return reshaped.reshape(-1, 1)

        self.state.offset_visits = _slice_offsets(self.state.offset_visits)
        self.state.offset_gradients = _slice_offsets(self.state.offset_gradients)
        self.state.offset_opacity = _slice_offsets(self.state.offset_opacity)

    def _compute_prune_mask(self, opt) -> torch.Tensor:
        """Identify anchors with insufficient contribution for removal."""
        if self.state.anchor_visits.numel() == 0:
            return torch.zeros((0,), dtype=torch.bool, device=self.device)
        pruning_mode = getattr(opt, "pruning_type", "mean")
        if pruning_mode == "max":
            low_opacity = (self.state.anchor_opacity < opt.min_opacity).view(-1)
        else:
            low_opacity = (
                self.state.anchor_opacity < opt.min_opacity * self.state.anchor_visits
            ).view(-1)
        visited = (
            self.state.anchor_visits.view(-1) > opt.update_interval * opt.success_threshold
        )
        return low_opacity & visited
