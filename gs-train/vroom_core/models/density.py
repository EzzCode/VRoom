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
    anchor_visits: torch.Tensor
    offset_gradients: torch.Tensor
    offset_visits: torch.Tensor
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
        selection = render_pkg["selection_mask"]
        visible = render_pkg["visible_mask"]
        opacity = render_pkg["opacity"]
        grad_points = render_pkg["viewspace_points"]
        pixel_hits = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]

        if visible.numel() == 0:
            return
        n_visible = int(visible.sum().item())

        # --- anchor-level opacity statistics ---
        per_offset_opacity = torch.zeros(
            (n_visible * self.n_offsets,), device=self.device
        )
        per_offset_opacity[selection] = opacity.detach().view(-1)
        opacity_matrix = per_offset_opacity.view(n_visible, self.n_offsets)

        pruning_mode = getattr(opt, "pruning_type", "mean")
        if pruning_mode == "max":
            batch_score = torch.abs(opacity_matrix.sum(dim=1, keepdim=True))
            self.state.anchor_opacity[visible] = torch.maximum(
                self.state.anchor_opacity[visible], batch_score
            )
        else:
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
        ).squeeze(1)
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
            off_opacity = (
                self.state.offset_opacity / self.state.offset_visits.clamp(min=1.0)
            ).nan_to_num(0.0).view(-1)
            scores = raw_grads * self.state.max_radii * torch.pow(off_opacity, 0.2)
            ready = (
                self.state.offset_visits.view(-1) > opt.update_interval * opt.success_threshold * 0.5
            )
            ready = torch.logical_and(ready, off_opacity > 0.15)
        else:
            scores = (
                self.state.offset_gradients / self.state.offset_visits.clamp(min=1.0)
            ).nan_to_num(0.0).view(-1)
            ready = (
                self.state.offset_visits.view(-1) > opt.update_interval * opt.success_threshold * 0.5
            )

        # --- grow new anchors across multiple scales ---
        added = self._hierarchical_grow(model, opt, scores, ready, iteration)

        # --- pad density state buffers after growing ---
        self._pad_offset_buffers(model, opt, ready)

        # --- prune low-contribution anchors ---
        # Note: _hierarchical_grow already extended anchor_opacity/visits for new
        # anchors, so _compute_prune_mask already returns the correct total size.
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
        self, model, opt, scores: torch.Tensor, ready: torch.Tensor, iteration: int
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
            above_threshold = (scores >= level_threshold) & ready

            # --- pad candidate mask to match current (possibly larger) field ---
            current_n_offsets = field.anchor.shape[0] * self.n_offsets
            growth_delta = current_n_offsets - above_threshold.shape[0]
            if growth_delta > 0:
                above_threshold = torch.cat(
                    [above_threshold, torch.zeros(growth_delta, dtype=torch.bool, device=self.device)],
                    dim=0,
                )

            # --- stochastic thinning: fewer candidates at deeper levels ---
            keep_prob = 1.0 - 0.5 ** (level + 1)
            stochastic_filter = torch.rand_like(above_threshold.float()) > keep_prob
            candidates = above_threshold & stochastic_filter.to(self.device)

            if not candidates.any():
                continue

            # Compute selected child positions without materializing the full
            # [N_anchors, n_offsets, 3] tensor — avoids OOM on large scenes
            candidate_indices = candidates.nonzero(as_tuple=True)[0]
            anchor_idx = candidate_indices // model.n_offsets
            offset_idx = candidate_indices % model.n_offsets
            selected_xyz = (
                field.anchor[anchor_idx]
                + field.offset[anchor_idx, offset_idx]
                  * anchor_extent[anchor_idx]
            )
            scale_divisor = max(hierarchy_factor ** level, 1)
            cell_size = field.voxel_size * (init_factor // scale_divisor)

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
        # Avoid repeat_interleave which creates [N_anchors * n_offsets, feat_dim] —
        # instead compute which anchor each candidate belongs to and index directly.
        candidate_indices = candidates.nonzero(as_tuple=True)[0]
        anchor_indices = candidate_indices // model.n_offsets
        candidate_features = field.feature[anchor_indices]  # [n_candidates, feat_dim]

        pooled = torch.zeros(
            (unique_grid.shape[0], field.feature.shape[1]), device=self.device
        )
        counts = torch.zeros((unique_grid.shape[0], 1), device=self.device)
        pooled.index_add_(0, inverse, candidate_features)
        counts.index_add_(
            0, inverse, torch.ones((inverse.shape[0], 1), device=self.device)
        )
        pooled = pooled / counts.clamp(min=1.0)
        novel_features = pooled[novel]

        # Pool labels from contributing offsets
        novel_labels = None
        if field.label_ids is not None:
            candidate_labels = field.label_ids.view(-1)[anchor_indices]
            # Vectorized majority vote per unique voxel
            n_unique = unique_grid.shape[0]
            n_classes = int(candidate_labels.max().item()) + 1 if candidate_labels.numel() > 0 else 1
            one_hot = torch.zeros((candidate_labels.shape[0], n_classes), dtype=torch.long, device=self.device)
            one_hot.scatter_(1, candidate_labels.view(-1, 1).long(), 1)
            votes = torch.zeros((n_unique, n_classes), dtype=torch.long, device=self.device)
            votes.scatter_add_(0, inverse.unsqueeze(1).expand(-1, n_classes), one_hot)
            pooled_labels = votes.argmax(dim=1)
            novel_labels = pooled_labels[novel].view(-1, 1)

        return novel_positions, novel_features, novel_labels

    @staticmethod
    def _check_voxel_overlap(
        occupied: torch.Tensor, candidates: torch.Tensor, chunk_size: int = 4096
    ) -> torch.Tensor:
        """Check which *candidates* coincide with any *occupied* voxel.

        Uses a hash-set approach: encode each (x,y,z) integer triple into a
        single int64 scalar, build a set from occupied voxels, then check
        candidates with isin(). O(N+M) instead of O(N*M).
        """
        # Encode 3D integer coordinates as a single int64 using cantor-style packing.
        # Shift to non-negative first (coordinates can be negative).
        def encode(t: torch.Tensor) -> torch.Tensor:
            # t: [N, 3] int64
            # Shift so all values are >= 0 (add a large offset)
            shift = 100_000
            x = t[:, 0].to(torch.int64) + shift
            y = t[:, 1].to(torch.int64) + shift
            z = t[:, 2].to(torch.int64) + shift
            # Pack into single int64: x * M^2 + y * M + z
            M = torch.tensor(200_001, dtype=torch.int64, device=t.device)
            return x * M * M + y * M + z

        occupied_keys = encode(occupied)    # [N_occupied]
        candidate_keys = encode(candidates) # [N_candidates]
        return torch.isin(candidate_keys, occupied_keys)

    # ------------------------------------------------------------------
    # Post-densification state management
    # ------------------------------------------------------------------

    def _pad_offset_buffers(self, model, opt, ready: torch.Tensor) -> None:
        """Zero-out visited offset slots and pad buffers for newly grown anchors."""
        total_offsets = model.field.anchor.shape[0] * self.n_offsets

        ready = ready[: self.state.offset_visits.shape[0]]
        self.state.offset_visits[ready] = 0
        self.state.offset_gradients[ready] = 0
        self.state.offset_opacity[ready] = 0

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
