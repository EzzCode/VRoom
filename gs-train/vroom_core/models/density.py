import torch
import torch.nn as nn

# Maps optimizer group name to AnchorCloud attribute name.
# density.py owns this mapping; gaussian_model.py must use the same group names.
_GROUP_TO_FIELD = {
    "anchor":   "anchors_positions",
    "offset":   "gaussians_offsets",
    "feature":  "anchor_features",
    "scaling":  "anchors_log_scales",
    "rotation": "anchors_rotations",
}
_FIELD_GROUPS = set(_GROUP_TO_FIELD)


class DensifcationController:
    def __init__(self, voxel_size, anchor_cloud, optimizer, num_gaussians_per_anchor=5):
        self.gaussian_gradients_acc = None
        self.gaussian_visits = None
        self.anchor_opacity_acc = None
        self.anchor_visits = None
        self.voxel_size = voxel_size
        self.anchor_cloud = anchor_cloud
        self.optimizer = optimizer
        self.num_gaussians_per_anchor = num_gaussians_per_anchor

    def reset_state(self):
        """Allocate zero accumulators sized to the current cloud."""
        device = self.anchor_cloud.anchors_positions.device
        n_anchors = self.anchor_cloud.anchors_positions.shape[0]
        n_gaussians = n_anchors * self.num_gaussians_per_anchor

        self.gaussian_gradients_acc = torch.zeros(n_gaussians, device=device)
        self.gaussian_visits = torch.zeros(n_gaussians, device=device)
        self.anchor_opacity_acc = torch.zeros(n_anchors, device=device)
        self.anchor_visits = torch.zeros(n_anchors, device=device)

    def _pad_state(self, n_new):
        """Pad accumulators with zeros for n_new newly added anchors (preserves existing stats)."""
        device = self.anchor_cloud.anchors_positions.device

        zeros_gaussians = torch.zeros(n_new * self.num_gaussians_per_anchor, device=device)
        zeros_anchors = torch.zeros(n_new, device=device)

        self.gaussian_gradients_acc = torch.cat([self.gaussian_gradients_acc, zeros_gaussians])
        self.gaussian_visits = torch.cat([self.gaussian_visits, zeros_gaussians])
        self.anchor_opacity_acc = torch.cat([self.anchor_opacity_acc, zeros_anchors])
        self.anchor_visits = torch.cat([self.anchor_visits, zeros_anchors])

    def _mask_state(self, keep_mask):
        """Trim accumulators to surviving anchors (preserves stats for kept anchors)."""
        keep_mask_for_gaussians = keep_mask.repeat_interleave(self.num_gaussians_per_anchor)

        self.gaussian_gradients_acc = self.gaussian_gradients_acc[keep_mask_for_gaussians]
        self.gaussian_visits = self.gaussian_visits[keep_mask_for_gaussians]
        self.anchor_opacity_acc = self.anchor_opacity_acc[keep_mask]
        self.anchor_visits = self.anchor_visits[keep_mask]

    def update_densification_state(
        self,
        visibility_mask,
        selection_mask,
        opacity,
        rendered_2d_points,
        width,
        height,
    ):
        """
        This will run every iteration in the loop and will update the state which include:
        1. Gaussain grads
        2. Gaussain visits
        3. anchor visits
        4. anchor opacity

        Note: opacity and rendered_2d_points cover only SELECTED gaussians
        (N_selected = N_vis * K filtered by selection_mask). selection_mask
        has shape [N_vis * K] and maps them back to the full visible set.
        """
        if self.gaussian_gradients_acc is None:
            self.reset_state()

        K = self.num_gaussians_per_anchor
        device = visibility_mask.device

        # accumulate the gradients every iteration and acc gaussian visits
        # global indices of all visible gaussians in the full [N_all * K] accumulator
        visibility_mask_for_gaussians = visibility_mask.repeat_interleave(K)
        vis_gaussian_global_indices = visibility_mask_for_gaussians.nonzero(as_tuple=True)[0]  # [N_vis * K]

        # narrow to selected gaussians (decoder selection_mask filters N_vis*K → N_selected)
        selected_global_indices = vis_gaussian_global_indices[selection_mask]  # [N_selected]

        opacity_mask = opacity.view(-1) > 0.5  # [N_selected]
        active_global_indices = selected_global_indices[opacity_mask]

        gaussians_gradients = rendered_2d_points.grad
        if gaussians_gradients is None:
            return
        gaussians_gradients = gaussians_gradients.reshape(-1, 2).clone()  # [N_selected, 2]
        scaler = torch.tensor(
            [width * 0.5, height * 0.5], device=gaussians_gradients.device
        )
        gaussians_gradients *= scaler
        grad_magnitude = torch.norm(gaussians_gradients, dim=-1)  # [N_selected]

        self.gaussian_gradients_acc[active_global_indices] += grad_magnitude[opacity_mask]
        self.gaussian_visits[active_global_indices] += 1  # increment visits for gaussians

        # accumlate anchor opacity
        # scatter selected gaussian opacities back to their parent visible anchors
        n_vis_anchors = int(visibility_mask.sum().item())
        vis_anchor_local_idx = torch.arange(n_vis_anchors, device=device).repeat_interleave(K)  # [N_vis * K]
        selected_anchor_local_idx = vis_anchor_local_idx[selection_mask]  # [N_selected]

        opacity_flat = opacity.view(-1).detach().clamp(min=0)  # [N_selected]
        opacity_sum = torch.zeros(n_vis_anchors, device=device)
        opacity_count = torch.zeros(n_vis_anchors, device=device)
        opacity_sum.scatter_add_(0, selected_anchor_local_idx, opacity_flat)
        opacity_count.scatter_add_(0, selected_anchor_local_idx, torch.ones_like(opacity_flat))
        anchors_opacity_avg = opacity_sum / opacity_count.clamp(min=1)

        self.anchor_opacity_acc[visibility_mask] += anchors_opacity_avg
        self.anchor_visits[visibility_mask] += 1  # increment visits for anchors

    def growing_operation(self):
        """
        Quantize the space and calculate the gradients of the gaussians in that space,
        if it is higher than a certain threshold we will add a new anchor in the center of that voxel
        """
        if self.gaussian_gradients_acc is None:
            return

        gradient_threshold = 0.0005
        # we are going to do growing for different quantization levels depending on gradient value
        level_2_threshold = gradient_threshold * 2  # level 2 resolution threshold
        level_3_threshold = gradient_threshold * 4  # level 3 resolution threshold
        quantization_size_L1 = self.voxel_size
        quantization_size_L2 = self.voxel_size / 4
        quantization_size_L3 = self.voxel_size / 16

        device = self.anchor_cloud.anchors_positions.device
        feat_dim = self.anchor_cloud.anchor_features.shape[1]

        # we do this process for each quantization level
        for level, quantization_size in enumerate(
            [quantization_size_L1, quantization_size_L2, quantization_size_L3]
        ):
            current_gaussian_gradients_acc = self.gaussian_gradients_acc / self.gaussian_visits.clamp(min=1)

            (
                unique_quantized_gaussians,
                gaussian_positions,
                average_gradient_per_voxel,
                inverse_gaussian_voxel_indices,
            ) = self._quantize(quantization_size, current_gaussian_gradients_acc)

            # apply per-level threshold
            if level == 0:
                current_threshold = gradient_threshold
            elif level == 1:
                current_threshold = level_2_threshold
            elif level == 2:
                current_threshold = level_3_threshold

            above_threshold_mask = average_gradient_per_voxel > current_threshold

            # apply random elimination to avoid too many anchors spawining
            if level == 1:  # 50% survive
                ratio_of_voxels_to_keep = 0.5
                random_mask = (
                    torch.rand_like(average_gradient_per_voxel)
                    < ratio_of_voxels_to_keep
                ).bool()
                above_threshold_mask = above_threshold_mask & random_mask
            elif level == 2:  # 70%
                ratio_of_voxels_to_keep = 0.7
                random_mask = (
                    torch.rand_like(average_gradient_per_voxel)
                    < ratio_of_voxels_to_keep
                ).bool()
                above_threshold_mask = above_threshold_mask & random_mask

            if not above_threshold_mask.any():
                continue

            # indices into unique_quantized_gaussians for above-threshold voxels
            above_threshold_indices = above_threshold_mask.nonzero(as_tuple=True)[0]
            voxel_above_threshold = unique_quantized_gaussians[above_threshold_indices]

            anchor_voxelized_grid = (
                self.anchor_cloud.anchors_positions.detach() / quantization_size
            ).to(torch.int64)

            # check for overlap so that we don't add new anchors in the same location
            # exact row-wise comparison via broadcasting (no hashing)
            overlap_mask = (
                voxel_above_threshold.unsqueeze(1) == anchor_voxelized_grid.unsqueeze(0)
            ).all(dim=-1).any(dim=-1)

            anchors_to_add = voxel_above_threshold[~overlap_mask]  # (M_new, 3) voxel coords
            anchors_to_add_indices = above_threshold_indices[~overlap_mask]  # into unique_quantized_gaussians

            if anchors_to_add.shape[0] == 0:
                continue

            n_new = anchors_to_add.shape[0]

            # set the anchors to added features to be the average for features of the contributing gaussians
            # and will be placed at the center of the voxel
            parent_to_gaussian_features = torch.repeat_interleave(
                self.anchor_cloud.anchor_features.detach(),
                self.num_gaussians_per_anchor,
                dim=0,
            )  # (N_all * K, F)

            parent_to_gaussian_labels = (
                torch.repeat_interleave(
                    self.anchor_cloud.semantic_labels,
                    self.num_gaussians_per_anchor,
                    dim=0,
                )
                if self.anchor_cloud.semantic_labels is not None
                else None
            )  # (N_all * K,) or None

            anchor_positions = anchors_to_add.float() * quantization_size  # (M_new, 3)
            anchor_feature = torch.zeros(n_new, feat_dim, device=device)
            anchor_label = (
                torch.zeros(n_new, dtype=torch.long, device=device)
                if parent_to_gaussian_labels is not None
                else None
            )

            for i, voxel_idx in enumerate(anchors_to_add_indices):
                member_mask = inverse_gaussian_voxel_indices == voxel_idx
                if member_mask.any():
                    anchor_feature[i] = parent_to_gaussian_features[member_mask].mean(dim=0)
                    if parent_to_gaussian_labels is not None:
                        anchor_label[i] = parent_to_gaussian_labels[member_mask].mode()[0]

            anchor_scale = quantization_size
            log_scale_val = float(torch.tensor(anchor_scale).log().item())
            anchor_log_scales = torch.full(
                (n_new, 6), log_scale_val, dtype=torch.float32, device=device
            )

            anchor_rotation = torch.zeros(n_new, 4, dtype=torch.float32, device=device)
            anchor_rotation[:, 0] = 1.0  # identity quaternion

            anchor_offsets = torch.zeros(
                n_new, self.num_gaussians_per_anchor, 3, dtype=torch.float32, device=device
            )

            extension_dict = {
                "anchor":   anchor_positions,
                "offset":   anchor_offsets,
                "feature":  anchor_feature,
                "scaling":  anchor_log_scales,
                "rotation": anchor_rotation,
            }

            # update optimizer and the anchor cloud data
            self._extend_optimizer(extension_dict)

            # update non-parameter data
            if parent_to_gaussian_labels is not None:
                existing_labels = self.anchor_cloud.semantic_labels
                self.anchor_cloud.semantic_labels = (
                    anchor_label if existing_labels is None
                    else torch.cat([existing_labels.view(-1), anchor_label.view(-1)])
                )
            self.anchor_cloud.visibility_mask = torch.ones(
                self.anchor_cloud.anchors_positions.shape[0],
                dtype=torch.bool, device=device,
            )

            # pad accumulators with zeros for new anchors (preserve existing stats)
            self._pad_state(n_new)

    def pruning_operation(self, opacity_threshold=0.01):
        """
        prune anchors with opacity less than a threshold
        """
        if self.anchor_visits is None:
            return

        prune_them_anchors_mask = (
            self.anchor_opacity_acc / self.anchor_visits.clamp(min=1) < opacity_threshold
        )

        if not prune_them_anchors_mask.any():
            return

        keep_mask = ~prune_them_anchors_mask

        # update optimizer and the anchor cloud data
        self._prune_optimizer(keep_mask)

        # update non-parameter data
        if self.anchor_cloud.semantic_labels is not None:
            self.anchor_cloud.semantic_labels = self.anchor_cloud.semantic_labels[keep_mask]

        self.anchor_cloud.visibility_mask = torch.ones(
            self.anchor_cloud.anchors_positions.shape[0],
            dtype=torch.bool,
            device=self.anchor_cloud.anchors_positions.device,
        )

        # mask accumulators to surviving anchors (preserve existing stats)
        self._mask_state(keep_mask)

    def _extend_optimizer(self, extension_dict):
        """
        For each anchor-cloud optimizer group:
          1. Extend Adam state buffers (exp_avg, exp_avg_sq) with zeros for new entries.
          2. Concatenate old param with extension to form a new Parameter.
          3. Replace group["params"][0] with the new Parameter.
          4. Mirror the new Parameter onto the AnchorCloud attribute.
        Called BEFORE any field attributes are changed externally.
        """
        for group in self.optimizer.param_groups:
            name = group.get("name")
            if name not in extension_dict:
                continue
            ext = extension_dict[name].to(dtype=group["params"][0].dtype)
            old_param = group["params"][0]
            state = self.optimizer.state.pop(old_param, {})

            for key in ("exp_avg", "exp_avg_sq"):
                if key in state:
                    state[key] = torch.cat([state[key], torch.zeros_like(ext)], dim=0)

            new_param = nn.Parameter(
                torch.cat([old_param.detach(), ext], dim=0),
                requires_grad=(name != "rotation"),
            )
            group["params"][0] = new_param
            if state:
                self.optimizer.state[new_param] = state

            setattr(self.anchor_cloud, _GROUP_TO_FIELD[name], new_param)

    def _prune_optimizer(self, keep_mask):
        """
        For each anchor-cloud optimizer group:
          1. Slice Adam state buffers to surviving rows.
          2. Create new Parameter from surviving rows.
          3. Replace group["params"][0] and the AnchorCloud attribute.
        Called BEFORE any field attributes are changed externally.
        """
        for group in self.optimizer.param_groups:
            name = group.get("name")
            if name not in _FIELD_GROUPS:
                continue
            old_param = group["params"][0]
            state = self.optimizer.state.pop(old_param, {})

            for key in ("exp_avg", "exp_avg_sq"):
                if key in state:
                    state[key] = state[key][keep_mask]

            new_param = nn.Parameter(
                old_param[keep_mask].detach().clone(),
                requires_grad=(name != "rotation"),
            )
            group["params"][0] = new_param
            if state:
                self.optimizer.state[new_param] = state

            setattr(self.anchor_cloud, _GROUP_TO_FIELD[name], new_param)

    def _quantize(self, quantization_size, gaussian_gradients):
        """
        Quantize the space based on a voxel size and return the average gradient per voxel, gaussians position and the grid
        """
        gaussian_positions = self.anchor_cloud.anchors_positions.unsqueeze(
            1
        ) + self.anchor_cloud.gaussians_offsets * torch.exp(
            self.anchor_cloud.anchors_log_scales
        )[:, :3].unsqueeze(1)  # (num_anchors, num_gaussians_per_anchor, 3)

        quantized_gaussians = (gaussian_positions.detach().view(-1, 3) / quantization_size).to(torch.int64)
        unique_quantized_gaussians, inverse_gaussian_voxel_indices = torch.unique(
            quantized_gaussians, dim=0, return_inverse=True
        )

        # average gradient per voxel via scatter_add (avoids Python loop)
        n_voxels = unique_quantized_gaussians.shape[0]
        average_gradient_per_voxel = torch.zeros(
            n_voxels, dtype=gaussian_gradients.dtype, device=gaussian_gradients.device
        )
        voxel_counts = torch.zeros(n_voxels, dtype=torch.float32, device=gaussian_gradients.device)

        average_gradient_per_voxel.scatter_add_(0, inverse_gaussian_voxel_indices, gaussian_gradients)
        voxel_counts.scatter_add_(0, inverse_gaussian_voxel_indices, torch.ones_like(gaussian_gradients))
        average_gradient_per_voxel = average_gradient_per_voxel / voxel_counts.clamp(min=1)

        return (
            unique_quantized_gaussians,
            gaussian_positions,
            average_gradient_per_voxel,
            inverse_gaussian_voxel_indices,
        )
