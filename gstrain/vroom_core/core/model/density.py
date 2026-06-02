import torch

from gstrain.vroom_core.utilities.training import extend_optimizer, prune_optimizer


class DensifcationController:
    def __init__(self, voxel_size, anchor_cloud, optimizer, num_gaussians_per_anchor=5, densifier_configs=None):
        self.gaussian_gradients_acc = None
        self.gaussian_visits = None
        self.anchor_opacity_acc = None
        self.anchor_visits = None
        self.voxel_size = voxel_size
        self.anchor_cloud = anchor_cloud
        self.optimizer = optimizer
        self.num_gaussians_per_anchor = num_gaussians_per_anchor
        self.densifier_configs = densifier_configs or {}

    def reset_state(self):
        """Reset the state of the densifcation controller"""
        device = self.anchor_cloud.anchors_positions.device
        n_anchors = self.anchor_cloud.anchors_positions.shape[0]
        n_gaussians = n_anchors * self.num_gaussians_per_anchor

        self.gaussian_gradients_acc = torch.zeros(n_gaussians, device=device)
        self.gaussian_visits = torch.zeros(n_gaussians, device=device)
        self.anchor_opacity_acc = torch.zeros(n_anchors, device=device)
        self.anchor_visits = torch.zeros(n_anchors, device=device)

    def _pad_state(self, n_new):
        """Pad accumulator states with n_new anchors of zero at the end of a growing iteration"""
        device = self.anchor_cloud.anchors_positions.device

        zeros_gaussians = torch.zeros(
            n_new * self.num_gaussians_per_anchor, device=device
        )
        zeros_anchors = torch.zeros(n_new, device=device)

        self.gaussian_gradients_acc = torch.cat(
            [self.gaussian_gradients_acc, zeros_gaussians]
        )
        self.gaussian_visits = torch.cat([self.gaussian_visits, zeros_gaussians])
        self.anchor_opacity_acc = torch.cat([self.anchor_opacity_acc, zeros_anchors])
        self.anchor_visits = torch.cat([self.anchor_visits, zeros_anchors])

    def _prune_state(self, keep_mask):
        """trim accumulator states to surviving anchors"""
        keep_mask_for_gaussians = keep_mask.repeat_interleave(
            self.num_gaussians_per_anchor
        )

        self.gaussian_gradients_acc = self.gaussian_gradients_acc[
            keep_mask_for_gaussians
        ]
        self.gaussian_visits = self.gaussian_visits[keep_mask_for_gaussians]
        self.anchor_opacity_acc = self.anchor_opacity_acc[keep_mask]
        self.anchor_visits = self.anchor_visits[keep_mask]

    def update_densification_state(
        self,
        visibility_mask,
        negative_opacity_filter,
        opacity,
        points_grad,
        width,
        height,
    ):
        """
        This will run every iteration in the loop and will update the state which include:
        1. Gaussain grads
        2. Gaussain visits
        3. anchor visits
        4. anchor opacity
        """
        if self.gaussian_gradients_acc is None:
            self.reset_state()

        negative_opacity_filter = negative_opacity_filter.view(-1)

        K = self.num_gaussians_per_anchor
        device = visibility_mask.device

        # accumulate the gradients every iteration and increment gaussian visits

        # get indices of all visible gaussians
        visibility_mask_for_gaussians = visibility_mask.repeat_interleave(K)
        vis_gaussian_indices = visibility_mask_for_gaussians.nonzero(as_tuple=True)[
            0
        ]  # [N_vis * K] where N_vis is the number of visible gaussians

        # select only gaussians that have postive opacity
        # the reason i apply the negative_opacity_filter first before the opacity mask
        # is to match the dim of vis_gaussian_indices(N_vis*K) to opacity(N_selected)
        selected_indices = vis_gaussian_indices[negative_opacity_filter]  # [N_selected]

        opacity_mask = opacity.view(-1) > 0.5  # [N_selected]
        active_indices = selected_indices[opacity_mask]  # [N_active]

        if points_grad is None:
            return
        # transform gradients from NDC space into pixel space
        gaussians_gradients = points_grad[..., :2].reshape(
            -1, 2
        )  # already detached and cloned in train_step
        scaler = torch.tensor(
            [width * 0.5, height * 0.5], device=gaussians_gradients.device
        )
        gaussians_gradients *= scaler
        grad_magnitude = torch.norm(gaussians_gradients, dim=-1)  # [N_selected]

        self.gaussian_gradients_acc[active_indices] += grad_magnitude[opacity_mask]
        self.gaussian_visits[active_indices] += (
            1  # increment the visits for the selected gaussians
        )

        # accumulate anchor opacity
        # create a lookup table for each gaussian to its visible parent anchor
        n_vis_anchors = int(visibility_mask.sum().item())
        vis_anchor_local_idx = torch.arange(
            n_vis_anchors, device=device
        ).repeat_interleave(
            K
        )  # [N_vis * K] local indices of visible anchors repeated for each gaussian
        selected_anchor_local_idx = vis_anchor_local_idx[
            negative_opacity_filter
        ]  # [N_selected]
        # now that we have look up table for each gaussian to its parents index
        # we can calculate average opacity per vis anchor
        opacity_flat = opacity.view(-1).detach().clamp(min=0)  # [N_selected]
        opacity_sum = torch.zeros(n_vis_anchors, device=device)
        opacity_count = torch.zeros(n_vis_anchors, device=device)
        opacity_sum.scatter_add_(
            0, selected_anchor_local_idx, opacity_flat
        )  # add the opacity of each gaussian to its bucket (anchor)
        opacity_count.scatter_add_(
            0, selected_anchor_local_idx, torch.ones_like(opacity_flat)
        )  # add 1 to each bucket per gaussian
        anchors_opacity_avg = opacity_sum / opacity_count.clamp(min=1)

        self.anchor_opacity_acc[visibility_mask] += (
            anchors_opacity_avg  # map the local opacities to the global accumlator uisng the visibility_mask
        )
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
            current_gaussian_gradients_acc = (
                self.gaussian_gradients_acc / self.gaussian_visits.clamp(min=1)
            )

            (
                unique_quantized_gaussians,
                gaussian_positions,
                average_gradient_per_voxel,
                inverse_gaussian_voxel_indices,
            ) = self._quantize(quantization_size, current_gaussian_gradients_acc)

            # threshold is based on the level
            if level == 0:
                current_threshold = gradient_threshold
            elif level == 1:
                current_threshold = level_2_threshold
            elif level == 2:
                current_threshold = level_3_threshold

            above_threshold_mask = average_gradient_per_voxel > current_threshold

            # apply random elimination to avoid too many anchors spawining
            # priorty is given for higher thresholds/levels
            if level == 1:  # 50% survive
                ratio_of_voxels_to_keep = self.densifier_configs.get("ratio_voxels_L2", 0.5)
                random_mask = (
                    torch.rand_like(average_gradient_per_voxel)
                    < ratio_of_voxels_to_keep
                ).bool()
                above_threshold_mask = above_threshold_mask & random_mask
            elif level == 2:  # 60%
                ratio_of_voxels_to_keep = self.densifier_configs.get("ratio_voxels_L3", 0.6)
                random_mask = (
                    torch.rand_like(average_gradient_per_voxel)
                    < ratio_of_voxels_to_keep
                ).bool()
                above_threshold_mask = above_threshold_mask & random_mask
            elif level == 3:  # 80%
                ratio_of_voxels_to_keep = self.densifier_configs.get("ratio_voxels_L4", 0.8)
                random_mask = (
                    torch.rand_like(average_gradient_per_voxel)
                    < ratio_of_voxels_to_keep
                ).bool()
                above_threshold_mask = (
                    above_threshold_mask & random_mask
                )  # apply the random mask and threshold mask
            if not above_threshold_mask.any():
                continue

            # indices into unique_quantized_gaussians for above_threshold voxels
            above_threshold_indices = above_threshold_mask.nonzero(as_tuple=True)[
                0
            ]  # indices of voxels above threshold
            voxel_above_threshold = unique_quantized_gaussians[
                above_threshold_indices
            ]  # voxels above threshold

            anchor_voxelized_grid = (
                self.anchor_cloud.anchors_positions.detach() / quantization_size
            ).to(torch.int64)

            # check for overlap so that we don't add new anchors in the same location
            overlap_mask = (
                (
                    voxel_above_threshold.unsqueeze(1)
                    == anchor_voxelized_grid.unsqueeze(0)
                )
                .all(dim=-1)
                .any(dim=-1)
            )

            anchors_to_add = voxel_above_threshold[
                ~overlap_mask
            ]  # voxels that dont overlap with existing anchors
            anchors_to_add_indices = above_threshold_indices[~overlap_mask]

            if anchors_to_add.shape[0] == 0:
                continue

            n_new = anchors_to_add.shape[0]

            # set the anchors to be added features to be the average for features of the contributing gaussians
            # and will be placed at the center of the voxel
            parent_to_gaussian_features = torch.repeat_interleave(
                self.anchor_cloud.anchor_features.detach(),
                self.num_gaussians_per_anchor,
                dim=0,
            )

            parent_to_gaussian_labels = (
                torch.repeat_interleave(
                    self.anchor_cloud.semantic_labels,
                    self.num_gaussians_per_anchor,
                    dim=0,
                )
                if self.anchor_cloud.semantic_labels is not None
                else None
            )

            anchor_positions = anchors_to_add.float() * quantization_size
            anchor_feature = torch.zeros(n_new, feat_dim, device=device)
            anchor_label = (
                torch.zeros(n_new, dtype=torch.long, device=device)
                if parent_to_gaussian_labels is not None
                else None
            )

            # feature and label handeling
            n_voxels = unique_quantized_gaussians.shape[0]
            voxel_to_new_anchor = torch.full(
                (n_voxels,), -1, dtype=torch.long, device=device
            )  # 1 tensor filled with -1 for length of voxels
            voxel_to_new_anchor[anchors_to_add_indices] = torch.arange(
                n_new, device=device
            )  # sets an id to the correct bucket for voxels that we will add a new anchor to,else its a -1

            # For each Gaussian, which slot does it belong to? if it maps to -1 its none
            # inverse_gaussian_voxel_indices: knows gaussian to voxel
            # voxel_to_new_anchor: knows voxel to new anchor id
            gauss_to_slot = voxel_to_new_anchor[
                inverse_gaussian_voxel_indices
            ]  # gauss_to_slot now knows: gaussian to new anchor id , size is number of gaussians
            contributes = gauss_to_slot >= 0
            contributing_slot = gauss_to_slot[
                contributes
            ]  # which anchor are we computing feature/label for

            # Average features
            contributing_features = parent_to_gaussian_features[
                contributes
            ]  # size is number of gaussians
            feature_sum = torch.zeros(n_new, feat_dim, device=device)
            feature_count = torch.zeros(n_new, device=device)
            feature_sum.scatter_add_(
                0,
                contributing_slot.unsqueeze(1).expand(
                    -1, feat_dim
                ),  # anchor ids are the buckets
                contributing_features,
            )
            feature_count.scatter_add_(
                0,
                contributing_slot,
                torch.ones(contributing_slot.shape[0], device=device),
            )
            anchor_feature = feature_sum / feature_count.clamp(min=1).unsqueeze(1)

            # majority vote for anchors labels
            if parent_to_gaussian_labels is not None:
                contributing_labels = parent_to_gaussian_labels[
                    contributes
                ].long()  # size is number of gaussians
                n_classes = (
                    int(contributing_labels.max().item()) + 1
                )  # beacuase indices start from zero
                # flatten contributing_labels for scatter add
                flat_idx = (
                    contributing_slot * n_classes + contributing_labels
                )  # slot * number of classes + label , Ex: slot:0 * number of classes:3 + label:1 = bucket:1 : (row * width) + col
                class_counts = torch.zeros(n_new * n_classes, device=device)
                class_counts.scatter_add_(
                    0, flat_idx, torch.ones(contributing_slot.shape[0], device=device)
                )
                anchor_label = class_counts.view(n_new, n_classes).argmax(dim=1)

            anchor_scale = quantization_size
            log_scale_val = float(torch.tensor(anchor_scale).log().item())
            anchor_log_scales = torch.full(
                (n_new, 6), log_scale_val, dtype=torch.float32, device=device
            )

            anchor_offsets = torch.zeros(
                n_new,
                self.num_gaussians_per_anchor,
                3,
                dtype=torch.float32,
                device=device,
            )

            extension_dict = {
                "anchors_positions": anchor_positions,
                "gaussians_offsets": anchor_offsets,
                "anchor_features": anchor_feature,
                "anchors_log_scales": anchor_log_scales,
            }

            # update optimizer and the anchor cloud data
            extend_optimizer(self.optimizer, self.anchor_cloud, extension_dict)

            if parent_to_gaussian_labels is not None:
                existing_labels = self.anchor_cloud.semantic_labels
                self.anchor_cloud.semantic_labels = (
                    anchor_label
                    if existing_labels is None
                    else torch.cat([existing_labels.view(-1), anchor_label.view(-1)])
                )
            self.anchor_cloud.visibility_mask = torch.ones(
                self.anchor_cloud.anchors_positions.shape[0],
                dtype=torch.bool,
                device=device,
            )

            # pad accumulators with zeros for new anchors
            self._pad_state(n_new)

    def pruning_operation(self, opacity_threshold=0.01):
        """
        prune anchors with opacity less than a threshold
        """
        if self.anchor_visits is None:
            return

        prune_them_anchors_mask = (self.anchor_visits > 0) & (
            self.anchor_opacity_acc / self.anchor_visits.clamp(min=1)
            < opacity_threshold
        )

        if not prune_them_anchors_mask.any():
            return

        keep_mask = ~prune_them_anchors_mask
        if keep_mask.sum() == 0:
            return

        # update optimizer and the anchor cloud data
        prune_optimizer(self.optimizer, self.anchor_cloud, keep_mask)

        # update semantic labels
        if self.anchor_cloud.semantic_labels is not None:
            self.anchor_cloud.semantic_labels = self.anchor_cloud.semantic_labels[
                keep_mask
            ]
        # update vis mask
        self.anchor_cloud.visibility_mask = torch.ones(
            self.anchor_cloud.anchors_positions.shape[0],
            dtype=torch.bool,
            device=self.anchor_cloud.anchors_positions.device,
        )

        # trim accumulators to surviving anchors
        self._prune_state(keep_mask)

    def _quantize(self, quantization_size, gaussian_gradients):
        """
        Quantize the space based on a voxel size and return the average gradient per voxel, gaussians position and the grid
        """
        gaussian_positions = self.anchor_cloud.anchors_positions.unsqueeze(
            1
        ) + self.anchor_cloud.gaussians_offsets * torch.exp(
            self.anchor_cloud.anchors_log_scales
        )[:, :3].unsqueeze(1)  # (num_anchors, num_gaussians_per_anchor, 3)

        quantized_gaussians = (
            gaussian_positions.detach().view(-1, 3) / quantization_size
        ).to(torch.int64)
        unique_quantized_gaussians, inverse_gaussian_voxel_indices = torch.unique(
            quantized_gaussians, dim=0, return_inverse=True
        )

        # average gradient per voxel with scatter_add
        n_voxels = unique_quantized_gaussians.shape[0]
        average_gradient_per_voxel = torch.zeros(
            n_voxels, dtype=gaussian_gradients.dtype, device=gaussian_gradients.device
        )
        voxel_counts = torch.zeros(
            n_voxels, dtype=torch.float32, device=gaussian_gradients.device
        )

        average_gradient_per_voxel.scatter_add_(
            0, inverse_gaussian_voxel_indices, gaussian_gradients
        )
        voxel_counts.scatter_add_(
            0, inverse_gaussian_voxel_indices, torch.ones_like(gaussian_gradients)
        )
        average_gradient_per_voxel = average_gradient_per_voxel / voxel_counts.clamp(
            min=1
        )

        return (
            unique_quantized_gaussians,
            gaussian_positions,
            average_gradient_per_voxel,
            inverse_gaussian_voxel_indices,
        )
