import torch
import torch.nn as nn


class GaussianDecoder(nn.Module):
    def __init__(self, feature_dim, anchor_cloud):
        super().__init__()
        hidden_dim = 64
        input_dim = feature_dim + 3 # where 3 is the dim of the direction vector

        self.number_gaussians_per_anchor = anchor_cloud.gaussians_per_anchor
        self.feature_dim = feature_dim
        number_gaussians_per_anchor = self.number_gaussians_per_anchor

        self.color_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, number_gaussians_per_anchor * 3),
            nn.Sigmoid(),
        )
        self.covariance_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, number_gaussians_per_anchor * 7), # 3 for scale and 4 for rotation
        )
        self.opacity_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, number_gaussians_per_anchor),
            nn.Tanh(),
        )

    def forward_pass(self, anchor_cloud, visible_anchors_mask, camera):
        visible_anchors = anchor_cloud.anchors_positions[visible_anchors_mask]
        num_visible = visible_anchors.shape[0]

        # create the feature vector
        anchor_to_camera_distance = visible_anchors -  camera.camera_center.to(visible_anchors.device) # distance matrix between each anchor and the camera
        viewing_vector = torch.nn.functional.normalize(anchor_to_camera_distance, dim=-1) # direction vector from camera to anchor

        # concatenate features and viewing vector along last dimension
        features = anchor_cloud.anchor_features[visible_anchors_mask]
        latent = torch.cat([features, viewing_vector], dim=-1)

        # predictions
        color_pred = self.color_network(latent).view(num_visible, self.number_gaussians_per_anchor, 3)
        opacity_pred = self.opacity_network(latent).view(num_visible, self.number_gaussians_per_anchor, 1)
        covariance_pred = self.covariance_network(latent).view(num_visible, self.number_gaussians_per_anchor, 7)

        # valid opacity mask
        negative_opacity_filter = (opacity_pred.squeeze(-1) > 0) # we cant have negative opacties

        color_pred = color_pred[negative_opacity_filter]
        opacity_pred = opacity_pred[negative_opacity_filter]
        covariance_pred = covariance_pred[negative_opacity_filter]

        anchor_positions = anchor_cloud.anchors_positions[visible_anchors_mask]
        gaussian_offsets = anchor_cloud.gaussians_offsets[visible_anchors_mask]
        anchor_scales = torch.exp(anchor_cloud.anchors_log_scales[visible_anchors_mask])

        # used to calculate the gaussians position
        valid_scales = anchor_scales.unsqueeze(1).expand(-1, self.number_gaussians_per_anchor, -1)[negative_opacity_filter][:, :3] # uses broadcasting to create a matrix where we copy scale of each anchor to its gaussians
        valid_offsets = gaussian_offsets.view(num_visible, self.number_gaussians_per_anchor, 3)[negative_opacity_filter]
        valid_anchors = anchor_positions.unsqueeze(1).expand(-1, self.number_gaussians_per_anchor, -1)[negative_opacity_filter]
        gaussian_positions = valid_anchors + (valid_offsets * valid_scales)

        normalized_rotations = torch.nn.functional.normalize(covariance_pred[:, 3:7], dim=-1)

        semantics_pred = None
        if anchor_cloud.semantic_labels is not None and anchor_cloud.semantic_manager is not None:
            visible_labels = anchor_cloud.semantic_labels[visible_anchors_mask]
            if visible_labels.dim() > 1:
                visible_labels = visible_labels.squeeze(-1)
            visible_label_indices = anchor_cloud.semantic_manager.build_lookup_table(visible_labels)
            visible_one_hot = anchor_cloud.semantic_manager.one_hot_encode(visible_label_indices).float()
            expanded_one_hot = visible_one_hot.unsqueeze(1).expand(-1, self.number_gaussians_per_anchor, -1)
            semantics_pred = expanded_one_hot[negative_opacity_filter]

        return {
            "gaussian_possitions": gaussian_positions,
            "normalized_rotations": normalized_rotations,
            "color": color_pred,
            "opacity": opacity_pred,
            "scaling": torch.exp(covariance_pred[:, :3]),
            "negative_opacity_filter": negative_opacity_filter,
            "semantics": semantics_pred
        }
