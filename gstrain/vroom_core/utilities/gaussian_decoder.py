import torch
import torch.nn as nn


class GaussianDecoder(nn.Module):
    def __init__(self, feature_dim, anchor_cloud):
        super().__init__()
        hidden_dim = 64
        input_dim = feature_dim + 3 # where 3 is the dim of the direction vector

        self.number_gaussians_per_anchor = anchor_cloud.gaussians_per_anchor
        self.feature_dim = feature_dim

        self.color_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.number_gaussians_per_anchor * 3),
            nn.Sigmoid(),
        )
        self.covariance_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.number_gaussians_per_anchor * 7),  # 3 for scale and 4 for rotation
        )
        self.opacity_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.number_gaussians_per_anchor),
            nn.Tanh(),
        )


    def forward_pass(self, anchor_cloud, visible_anchors_mask, camera):
        visible_anchors = anchor_cloud.anchors_positions[visible_anchors_mask]
        num_visible = visible_anchors.shape[0]

        # create the feature vector
        anchor_to_camera_distance = visible_anchors - camera.camera_center.to(visible_anchors.device)  # distance matrix between each anchor and the camera
        viewing_vector = torch.nn.functional.normalize(anchor_to_camera_distance, dim=-1)  # direction vector from camera to anchor

        # concatenate features and viewing vector along last dimension
        features = anchor_cloud.anchor_features[visible_anchors_mask]
        latent = torch.cat([features, viewing_vector], dim=-1)

        # predictions
        color_pred = self.color_network(latent).view(num_visible, self.number_gaussians_per_anchor, 3)
        opacity_pred = self.opacity_network(latent).view(num_visible, self.number_gaussians_per_anchor, 1)
        covariance_pred = self.covariance_network(latent).view(num_visible, self.number_gaussians_per_anchor, 7)

        # valid opacity mask
        negative_opacity_filter = (opacity_pred.squeeze(-1) > 0)  # we cant have negative opacities

        color_pred = color_pred[negative_opacity_filter]
        opacity_pred = opacity_pred[negative_opacity_filter]
        covariance_pred = covariance_pred[negative_opacity_filter]

        # gaussian scales
        anchor_scales = torch.exp(anchor_cloud.anchors_log_scales[visible_anchors_mask])
        anchor_scale_xyz = anchor_scales[:, :3] 
        anchor_scale_repeated = anchor_scale_xyz.unsqueeze(1).expand(-1, self.number_gaussians_per_anchor, -1)
        anchor_scale_filtered = anchor_scale_repeated[negative_opacity_filter]
        scaling_multiplier = torch.nn.functional.softplus(covariance_pred[:, :3])
        scaling = anchor_scale_filtered * scaling_multiplier

        return {
            "color": color_pred,
            "opacity": opacity_pred,
            "scaling": scaling,
            "rotations": covariance_pred[:, 3:7],
            "negative_opacity_filter": negative_opacity_filter,
        }
