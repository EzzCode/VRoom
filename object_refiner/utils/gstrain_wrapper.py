import numpy as np
import torch
import functools
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict

from gstrain.vroom_core.core.model.anchor_field import AnchorCloud, AnchorCloudData
from gstrain.vroom_core.utilities.gaussian_decoder import GaussianDecoder
from gstrain.vroom_core.utilities.utils import CheckpointManager, SemanticsManager, projection_matrix
from gstrain.vroom_core.utilities.render import apply_frustum_culling as _culling
from gstrain.vroom_core.utilities.render import render as _render
from gstrain.vroom_core.core.training.orchestration import prepare_gaussian_space_props


class Camera:
    def __init__(self, R, T, K, width, height, uid=0):
        self.uid = uid
        self.R = np.asarray(R, dtype=np.float32)
        self.T = np.asarray(T, dtype=np.float32).flatten()
        self.image_width = int(width)
        self.image_height = int(height)
        self.resolution_scale = 1.0

        K = np.asarray(K, dtype=np.float32)
        self.fx = float(K[0, 0])
        self.fy = float(K[1, 1])
        self.cx = float(K[0, 2])
        self.cy = float(K[1, 2])
        self.FoVx = 2.0 * np.arctan(width / (2.0 * self.fx))
        self.FoVy = 2.0 * np.arctan(height / (2.0 * self.fy))

        Rt = np.eye(4, dtype=np.float32)
        Rt[:3, :3] = self.R
        Rt[:3, 3] = self.T
        self.world_view_transform = torch.tensor(Rt).transpose(0, 1).cuda()
        self.projection_matrix = (
            projection_matrix(znear=0.01, zfar=100.0, fov_x=self.FoVx, fov_y=self.FoVy)
            .transpose(0, 1)
            .cuda()
        )
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0)
            .bmm(self.projection_matrix.unsqueeze(0))
            .squeeze(0)
        )
        self.camera_center = self.world_view_transform.inverse()[3, :3]


def make_camera(R, T, K, width, height, uid=0):
    return Camera(R, T, K, width, height, uid=uid)


@dataclass
class VRoomGaussians:
    anchor_cloud: AnchorCloud
    decoder: GaussianDecoder
    checkpoint_manager: CheckpointManager
    gaussian_type: str
    feature_dim: int
    gaussians_per_anchor: int
    quantization_size: float
    render_mode: str
    tile_size_2dgs: int
    optim_params: Dict = field(default_factory=dict)


def build_vroom_gaussians(kwargs: dict, device="cuda") -> VRoomGaussians:
    gaussian_type = str(kwargs.get("gaussian_type", "2D"))
    feature_dim = int(kwargs.get("feature_dim", 32))
    gaussians_per_anchor = int(kwargs.get("gaussians_per_anchor", 10))
    quantization_size = float(kwargs.get("quantization_size", 0.001))
    render_mode = str(kwargs.get("render_mode", "RGB+ED"))
    tile_size_2dgs = int(kwargs.get("tile_size_2dgs", 8))

    knn_k = kwargs.get("knn_k")
    knn_chunk_size = kwargs.get("knn_chunk_size")
    min_quantization_size = kwargs.get("min_quantization_size")

    anchor_cloud = AnchorCloud(
        gaussians_per_anchor=gaussians_per_anchor,
        feature_dim=feature_dim,
        quantization_size=quantization_size,
        knn_k=knn_k,
        knn_chunk_size=knn_chunk_size,
        min_quantization_size=min_quantization_size,
        device=device,
    )
    decoder = GaussianDecoder(
        feature_dim=feature_dim,
        anchor_cloud=anchor_cloud,
    ).to(device)
    checkpoint_manager = CheckpointManager(anchor_cloud, decoder)

    return VRoomGaussians(
        anchor_cloud=anchor_cloud,
        decoder=decoder,
        checkpoint_manager=checkpoint_manager,
        gaussian_type=gaussian_type,
        feature_dim=feature_dim,
        gaussians_per_anchor=gaussians_per_anchor,
        quantization_size=quantization_size,
        render_mode=render_mode,
        tile_size_2dgs=tile_size_2dgs,
    )


def load_vroom_checkpoint(gaussians: VRoomGaussians, ply_path: str, mlp_dir: str) -> None:
    payload = gaussians.checkpoint_manager.load_anchor_field(ply_path)

    if payload["log_scaling"].numel() > 0:
        quantization_size = float(torch.exp(payload["log_scaling"][:, :3]).mean().item())
    else:
        avs = gaussians.anchor_cloud.quantization_size
        quantization_size = float(avs) if avs is not None and avs > 0 else 1.0

    seeds = AnchorCloudData(
        anchors_positions=payload["anchor"],
        gaussians_offsets=payload["offset"],
        anchor_features=payload["feature"],
        anchors_log_scales=payload["log_scaling"],
        anchors_rotations=payload["rotation"],
        labels=payload["labels"],
        semantic_manager=None
        if payload["labels"] is None
        else SemanticsManager(torch.unique(payload["labels"].view(-1))),
        quantization_size=quantization_size,
    )
    gaussians.anchor_cloud.set_anchors_cloud(seeds)
    gaussians.checkpoint_manager.load_decoder(mlp_dir)


def save_vroom_checkpoint(gaussians: VRoomGaussians, ply_path: str, mlp_dir: str) -> None:
    gaussians.checkpoint_manager.save_anchor_cloud(ply_path)
    gaussians.checkpoint_manager.save_decoder(
        mlp_dir,
        gaussian_type=gaussians.gaussian_type,
        render_mode=gaussians.render_mode,
        tile_Size=gaussians.tile_size_2dgs,
    )


def prefilter_anchors(gaussians: VRoomGaussians, cam: Camera) -> torch.Tensor:
    """Return a boolean mask (N,) of anchors visible from cam."""
    return _culling(cam, gaussians.anchor_cloud, gaussians.gaussian_type).squeeze()


def render_rgba(
    gaussians: VRoomGaussians,
    cam: Camera,
    bg_white=True,
    object_label_id=None,
    exclude_label_id=None,
    training=False,
    visible_mask=None,
):
    """Render gaussians from cam. Returns {'rgb': (3,H,W) cuda, 'alpha': (H,W) cuda}."""
    bg = torch.full((3,), 1.0 if bg_white else 0.0, dtype=torch.float32, device="cuda")

    if visible_mask is None:
        visible_mask = _culling(cam, gaussians.anchor_cloud, gaussians.gaussian_type).squeeze()

    # Combine with label/object mask if requested
    object_mask = None
    if object_label_id is not None:
        if gaussians.anchor_cloud.semantic_labels is not None:
            object_mask = (gaussians.anchor_cloud.semantic_labels.squeeze() == int(object_label_id))
    elif exclude_label_id is not None:
        if gaussians.anchor_cloud.semantic_labels is not None:
            object_mask = (gaussians.anchor_cloud.semantic_labels.squeeze() != int(exclude_label_id))

    if object_mask is not None:
        visible_mask = visible_mask & object_mask.to(visible_mask.device)

    # Decode anchors into neural Gaussians
    decoded_output = gaussians.decoder.forward_pass(
        anchor_cloud=gaussians.anchor_cloud,
        visible_anchors_mask=visible_mask,
        camera=cam,
    )

    # Prepare positions and rotations
    gaussian_positions, normalized_rotations = prepare_gaussian_space_props(
        anchor_cloud=gaussians.anchor_cloud,
        visible_anchors_mask=visible_mask,
        negative_opacity_filter=decoded_output["negative_opacity_filter"],
        rotations_pred=decoded_output["rotations"],
    )

    # Render
    pkg = _render(
        viewpoint_camera=cam,
        decoded_output=decoded_output,
        gaussian_positions=gaussian_positions,
        normalized_rotations=normalized_rotations,
        background_color=bg,
        gaussian_type=gaussians.gaussian_type,
        tile_Size=gaussians.tile_size_2dgs,
    )

    rgb = torch.clamp(pkg["render"], 0.0, 1.0)
    alpha_raw = pkg["render_alphas"]
    alpha = alpha_raw[0] if alpha_raw.ndim == 3 else alpha_raw

    return_pkg = {"rgb": rgb, "alpha": alpha}
    if training:
        return_pkg.update({
            "render": rgb,
            "render_alphas": alpha_raw,
            "scaling": pkg["scaling"],
            "opacity": pkg["opacity"],
            "rendered_2d_points": pkg["rendered_2d_points"],
            "visible_anchors_mask": visible_mask,
            "negative_opacity_filter": decoded_output["negative_opacity_filter"],
            "render_depth": pkg.get("render_depth"),
        })
        if gaussians.gaussian_type == "2D":
            return_pkg.update({
                "render_normals": pkg.get("render_normals"),
                "render_normals_from_depth": pkg.get("render_normals_from_depth"),
                "render_distort": pkg.get("render_distort"),
            })
    return return_pkg


def ssim_loss(prediction: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    @functools.lru_cache(maxsize=4)
    def kernel(size: int, channels: int, device_str: str, dtype):
        dist = torch.distributions.Normal(loc=size // 2, scale=1.5)
        coords = torch.arange(size, dtype=torch.float32)
        weights = dist.log_prob(coords).exp()
        weights = weights / weights.sum()
        kernel2d = weights[:, None] @ weights[None, :]
        return kernel2d.unsqueeze(0).unsqueeze(0).expand(channels, 1, size, size).contiguous().to(device=device_str, dtype=dtype)

    if prediction.dim() == 3:
        prediction = prediction.unsqueeze(0)
        target = target.unsqueeze(0)
    prediction = prediction.clamp(0.0, 1.0)
    target = target.clamp(0.0, 1.0)
    channels = prediction.shape[1]
    padding = window_size // 2
    window = kernel(window_size, channels, str(prediction.device), prediction.dtype)
    mu_a = F.conv2d(prediction, window, padding=padding, groups=channels)
    mu_b = F.conv2d(target, window, padding=padding, groups=channels)
    sigma_a = F.conv2d(prediction * prediction, window, padding=padding, groups=channels) - mu_a.pow(2)
    sigma_b = F.conv2d(target * target, window, padding=padding, groups=channels) - mu_b.pow(2)
    sigma_ab = F.conv2d(prediction * target, window, padding=padding, groups=channels) - (mu_a * mu_b)
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    numerator = (2.0 * mu_a * mu_b + c1) * (2.0 * sigma_ab + c2)
    denominator = (mu_a.pow(2) + mu_b.pow(2) + c1) * (sigma_a + sigma_b + c2)
    return 1.0 - (numerator / denominator).mean()
