import torch
from cuda_rasterizer import rasterize_2dgs, frustum_cull_2dgs


def render(
    viewpoint_camera,
    decoded_output,
    gaussian_positions,
    normalized_rotations,
    background_color,
    gaussian_type="2D",
    tile_Size=8,
    semantics=None,
):
    xyz = gaussian_positions
    color = decoded_output["color"]
    opacity = decoded_output["opacity"]
    scaling = decoded_output["scaling"]
    rot = normalized_rotations

    render_device = xyz.device
    background_color = background_color.to(render_device)
    K = torch.tensor(
        [
            [viewpoint_camera.fx, 0, viewpoint_camera.cx],
            [0, viewpoint_camera.fy, viewpoint_camera.cy],
            [0, 0, 1],
        ],
        dtype=torch.float32,
        device=render_device,
    )
    viewmat = (
        viewpoint_camera.world_view_transform.transpose(0, 1).to(render_device).float()
    )

    if gaussian_type != "2D":
        raise ValueError(f"gaussian_type {gaussian_type} is not supported, use 2D")

    # diff-surfel-rasterization wrapper processes N channels by looping in chunks of 3.
    # Format: [R, G, B, S1, ..., SF, D]
    if semantics is not None:
        combined_colors = torch.cat([color, semantics.detach()], dim=-1)
    else:
        combined_colors = color

    (rendered, render_alphas), info = rasterize_2dgs(
        points_world_space=xyz,
        quats=rot,
        scale_vecs=scaling,
        opacities=opacity.squeeze(-1),
        colors_feat=combined_colors,
        w2cam_mats=viewmat[None],
        cam_intrinsics=K[None],
        img_W=int(viewpoint_camera.image_width),
        img_H=int(viewpoint_camera.image_height),
        backgrounds=background_color[None],
        near_plane=0.01,
        far_plane=100.0,
    )

    # Unify output: Pack RGB + Depth into the standard 4-channel slot
    # and isolate semantics.
    render_colors = torch.cat([rendered[..., :3], rendered[..., -1:]], dim=-1)
    render_semantics = rendered[..., 3:-1]
    if render_semantics.shape[-1] == 0:
        render_semantics = None

    # Extract 2DGS specific maps from info
    render_normals = info["normals_rend"]
    render_normals_from_depth = info["normals_surf"]
    render_distort = info["render_distloss"]
    render_median = torch.zeros_like(render_alphas)

    if render_colors.shape[-1] == 4:
        colors, depths = render_colors[..., 0:3], render_colors[..., 3:4]
        depth = depths[0].permute(2, 0, 1)
    else:
        colors = render_colors
        depth = None

    rendered_image = colors[0].permute(2, 0, 1)
    radii = info["radii"].squeeze(0)
    try:
        info["means2d"].retain_grad()
    except RuntimeError:
        pass

    render_alphas = render_alphas[0].permute(2, 0, 1)
    if render_semantics is not None:
        render_semantics = render_semantics[0].permute(2, 0, 1)

    return_dict = {
        "render": rendered_image,
        "scaling": scaling,
        "rendered_2d_points": info["means2d"],
        "visibility_filter": radii > 0,
        "opacity": opacity,
        "render_depth": depth,
        "radii": radii,
        "render_alphas": render_alphas,
        "render_semantics": render_semantics,
    }
    if gaussian_type == "2D":
        return_dict.update(
            {
                "render_normals": render_normals,
                "render_normals_from_depth": render_normals_from_depth,
                "render_distort": render_distort,
            }
        )
    return return_dict


def apply_frustum_culling(viewpoint_camera, anchor_cloud, gaussian_type="2D"):
    """Project visible anchors and return a tightened visibility mask."""
    means = anchor_cloud.anchors_positions[anchor_cloud.visibility_mask]
    scales = torch.exp(anchor_cloud.anchors_log_scales[anchor_cloud.visibility_mask])[
        :, :3
    ]
    import torch.nn.functional as F

    quats = F.normalize(
        anchor_cloud.anchors_rotations[anchor_cloud.visibility_mask], dim=-1
    )
    render_device = means.device

    Ks = torch.tensor(
        [
            [viewpoint_camera.fx, 0, viewpoint_camera.cx],
            [0, viewpoint_camera.fy, viewpoint_camera.cy],
            [0, 0, 1],
        ],
        dtype=torch.float32,
        device=render_device,
    )[None]
    viewmats = (
        viewpoint_camera.world_view_transform.transpose(0, 1)
        .to(render_device)
        .float()[None]
    )

    if gaussian_type != "2D":
        raise ValueError(f"gaussian_type {gaussian_type} is not supported, use 2D")

    proj_results = frustum_cull_2dgs(
        points_world_space=means,
        quats=quats,
        scale_vecs=scales,
        w2cam_mats=viewmats,
        cam_intrinsics=Ks,
        img_W=int(viewpoint_camera.image_width),
        img_H=int(viewpoint_camera.image_height),
        near_plane=0.01,
        far_plane=100.0,
    )

    radii = proj_results[0]
    visible_mask = anchor_cloud.visibility_mask.clone()
    visible_mask[anchor_cloud.visibility_mask] = radii.squeeze(0) > 0
    return visible_mask
