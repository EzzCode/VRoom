import gsplat
import torch
from gsplat.cuda._wrapper import fully_fused_projection, fully_fused_projection_2dgs
from diff_surfel_rasterization import rasterization_2dgs, rasterization_2dgs_inria_wrapper


def render(viewpoint_camera, pc, pipe, bg_color, visible_mask=None, training=True, object_mask=None):
    """Rasterize visible neural Gaussians using gsplat."""
    if object_mask is None:
        xyz, offset, color, opacity, scaling, rot, selection_mask, semantics = pc.generate_neural_gaussians(
            viewpoint_camera, visible_mask, training
        )
    else:
        xyz, offset, color, opacity, scaling, rot, selection_mask, semantics = pc.generate_neural_gaussians(
            viewpoint_camera, visible_mask & object_mask, training
        )

    render_device = xyz.device
    bg_color = bg_color.to(render_device)
    K = torch.tensor(
        [
            [viewpoint_camera.fx, 0, viewpoint_camera.cx],
            [0, viewpoint_camera.fy, viewpoint_camera.cy],
            [0, 0, 1],
        ],
        dtype=torch.float32,
        device=render_device,
    )
    viewmat = viewpoint_camera.world_view_transform.transpose(0, 1).to(render_device)

    if pc.gs_attr == "3D":
        render_colors, render_alphas, render_semantics, info = gsplat.rasterization(
            means=xyz,
            quats=rot,
            scales=scaling,
            opacities=opacity.squeeze(-1),
            colors=color,
            viewmats=viewmat[None],
            Ks=K[None],
            width=int(viewpoint_camera.image_width),
            height=int(viewpoint_camera.image_height),
            backgrounds=bg_color[None],
            packed=False,
            render_mode=pc.render_mode,
            features=semantics.detach(),
        )
    elif pc.gs_attr == "2D": 
        # gsplat's Inria wrapper processes N channels by looping in chunks of 3.
        # Format: [R, G, B, S1, ..., SF, D]
        combined_colors = torch.cat([color, semantics.detach()], dim=-1)
        
        (rendered, render_alphas), info = rasterization_2dgs_inria_wrapper(
            means=xyz,
            quats=rot,
            scales=scaling,
            opacities=opacity.squeeze(-1),
            colors=combined_colors,
            viewmats=viewmat[None],
            Ks=K[None],
            width=int(viewpoint_camera.image_width),
            height=int(viewpoint_camera.image_height),
            backgrounds=bg_color[None],
            near_plane=pc.near_plane if hasattr(pc, "near_plane") else 0.01,
            far_plane=pc.far_plane if hasattr(pc, "far_plane") else 100.0,
        )
        
        # Unify output: Pack RGB + Depth into the standard 4-channel slot
        # and isolate semantics.
        render_colors = torch.cat([rendered[..., :3], rendered[..., -1:]], dim=-1)
        render_semantics = rendered[..., 3:-1]
        
        # Extract 2DGS specific maps from meta
        render_normals = info["normals_rend"]
        render_normals_from_depth = info["normals_surf"]
        render_distort = info["render_distloss"]
        render_median = torch.zeros_like(render_alphas)
    else:
        raise ValueError(f"Unknown gs_attr: {pc.gs_attr}")

    # --- Unified Output Processing ---
    # Extract depth if the rasterizer returned a 4-channel RGB+D image
    if render_colors.shape[-1] == 4:
        colors, depths = render_colors[..., 0:3], render_colors[..., 3:4]
        depth = depths[0].permute(2, 0, 1) # [1, H, W]
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

    return_dict = {
        "render": rendered_image,
        "scaling": scaling,
        "viewspace_points": info["means2d"],
        "visibility_filter": radii > 0,
        "visible_mask": visible_mask,
        "selection_mask": selection_mask,
        "opacity": opacity,
        "render_depth": depth,
        "radii": radii,
        "render_alphas": render_alphas,
        "render_semantics": render_semantics,
    }
    if pc.gs_attr == "2D":
        return_dict.update(
            {
                "render_normals": render_normals,
                "render_normals_from_depth": render_normals_from_depth,
                "render_distort": render_distort,
            }
        )
    return return_dict


def prefilter_voxel(viewpoint_camera, pc):
    """Project visible anchors and return a tightened visibility mask."""
    means = pc.get_anchor[pc._anchor_mask]
    scales = pc.get_scaling[pc._anchor_mask][:, :3]
    quats = pc.get_rotation[pc._anchor_mask]
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
    viewmats = viewpoint_camera.world_view_transform.transpose(0, 1).to(render_device)[None]

    if pc.gs_attr == "3D":
        proj_results = fully_fused_projection(
            means,
            None,
            quats,
            scales,
            viewmats,
            Ks,
            int(viewpoint_camera.image_width),
            int(viewpoint_camera.image_height),
            eps2d=0.3,
            packed=False,
            near_plane=0.01,
            far_plane=1e10,
            radius_clip=0.0,
            sparse_grad=False,
            calc_compensations=False,
        )
    elif pc.gs_attr == "2D":
        proj_results = fully_fused_projection_2dgs(
            means,
            quats,
            scales,
            viewmats,
            Ks,
            int(viewpoint_camera.image_width),
            int(viewpoint_camera.image_height),
            eps2d=0.3,
            packed=False,
            near_plane=0.01,
            far_plane=1e10,
            radius_clip=0.0,
            sparse_grad=False,
        )
    else:
        raise ValueError(f"Unknown gs_attr: {pc.gs_attr}")

    radii = proj_results[0]
    visible_mask = pc._anchor_mask.clone()
    visible_mask[pc._anchor_mask] = radii.squeeze(0) > 0
    return visible_mask
