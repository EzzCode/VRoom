from __future__ import annotations

import torch

from gaussian_renderer.render import prefilter_voxel, render as _render


class RasterizerAPI:

    def cull_invisible_anchors(
        self,
        camera_view,
        anchor_cloud,
        decoder,
    ) -> torch.Tensor:
        return prefilter_voxel(camera_view, anchor_cloud, decoder)

    def render(
        self,
        camera_view,
        anchor_cloud,
        decoder,
        bg_color: torch.Tensor,
        visible_mask: torch.Tensor | None = None,
        pipe=None,
    ) -> dict:

        if visible_mask is None:
            visible_mask = torch.ones(anchor_cloud.anchors_positions.shape[0], dtype=torch.bool, device=anchor_cloud.device)

        decoded_output = decoder.forward_pass(anchor_cloud, visible_mask, camera_view)

        gs_attr = decoder.gs_attr
        render_mode = decoder.render_mode
        tile_size_2dgs = decoder.tile_size_2dgs

        pkg = _render(
            viewpoint_camera=camera_view,
            decoded_output=decoded_output,
            bg_color=bg_color,
            gs_attr=gs_attr,
            render_mode=render_mode,
            tile_size_2dgs=tile_size_2dgs,
            semantics=None,
        )

        pkg["rendered_2d_points"] = pkg.get("viewspace_points")
        pkg["visible_mask"] = visible_mask
        pkg["selection_mask"] = decoded_output["negative_opacity_filter"]
        return pkg
