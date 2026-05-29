from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from tqdm import tqdm

from vroom_core.utilities.gaussian_renderer.render import prefilter_voxel, render

from vroom_core.utilities.utils.runtime import ensure_directory


@dataclass(frozen=True)
class MeshFusionOptions:
    voxel_size: float = -1.0
    sdf_trunc: float = -1.0
    depth_trunc: float = -1.0
    mesh_res: int = 256
    cluster_keep: int = 10
    min_cluster_triangles: int = 50
    mask_background: bool = True


@dataclass
class _RenderCapture:
    camera: object
    rgb: torch.Tensor
    depth: torch.Tensor
    alpha_mask: torch.Tensor | None


@dataclass(frozen=True)
class MeshExportResult:
    label_id: int
    raw_path: Path
    filtered_path: Path
    num_vertices: int
    num_vertices_filtered: int


class _RenderPipe:
    def __init__(self, add_prefilter: bool = True) -> None:
        self.add_prefilter = add_prefilter


class ObjectMeshExporter:
    def __init__(self, anchor_cloud, decoder, background: torch.Tensor, add_prefilter: bool = True, gaussian_type: str = "3D", render_mode: str = "RGB+ED", tile_size_2dgs: int = 8) -> None:
        self.anchor_cloud = anchor_cloud
        self.decoder = decoder
        self.background = background.to(anchor_cloud.device)
        self.pipe = _RenderPipe(add_prefilter=add_prefilter)
        self.gaussian_type = gaussian_type
        self.render_mode = render_mode
        self.tile_size_2dgs = tile_size_2dgs

    def available_labels(self, skip_zero: bool = True) -> list[int]:
        if self.anchor_cloud.semantic_labels is None:
            return []
        labels = torch.unique(self.anchor_cloud.semantic_labels.view(-1)).detach().cpu().tolist()
        labels = [int(label) for label in labels]
        if skip_zero:
            labels = [label for label in labels if label != 0]
        return sorted(labels)

    @torch.no_grad()
    def export_label_mesh(self, cameras: Iterable, label_id: int, output_dir: str | Path, options: MeshFusionOptions) -> MeshExportResult:
        label_mask = self._label_mask(label_id)
        captures = self._capture_views(cameras, label_mask)
        if not captures:
            raise RuntimeError(f"No visible geometry found for label {label_id}.")

        o3d = self._require_open3d()
        depth_trunc = self._resolve_depth_trunc(captures, options.depth_trunc)
        voxel_size = options.voxel_size if options.voxel_size > 0 else depth_trunc / max(options.mesh_res, 1)
        sdf_trunc = options.sdf_trunc if options.sdf_trunc > 0 else 5.0 * voxel_size

        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=float(voxel_size),
            sdf_trunc=float(sdf_trunc),
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        )

        for capture in tqdm(captures, desc=f"Fuse label {label_id}", dynamic_ncols=True):
            rgb = np.clip(capture.rgb.permute(1, 2, 0).numpy(), 0.0, 1.0)
            depth = capture.depth.squeeze(0).numpy().copy()
            if options.mask_background and capture.alpha_mask is not None:
                alpha = capture.alpha_mask.squeeze(0).numpy()
                depth[alpha < 0.5] = 0.0

            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(np.ascontiguousarray((rgb * 255.0).astype(np.uint8))),
                o3d.geometry.Image(np.ascontiguousarray(depth.astype(np.float32))),
                depth_scale=1.0,
                depth_trunc=float(depth_trunc),
                convert_rgb_to_intensity=False,
            )
            volume.integrate(rgbd, self._intrinsic(capture.camera, o3d), self._extrinsic(capture.camera))

        mesh = volume.extract_triangle_mesh()
        filtered = self._keep_largest_clusters(mesh, options.cluster_keep, options.min_cluster_triangles)

        label_dir = Path(output_dir) / f"label_{label_id}"
        ensure_directory(str(label_dir))
        raw_path = label_dir / "raw.ply"
        filtered_path = label_dir / "filtered.ply"
        o3d.io.write_triangle_mesh(str(raw_path), mesh)
        o3d.io.write_triangle_mesh(str(filtered_path), filtered)

        return MeshExportResult(
            label_id=label_id,
            raw_path=raw_path,
            filtered_path=filtered_path,
            num_vertices=len(mesh.vertices),
            num_vertices_filtered=len(filtered.vertices),
        )

    def _label_mask(self, label_id: int) -> torch.Tensor:
        if self.anchor_cloud.semantic_labels is None:
            raise RuntimeError("This checkpoint has no semantic labels, so per-object mesh export is unavailable.")
        return (self.anchor_cloud.semantic_labels.view(-1) == int(label_id)).to(self.anchor_cloud.device)

    @torch.no_grad()
    def _capture_views(self, cameras: Iterable, label_mask: torch.Tensor) -> list[_RenderCapture]:
        captures: list[_RenderCapture] = []
        for camera in tqdm(list(cameras), desc="Render object views", dynamic_ncols=True):
            visible = prefilter_voxel(camera, self.anchor_cloud, self.gaussian_type).squeeze() if self.pipe.add_prefilter else self.anchor_cloud.visibility_mask
            visible = visible & label_mask
            if visible.numel() == 0 or not bool(visible.any().item()):
                continue
            decoded_output = self.decoder.forward_pass(
                anchor_cloud=self.anchor_cloud,
                visible_anchors_mask=visible,
                camera=camera,
            )
            from vroom_core.core.training.orchestration import prepare_gaussian_space_props
            gaussian_positions, normalized_rotations = prepare_gaussian_space_props(
                anchor_cloud=self.anchor_cloud,
                visible_anchors_mask=visible,
                negative_opacity_filter=decoded_output["negative_opacity_filter"],
                rotations_pred=decoded_output["rotations"],
            )

            render_pkg = render(
                viewpoint_camera=camera,
                decoded_output=decoded_output,
                gaussian_positions=gaussian_positions,
                normalized_rotations=normalized_rotations,
                bg_color=self.background,
                gaussian_type=self.gaussian_type,
                render_mode=self.render_mode,
                tile_size_2dgs=self.tile_size_2dgs,
                semantics=None,
            )
            depth = render_pkg.get("render_depth")
            if depth is None:
                raise RuntimeError("The active render mode does not produce depth. Use a render mode such as 'RGB+ED' or 'RGB+D'.")
            captures.append(
                _RenderCapture(
                    camera=camera,
                    rgb=torch.clamp(render_pkg["render"].detach(), 0.0, 1.0).cpu(),
                    depth=depth.detach().cpu(),
                    alpha_mask=None if camera.alpha_mask is None else camera.alpha_mask.detach().cpu(),
                )
            )
        return captures

    def _resolve_depth_trunc(self, captures: list[_RenderCapture], configured: float) -> float:
        if configured > 0:
            return float(configured)
        valid_depths = []
        for capture in captures:
            depth = capture.depth
            valid = depth[depth > 0]
            if valid.numel() > 0:
                valid_depths.append(valid)
        if not valid_depths:
            return 3.0
        merged = torch.cat(valid_depths)
        if merged.numel() > 1_000_000:
            merged = merged[torch.randperm(merged.numel())[:1_000_000]]
        return float(torch.quantile(merged, 0.99).item() * 1.1)

    def _require_open3d(self):
        try:
            import open3d as o3d
        except ImportError as exc:
            raise RuntimeError("open3d is required for mesh export. Install it in the environment used for export.") from exc
        return o3d

    def _intrinsic(self, camera, o3d):
        return o3d.camera.PinholeCameraIntrinsic(
            width=int(camera.image_width),
            height=int(camera.image_height),
            fx=float(camera.fx),
            fy=float(camera.fy),
            cx=float(camera.cx),
            cy=float(camera.cy),
        )

    def _extrinsic(self, camera) -> np.ndarray:
        return np.asarray(camera.world_view_transform.transpose(0, 1).detach().cpu().numpy())

    def _keep_largest_clusters(self, mesh, cluster_keep: int, min_triangles: int):
        o3d = self._require_open3d()
        del o3d  # Only used for import validation.
        import copy

        filtered = copy.deepcopy(mesh)
        triangle_clusters, cluster_triangles, _ = filtered.cluster_connected_triangles()
        triangle_clusters = np.asarray(triangle_clusters)
        cluster_triangles = np.asarray(cluster_triangles)
        if cluster_triangles.size == 0:
            return filtered
        cluster_keep = max(min(cluster_keep, cluster_triangles.size), 1)
        threshold = np.sort(cluster_triangles)[-cluster_keep]
        threshold = max(int(threshold), int(min_triangles))
        remove_mask = cluster_triangles[triangle_clusters] < threshold
        filtered.remove_triangles_by_mask(remove_mask)
        filtered.remove_unreferenced_vertices()
        filtered.remove_degenerate_triangles()
        return filtered
