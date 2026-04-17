"""Scene discovery and assembly for VRoom."""

from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from plyfile import PlyData, PlyElement

from vroom_core.data.camera_system import FrameRecord, RenderCamera
from vroom_core.data.colmap_io import read_extrinsics_binary, read_extrinsics_text, read_intrinsics_binary, read_intrinsics_text, quaternion_to_rotation
from vroom_core.utils.geometry import PointCloudSample, focal_to_fov, fov_to_focal, world_to_view_matrix


@dataclass(frozen=True)
class SceneLayout:
    root: Path
    image_dir: Path
    mask_dir: Path | None
    depth_dir: Path | None
    image_file: Path
    camera_file: Path
    binary: bool
    point_cloud_file: Path
    depth_param_file: Path | None


@dataclass(frozen=True)
class SceneBundle:
    point_cloud: PointCloudSample
    train_records: list[FrameRecord]
    test_records: list[FrameRecord]
    normalization: dict
    ply_path: str


def discover_colmap_scene(root: str, images: str, depths: str, masks: str, add_mask: bool, add_depth: bool) -> SceneLayout:
    base = Path(root)
    image_dir = base / images
    mask_dir = (base / masks) if add_mask else None
    depth_dir = (base / depths) if add_depth else None
    metadata_candidates = [
        (base / "sparse/0/images.bin", base / "sparse/0/cameras.bin", True),
        (base / "sparse/0/images.txt", base / "sparse/0/cameras.txt", False),
        (base / "colmap/images.txt", base / "colmap/cameras_undistorted.txt", False),
    ]
    if "3dovs" in root or "lerf_ovs" in root:
        point_cloud_candidates = [base / "sparse/0/points3D_deva.ply"]
    elif "scannet" in root or "mipnerf360" in root:
        point_cloud_candidates = [base / "points3D.ply"]
    else:
        point_cloud_candidates = [base / "sparse/0/points3D.ply"]

    image_file = camera_file = None
    binary = True
    for candidate_image, candidate_camera, is_binary in metadata_candidates:
        if candidate_image.exists() and candidate_camera.exists():
            image_file, camera_file, binary = candidate_image, candidate_camera, is_binary
            break
    if image_file is None or camera_file is None:
        raise FileNotFoundError(f"No COLMAP camera files found under {root}")
    point_cloud_file = next((candidate for candidate in point_cloud_candidates if candidate.exists()), None)
    if point_cloud_file is None:
        raise FileNotFoundError(f"No supported point cloud file found under {root}")
    depth_param_file = base / "sparse/0/depth_params.json"
    return SceneLayout(
        root=base,
        image_dir=image_dir,
        mask_dir=mask_dir,
        depth_dir=depth_dir,
        image_file=image_file,
        camera_file=camera_file,
        binary=binary,
        point_cloud_file=point_cloud_file,
        depth_param_file=depth_param_file if depth_param_file.exists() else None,
    )


def compute_nerf_normalization(camera_records: list[FrameRecord]) -> dict:
    centers = []
    for record in camera_records:
        world_to_camera = world_to_view_matrix(record.rotation, record.translation)
        camera_to_world = np.linalg.inv(world_to_camera)
        centers.append(camera_to_world[:3, 3:4])
    stacked = np.hstack(centers)
    center = np.mean(stacked, axis=1, keepdims=True)
    radius = np.max(np.linalg.norm(stacked - center, axis=0, keepdims=True)) * 1.1
    return {"translate": -center.flatten(), "radius": radius}


def load_point_cloud(path: str) -> PointCloudSample:
    ply = PlyData.read(path)
    vertex = ply["vertex"]
    points = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1)
    colors = (
        np.stack([vertex["red"], vertex["green"], vertex["blue"]], axis=1).astype(np.float32) / 255.0
        if {"red", "green", "blue"}.issubset(vertex.data.dtype.names)
        else np.zeros_like(points, dtype=np.float32)
    )
    normals = (
        np.stack([vertex["nx"], vertex["ny"], vertex["nz"]], axis=1).astype(np.float32)
        if {"nx", "ny", "nz"}.issubset(vertex.data.dtype.names)
        else np.zeros_like(points, dtype=np.float32)
    )
    labels = np.asarray(vertex["label"]).astype(np.uint8) if "label" in vertex.data.dtype.names else np.zeros(points.shape[0], dtype=np.uint8)
    return PointCloudSample(points=points, colors=colors, normals=normals, label_ids=labels)


def write_point_cloud(path: str, points: np.ndarray, colors: np.ndarray, labels: np.ndarray) -> None:
    labels = labels.reshape(-1, 1)
    normals = np.zeros_like(points)
    rgb = np.clip(colors * 255.0 if colors.dtype.kind == "f" else colors, 0, 255).astype(np.uint8)
    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"), ("label", "u1"),
    ]
    packed = np.concatenate([points, normals, rgb, labels], axis=1)
    structured = np.empty(points.shape[0], dtype=dtype)
    structured[:] = list(map(tuple, packed))
    PlyData([PlyElement.describe(structured, "vertex")]).write(path)


def read_camera_records(layout: SceneLayout) -> list[FrameRecord]:
    depth_params = None
    if layout.depth_param_file is not None:
        with open(layout.depth_param_file, "r", encoding="utf-8") as handle:
            depth_params = json.load(handle)

    extrinsics = read_extrinsics_binary(str(layout.image_file)) if layout.binary else read_extrinsics_text(str(layout.image_file))
    intrinsics = read_intrinsics_binary(str(layout.camera_file)) if layout.binary else read_intrinsics_text(str(layout.camera_file))

    def build_record(extrinsic):
        intr = intrinsics[extrinsic.camera_id]
        if intr.model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL"}:
            fx = fy = intr.params[0]
            cx, cy = intr.params[1], intr.params[2]
        elif intr.model in {"PINHOLE", "OPENCV"}:
            fx, fy, cx, cy = intr.params[:4]
        else:
            raise ValueError(f"Unsupported COLMAP camera model: {intr.model}")

        image_path = layout.image_dir / extrinsic.name
        if not image_path.exists():
            return None
        image_name = image_path.stem
        mask_path = layout.mask_dir / f"{image_name}.png"
        alpha_mask = Image.open(mask_path) if layout.mask_dir is not None and mask_path.exists() else None
        depth = None
        if layout.depth_dir is not None:
            depth_path = layout.depth_dir / extrinsic.name.replace(".JPG", ".png").replace(".jpg", ".png")
            if depth_path.exists():
                depth = cv2.imread(str(depth_path), -1).astype(np.float32) / float(2**16)
        params = depth_params.get(image_name) if depth_params is not None else None
        return FrameRecord(
            uid=intr.id,
            rotation=quaternion_to_rotation(extrinsic.qvec).T,
            translation=np.asarray(extrinsic.tvec),
            fov_y=focal_to_fov(fy, intr.height),
            fov_x=focal_to_fov(fx, intr.width),
            cx=float(cx),
            cy=float(cy),
            image=Image.open(image_path),
            image_path=str(image_path),
            image_name=image_name,
            width=int(intr.width),
            height=int(intr.height),
            alpha_mask=alpha_mask,
            depth=depth,
            depth_params=params,
        )

    with concurrent.futures.ThreadPoolExecutor() as pool:
        records = [record for record in pool.map(build_record, extrinsics.values()) if record is not None]
    return sorted(records, key=lambda record: record.image_path)


def split_records(records: list[FrameRecord], eval_mode: bool, llffhold: int) -> tuple[list[FrameRecord], list[FrameRecord]]:
    if eval_mode:
        return (
            [record for index, record in enumerate(records) if index % llffhold != 0],
            [record for index, record in enumerate(records) if index % llffhold == 0],
        )
    return (
        [record for record in records if "test" not in record.image_name],
        [record for record in records if "test" in record.image_name],
    )


def load_colmap_bundle(root: str, eval_mode: bool, images: str, depths: str, masks: str, add_mask: bool, add_depth: bool, llffhold: int = 32) -> SceneBundle:
    layout = discover_colmap_scene(root, images, depths, masks, add_mask, add_depth)
    records = read_camera_records(layout)
    train_records, test_records = split_records(records, eval_mode, llffhold)
    return SceneBundle(
        point_cloud=load_point_cloud(str(layout.point_cloud_file)),
        train_records=train_records,
        test_records=test_records,
        normalization=compute_nerf_normalization(train_records),
        ply_path=str(layout.point_cloud_file),
    )


def build_camera(record: FrameRecord, uid: int, resolution_scale, args):
    original_width, original_height = record.image.size
    if args.resolution in [1, 2, 4, 8]:
        resolution = round(original_width / (resolution_scale * args.resolution)), round(original_height / (resolution_scale * args.resolution))
    else:
        if args.resolution == -1 and original_width > 1600:
            downsample = original_width / 1600
        elif args.resolution == -1:
            downsample = 1.0
        else:
            downsample = original_width / args.resolution
        scale = float(downsample) * float(resolution_scale)
        resolution = (int(original_width / scale), int(original_height / scale))
    return RenderCamera(
        FrameRecord(
            uid=uid,
            rotation=record.rotation,
            translation=record.translation,
            fov_y=record.fov_y,
            fov_x=record.fov_x,
            cx=record.cx,
            cy=record.cy,
            image=record.image,
            image_path=record.image_path,
            image_name=record.image_name,
            width=record.width,
            height=record.height,
            alpha_mask=record.alpha_mask,
            depth=record.depth,
            depth_params=record.depth_params,
        ),
        resolution=resolution,
        resolution_scale=resolution_scale,
        data_device=args.data_device,
        data_format=args.data_format,
        scene_translation=np.asarray(args.center, dtype=np.float32),
        scene_scale=float(args.scale),
    )


def build_camera_list(records, resolution_scale, args):
    return [build_camera(record, index, resolution_scale, args) for index, record in enumerate(records)]


def camera_to_json(index, record: FrameRecord):
    camera_to_world = np.linalg.inv(world_to_view_matrix(record.rotation, record.translation))
    return {
        "id": index,
        "img_name": record.image_name,
        "width": record.width,
        "height": record.height,
        "position": camera_to_world[:3, 3].tolist(),
        "rotation": camera_to_world[:3, :3].tolist(),
        "fy": fov_to_focal(record.fov_y, record.height),
        "fx": fov_to_focal(record.fov_x, record.width),
    }


class TrainingScene:
    def __init__(self, args, gaussians, load_iteration=None, shuffle=True, logger=None, weed_ratio=0.0):
        self.model_path = args.model_path
        self.resolution_scales = args.resolution_scales
        self.gaussians = gaussians
        self.gaussians.weed_ratio = weed_ratio
        self.background = self._background_from_args(args)

        if args.data_format != "colmap":
            raise NotImplementedError("VRoom core currently supports COLMAP datasets only.")

        bundle = load_colmap_bundle(args.source_path, args.eval, args.images, args.depths, args.masks, args.add_mask, args.add_depth, args.llffhold)
        if shuffle:
            rng = np.random.default_rng(0)
            rng.shuffle(bundle.train_records)
            rng.shuffle(bundle.test_records)

        self.cameras_extent = bundle.normalization["radius"]
        self.gaussians.set_appearance(len(bundle.train_records))
        self.train_cameras = {}
        self.test_cameras = {}

        if load_iteration:
            iteration_dir = os.path.join(self.model_path, "point_cloud", f"iteration_{load_iteration}")
            self.gaussians.load_ply(os.path.join(iteration_dir, "point_cloud.ply"))
            self.gaussians.load_mlp_checkpoints(iteration_dir)
        else:
            sampled = self.save_input_point_cloud(bundle.point_cloud, args.ratio, os.path.join(self.model_path, "input.ply"))
            with open(os.path.join(self.model_path, "cameras.json"), "w", encoding="utf-8") as handle:
                json.dump([camera_to_json(index, record) for index, record in enumerate(bundle.test_records + bundle.train_records)], handle)
            self.gaussians.initialize_anchors(sampled, self.cameras_extent, args.global_appearance, logger)

        for scale in self.resolution_scales:
            self.train_cameras[scale] = build_camera_list(bundle.train_records, scale, args)
            self.test_cameras[scale] = build_camera_list(bundle.test_records, scale, args)

    def _background_from_args(self, args):
        if args.random_background:
            return torch.rand(3, dtype=torch.float32, device=self.gaussians.device)
        if args.white_background:
            return torch.ones(3, dtype=torch.float32, device=self.gaussians.device)
        return torch.zeros(3, dtype=torch.float32, device=self.gaussians.device)

    def save_input_point_cloud(self, point_cloud: PointCloudSample, ratio: int, path: str) -> PointCloudSample:
        stride = max(int(ratio), 1)
        sampled = PointCloudSample(
            points=point_cloud.points[::stride],
            colors=point_cloud.colors[::stride],
            normals=point_cloud.normals[::stride],
            label_ids=point_cloud.label_ids[::stride],
        )
        write_point_cloud(path, sampled.points, sampled.colors, sampled.label_ids)
        return sampled

    def getTrainCameras(self):
        cameras = []
        for scale in self.resolution_scales:
            cameras.extend(self.train_cameras[scale])
        return cameras

    def getTestCameras(self):
        cameras = []
        for scale in self.resolution_scales:
            cameras.extend(self.test_cameras[scale])
        return cameras

