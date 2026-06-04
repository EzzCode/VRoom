from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass
import json
import os
from pathlib import Path
import struct
from typing import Any, Optional

import cv2
import numpy as np
import torch
from torch import nn
from PIL import Image
from plyfile import PlyData, PlyElement

from gstrain.vroom_core.utilities.utils.utils import (
    pil_image_to_tensor,
    projection_matrix,
    world_to_view_matrix,
    PointCloudSample,
    focal_to_fov,
    fov_to_focal,
    SemanticsManager,
    CheckpointManager,
)
from gstrain.vroom_core.core.model.anchor_field import AnchorCloudData


@dataclass(frozen=True)
class FrameRecord:
    uid: int
    rotation: np.ndarray
    translation: np.ndarray
    fov_y: float
    fov_x: float
    cx: float
    cy: float
    image: Image.Image
    image_path: str
    image_name: str
    width: int
    height: int
    alpha_mask: Optional[Image.Image] = None
    depth: Optional[np.ndarray] = None
    depth_params: Optional[dict] = None


class RenderCamera(nn.Module):
    def __init__(
        self,
        record: FrameRecord,
        resolution: tuple[int, int],
        resolution_scale: float,
        data_device: str = "cuda",
        data_format: str = "colmap",
        scene_translation: np.ndarray | None = None,
        scene_scale: float = 1.0,
        args: Optional[Any] = None,
        alias_merge_map: Optional[dict[int, int]] = None,
    ) -> None:
        super().__init__()
        self.uid = record.uid
        self.colmap_id = record.uid
        self.R = record.rotation
        self.T = record.translation
        self.FoVx = record.fov_x
        self.FoVy = record.fov_y
        self.image_name = record.image_name
        self.image_path = record.image_path
        self.resolution_scale = resolution_scale
        self.width = record.width
        self.height = record.height
        self.data_device = torch.device(data_device)
        self.znear = 0.01
        self.zfar = 100.0
        self.args = args
        self.alias_merge_map = alias_merge_map

        rgba = pil_image_to_tensor(record.image, resolution).to(self.data_device)
        self.original_image = rgba[:3].clamp(0.0, 1.0)
        self.alpha_mask = self._resolve_alpha_mask(record, resolution, rgba)
        self.image_width = int(self.original_image.shape[2])
        self.image_height = int(self.original_image.shape[1])
        self.invdepthmap = None
        self.depth_mask = None

        translation = (
            np.zeros(3, dtype=np.float32)
            if scene_translation is None
            else np.asarray(scene_translation, dtype=np.float32)
        )
        world_view = world_to_view_matrix(
            self.R, self.T, translation, float(scene_scale)
        )
        self.world_view_transform = torch.tensor(
            world_view, dtype=torch.float32, device=self.data_device
        ).transpose(0, 1)
        self.projection_matrix = (
            projection_matrix(self.znear, self.zfar, self.FoVx, self.FoVy)
            .transpose(0, 1)
            .to(self.data_device)
        )
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0) @ self.projection_matrix.unsqueeze(0)
        ).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

        self.cx = record.cx * resolution[0] / record.image.size[0]
        self.cy = record.cy * resolution[1] / record.image.size[1]
        self.fx = self.image_width / (2.0 * np.tan(self.FoVx * 0.5))
        self.fy = self.image_height / (2.0 * np.tan(self.FoVy * 0.5))
        self.c2w = self.world_view_transform.transpose(0, 1).inverse()
        self.object_mask = self._load_object_mask(resolution)

    def _resolve_alpha_mask(
        self, record: FrameRecord, resolution: tuple[int, int], rgba: torch.Tensor
    ) -> torch.Tensor:
        if record.alpha_mask is not None:
            return pil_image_to_tensor(record.alpha_mask, resolution).to(
                self.data_device
            )
        if rgba.shape[0] == 4:
            return rgba[3:4]
        return torch.ones_like(rgba[:1])

    def _load_object_mask(self, resolution):
        source = Path(self.image_path)
        candidates = [
            Path(str(source).replace("images", "object_mask_deva")).with_suffix(".png"),
            Path(str(source).replace("images_all", "object_mask")).with_suffix(".png"),
            Path(str(source).replace("images", "object_mask")).with_suffix(".png"),
        ]
        if self.args is not None:
            frames_dir = getattr(self.args, "frames", "images")
            masks_dir = getattr(self.args, "masks", "masks")
            candidate = Path(str(source).replace(f"/{frames_dir}/", f"/{masks_dir}/")).with_suffix(".png")
            candidates.insert(0, candidate)
            candidate2 = Path(str(source).replace(frames_dir, masks_dir)).with_suffix(".png")
            candidates.insert(0, candidate2)

        for candidate in candidates:
            if candidate == source or not candidate.exists():
                continue
            image = Image.open(candidate).convert("L")
            array = np.array(image.resize(resolution), dtype=np.uint8, copy=True)
            if self.alias_merge_map is not None:
                for src_val, dst_val in self.alias_merge_map.items():
                    array[array == src_val] = dst_val
            return torch.from_numpy(array)
        return torch.zeros((resolution[1], resolution[0]), dtype=torch.uint8)


@dataclass(frozen=True)
class ColmapCamera:
    id: int
    model: str
    width: int
    height: int
    params: np.ndarray


@dataclass(frozen=True)
class ColmapImage:
    id: int
    qvec: np.ndarray
    tvec: np.ndarray
    camera_id: int
    name: str
    xys: np.ndarray
    point3D_ids: np.ndarray


CAMERA_MODEL_SPECS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}


def quaternion_to_rotation(qvec: np.ndarray) -> np.ndarray:
    w, x, y, z = [float(value) for value in qvec]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def _read_exact(handle, count: int) -> bytes:
    payload = handle.read(count)
    if len(payload) != count:
        raise EOFError("Unexpected end of COLMAP file.")
    return payload


def _unpack(handle, fmt: str):
    return struct.unpack("<" + fmt, _read_exact(handle, struct.calcsize("<" + fmt)))


def read_intrinsics_text(path: str) -> dict[int, ColmapCamera]:
    cameras: dict[int, ColmapCamera] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split()
            camera_id = int(fields[0])
            cameras[camera_id] = ColmapCamera(
                id=camera_id,
                model=fields[1],
                width=int(fields[2]),
                height=int(fields[3]),
                params=np.asarray(
                    [float(value) for value in fields[4:]], dtype=np.float64
                ),
            )
    return cameras


def read_extrinsics_text(path: str) -> dict[int, ColmapImage]:
    images: dict[int, ColmapImage] = {}
    with open(path, "r", encoding="utf-8") as handle:
        while True:
            header = handle.readline()
            if not header:
                break
            stripped = header.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split()
            image_id = int(fields[0])
            qvec = np.asarray([float(value) for value in fields[1:5]], dtype=np.float64)
            tvec = np.asarray([float(value) for value in fields[5:8]], dtype=np.float64)
            feature_line = handle.readline().strip().split()
            if feature_line:
                xys = np.column_stack(
                    [
                        np.asarray(
                            [float(value) for value in feature_line[0::3]],
                            dtype=np.float64,
                        ),
                        np.asarray(
                            [float(value) for value in feature_line[1::3]],
                            dtype=np.float64,
                        ),
                    ]
                )
                point_ids = np.asarray(
                    [int(value) for value in feature_line[2::3]], dtype=np.int64
                )
            else:
                xys = np.zeros((0, 2), dtype=np.float64)
                point_ids = np.zeros((0,), dtype=np.int64)
            images[image_id] = ColmapImage(
                id=image_id,
                qvec=qvec,
                tvec=tvec,
                camera_id=int(fields[8]),
                name=fields[9],
                xys=xys,
                point3D_ids=point_ids,
            )
    return images


def read_intrinsics_binary(path: str) -> dict[int, ColmapCamera]:
    cameras: dict[int, ColmapCamera] = {}
    with open(path, "rb") as handle:
        (count,) = _unpack(handle, "Q")
        for _ in range(count):
            camera_id, model_id, width, height = _unpack(handle, "iiQQ")
            model_name, param_count = CAMERA_MODEL_SPECS[model_id]
            params = np.asarray(_unpack(handle, "d" * param_count), dtype=np.float64)
            cameras[camera_id] = ColmapCamera(
                camera_id, model_name, width, height, params
            )
    return cameras


def read_extrinsics_binary(path: str) -> dict[int, ColmapImage]:
    images: dict[int, ColmapImage] = {}
    with open(path, "rb") as handle:
        (count,) = _unpack(handle, "Q")
        for _ in range(count):
            unpacked = _unpack(handle, "idddddddi")
            image_id = unpacked[0]
            qvec = np.asarray(unpacked[1:5], dtype=np.float64)
            tvec = np.asarray(unpacked[5:8], dtype=np.float64)
            camera_id = unpacked[8]
            name_bytes = bytearray()
            while True:
                (char,) = _unpack(handle, "c")
                if char == b"\x00":
                    break
                name_bytes.extend(char)
            (num_points,) = _unpack(handle, "Q")
            if num_points > 0:
                packed = _unpack(handle, "ddq" * num_points)
                xys = np.column_stack(
                    [
                        np.asarray(packed[0::3], dtype=np.float64),
                        np.asarray(packed[1::3], dtype=np.float64),
                    ]
                )
                point_ids = np.asarray(packed[2::3], dtype=np.int64)
            else:
                xys = np.zeros((0, 2), dtype=np.float64)
                point_ids = np.zeros((0,), dtype=np.int64)
            images[image_id] = ColmapImage(
                id=image_id,
                qvec=qvec,
                tvec=tvec,
                camera_id=camera_id,
                name=name_bytes.decode("utf-8"),
                xys=xys,
                point3D_ids=point_ids,
            )
    return images


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


def load_points3D_bin_helper(path: Path) -> dict:
    points = {}
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            blob = struct.unpack("<QdddBBBd", f.read(43))
            pid, xyz, rgb = blob[0], blob[1:4], blob[4:7]
            (tlen,) = struct.unpack("<Q", f.read(8))
            f.seek(8 * tlen, 1)  # Skip 2D-3D tracks
            points[pid] = (np.array(xyz), np.array(rgb, dtype=np.uint8))
    return points


def discover_colmap_scene(
    root: str, images: str, depths: str, masks: str, add_mask: bool, add_depth: bool
) -> SceneLayout:
    base = Path(root)
    image_dir = base / images
    mask_dir = (base / masks) if add_mask else None
    depth_dir = None
    metadata_candidates = [
        (base / "sparse/0/images.bin", base / "sparse/0/cameras.bin", True),
        (base / "sparse/0/images.txt", base / "sparse/0/cameras.txt", False),
        (base / "colmap/images.txt", base / "colmap/cameras_undistorted.txt", False),
    ]
    point_cloud_candidates = []
    if "3dovs" in root or "lerf_ovs" in root:
        point_cloud_candidates.append(base / "sparse/0/points3D_deva.ply")
    elif "scannet" in root or "mipnerf360" in root:
        point_cloud_candidates.append(base / "points3D.ply")
    point_cloud_candidates.extend([
        base / "labeled_output/points3D_labeled.ply",
        base / "sparse/0/points3D.ply",
    ])

    image_file = camera_file = None
    binary = True
    for candidate_image, candidate_camera, is_binary in metadata_candidates:
        if candidate_image.exists() and candidate_camera.exists():
            image_file, camera_file, binary = (
                candidate_image,
                candidate_camera,
                is_binary,
            )
            break
    if image_file is None or camera_file is None:
        raise FileNotFoundError(f"No COLMAP camera files found under {root}")
    point_cloud_file = next(
        (candidate for candidate in point_cloud_candidates if candidate.exists()), None
    )
    if point_cloud_file is None:
        bin_candidate = base / "sparse/0/points3D.bin"
        if bin_candidate.exists():
            ply_path = base / "sparse/0/points3D.ply"
            try:
                pts_dict = load_points3D_bin_helper(bin_candidate)
                xyz = np.array([v[0] for v in pts_dict.values()], dtype=np.float32)
                rgb = np.array([v[1] for v in pts_dict.values()], dtype=np.float32) / 255.0
                labels = np.zeros(len(xyz), dtype=np.uint8)
                write_point_cloud(str(ply_path), xyz, rgb, labels)
                point_cloud_file = ply_path
            except Exception as e:
                raise FileNotFoundError(f"Failed to auto-convert points3D.bin to points3D.ply: {e}")
        else:
            raise FileNotFoundError(f"No supported point cloud file (ply or bin) found under {root}")
    depth_param_file = None
    return SceneLayout(
        root=base,
        image_dir=image_dir,
        mask_dir=mask_dir,
        depth_dir=depth_dir,
        image_file=image_file,
        camera_file=camera_file,
        binary=binary,
        point_cloud_file=point_cloud_file,
        depth_param_file=depth_param_file,
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
        np.stack([vertex["red"], vertex["green"], vertex["blue"]], axis=1).astype(
            np.float32
        )
        / 255.0
        if {"red", "green", "blue"}.issubset(vertex.data.dtype.names)
        else np.zeros_like(points, dtype=np.float32)
    )
    normals = (
        np.stack([vertex["nx"], vertex["ny"], vertex["nz"]], axis=1).astype(np.float32)
        if {"nx", "ny", "nz"}.issubset(vertex.data.dtype.names)
        else np.zeros_like(points, dtype=np.float32)
    )
    labels = (
        np.asarray(vertex["label"]).astype(np.uint8)
        if "label" in vertex.data.dtype.names
        else np.zeros(points.shape[0], dtype=np.uint8)
    )
    return PointCloudSample(
        points=points, colors=colors, normals=normals, label_ids=labels
    )


def write_point_cloud(
    path: str, points: np.ndarray, colors: np.ndarray, labels: np.ndarray
) -> None:
    labels = labels.reshape(-1, 1)
    normals = np.zeros_like(points)
    rgb = np.clip(
        colors * 255.0 if colors.dtype.kind == "f" else colors, 0, 255
    ).astype(np.uint8)
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
        ("label", "u1"),
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

    extrinsics = (
        read_extrinsics_binary(str(layout.image_file))
        if layout.binary
        else read_extrinsics_text(str(layout.image_file))
    )
    intrinsics = (
        read_intrinsics_binary(str(layout.camera_file))
        if layout.binary
        else read_intrinsics_text(str(layout.camera_file))
    )

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
        alpha_mask = (
            Image.open(layout.mask_dir / extrinsic.name)
            if layout.mask_dir is not None
            and (layout.mask_dir / extrinsic.name).exists()
            else None
        )
        depth = None
        if layout.depth_dir is not None:
            depth_path = layout.depth_dir / extrinsic.name.replace(
                ".JPG", ".png"
            ).replace(".jpg", ".png")
            if depth_path.exists():
                depth = cv2.imread(str(depth_path), -1).astype(np.float32) / float(
                    2**16
                )
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
        records = [
            record
            for record in pool.map(build_record, extrinsics.values())
            if record is not None
        ]
    return sorted(records, key=lambda record: record.image_path)


def split_records(
    records: list[FrameRecord], eval_mode: bool, llffhold: int
) -> tuple[list[FrameRecord], list[FrameRecord]]:
    if eval_mode:
        return (
            [record for index, record in enumerate(records) if index % llffhold != 0],
            [record for index, record in enumerate(records) if index % llffhold == 0],
        )
    return (
        [record for record in records if "test" not in record.image_name],
        [record for record in records if "test" in record.image_name],
    )


def load_colmap_bundle(
    root: str,
    eval_mode: bool,
    images: str,
    depths: str,
    masks: str,
    add_mask: bool,
    add_depth: bool,
    llffhold: int = 32,
) -> SceneBundle:
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


def build_camera(record: FrameRecord, uid: int, resolution_scale, args, alias_merge_map=None):
    original_width, original_height = record.image.size
    if args.image_downscale_factor in [1, 2, 4, 8]:
        resolution = (
            round(original_width / (resolution_scale * args.image_downscale_factor)),
            round(original_height / (resolution_scale * args.image_downscale_factor)),
        )
    else:
        if args.image_downscale_factor == -1 and original_width > 1600:
            downsample = original_width / 1600
        elif args.image_downscale_factor == -1:
            downsample = 1.0
        else:
            downsample = original_width / args.image_downscale_factor
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
        data_device=args.dataset_storage_device,
        data_format=args.data_format,
        scene_translation=np.asarray(args.camera_center, dtype=np.float32),
        scene_scale=float(args.camera_scale),
        args=args,
        alias_merge_map=alias_merge_map,
    )


def build_camera_list(records, resolution_scale, args, alias_merge_map=None):
    return [
        build_camera(record, index, resolution_scale, args, alias_merge_map)
        for index, record in enumerate(records)
    ]


def camera_to_json(index, record: FrameRecord):
    camera_to_world = np.linalg.inv(
        world_to_view_matrix(record.rotation, record.translation)
    )
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
    def __init__(
        self,
        args,
        anchor_cloud,
        decoder,
        load_iteration=None,
        shuffle=True,
        logger=None,
        weed_ratio=0.0,
    ):
        self.model_path = args.model_path
        self.multiscale_factors = args.multiscale_factors
        self.anchor_cloud = anchor_cloud
        self.decoder = decoder
        self.weed_ratio = weed_ratio
        self.background = self._background_from_args(args)

        if args.data_format != "colmap":
            raise NotImplementedError(
                "VRoom core currently supports COLMAP datasets only."
            )

        bundle = load_colmap_bundle(
            args.dataset_path,
            args.eval,
            args.frames,
            args.depths,
            args.masks,
            args.add_mask,
            args.add_depth,
            args.llffhold,
        )
        if shuffle:
            rng = np.random.default_rng(0)
            rng.shuffle(bundle.train_records)
            rng.shuffle(bundle.test_records)

        self.cameras_extent = bundle.normalization["radius"]
        self.train_cameras = {}
        self.test_cameras = {}

        self.alias_merge_map = None
        merge_map_candidates = [
            os.path.join(args.dataset_path, "labeled_output", "alias_merge_map.json"),
            os.path.join(args.dataset_path, "alias_merge_map.json"),
        ]
        for candidate in merge_map_candidates:
            if os.path.exists(candidate):
                try:
                    with open(candidate, "r", encoding="utf-8") as f:
                        raw_map = json.load(f)
                        self.alias_merge_map = {int(k): int(v) for k, v in raw_map.items()}
                    if logger:
                        logger.info(f"Loaded alias merge map from {candidate} with {len(self.alias_merge_map)} entries.")
                    break
                except Exception as e:
                    if logger:
                        logger.warning(f"Failed to load alias merge map from {candidate}: {e}")

        if load_iteration:
            checkpoints = CheckpointManager(self.anchor_cloud, self.decoder)
            iteration_dir = os.path.join(
                self.model_path, "point_cloud", f"iteration_{load_iteration}"
            )
            payload = checkpoints.load_anchor_field(
                os.path.join(iteration_dir, "point_cloud.ply")
            )

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
                quantization_size=float(torch.exp(payload["log_scaling"][:, :3]).mean().item())
                if payload["log_scaling"].numel() > 0
                else 1.0,
            )
            self.anchor_cloud.set_anchors_cloud(seeds)
            checkpoints.load_decoder(Path(iteration_dir))
        else:
            sampled = self.save_input_point_cloud(
                bundle.point_cloud,
                args.pc_downsampling_ratio,
                os.path.join(self.model_path, "input.ply"),
            )
            with open(
                os.path.join(self.model_path, "cameras.json"), "w", encoding="utf-8"
            ) as handle:
                json.dump(
                    [
                        camera_to_json(index, record)
                        for index, record in enumerate(
                            bundle.test_records + bundle.train_records
                        )
                    ],
                    handle,
                )

            self.anchor_cloud.initialize_anchors(sampled)

        for scale in self.multiscale_factors:
            self.train_cameras[scale] = build_camera_list(
                bundle.train_records, scale, args, self.alias_merge_map
            )
            self.test_cameras[scale] = build_camera_list(
                bundle.test_records, scale, args, self.alias_merge_map
            )

    def _background_from_args(self, args):
        if args.random_background:
            return torch.rand(3, dtype=torch.float32, device=self.anchor_cloud.device)
        if args.white_background:
            return torch.ones(3, dtype=torch.float32, device=self.anchor_cloud.device)
        return torch.zeros(3, dtype=torch.float32, device=self.anchor_cloud.device)

    def save_input_point_cloud(
        self, point_cloud: PointCloudSample, ratio: int, path: str
    ) -> PointCloudSample:
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
        for scale in self.multiscale_factors:
            cameras.extend(self.train_cameras[scale])
        return cameras

    def getTestCameras(self):
        cameras = []
        for scale in self.multiscale_factors:
            cameras.extend(self.test_cameras[scale])
        return cameras
