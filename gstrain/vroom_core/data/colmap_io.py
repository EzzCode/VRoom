"""VRoom COLMAP readers."""

from __future__ import annotations

from dataclasses import dataclass
import struct

import numpy as np


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
                params=np.asarray([float(value) for value in fields[4:]], dtype=np.float64),
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
                        np.asarray([float(value) for value in feature_line[0::3]], dtype=np.float64),
                        np.asarray([float(value) for value in feature_line[1::3]], dtype=np.float64),
                    ]
                )
                point_ids = np.asarray([int(value) for value in feature_line[2::3]], dtype=np.int64)
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
            cameras[camera_id] = ColmapCamera(camera_id, model_name, width, height, params)
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

