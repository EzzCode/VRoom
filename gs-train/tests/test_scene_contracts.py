import struct
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vroom_core.data.camera_system import FrameRecord as CameraRecord
from vroom_core.data.colmap_io import quaternion_to_rotation, read_extrinsics_binary, read_extrinsics_text, read_intrinsics_binary, read_intrinsics_text
from vroom_core.data.scene_pipeline import compute_nerf_normalization


def _dummy_image():
    return Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))


def test_quaternion_to_rotation_identity():
    rotation = quaternion_to_rotation(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64))
    assert np.allclose(rotation, np.eye(3), atol=1e-6)


def test_text_colmap_readers_parse_expected_fields(tmp_path):
    cameras = tmp_path / "cameras.txt"
    images = tmp_path / "images.txt"

    cameras.write_text("# cameras\n1 PINHOLE 640 480 500.0 510.0 320.0 240.0\n")
    images.write_text(
        "# images\n"
        "1 1 0 0 0 0 0 0 1 frame.png\n"
        "0.0 0.0 -1\n"
    )

    intrinsics = read_intrinsics_text(str(cameras))
    extrinsics = read_extrinsics_text(str(images))

    assert intrinsics[1].width == 640
    assert intrinsics[1].height == 480
    assert np.allclose(intrinsics[1].params, np.array([500.0, 510.0, 320.0, 240.0]))
    assert extrinsics[1].name == "frame.png"
    assert np.allclose(extrinsics[1].qvec, np.array([1.0, 0.0, 0.0, 0.0]))


def test_binary_colmap_readers_parse_expected_fields(tmp_path):
    camera_path = tmp_path / "cameras.bin"
    image_path = tmp_path / "images.bin"

    with open(camera_path, "wb") as handle:
        handle.write(struct.pack("<Q", 1))
        handle.write(struct.pack("<iiQQ", 1, 1, 640, 480))
        handle.write(struct.pack("<4d", 500.0, 510.0, 320.0, 240.0))

    with open(image_path, "wb") as handle:
        handle.write(struct.pack("<Q", 1))
        handle.write(struct.pack("<i", 1))
        handle.write(struct.pack("<4d", 1.0, 0.0, 0.0, 0.0))
        handle.write(struct.pack("<3d", 0.0, 0.0, 0.0))
        handle.write(struct.pack("<i", 1))
        handle.write(b"frame.png\x00")
        handle.write(struct.pack("<Q", 0))

    intrinsics = read_intrinsics_binary(str(camera_path))
    extrinsics = read_extrinsics_binary(str(image_path))

    assert intrinsics[1].model == "PINHOLE"
    assert intrinsics[1].width == 640
    assert extrinsics[1].camera_id == 1
    assert extrinsics[1].name == "frame.png"


def test_nerf_normalization_centers_camera_cloud():
    image = _dummy_image()
    records = [
        CameraRecord(
            uid=0,
            rotation=np.eye(3, dtype=np.float32),
            translation=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            fov_y=0.7,
            fov_x=0.7,
            cx=4.0,
            cy=4.0,
            image=image,
            image_path="frame_0.png",
            image_name="frame_0",
            width=8,
            height=8,
        ),
        CameraRecord(
            uid=1,
            rotation=np.eye(3, dtype=np.float32),
            translation=np.array([-2.0, 0.0, 0.0], dtype=np.float32),
            fov_y=0.7,
            fov_x=0.7,
            cx=4.0,
            cy=4.0,
            image=image,
            image_path="frame_1.png",
            image_name="frame_1",
            width=8,
            height=8,
        ),
    ]

    normalization = compute_nerf_normalization(records)

    assert np.allclose(normalization["translate"], np.array([-1.0, 0.0, 0.0]), atol=1e-6)
    assert np.isclose(normalization["radius"], 1.1, atol=1e-6)
