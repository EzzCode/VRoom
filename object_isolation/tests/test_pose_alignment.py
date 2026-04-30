"""Unit tests for object_isolation.pose_alignment.

Run with:
    python -m pytest object_isolation/tests/test_pose_alignment.py -v
or:
    python -m unittest object_isolation.tests.test_pose_alignment

The tests run on CPU (numpy only) and do not need a GPU or ObjectGS.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

_VROOM_ROOT = Path(__file__).resolve().parents[2]
if str(_VROOM_ROOT) not in sys.path:
    sys.path.insert(0, str(_VROOM_ROOT))

from object_isolation import pose_alignment as pa


def _look_at_w2c(camera_pos, target, up):
    """Reference look-at (world->camera, OpenCV/COLMAP convention).
    Returns ``(R_w2c, T_w2c)`` such that
        p_camera = R_w2c @ p_world + T_w2c
    """
    forward = (np.asarray(target) - np.asarray(camera_pos)).astype(np.float64)
    forward /= np.linalg.norm(forward)
    right = np.cross(np.asarray(up, dtype=np.float64), forward)
    right /= np.linalg.norm(right)
    cam_down = np.cross(forward, right)
    cam_down /= np.linalg.norm(cam_down)
    R = np.stack([right, cam_down, forward], axis=0)
    T = -R @ np.asarray(camera_pos, dtype=np.float64)
    return R, T


class TestPoseAlignment(unittest.TestCase):

    def setUp(self):
        # Object at world origin, +Z up, radius 1.0
        self.center = np.array([0.0, 0.0, 0.0])
        self.up = np.array([0.0, 0.0, 1.0])
        self.radius = 1.0

        # Reference camera at (1.5, 0, 0.5) looking at origin
        self.ref_pos = np.array([1.5, 0.0, 0.5])
        self.R_ref, self.T_ref = _look_at_w2c(self.ref_pos, self.center, self.up)

    def test_W_to_O_maps_center_to_origin(self):
        T_W_O = pa.compute_W_to_O(self.center, self.up, self.R_ref)
        c_h = np.array([self.center[0], self.center[1], self.center[2], 1.0])
        c_o = (T_W_O @ c_h)[:3]
        np.testing.assert_allclose(c_o, [0.0, 0.0, 0.0], atol=1e-9)

    def test_W_to_O_basis_is_orthonormal(self):
        T_W_O = pa.compute_W_to_O(self.center, self.up, self.R_ref)
        R = T_W_O[:3, :3]
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-9)
        self.assertAlmostEqual(np.linalg.det(R), 1.0, places=6)

    def test_O_to_V_reads_back_reference_distance(self):
        T_W_O = pa.compute_W_to_O(self.center, self.up, self.R_ref)
        az0, el0, r_v = pa.compute_O_to_V(T_W_O, self.R_ref, self.T_ref)
        self.assertAlmostEqual(r_v, float(np.linalg.norm(self.ref_pos - self.center)), places=6)

    def test_round_trip_reference_pose(self):
        """With the convention `tile=(0,0) == input view`, lifting a tile
        with zero deltas must recover the reference camera's own pose."""
        T_W_O = pa.compute_W_to_O(self.center, self.up, self.R_ref)
        az0, el0, r_v = pa.compute_O_to_V(T_W_O, self.R_ref, self.T_ref)

        R_back, T_back = pa.z123_tile_to_W(
            tile_az_deg=0.0, tile_el_deg=0.0,
            az0_deg=az0, el0_deg=el0,
            r_v=r_v, T_W_O=T_W_O,
            azimuth_sign=+1, elevation_sign=+1,
        )
        np.testing.assert_allclose(R_back, self.R_ref, atol=1e-6)
        np.testing.assert_allclose(T_back, self.T_ref, atol=1e-6)

    def test_synthetic_delta_az90(self):
        """Tile delta of (+90° az, 0 el) should rotate the camera 90° around
        the up axis from the reference position, in the O frame."""
        T_W_O = pa.compute_W_to_O(self.center, self.up, self.R_ref)
        az0, el0, r_v = pa.compute_O_to_V(T_W_O, self.R_ref, self.T_ref)

        # Tile delta (90, 0) with signs (+1, +1) -> spherical (az0+90, el0)
        R, T = pa.z123_tile_to_W(
            tile_az_deg=90.0, tile_el_deg=0.0,
            az0_deg=az0, el0_deg=el0,
            r_v=r_v, T_W_O=T_W_O,
            azimuth_sign=+1, elevation_sign=+1,
        )
        az_new = np.radians(az0 + 90.0)
        el_new = np.radians(el0)
        expected_C_o = np.array([
            r_v * np.cos(el_new) * np.cos(az_new),
            r_v * np.cos(el_new) * np.sin(az_new),
            r_v * np.sin(el_new),
        ])
        T_O_W = np.linalg.inv(T_W_O)
        expected_center = (T_O_W @ np.array([*expected_C_o, 1.0]))[:3]
        cam_center = -R.T @ T
        np.testing.assert_allclose(cam_center, expected_center, atol=1e-6)

    def test_camera_always_looks_at_origin_in_O(self):
        """For any tile, transforming origin-of-O into the novel camera frame
        must yield a point on the camera's +Z axis (i.e. the camera is
        looking at the object center)."""
        trace = pa.build_full_pose_chain(
            object_center=self.center,
            object_up_world=self.up,
            object_radius=self.radius,
            reference_R_w2c=self.R_ref,
            reference_T_w2c=self.T_ref,
            azimuth_sign=-1, elevation_sign=+1,
        )
        for t in trace.tiles:
            R = np.asarray(t["R_w2c"])
            T = np.asarray(t["T_w2c"])
            # World origin (which is also object_center here) in camera frame
            p_cam = R @ self.center + T
            # The vector from camera to object_center, expressed in camera
            # coords, should be along +Z (the camera-forward axis).
            self.assertGreater(p_cam[2], 0.0, "object should be in front of cam")
            np.testing.assert_allclose(
                p_cam[:2], [0.0, 0.0], atol=1e-6,
                err_msg="object center should land on camera optical axis",
            )

    def test_distance_lock_uses_synthetic_radius(self):
        """The default ``r_v_synth = 1.5 * object_radius`` (Zero123++
        canonical) must be used for novel-view positions, NOT the reference
        camera's real distance."""
        radius = 0.7
        ref_pos = np.array([5.0, 0.0, 0.3])  # very far real-world reference
        R_ref, T_ref = _look_at_w2c(ref_pos, self.center, self.up)

        trace = pa.build_full_pose_chain(
            object_center=self.center,
            object_up_world=self.up,
            object_radius=radius,
            reference_R_w2c=R_ref,
            reference_T_w2c=T_ref,
            azimuth_sign=-1, elevation_sign=+1,
        )
        expected = pa.zero123_native_distance(radius)  # 1.5 * 0.7 = 1.05
        self.assertAlmostEqual(trace.r_v_synth, expected, places=6)
        # Every novel camera should sit at distance r_v_synth from object_center
        for t in trace.tiles:
            R = np.asarray(t["R_w2c"])
            T = np.asarray(t["T_w2c"])
            cam_center = -R.T @ T
            d = float(np.linalg.norm(cam_center - self.center))
            self.assertAlmostEqual(d, expected, places=5)

    def test_zero123_tile_schedule_shape(self):
        sched = pa.zero123plus_v12_tile_schedule()
        self.assertEqual(len(sched), 6)
        azs = [a for a, _ in sched]
        els = [e for _, e in sched]
        self.assertEqual(els, [20.0, -10.0, 20.0, -10.0, 20.0, -10.0])
        self.assertEqual(azs, [30.0, 90.0, 150.0, 210.0, 270.0, 330.0])


if __name__ == "__main__":
    unittest.main()
