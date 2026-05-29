"""
Frame W - World
Frame L - Local Object (centered on object centroid, axis-aligned to scene up)
Frame V - Stable Diffusion Virtual Camera

W2C - World to Camera
L2V - Local to Virtual
"""
import numpy as np
from dataclasses import dataclass
from .helpers import normalize


def look_at(eye, target, up):
    """Axes: x-right, y-down, z-forward (right hand)"""
    eye = np.asarray(eye, dtype=np.float32)
    forward = normalize(np.asarray(target) - eye)
    up = normalize(up)
    right = normalize(np.cross(forward, up))
    cam_up = np.cross(right, forward)
    R_w2c = np.stack([right, -cam_up, forward], axis=0)
    T_w2c = -R_w2c @ eye
    return R_w2c, T_w2c


# L→V mapping: +X_L (orbit front) → +Z_V, +Z_L (up) → +Y_V, +Y_L (right) → +X_V
R_L2V = np.array([
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0],
], dtype=np.float32)


def orbit_position(azimuth_deg, elevation_deg):
    """
    azimuth=0 -> +Z_V (front), positive azimuth clockwise from above looking down
    elevation > 0 -> lifted toward +Y_V.
    """
    azimuth = np.deg2rad(float(azimuth_deg))
    elevation = np.deg2rad(float(elevation_deg))
    return np.array([
        np.sin(azimuth) * np.cos(elevation),
        np.sin(elevation),
        np.cos(azimuth) * np.cos(elevation),
    ], dtype=np.float32)


@dataclass
class ObjectFrame:
    """Coordinate frame anchored to an object: bridges World, Local, and Virtual spaces."""
    centroid: np.ndarray   
    up: np.ndarray         
    base_dir: np.ndarray   
    radius: float          
    R: np.ndarray = None   

    def __post_init__(self):
        up = normalize(self.up)
        base = np.asarray(self.base_dir).reshape(3)
        # project base onto the plane perpendicular to up
        base = base - float(np.dot(base, up)) * up
        base = normalize(base)
        right = np.cross(up, base)              # +Y_L = Z_L × X_L
        base = np.cross(right, up)              # +X_L re-orthogonalized
        self.up = up
        self.base_dir = base
        self.centroid = np.asarray(self.centroid, dtype=np.float32).reshape(3)
        self.radius = float(self.radius)
        self.R = np.stack([base, right, up], axis=0)   # rows: X_L, Y_L, Z_L

    def world_to_local(self, pts):
        pts = np.asarray(pts, dtype=np.float32).reshape(-1, 3)
        return (self.R @ (pts - self.centroid).T).T

    def local_to_world(self, pts):
        pts = np.asarray(pts, dtype=np.float32).reshape(-1, 3)
        return (self.R.T @ pts.T).T + self.centroid

    def virtual_to_world_camera(self, azimuth_deg, elevation_deg):
        """Returns (R_w2c, T_w2c, camera_pos_world) for a Virtual orbit view."""
        c_V = orbit_position(azimuth_deg, elevation_deg)
        c_L = (R_L2V.T @ c_V) * self.radius
        c_W = self.local_to_world(c_L.reshape(1, 3))[0]
        R_w2c, T_w2c = look_at(c_W, self.centroid, self.up)
        return R_w2c, T_w2c, c_W

    def world_to_virtual(self, camera_pos):
        """Map a world camera position to (azimuth_deg, elevation_deg) in V."""
        c_L = self.world_to_local(camera_pos)[0]
        c_V = R_L2V @ c_L
        r = float(np.linalg.norm(c_V))
        if r < 1e-9:
            return 0.0, 0.0
        c_V = c_V / r
        az = float(np.degrees(np.arctan2(c_V[0], c_V[2])))
        el = float(np.degrees(np.arcsin(np.clip(c_V[1], -1.0, 1.0))))
        return az, el


def scale_intrinsics(K, src_width, src_height, out_size=576):
    """Scale camera intrinsics K to a square output resolution.
    Preserves angular FoV; centers the principal point."""
    K = np.asarray(K, dtype=np.float32).reshape(3, 3)
    fx = float(K[0, 0]) * (out_size / float(src_width))
    fy = float(K[1, 1]) * (out_size / float(src_height))
    cx = out_size / 2.0
    cy = out_size / 2.0
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
