"""
Coordinate frames for the object-centric isolation pipeline.

This file owns ALL transform math. Three frames are tracked explicitly:

    Frame W  — World (Scaffold-GS / ObjectGS scene coords)
    Frame L  — Local Object (centered on object centroid, axis-aligned to scene up)
    Frame V  — Diffusion Virtual Camera (canonical convention of the diffusion prior)

Conventions (cameras.json, ObjectGS):
    rotation  = R_c2w   (camera-to-world rotation; columns are camera axes in world)
    position  = C_W     (camera world center)

Derived:
    R_w2c = R_c2w.T
    T_w2c = -R_w2c @ C_W
    point_in_camera = R_w2c @ point_in_world + T_w2c

ASCII layout:

           up_W  (snapped world up)
            |
            |        +Z_L = up_W
            |       /
            |      /
            |     +-------> +X_L = base_dir_W   (orbit zero-azimuth)
            |    /              (median training-camera direction in horizontal plane)
            |   /
            |  c_obj  <-- object centroid (origin of L)
            |
        ---origin of W (cameras.json frame)


Frame V (SV3D canonical):
    +Y_V = up
    +Z_V = into-scene from front camera
    +X_V = right-hand
    Object at origin. Camera at radius `r_norm` (unit-radius orbit).
    A view is parameterized by (azimuth_deg, elevation_deg):
        - azimuth_deg = 0  →  camera at +Z_V looking toward origin
        - azimuth_deg sweeps clockwise looking down +Y_V

The fixed L→V mapping below maps Local +X (front) onto Virtual +Z (front) so
that azimuth 0 in V corresponds to the orbit zero-azimuth in L.

NUMPY-ONLY by design — no torch — so this file can be unit-tested without GPU.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _unit(v: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    """Return v normalized to unit length, or fallback if degenerate."""
    v = np.asarray(v, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n < 1e-12:
        if fallback is None:
            raise ValueError("Zero vector cannot be normalized and no fallback given.")
        return np.asarray(fallback, dtype=np.float64).reshape(3)
    return v / n


def look_at_w2c(camera_center_W: np.ndarray, target_W: np.ndarray, up_W: np.ndarray
                ) -> tuple[np.ndarray, np.ndarray]:
    """COLMAP-convention look_at: returns (R_w2c, T_w2c).

    Camera looks from `camera_center_W` at `target_W` with world up `up_W`.
    Camera axes (rows of R_w2c, expressed in world coords):
        right   =  +X_cam
        -up_cam =  +Y_cam   (image y is downward, OpenCV convention)
        forward =  +Z_cam
    """
    eye = np.asarray(camera_center_W, dtype=np.float64).reshape(3)
    tgt = np.asarray(target_W, dtype=np.float64).reshape(3)
    up = _unit(up_W, np.array([0, 0, 1.0]))

    forward = _unit(tgt - eye, np.array([0, 0, 1.0]))
    # Guard: if forward is parallel to up, pick a different up.
    if abs(float(np.dot(forward, up))) > 0.999:
        up = _unit(np.array([1.0, 0.0, 0.0]) if abs(forward[0]) < 0.9
                   else np.array([0.0, 1.0, 0.0]))
    # OpenCV / COLMAP: x-right, y-down, z-forward (right-handed). With
    # world up `up` and forward `f`, the image-right axis is f × up
    # (not up × f — that would mirror the image horizontally).
    right = _unit(np.cross(forward, up))
    cam_up = _unit(np.cross(right, forward))  # orthonormalize

    R_w2c = np.stack([right, -cam_up, forward], axis=0).astype(np.float32)  # rows
    T_w2c = (-R_w2c @ eye).astype(np.float32)
    return R_w2c, T_w2c


# ─────────────────────────────────────────────────────────────────────────────
# Frame W ↔ Frame L  (World ↔ Local Object)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class WorldLocal:
    """Rigid transform between world frame W and object-local frame L.

    L is defined by:
        - origin     = `centroid_W` (object centroid in world)
        - +Z_L       = `up_W`          (world up, snapped)
        - +X_L       = `base_dir_W`    (orbit zero-azimuth in horizontal plane)
        - +Y_L       = up × X          (right-hand)

    `radius` is the canonical orbit radius in world units; used for V-frame scaling.

    All operations are pure numpy. Points are float64; rotations float32.
    """
    centroid_W: np.ndarray   # shape (3,)
    up_W: np.ndarray         # shape (3,) unit
    base_dir_W: np.ndarray   # shape (3,) unit, ⊥ up_W
    radius: float

    # Cached rotation: R_WL has rows = [X_L, Y_L, Z_L] in world coords.
    # Mapping convention:
    #     point_in_L = R_WL @ (point_in_W - centroid_W)
    R_WL: np.ndarray = None  # set in __post_init__

    def __post_init__(self):
        up = _unit(self.up_W)
        # Project base_dir onto plane ⊥ up (numerical hygiene).
        base = np.asarray(self.base_dir_W, dtype=np.float64).reshape(3)
        base = base - float(np.dot(base, up)) * up
        base = _unit(base, np.array([1.0, 0.0, 0.0]) if abs(up[0]) < 0.9
                     else np.array([0.0, 1.0, 0.0]))
        right = _unit(np.cross(up, base))    # +Y_L
        # Re-orthogonalize base from up × right to guarantee right-handedness.
        base = _unit(np.cross(right, up))    # +X_L (same plane, exactly orthogonal)

        self.up_W = up.astype(np.float64)
        self.base_dir_W = base.astype(np.float64)
        self.R_WL = np.stack([base, right, up], axis=0).astype(np.float64)
        self.centroid_W = np.asarray(self.centroid_W, dtype=np.float64).reshape(3)
        self.radius = float(self.radius)

    # ── Points ────────────────────────────────────────────────────────────
    def world_to_local_pts(self, pts_W: np.ndarray) -> np.ndarray:
        """(N, 3) world → (N, 3) local."""
        pts = np.asarray(pts_W, dtype=np.float64).reshape(-1, 3)
        return (self.R_WL @ (pts - self.centroid_W).T).T

    def local_to_world_pts(self, pts_L: np.ndarray) -> np.ndarray:
        """(N, 3) local → (N, 3) world."""
        pts = np.asarray(pts_L, dtype=np.float64).reshape(-1, 3)
        return (self.R_WL.T @ pts.T).T + self.centroid_W

    # ── Rotations ─────────────────────────────────────────────────────────
    def world_to_local_rot(self, R_world: np.ndarray) -> np.ndarray:
        """A rotation acting on world vectors → equivalent rotation in L."""
        return self.R_WL @ np.asarray(R_world, dtype=np.float64) @ self.R_WL.T

    def local_to_world_rot(self, R_local: np.ndarray) -> np.ndarray:
        return self.R_WL.T @ np.asarray(R_local, dtype=np.float64) @ self.R_WL


# ─────────────────────────────────────────────────────────────────────────────
# Frame L ↔ Frame V  (Local ↔ Diffusion Virtual Camera)
# ─────────────────────────────────────────────────────────────────────────────

# Fixed permutation matrix mapping Local axes to Virtual axes.
#
# We choose:    +X_L (front)  →  +Z_V (front)
#               +Z_L (up)     →  +Y_V (up)
#               +Y_L (right)  →  +X_V (right)
#
# So a point in V is `R_LV @ point_in_L`. R_LV is orthonormal, det = +1.
#
# Why: Diffusion priors (SV3D, Zero123++) all use a +Y-up, object-at-origin
# canonical pose where azimuth 0 places the camera at +Z looking at the origin.
# Aligning the Local "front" (+X_L = orbit zero-azimuth, i.e. the median
# training-camera direction) with Virtual "front" (+Z_V) means a hallucinated
# view at azimuth 0 in V back-projects to the input camera in L.
R_LV: np.ndarray = np.array([
    [0.0, 1.0, 0.0],   # X_V = Y_L
    [0.0, 0.0, 1.0],   # Y_V = Z_L
    [1.0, 0.0, 0.0],   # Z_V = X_L
], dtype=np.float64)
# Sanity: R_LV is orthonormal with det +1.
assert abs(np.linalg.det(R_LV) - 1.0) < 1e-9
assert np.allclose(R_LV @ R_LV.T, np.eye(3), atol=1e-9)


def sv3d_view_position_V(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    """Camera position in V on the unit-radius orbit.

    azimuth_deg = 0  →  position (0, 0, 1)         (in front, looking at origin)
    azimuth sweeps positively to the right when looking down +Y_V (clockwise from above)
    elevation_deg > 0  →  camera lifted toward +Y_V
    """
    az = np.deg2rad(float(azimuth_deg))
    el = np.deg2rad(float(elevation_deg))
    cos_e, sin_e = np.cos(el), np.sin(el)
    # Horizontal (XZ) ring rotated by azimuth, then lifted by elevation.
    # az = 0 → +Z; az = 90° → +X; az = 180° → -Z (back).
    horiz = np.array([np.sin(az) * cos_e, 0.0, np.cos(az) * cos_e], dtype=np.float64)
    vert = np.array([0.0, sin_e, 0.0], dtype=np.float64)
    return horiz + vert  # unit radius


@dataclass
class LocalSV3D:
    """Builder for SV3D virtual cameras → world cameras for ObjectGS rendering.

    Holds a `WorldLocal` (W↔L) and uses the fixed `R_LV` (L↔V).

    SV3D output frames carry a (azimuth_deg, elevation_deg) per frame in the
    Virtual frame. We recover a world-space camera (R_w2c, T_w2c, C_W) via:

        1. C_V = sv3d_view_position_V(az, el)            # unit radius in V
        2. C_L = R_LV.T @ C_V * radius                   # world units in L
        3. C_W = R_WL.T @ C_L + centroid_W               # world coords
        4. R_w2c, T_w2c = look_at_w2c(C_W, centroid_W, up_W)

    Intrinsics K must be supplied separately (see `make_K_for_sv3d_output`).
    """
    world_local: WorldLocal

    # ── V → W chain ───────────────────────────────────────────────────────
    def sv3d_camera_in_V(self, azimuth_deg: float, elevation_deg: float) -> np.ndarray:
        return sv3d_view_position_V(azimuth_deg, elevation_deg)

    def sv3d_camera_in_L(self, azimuth_deg: float, elevation_deg: float) -> np.ndarray:
        c_V = self.sv3d_camera_in_V(azimuth_deg, elevation_deg)
        return (R_LV.T @ c_V) * self.world_local.radius

    def sv3d_camera_in_W(self, azimuth_deg: float, elevation_deg: float) -> np.ndarray:
        c_L = self.sv3d_camera_in_L(azimuth_deg, elevation_deg)
        return self.world_local.local_to_world_pts(c_L.reshape(1, 3))[0]

    def sv3d_view_to_world_camera(
        self, azimuth_deg: float, elevation_deg: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (R_w2c (3,3) float32, T_w2c (3,) float32, C_W (3,) float64)."""
        C_W = self.sv3d_camera_in_W(azimuth_deg, elevation_deg)
        R_w2c, T_w2c = look_at_w2c(C_W, self.world_local.centroid_W, self.world_local.up_W)
        return R_w2c, T_w2c, C_W

    # ── Inverse: a world camera → its (az, el) in V ───────────────────────
    def world_camera_to_sv3d_view(self, C_W: np.ndarray) -> tuple[float, float]:
        """Map a world camera center to its (azimuth_deg, elevation_deg) in V.

        Useful for: tagging real training cameras with the V-frame angle they
        correspond to, so the hallucination phase can identify uncovered
        sectors.
        """
        c_L = self.world_local.world_to_local_pts(np.asarray(C_W).reshape(1, 3))[0]
        # Project onto the unit sphere in L, then into V.
        c_V = R_LV @ c_L
        r = float(np.linalg.norm(c_V))
        if r < 1e-9:
            return 0.0, 0.0
        c_V = c_V / r
        # az in V: angle in XZ plane measured from +Z, sweeping toward +X.
        az = np.degrees(np.arctan2(c_V[0], c_V[2]))
        el = np.degrees(np.arcsin(np.clip(c_V[1], -1.0, 1.0)))
        return float(az), float(el)


# ─────────────────────────────────────────────────────────────────────────────
# Intrinsics for SV3D output
# ─────────────────────────────────────────────────────────────────────────────

def make_K_for_sv3d_output(
    reference_K: np.ndarray,
    reference_width: int, reference_height: int,
    output_size: int = 576,
) -> np.ndarray:
    """Scale a real-camera K to SV3D's square output resolution.

    SV3D emits 576×576 frames by default. The principal point is centered.
    We preserve the angular FoV by scaling fx/fy by output_size / reference_*.
    """
    K = np.asarray(reference_K, dtype=np.float64).reshape(3, 3)
    sx = output_size / float(reference_width)
    sy = output_size / float(reference_height)
    fx = float(K[0, 0]) * sx
    fy = float(K[1, 1]) * sy
    cx = output_size / 2.0
    cy = output_size / 2.0
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
