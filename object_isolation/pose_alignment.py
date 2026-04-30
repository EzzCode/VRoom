"""Pose alignment math: World <-> Local-Object <-> Zero123++ Virtual.

Each transform is an isolated, named, well-commented function. Every
intermediate is exposed so a debug trace JSON can record the full chain.

==============================================================================
Frame definitions (CANONICAL — do not silently change)
==============================================================================

W (World)
    The room ObjectGS world frame. Right-handed.
    cameras.json stores ``rotation = R_w2c`` and ``position = T_w2c`` directly,
    so a world point ``p_w`` maps to camera coords as ``p_c = R_w2c @ p_w + T_w2c``.
    The world-space camera center is therefore ``C = -R_w2c.T @ T_w2c``.

O (Local-Object)
    Origin = ``object_center`` (W-frame).
    +Z = ``object_up_world`` (camera-local-up consensus, see
         ``estimate_scene_up_from_cameras`` reused from
         ``target_replenishment.core.diagnostics``).
    +X = projection of the reference camera's "right" direction onto the plane
         perpendicular to +Z, normalized.
    +Y = +Z x +X (right-handed).
    Right-handed.

V (Zero123++ Virtual)
    Camera-centric frame for a single Zero123++ tile. Defined by spherical
    coordinates in O:
        x = r cos(el) cos(az)
        y = r cos(el) sin(az)
        z = r sin(el)
    Camera at that position, look-at origin of O, with up = O's +Z.
    The image plane orientation matches Zero123++ output convention. The
    sign convention of (az, el) wrt Zero123++'s tile schedule is captured
    by ``azimuth_sign`` and ``elevation_sign`` (validated combo:
    ``azimuth_sign=-1, elevation_sign=+1`` per repo memory).

==============================================================================
The pipeline (one tile from Zero123++ -> a real-world camera pose)
==============================================================================

    T_W_O = compute_W_to_O(object_center, up_world, ref_R_w2c)
    az0, el0, r_v = compute_O_to_V(T_W_O, ref_R_w2c, ref_T_w2c)
    R_w2c_novel, T_w2c_novel = z123_tile_to_W(
        tile_az_deg, tile_el_deg, az0_deg=az0, el0_deg=el0, r_v=r_v,
        T_W_O=T_W_O, azimuth_sign=-1, elevation_sign=+1,
    )

==============================================================================
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np

# ── Small linear-algebra helpers ─────────────────────────────────────────────


def _normalize(v: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n < 1e-10:
        return np.asarray(fallback, dtype=np.float64)
    return v / n


def _make_homog(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Pack a (3x3) rotation and (3,) translation into a 4x4 homogeneous
    transform. Convention: ``T @ [p; 1] = [R @ p + t; 1]``."""
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    t = np.asarray(t, dtype=np.float64).reshape(3)
    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = R
    M[:3, 3] = t
    return M


def _invert_rigid(M: np.ndarray) -> np.ndarray:
    """Invert a 4x4 rigid transform (rotation + translation only)."""
    R = M[:3, :3]
    t = M[:3, 3]
    Mi = np.eye(4, dtype=np.float64)
    Mi[:3, :3] = R.T
    Mi[:3, 3] = -R.T @ t
    return Mi


def _look_at_R(camera_pos: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Compute a world-to-camera rotation R such that the camera at
    ``camera_pos`` looks at ``target`` with the given world up vector.

    Uses the OpenCV/COLMAP convention: camera +Z is forward (into the scene),
    camera +X is right, camera +Y is down. So the rows of R are the camera
    axes expressed in world coordinates.
    """
    camera_pos = np.asarray(camera_pos, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = _normalize(up, np.array([0.0, 0.0, 1.0]))

    forward = _normalize(target - camera_pos, np.array([0.0, 0.0, 1.0]))
    # Camera right = up x forward (we will then re-derive up to be orthogonal)
    right = _normalize(np.cross(up, forward), np.array([1.0, 0.0, 0.0]))
    # OpenCV convention: camera +Y is *down*, so cam_down = forward x right
    cam_down = _normalize(np.cross(forward, right), np.array([0.0, 1.0, 0.0]))

    # R_w2c rows = camera axes in world coords: [right; cam_down; forward]
    R = np.stack([right, cam_down, forward], axis=0)
    return R


# ── Step A: compute W -> O ───────────────────────────────────────────────────


def compute_W_to_O(
    object_center: np.ndarray,
    object_up_world: np.ndarray,
    reference_R_w2c: np.ndarray,
) -> np.ndarray:
    """Build the homogeneous transform from World (W) to Local-Object (O).

    ``T_W_O @ [p_w; 1] = [p_o; 1]``.

    Reference camera's "right" direction in world coords is the first row of
    ``reference_R_w2c.T``, i.e. ``reference_R_w2c[0, :]`` (the row of R_w2c
    corresponding to camera-X is the world-frame direction of camera-X).
    Wait — ``R_w2c`` rotates world to camera, so its rows give the camera
    axes expressed in world coords:
        cam_right_in_world = R_w2c[0, :]  (camera +X axis)
    We project this vector onto the plane perpendicular to ``object_up_world``
    to get O's +X.
    """
    center = np.asarray(object_center, dtype=np.float64).reshape(3)
    up = _normalize(object_up_world, np.array([0.0, 0.0, 1.0]))
    R_ref = np.asarray(reference_R_w2c, dtype=np.float64).reshape(3, 3)

    # Reference camera +X (right) direction expressed in world coords
    ref_right_world = R_ref[0, :]
    # Project onto plane perpendicular to up
    x_axis = ref_right_world - float(ref_right_world @ up) * up
    # Fallback: if the reference's right is nearly parallel to up, use Y
    if np.linalg.norm(x_axis) < 1e-6:
        alt = np.array([1.0, 0.0, 0.0]) if abs(up[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        x_axis = alt - float(alt @ up) * up
    x_axis = _normalize(x_axis, np.array([1.0, 0.0, 0.0]))

    z_axis = up
    y_axis = _normalize(np.cross(z_axis, x_axis), np.array([0.0, 1.0, 0.0]))
    # Re-orthogonalize x_axis to guarantee a clean basis
    x_axis = _normalize(np.cross(y_axis, z_axis), x_axis)

    # Rows of R_OW are O's axes in world coords; that is the world->O rotation.
    R_OW = np.stack([x_axis, y_axis, z_axis], axis=0)
    t_OW = -R_OW @ center

    return _make_homog(R_OW, t_OW)


# ── Step B: compute O -> V (read off reference camera spherical coords) ─────


def compute_O_to_V(
    T_W_O: np.ndarray,
    reference_R_w2c: np.ndarray,
    reference_T_w2c: np.ndarray,
) -> tuple[float, float, float]:
    """Read off the spherical coordinates ``(az0_deg, el0_deg, r_v)`` of the
    reference camera *expressed in the O frame*.

    These are the Zero123++ "input view" coordinates and are subtracted from
    each generated tile's (az, el) so the tiles are interpreted as deltas
    from the input pose.
    """
    R_ref = np.asarray(reference_R_w2c, dtype=np.float64).reshape(3, 3)
    T_ref = np.asarray(reference_T_w2c, dtype=np.float64).reshape(3)

    # Reference camera center in world: C_w = -R^T @ T
    C_w = -R_ref.T @ T_ref
    C_w_h = np.array([C_w[0], C_w[1], C_w[2], 1.0])
    C_o = (T_W_O @ C_w_h)[:3]

    r = float(np.linalg.norm(C_o))
    if r < 1e-8:
        # Reference camera at object center is degenerate; force a small radius
        # so downstream math doesn't divide by zero. Caller should ensure this
        # never happens by picking a reasonable reference.
        return 0.0, 0.0, 1.0

    el = float(np.degrees(np.arcsin(np.clip(C_o[2] / r, -1.0, 1.0))))
    az = float(np.degrees(np.arctan2(C_o[1], C_o[0])))
    return az, el, r


# ── Step C: lift one Zero123++ tile (az, el) -> world camera pose ────────────


def z123_tile_to_W(
    tile_az_deg: float,
    tile_el_deg: float,
    az0_deg: float,
    el0_deg: float,
    r_v: float,
    T_W_O: np.ndarray,
    azimuth_sign: int = -1,
    elevation_sign: int = +1,
) -> tuple[np.ndarray, np.ndarray]:
    """Map a Zero123++ tile's ``(az, el)`` to a world-frame camera pose
    ``(R_w2c, T_w2c)``.

    Pipeline (each line is a frame transform — DO NOT collapse):
        1. Zero123++ tile coordinates are deltas from the input/reference
           view (input is conceptually at tile=(0,0)). The novel camera's
           spherical position in O is therefore the reference's spherical
           position plus the (signed) tile offset:
               az_in_O = az0 + azimuth_sign   * tile_az
               el_in_O = el0 + elevation_sign * tile_el
        2. Place the camera in O at spherical (az_in_O, el_in_O, r_v):
               C_o = r_v * (cos(el) cos(az), cos(el) sin(az), sin(el))
        3. R_w2c_in_O = look-at origin of O with up = O's +Z (i.e. [0,0,1] in O).
        4. Convert (R_w2c_in_O, C_o) to a 4x4 transform T_O_C.
        5. Compose: T_W_C = T_O_C @ T_W_O. Extract R, t.

    With this convention, ``tile_az_deg=0, tile_el_deg=0`` recovers the
    reference camera's pose exactly (verified by unit test).
    """
    # 1. Spherical position in O (reference offset by signed tile deltas)
    az = np.radians(az0_deg + azimuth_sign * tile_az_deg)
    el = np.radians(el0_deg + elevation_sign * tile_el_deg)
    r = float(r_v)

    # 2. Camera position in O (O-frame coords)
    C_o = np.array(
        [r * np.cos(el) * np.cos(az), r * np.cos(el) * np.sin(az), r * np.sin(el)],
        dtype=np.float64,
    )

    # 3. R_OC: O-to-camera rotation. Look at origin with up = O's +Z.
    R_OC = _look_at_R(camera_pos=C_o, target=np.zeros(3), up=np.array([0.0, 0.0, 1.0]))
    # 4. T_O_C: O -> camera (rotates O coords into camera coords, then translates)
    t_OC = -R_OC @ C_o
    T_O_C = _make_homog(R_OC, t_OC)

    # 5. Compose W -> O -> camera
    T_W_O = np.asarray(T_W_O, dtype=np.float64).reshape(4, 4)
    T_W_C = T_O_C @ T_W_O

    R_w2c = T_W_C[:3, :3]
    T_w2c = T_W_C[:3, 3]
    return R_w2c, T_w2c


# ── Zero123++ tile schedule (default = sudo-ai/zero123plus v1.2) ────────────


def zero123plus_v12_tile_schedule() -> list[tuple[float, float]]:
    """Return the 6 ``(azimuth_deg, elevation_deg)`` of Zero123++ v1.2 tiles.

    Per the model card and validated in repo memory:
        elevations follow the pattern [20, -10, 20, -10, 20, -10]
        azimuths   = [30, 90, 150, 210, 270, 330]
    These are *deltas from the input view* (which is itself at (0, 0)) in
    the conventions of the Zero123++ training pipeline.
    """
    azimuths = [30.0, 90.0, 150.0, 210.0, 270.0, 330.0]
    elevations = [20.0, -10.0, 20.0, -10.0, 20.0, -10.0]
    return list(zip(azimuths, elevations))


# ── Zero123++ native intrinsics (Phase 3.5.1) ────────────────────────────────


def zero123_native_K(image_size: int = 512, fov_deg: float = 49.1) -> np.ndarray:
    """Zero123++ canonical pinhole K for an ``image_size x image_size`` tile.

    Default ``fov_deg=49.1`` is the Objaverse-render vertical FOV used by
    Zero123++ training. Refined per backend by ``zero123_calibrate.py``.
    """
    f = 0.5 * image_size / np.tan(0.5 * np.radians(fov_deg))
    cx = cy = 0.5 * image_size
    return np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def zero123_native_distance(object_radius: float, normalized_distance: float = 1.5) -> float:
    """Synthetic camera distance in the room's metric scale.

    Zero123++ trains on objects normalized to a unit sphere placed at distance
    ``normalized_distance`` from the camera. To preserve Zero123++'s
    pixel<->angle mapping we place the virtual camera at
    ``r_v_synth = normalized_distance * object_radius``.

    Empirically refined per backend via ``zero123_calibrate.py``.
    """
    return float(normalized_distance) * float(object_radius)


# ── Bundled trace dataclass ─────────────────────────────────────────────────


@dataclass
class PoseTrace:
    """Snapshot of every intermediate produced for one object run.

    Persist via ``json.dump(asdict(trace), f, indent=2)`` (numpy arrays are
    converted to lists in :meth:`to_dict`).
    """

    object_center: np.ndarray
    object_up_world: np.ndarray
    object_radius: float
    reference_R_w2c: np.ndarray
    reference_T_w2c: np.ndarray
    T_W_O: np.ndarray
    az0_deg: float
    el0_deg: float
    r_v: float  # real-world reference distance (NOT used for novel views)
    r_v_synth: float  # Zero123++ synthetic distance (USED for novel views)
    azimuth_sign: int
    elevation_sign: int
    tiles: list = field(default_factory=list)

    def to_dict(self) -> dict:
        out = {}
        for k, v in asdict(self).items():
            if isinstance(v, np.ndarray):
                out[k] = v.tolist()
            else:
                out[k] = v
        return out


def build_full_pose_chain(
    object_center: np.ndarray,
    object_up_world: np.ndarray,
    object_radius: float,
    reference_R_w2c: np.ndarray,
    reference_T_w2c: np.ndarray,
    tile_schedule: Optional[list[tuple[float, float]]] = None,
    azimuth_sign: int = -1,
    elevation_sign: int = +1,
    r_v_synth: Optional[float] = None,
) -> PoseTrace:
    """End-to-end: build T_W_O, read (az0, el0, r_v), and for every Zero123++
    tile record (R_w2c, T_w2c) in W. The novel-view *position* uses the
    synthetic distance ``r_v_synth`` (Phase 3.5.1) by default, NOT the
    reference camera's real distance ``r_v`` — this preserves Zero123++'s
    pixel<->angle mapping. Pass ``r_v_synth=None`` to default to
    ``zero123_native_distance(object_radius)``.
    """
    if tile_schedule is None:
        tile_schedule = zero123plus_v12_tile_schedule()
    if r_v_synth is None:
        r_v_synth = zero123_native_distance(object_radius)

    T_W_O = compute_W_to_O(object_center, object_up_world, reference_R_w2c)
    az0, el0, r_v = compute_O_to_V(T_W_O, reference_R_w2c, reference_T_w2c)

    tiles = []
    for tile_az, tile_el in tile_schedule:
        R, T = z123_tile_to_W(
            tile_az_deg=tile_az,
            tile_el_deg=tile_el,
            az0_deg=az0,
            el0_deg=el0,
            r_v=r_v_synth,  # <-- metric scale lock: use synthetic distance
            T_W_O=T_W_O,
            azimuth_sign=azimuth_sign,
            elevation_sign=elevation_sign,
        )
        tiles.append({
            "tile_az_deg": float(tile_az),
            "tile_el_deg": float(tile_el),
            "R_w2c": R.tolist(),
            "T_w2c": T.tolist(),
        })

    return PoseTrace(
        object_center=np.asarray(object_center, dtype=np.float64),
        object_up_world=np.asarray(object_up_world, dtype=np.float64),
        object_radius=float(object_radius),
        reference_R_w2c=np.asarray(reference_R_w2c, dtype=np.float64),
        reference_T_w2c=np.asarray(reference_T_w2c, dtype=np.float64),
        T_W_O=T_W_O,
        az0_deg=az0,
        el0_deg=el0,
        r_v=r_v,
        r_v_synth=float(r_v_synth),
        azimuth_sign=int(azimuth_sign),
        elevation_sign=int(elevation_sign),
        tiles=tiles,
    )
