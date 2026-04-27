"""
View Alignment — Convert Era3D output angles to Scaffold-GS world-space cameras.

Maps Era3D's known azimuth offsets (relative to the input view) back to
world-space camera poses (R, T, K) compatible with ObjectGS rendering.

Era3D outputs 6 views at orthographic projection, 0° elevation, with
azimuths spaced every 60° relative to the input view.

Public API:
    compute_novel_cameras(coverage_result, era3d_views) -> list[dict]
"""

__all__ = ['compute_novel_cameras']

import logging
import numpy as np

logger = logging.getLogger(__name__)

# Zero123++ v1.2 outputs 6 views at these azimuth offsets from the input view.
# IMPORTANT: v1.2 also uses per-view ELEVATION offsets that alternate
# +20° / -10° relative to the (assumed-horizontal) input frame. Ignoring this
# is a major source of "novel views look right but optimizer makes geometry
# worse" — the supervision camera ends up tilted vs. what was rendered.
# Source: Zero123++ paper / sudo-ai/zero123plus-v1.2 model card.
ERA3D_AZIMUTH_OFFSETS_DEG = [30, 90, 150, 210, 270, 330]
ZERO123PP_ELEVATION_OFFSETS_DEG = [20, -10, 20, -10, 20, -10]


def look_at(camera_pos: np.ndarray, target_pos: np.ndarray, up_vector: np.ndarray):
    """Construct COLMAP-convention R, T from camera position looking at target.

    COLMAP convention: R transforms world → camera, rows are (right, -up, forward).
    Matches the look_at() in render_360.py.
    """
    forward = target_pos - camera_pos
    forward = forward / (np.linalg.norm(forward) + 1e-8)

    right = np.cross(up_vector, forward)
    right = right / (np.linalg.norm(right) + 1e-8)

    up = np.cross(forward, right)
    up = up / (np.linalg.norm(up) + 1e-8)

    # R: world → camera. Rows = right, -up, forward (COLMAP Y-down)
    R = np.vstack((right, -up, forward)).astype(np.float32)
    T = (-R @ camera_pos).astype(np.float32)
    return R, T


def compute_novel_cameras(
    object_center: np.ndarray,
    input_azimuth: float,
    orbit_radius: float,
    up_vector: np.ndarray,
    reference_K: np.ndarray,
    reference_width: int,
    reference_height: int,
    output_size: int = 512,
    gap_azimuths: list = None,
    azimuth_sign: int = 1,
    elevation_sign: int = 1,
) -> list:
    """Compute world-space cameras for Era3D's 6 output views.

    Args:
        object_center: (3,) object centroid in world space.
        input_azimuth: Azimuth of the input camera relative to object (radians).
        orbit_radius: Distance from object center to place cameras.
        up_vector: (3,) world up direction.
        reference_K: (3,3) intrinsic matrix from a training camera.
        reference_width: Training image width.
        reference_height: Training image height.
        output_size: Era3D output resolution (512).
        gap_azimuths: If provided, only keep views whose azimuth falls in
                      an uncovered sector. None = keep all 6.
        azimuth_sign: +1 or -1. Flip if Zero123++ rotates opposite to our
                      basis_v = up × basis_h handedness convention.
        elevation_sign: +1 or -1. Flip if our up_vector estimate points
                      opposite to Zero123++'s assumed up.

    Returns:
        List of camera dicts with R, T, K, position, width, height, azimuth_deg.
    """
    # Compute horizontal basis perpendicular to up
    if abs(up_vector[0]) < 0.9:
        arbitrary = np.array([1, 0, 0], dtype=np.float32)
    else:
        arbitrary = np.array([0, 1, 0], dtype=np.float32)
    basis_h = np.cross(up_vector, arbitrary)
    basis_h /= np.linalg.norm(basis_h)
    basis_v = np.cross(up_vector, basis_h)
    basis_v /= np.linalg.norm(basis_v)

    # Scale intrinsics to output_size
    scale_x = output_size / reference_width
    scale_y = output_size / reference_height
    K_scaled = np.array([
        [reference_K[0, 0] * scale_x, 0, output_size / 2.0],
        [0, reference_K[1, 1] * scale_y, output_size / 2.0],
        [0, 0, 1],
    ], dtype=np.float32)

    cameras = []
    for az_offset_deg, el_offset_deg in zip(
        ERA3D_AZIMUTH_OFFSETS_DEG, ZERO123PP_ELEVATION_OFFSETS_DEG
    ):
        az_world = input_azimuth + np.radians(az_offset_deg) * azimuth_sign
        el_rad = np.radians(el_offset_deg) * elevation_sign
        cos_el = float(np.cos(el_rad))
        sin_el = float(np.sin(el_rad))

        # Position on a tilted orbit: horizontal component scaled by cos(elev),
        # vertical component along world up scaled by sin(elev). Matches the
        # canonical (azimuth, elevation, radius) parameterization used by
        # Zero123++ to place its synthesized cameras.
        horizontal = (
            np.cos(az_world) * basis_h
            + np.sin(az_world) * basis_v
        )
        cam_pos = (
            object_center
            + orbit_radius * (cos_el * horizontal + sin_el * up_vector)
        ).astype(np.float32)

        R, T = look_at(cam_pos, object_center, up_vector)

        cam_dict = {
            'R': R,
            'T': T,
            'K': K_scaled.copy(),
            'position': cam_pos,
            'width': output_size,
            'height': output_size,
            'azimuth_offset_deg': az_offset_deg,
            'elevation_offset_deg': el_offset_deg,
            'azimuth_world_rad': float(az_world),
        }
        cameras.append(cam_dict)

    # Filter to gap sector if requested
    if gap_azimuths is not None and len(gap_azimuths) > 0:
        gap_set = set(gap_azimuths)
        filtered = []
        for cam in cameras:
            az = cam['azimuth_world_rad']
            # Normalize to [-pi, pi]
            az_norm = (az + np.pi) % (2 * np.pi) - np.pi
            # Check if this azimuth falls in any gap bin
            in_gap = _azimuth_in_gap(az_norm, gap_azimuths, bin_width=np.radians(10))
            if in_gap:
                filtered.append(cam)
                logger.info(
                    f"  View az_offset={cam['azimuth_offset_deg']}° "
                    f"(world={np.degrees(az_norm):.1f}°) → IN GAP, keeping"
                )
            else:
                logger.info(
                    f"  View az_offset={cam['azimuth_offset_deg']}° "
                    f"(world={np.degrees(az_norm):.1f}°) → covered, skipping"
                )

        if not filtered:
            # If no views fall in the gap, keep the 3 furthest from input
            logger.warning("No Era3D views fall in gap bins. Keeping 3 views furthest from input.")
            by_offset = sorted(cameras, key=lambda c: abs(c['azimuth_offset_deg'] - 180))
            filtered = by_offset[:3]

        cameras = filtered

    logger.info(f"Aligned {len(cameras)} novel cameras at orbit_radius={orbit_radius:.2f}")
    return cameras


def _azimuth_in_gap(azimuth: float, gap_azimuths: list, bin_width: float) -> bool:
    """Check if an azimuth falls within any gap bin."""
    for gap_az in gap_azimuths:
        diff = abs(azimuth - gap_az)
        # Handle wraparound
        diff = min(diff, 2 * np.pi - diff)
        if diff < bin_width:
            return True
    return False
