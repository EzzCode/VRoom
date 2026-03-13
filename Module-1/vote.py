"""
Point Cloud Voting for Object Labeling

Projects 3D points onto 2D mask images and assigns per-point object labels
through multi-view voting (majority, probability, or correspondence-based).

Usage:
    python voter/vote.py --data_path data --algorithm majority
"""

import struct
import numpy as np
import cv2
import argparse
import os
import sys
from collections import Counter, defaultdict
from plyfile import PlyData, PlyElement
from sklearn.cluster import DBSCAN

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from colmap_loader import (
    read_intrinsics_binary, read_extrinsics_binary,
    read_intrinsics_text, read_extrinsics_text,
)


##### COLMAP points3D reader (preserves track data for corr voting) ###########################################################

def load_points3D_bin(path):
    """Load COLMAP points3D.bin -> dict[id -> (xyz, rgb, error, tracks)]."""
    points = {}
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            blob = struct.unpack("<QdddBBBd", f.read(43))
            pid, xyz, rgb, err = blob[0], blob[1:4], blob[4:7], blob[7]
            (tlen,) = struct.unpack("<Q", f.read(8))
            raw = struct.unpack(f"<{'ii' * tlen}", f.read(8 * tlen))
            tracks = [(raw[i], raw[i + 1]) for i in range(0, len(raw), 2)]
            points[pid] = (np.array(xyz), np.array(rgb, dtype=np.uint8), err, tracks)
    return points


def load_points3D_txt(path):
    """Load COLMAP points3D.txt -> dict[id -> (xyz, rgb, error, tracks=[])]."""
    points = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            pid = int(parts[0])
            xyz = np.array([float(x) for x in parts[1:4]])
            rgb = np.array([int(x) for x in parts[4:7]], dtype=np.uint8)
            err = float(parts[7])
            # Parse tracks if present: pairs of (image_id, point2D_idx)
            track_parts = parts[8:]
            tracks = [(int(track_parts[i]), int(track_parts[i + 1]))
                       for i in range(0, len(track_parts), 2)]
            points[pid] = (xyz, rgb, err, tracks)
    return points


##### Camera intrinsics handling ###########################################################

# COLMAP models where params[0] is a shared focal length f, then cx, cy
_SHARED_FOCAL_MODELS = {
    "SIMPLE_PINHOLE", "SIMPLE_RADIAL", "SIMPLE_RADIAL_FISHEYE",
    "RADIAL", "RADIAL_FISHEYE",
}

def intrinsics_from_camera(cam):
    """Return (fx, fy, cx, cy) regardless of COLMAP camera model."""
    p = cam.params
    if cam.model in _SHARED_FOCAL_MODELS:
        return p[0], p[0], p[1], p[2]
    return p[0], p[1], p[2], p[3]      # PINHOLE / OPENCV / etc.


##### Batch projection ###########################################################

def quat_to_R(q):
    """Quaternion [w,x,y,z] -> 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),  2*(x*y - z*w),      2*(x*z + y*w)],
        [2*(x*y + z*w),      1 - 2*(x*x + z*z),  2*(y*z - x*w)],
        [2*(x*z - y*w),      2*(y*z + x*w),       1 - 2*(x*x + y*y)],
    ])


def project_all(pts_xyz, R, t, fx, fy, cx, cy, w, h):
    """
    Vectorised projection of Nx3 world points -> pixel coords.
    Returns (u, v, mask) where mask flags points inside the image.
    """
    cam = (R @ pts_xyz.T + t).T                    # Nx3 in camera frame
    z = cam[:, 2]
    valid = z > 0
    u = np.full(len(pts_xyz), -1.0)
    v = np.full(len(pts_xyz), -1.0)
    u[valid] = fx * cam[valid, 0] / z[valid] + cx
    v[valid] = fy * cam[valid, 1] / z[valid] + cy
    ui, vi = np.round(u).astype(int), np.round(v).astype(int)
    in_bounds = valid & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
    return ui, vi, in_bounds


##### Mask I/O ###########################################################

def _mask_path(mask_dir, image_name):
    """Derive mask .png path from a COLMAP image name (.jpg/.JPG)."""
    name = os.path.splitext(image_name)[0] + ".png"
    return os.path.join(mask_dir, name)


def load_mask(mask_dir, image_name):
    """Load a single-channel label mask, or None if missing."""
    path = _mask_path(mask_dir, image_name)
    return cv2.imread(path, cv2.IMREAD_UNCHANGED)


##### Vote collection ###########################################################

def collect_projection_votes(images, cameras, points, mask_dir):
    """
    For every 3D point, project into every view and read the label.
    Returns dict[point_id -> list[label]].
    """
    pid_list = list(points.keys())
    pts_xyz = np.array([points[p][0] for p in pid_list])       # Nx3

    votes = defaultdict(list)

    for img in images.values():
        mask = load_mask(mask_dir, img.name)
        if mask is None:
            print(f"  WARN: mask missing for {img.name}, skipping")
            continue

        R = quat_to_R(img.qvec)
        t = img.tvec.reshape(3, 1)
        fx, fy, cx, cy = intrinsics_from_camera(cameras[img.camera_id])
        h, w = mask.shape[:2]

        ui, vi, ok = project_all(pts_xyz, R, t, fx, fy, cx, cy, w, h)

        for idx in np.where(ok)[0]:
            votes[pid_list[idx]].append(int(mask[vi[idx], ui[idx]]))

        print(f"  {img.name}: {ok.sum()} / {len(pts_xyz)} points visible")

    return votes


def collect_correspondence_votes(images, points, mask_dir):
    """
    Use COLMAP's 2D <-> 3D correspondence tracks instead of re-projecting.
    Returns dict[point_id -> list[label]].
    """
    votes = defaultdict(list)
    mask_cache = {}

    for pid, (xyz, rgb, err, tracks) in points.items():
        for img_id, pt2d_idx in tracks:
            if img_id not in images:
                continue
            img = images[img_id]

            # Lazy-load and cache masks
            if img_id not in mask_cache:
                mask_cache[img_id] = load_mask(mask_dir, img.name)
            mask = mask_cache[img_id]
            if mask is None:
                continue

            if pt2d_idx >= len(img.xys):
                continue
            u, v = int(round(img.xys[pt2d_idx][0])), int(round(img.xys[pt2d_idx][1]))
            h, w = mask.shape[:2]
            if 0 <= u < w and 0 <= v < h:
                votes[pid].append(int(mask[v, u]))

    return votes


##### Label resolution strategies ###########################################################

def resolve_majority(label_list):
    """Pick the most frequent label."""
    return Counter(label_list).most_common(1)[0][0]


def resolve_probability(label_list):
    """Sample one label proportional to its frequency."""
    counts = Counter(label_list)
    labels, freqs = zip(*counts.items())
    probs = np.array(freqs, dtype=float)
    probs /= probs.sum()
    return int(np.random.choice(labels, p=probs))


def resolve_correspondence(label_list):
    """Majority vote, but ignore background (label 0)."""
    fg = [l for l in label_list if l != 0]
    if not fg:
        return 0
    return Counter(fg).most_common(1)[0][0]


_RESOLVERS = {
    "majority": resolve_majority,
    "prob":     resolve_probability,
    "corr":     resolve_correspondence,
}


##### PLY output ###########################################################

def save_labeled_ply(path, xyz, rgb, labels):
    """Write a PLY with (x,y,z, nx,ny,nz, r,g,b, label)."""
    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ("label", "u1"),
    ]
    n = len(xyz)
    normals = np.zeros((n, 3), dtype=np.float32)
    arr = np.empty(n, dtype=dtype)
    for i, row in enumerate(
        np.hstack([xyz, normals, rgb, labels.reshape(-1, 1)])
    ):
        arr[i] = tuple(row)
    PlyData([PlyElement.describe(arr, "vertex")]).write(path)


def prune_3d_outliers(xyz, labels, min_points=10):
    """
    Looks at the 3D points for each label. 
    If total points for a label is less than min_points, it is classified as background.
    """
    cleaned_labels = labels.copy()
    unique_labels = np.unique(labels)
    
    for lbl in unique_labels:
        if lbl == 0:
            continue # Skip background
            
        # Get the 3D coordinates of all points assigned to this label
        mask = (labels == lbl)
        obj_xyz = xyz[mask]
        
        if len(obj_xyz) < min_points:
            cleaned_labels[mask] = 0
            
    return cleaned_labels
##### Pipeline ###########################################################

def run_voting(args):
    data = args.data_path
    mask_dir = os.path.join(data, args.mask_dir)

    if not os.path.isdir(mask_dir):
        sys.exit(f"ERROR: mask dir not found: {mask_dir}")

    # Load COLMAP data ###########################################################
    sp = os.path.join(data, args.sparse_dir)
    if not os.path.isdir(sp):
        sys.exit(f"ERROR: sparse dir not found: {sp}")

    try:
        cameras = read_intrinsics_binary(os.path.join(sp, "cameras.bin"))
        images  = read_extrinsics_binary(os.path.join(sp, "images.bin"))
        points  = load_points3D_bin(os.path.join(sp, "points3D.bin"))
        print(f"Loaded binary COLMAP from {sp}")
    except Exception as e:
        cameras = read_intrinsics_text(os.path.join(sp, "cameras.txt"))
        images  = read_extrinsics_text(os.path.join(sp, "images.txt"))
        points  = load_points3D_txt(os.path.join(sp, "points3D.txt"))
        print(f"Loaded text COLMAP from {sp}")

    print(f"{len(cameras)} cam(s), {len(images)} imgs, {len(points)} pts")
    for cid, cam in cameras.items():
        fx, fy, cx, cy = intrinsics_from_camera(cam)
        print(f"  cam {cid}: {cam.model}  {cam.width}x{cam.height}  "
              f"f=({fx:.1f},{fy:.1f})  c=({cx:.1f},{cy:.1f})")

    # Collect votes ###########################################################
    algo = args.algorithm
    print(f"\nVoting strategy: {algo}")

    if algo == "corr":
        votes = collect_correspondence_votes(images, points, mask_dir)
    else:
        votes = collect_projection_votes(images, cameras, points, mask_dir)

    resolver = _RESOLVERS[algo]

    # Resolve labels ###########################################################
    pid_order = list(points.keys())
    xyz_arr = np.array([points[p][0] for p in pid_order], dtype=np.float32)
    rgb_arr = np.array([points[p][1] for p in pid_order], dtype=np.uint8)

    labels = np.zeros(len(pid_order), dtype=np.uint8)
    for i, pid in enumerate(pid_order):
        if pid in votes and votes[pid]:
            labels[i] = resolver(votes[pid])

    unique, counts = np.unique(labels, return_counts=True)
    print(f"\nLabel distribution ({len(unique)} labels):")
    for lbl, cnt in zip(unique, counts):
        print(f"  label {lbl:3d}: {cnt:6d} pts ({100*cnt/len(labels):.1f}%)")

    # Create output folder ###########################################################
    out_dir = os.path.join(data, args.output_dir)
    obj_dir = os.path.join(out_dir, "object_clouds")
    os.makedirs(obj_dir, exist_ok=True)

    # Prune outliers ###########################################################
    labels = prune_3d_outliers(xyz_arr, labels, min_points=args.min_points)

    unique, counts = np.unique(labels, return_counts=True)
    print(f"\nPruned {len(xyz_arr) - len(unique)} outliers")
    print(f"\nLabel distribution ({len(unique)} labels):")
    for lbl, cnt in zip(unique, counts):
        print(f"  label {lbl:3d}: {cnt:6d} pts ({100*cnt/len(labels):.1f}%)") 

    # Labeled PLY original RGB + label property
    labeled_path = os.path.join(out_dir, "points3D_labeled.ply")
    save_labeled_ply(labeled_path, xyz_arr, rgb_arr, labels)
    print(f"\nSaved -> {labeled_path}")

    # Visualization PLY auto-generate distinct colors per label
    def label_to_color(lbl):
        """Map label to a distinct RGB color using golden-angle HSV spacing."""
        if lbl == 0:
            return (150, 150, 150)  # gray for background
        # Golden angle (~137.5 deg) gives maximally spaced hues
        hue = ((lbl * 137.508) % 360) / 360.0
        sat, val = 0.75, 0.9
        # HSV -> RGB
        import colorsys
        r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
        return (int(r * 255), int(g * 255), int(b * 255))

    vis_rgb = np.array([label_to_color(l) for l in labels], dtype=np.uint8)
    vis_path = os.path.join(out_dir, "points3D_vis.ply")
    save_labeled_ply(vis_path, xyz_arr, vis_rgb, labels)
    print(f"Saved -> {vis_path}")

    # Per-object clouds one PLY per label (skip background 0)
    for lbl in sorted(set(labels)):
        if lbl == 0:
            continue
        mask = labels == lbl
        obj_path = os.path.join(obj_dir, f"label_{lbl:02d}.ply")
        save_labeled_ply(obj_path, xyz_arr[mask], rgb_arr[mask], labels[mask])
        print(f"Saved -> {obj_path}  ({mask.sum()} pts)")

    print(f"\nAll outputs in: {out_dir}")




##### CLI ###########################################################

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Point cloud object labeling via multi-view voting.")
    p.add_argument("--data_path",   required=True, help="Scene directory (contains COLMAP data and masks)")
    p.add_argument("--sparse_dir", default="sparse/0", help="COLMAP sparse model dir relative to data_path")
    p.add_argument("--mask_dir",   default="object_mask", help="Mask folder name")
    p.add_argument("--algorithm",  default="majority", choices=["majority", "prob", "corr"])
    p.add_argument("--output_dir", default="output", help="Output folder name inside data_path")
    p.add_argument("--min_points", type=int, default=10, help="Minimum number of points to be considered an object")
    run_voting(p.parse_args())
