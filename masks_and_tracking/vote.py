import json
import logging
import argparse

from pathlib import Path

import cv2
import numpy as np
from plyfile import PlyData, PlyElement
from masks_and_tracking.helpers import label_to_color

from sfm.colmap_loader import (
    read_intrinsics_binary, read_extrinsics_binary, read_points3D_binary,
    read_intrinsics_text, read_extrinsics_text, read_points3D_text,
)

logger = logging.getLogger(__name__)

_SPARSE_DIR= "sparse/0"
_MASK_DIR = "object_mask"
_OUTPUT_DIR = "output"
_MIN_POINTS = 10
_ALIAS_IOU_THRESH = 0.5
_ALIAS_MIN_COVISIBILITY = 20

def intrinsics(cam):
    param = cam.params
    if cam.model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "SIMPLE_RADIAL_FISHEYE", "RADIAL", "RADIAL_FISHEYE"}:
        fx = fy = param[0]
        cx, cy = param[1], param[2]
        return fx, fy, cx, cy
    fx, fy, cx, cy = param[0], param[1], param[2], param[3]
    return fx, fy, cx, cy

    
def majority_voting(images, cameras, pts_xyz, mask_dir):
    votes = {}
    for img in images.values():
        path = Path(mask_dir) / (Path(img.name).stem + ".png")
        mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            logger.warning("mask missing for %s, skipping", img.name)
            continue
        qw, qx, qy, qz = img.qvec
        R = np.array([
            [1 - 2*(qy*qy + qz*qz),  2*(qx*qy - qz*qw),      2*(qx*qz + qy*qw)],
            [2*(qx*qy + qz*qw),      1 - 2*(qx*qx + qz*qz),  2*(qy*qz - qx*qw)],
            [2*(qx*qz - qy*qw),      2*(qy*qz + qx*qw),       1 - 2*(qx*qx + qy*qy)],
        ])
        t = img.tvec.reshape(3, 1)
        fx, fy, cx, cy = intrinsics(cameras[img.camera_id])
        h, w = mask.shape[:2]
        cam_pts = (R @ pts_xyz.T + t).T
        z = cam_pts[:, 2]
        valid = z > 0
        u = np.full(len(pts_xyz), -1.0)
        v = np.full(len(pts_xyz), -1.0)
        u[valid] = fx * cam_pts[valid, 0] / z[valid] + cx
        v[valid] = fy * cam_pts[valid, 1] / z[valid] + cy
        ui, vi = np.round(u).astype(int), np.round(v).astype(int)
        ok = valid & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
        for idx in np.where(ok)[0]:
            if idx not in votes:
                votes[idx] = []
            votes[idx].append(int(mask[vi[idx], ui[idx]]))
        logger.info("%s: %d / %d points visible", img.name, ok.sum(), len(pts_xyz))
    return votes


def find_parent(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def merge_aliases(labels, votes, num_points, iou_thresh=0.75, min_covisibility=20):
    label_sets = {}
    for i in range(num_points):
        if i in votes:
            for lbl in votes[i]:
                if lbl != 0:
                    lbl_val = int(lbl)
                    if lbl_val not in label_sets:
                        label_sets[lbl_val] = set()
                    label_sets[lbl_val].add(i)

    if len(label_sets) < 2:
        return labels, {}

    sorted_labels = sorted(label_sets.keys())
    logger.info("Labels in raw votes: %s", sorted_labels)
    for lbl in sorted_labels:
        logger.info("  ID %3d: %5d points (raw vote membership)", lbl, len(label_sets[lbl]))

    parent = {id: id for id in label_sets}
    merge_count = 0
    for i, label_a in enumerate(sorted_labels):
        for label_b in sorted_labels[i + 1:]:
            if find_parent(parent, label_a) == find_parent(parent, label_b):
                continue
            set_a, set_b = label_sets[label_a], label_sets[label_b]
            shared = len(set_a & set_b)
            if shared < min_covisibility:
                continue
            intersection = len(set_a & set_b)
            union = len(set_a | set_b)
            iou = intersection / union if union > 0 else 0.0
            if iou >= iou_thresh:
                a, b = find_parent(parent, label_a), find_parent(parent, label_b)
                if a != b:
                    if a > b:
                        a, b = b, a
                    parent[b] = a
                merge_count += 1
                logger.info("Alias merge: ID %d -> ID %d  (3D IoU=%.3f, shared_pts=%d)", label_b, label_a, iou, shared)

    if merge_count == 0:
        logger.info("No aliases detected.")
        return labels, {}

    merge_map = {lbl: find_parent(parent, lbl) for lbl in sorted_labels if find_parent(parent, lbl) != lbl}
    remap  = labels.copy()
    for old_id, new_id in merge_map.items():
        remap[labels == old_id] = new_id

    logger.info("Merged %d alias pairs into %d unique objects.", merge_count, len({find_parent(parent, l) for l in sorted_labels}))
    return remap, merge_map


def save_ply(path, xyz, rgb, labels):
    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ("label", "u1"),
    ]
    n = len(xyz)
    normals = np.zeros((n, 3), dtype=np.float32)
    arr = np.empty(n, dtype=dtype)
    for i, row in enumerate(np.hstack([xyz, normals, rgb, labels.reshape(-1, 1)])):
        arr[i] = tuple(row)
    PlyData([PlyElement.describe(arr, "vertex")]).write(path)


def run_voting(args):
    data = args.data_path
    mask_dir = Path(data) / args.mask_dir

    if not mask_dir.is_dir():
        raise FileNotFoundError(f"mask dir not found: {mask_dir}")

    sp = Path(data) / args.sparse_dir
    if not sp.is_dir():
        raise FileNotFoundError(f"sparse dir not found: {sp}")

    try:
        cameras = read_intrinsics_binary(str(sp / "cameras.bin"))
        images  = read_extrinsics_binary(str(sp / "images.bin"))
        xyz_arr, rgb_arr, _ = read_points3D_binary(str(sp / "points3D.bin"))
        logger.info("Loaded binary COLMAP from %s", sp)
    except Exception:
        cameras = read_intrinsics_text(str(sp / "cameras.txt"))
        images  = read_extrinsics_text(str(sp / "images.txt"))
        xyz_arr, rgb_arr, _ = read_points3D_text(str(sp / "points3D.txt"))
        logger.info("Loaded text COLMAP from %s", sp)

    xyz_arr = xyz_arr.astype(np.float32)
    rgb_arr = rgb_arr.astype(np.uint8)

    logger.info("%d cam(s), %d imgs, %d pts", len(cameras), len(images), len(xyz_arr))
    for cid, cam in cameras.items():
        fx, fy, cx, cy = intrinsics(cam)
        logger.info("  cam %d: %s  %dx%d  f=(%.1f,%.1f)  c=(%.1f,%.1f)", cid, cam.model, cam.width, cam.height, fx, fy, cx, cy)

    logger.info("Voting strategy: majority projection")
    votes = majority_voting(images, cameras, xyz_arr, mask_dir)

    labels = np.zeros(len(xyz_arr), dtype=np.uint8)
    for i in range(len(xyz_arr)):
        if i in votes and votes[i]:
            counts = {}
            for val in votes[i]:
                counts[val] = counts.get(val, 0) + 1
            labels[i] = max(counts, key=lambda k: counts[k])

    unique, counts = np.unique(labels, return_counts=True)
    logger.info("Raw label distribution (%d labels):", len(unique))
    for lbl, cnt in zip(unique, counts):
        logger.info("  label %3d: %6d pts (%.1f%%)", lbl, cnt, 100 * cnt / len(labels))

    if not args.disable_alias_merge:
        logger.info("--- 3D Alias Merging ---")
        labels, merge_map = merge_aliases(
            labels, votes, len(xyz_arr),
            iou_thresh=args.alias_iou_thresh,
            min_covisibility=args.alias_min_covisibility,
        )
        if merge_map:
            map_path = Path(data) / args.output_dir / "alias_merge_map.json"
            map_path.parent.mkdir(parents=True, exist_ok=True)
            with open(map_path, "w") as f:
                json.dump({str(k): int(v) for k, v in merge_map.items()}, f, indent=2)
            logger.info("Saved alias map -> %s", map_path)
            unique, counts = np.unique(labels, return_counts=True)
            logger.info("Post-merge label distribution (%d labels):", len(unique))
            for lbl, cnt in zip(unique, counts):
                logger.info("  label %3d: %6d pts (%.1f%%)", lbl, cnt, 100 * cnt / len(labels))

    cleaned = labels.copy()
    for lbl in np.unique(labels):
        if lbl != 0:
            mask = labels == lbl
            if mask.sum() < args.min_points:
                cleaned[mask] = 0
    labels = cleaned

    unique, counts = np.unique(labels, return_counts=True)
    logger.info("Final label distribution (%d labels):", len(unique))
    for lbl, cnt in zip(unique, counts):
        logger.info("  label %3d: %6d pts (%.1f%%)", lbl, cnt, 100 * cnt / len(labels))

    out_dir = Path(data) / args.output_dir
    obj_dir = out_dir / "object_clouds"
    obj_dir.mkdir(parents=True, exist_ok=True)

    labeled_path = out_dir / "points3D_labeled.ply"
    save_ply(labeled_path, xyz_arr, rgb_arr, labels)
    logger.info("Saved -> %s", labeled_path)

    vis_rgb  = np.array([label_to_color(l) for l in labels], dtype=np.uint8)
    vis_path = out_dir / "points3D_vis.ply"
    save_ply(vis_path, xyz_arr, vis_rgb, labels)
    logger.info("Saved -> %s", vis_path)

    for lbl in sorted(set(labels)):
        if lbl == 0:
            continue
        mask     = labels == lbl
        obj_path = obj_dir / f"label_{lbl:02d}.ply"
        save_ply(obj_path, xyz_arr[mask], rgb_arr[mask], labels[mask])
        logger.info("Saved -> %s  (%d pts)", obj_path, mask.sum())

    logger.info("All outputs in: %s", out_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S")
    p = argparse.ArgumentParser(description="Point cloud object labeling via multi-view voting.")
    p.add_argument("--data_path", required=True, help="Scene directory (contains COLMAP data and masks)")
    p.add_argument("--sparse_dir", default=_SPARSE_DIR, help="COLMAP sparse model dir relative to data_path")
    p.add_argument("--mask_dir", default=_MASK_DIR, help="Mask folder name")
    p.add_argument("--output_dir", default=_OUTPUT_DIR, help="Output folder name inside data_path")
    p.add_argument("--min_points", type=int, default=_MIN_POINTS, help="Minimum number of points to be considered an object")
    p.add_argument("--disable_alias_merge", action="store_true", help="Disable 3D alias merging")
    p.add_argument("--alias_iou_thresh", type=float, default=_ALIAS_IOU_THRESH, help="Min 3D point IoU to merge two tracker IDs")
    p.add_argument("--alias_min_covisibility", type=int, default=_ALIAS_MIN_COVISIBILITY, help="Min shared points for alias merge candidates")
    run_voting(p.parse_args())