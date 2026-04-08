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
import json
from collections import Counter, defaultdict
from itertools import combinations
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
    """Derive the exact mask .png path from the COLMAP image stem.

    This pipeline now requires a strict one-to-one name contract:
    `image_name` -> `<stem>.png` in `mask_dir`.
    """
    stem = os.path.splitext(image_name)[0]
    return os.path.join(mask_dir, stem + ".png")


def load_mask(mask_dir, image_name):
    """Load a single-channel label mask, failing fast if the exact file is missing."""
    path = _mask_path(mask_dir, image_name)
    mask = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(
            f"Missing mask for {image_name}: {path}. "
            "The mask filename must match the image stem exactly."
        )
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    if mask.dtype != np.uint16:
        raise ValueError(
            f"Unsupported id_map dtype at {path}: {mask.dtype}. "
            "Expected uint16 from object_tracker.py"
        )
    return mask


##### Vote collection ###########################################################

def collect_projection_votes(images, cameras, points, mask_dir, temporal_decay=0.02):
    """
    For every 3D point, project into every view and read the label.
    Returns dict[point_id -> list[(label, weight)]].
    """
    pid_list = list(points.keys())
    pts_xyz = np.array([points[p][0] for p in pid_list])       # Nx3

    votes = defaultdict(list)

    ordered_images = sorted(images.values(), key=lambda x: x.name)
    n_images = len(ordered_images)

    for frame_idx, img in enumerate(ordered_images):
        mask = load_mask(mask_dir, img.name)

        R = quat_to_R(img.qvec)
        t = img.tvec.reshape(3, 1)
        fx, fy, cx, cy = intrinsics_from_camera(cameras[img.camera_id])
        h, w = mask.shape[:2]

        ui, vi, ok = project_all(pts_xyz, R, t, fx, fy, cx, cy, w, h)

        frame_age = (n_images - 1) - frame_idx
        frame_weight = np.exp(-temporal_decay * frame_age)

        for idx in np.where(ok)[0]:
            pid = pid_list[idx]
            label = int(mask[vi[idx], ui[idx]])
            point_error = float(points[pid][2])
            point_weight = 1.0 / (1.0 + point_error)
            votes[pid].append((label, frame_weight * point_weight))

        print(f"  {img.name}: {ok.sum()} / {len(pts_xyz)} points visible")

    return votes


def collect_correspondence_votes(images, points, mask_dir, temporal_decay=0.02):
    """
    Use COLMAP's 2D <-> 3D correspondence tracks instead of re-projecting.
    Returns dict[point_id -> list[(label, weight)]].
    """
    votes = defaultdict(list)
    mask_cache = {}

    image_name_order = sorted([img.name for img in images.values()])
    image_rank = {name: idx for idx, name in enumerate(image_name_order)}
    n_images = max(1, len(image_name_order))

    for pid, (xyz, rgb, err, tracks) in points.items():
        for img_id, pt2d_idx in tracks:
            if img_id not in images:
                continue
            img = images[img_id]

            # Lazy-load and cache masks
            if img_id not in mask_cache:
                mask_cache[img_id] = load_mask(mask_dir, img.name)
            mask = mask_cache[img_id]

            if pt2d_idx >= len(img.xys):
                continue
            u, v = int(round(img.xys[pt2d_idx][0])), int(round(img.xys[pt2d_idx][1]))
            h, w = mask.shape[:2]
            if 0 <= u < w and 0 <= v < h:
                frame_idx = image_rank.get(img.name, n_images - 1)
                frame_age = (n_images - 1) - frame_idx
                frame_weight = np.exp(-temporal_decay * frame_age)
                point_weight = 1.0 / (1.0 + float(err))
                votes[pid].append((int(mask[v, u]), frame_weight * point_weight))

    return votes


def collect_correspondence_label_evidence(images, points, mask_dir, temporal_decay=0.02):
    """Collect correspondence-first alias evidence keyed by COLMAP point identity.

    Returns:
        dict[point_id -> dict[label -> {'views': set[int], 'obs': int, 'weight': float}]]
    """
    mask_cache = {}
    evidence = {}

    image_name_order = sorted([img.name for img in images.values()])
    image_rank = {name: idx for idx, name in enumerate(image_name_order)}
    n_images = max(1, len(image_name_order))

    for pid, (_, _, err, tracks) in points.items():
        per_label = defaultdict(lambda: {"views": set(), "obs": 0, "weight": 0.0})
        for img_id, pt2d_idx in tracks:
            if img_id not in images:
                continue
            img = images[img_id]

            if img_id not in mask_cache:
                mask_cache[img_id] = load_mask(mask_dir, img.name)
            mask = mask_cache[img_id]
            if pt2d_idx >= len(img.xys):
                continue

            u = int(round(img.xys[pt2d_idx][0]))
            v = int(round(img.xys[pt2d_idx][1]))
            h, w = mask.shape[:2]
            if not (0 <= u < w and 0 <= v < h):
                continue

            label = int(mask[v, u])
            if label == 0:
                continue

            frame_idx = image_rank.get(img.name, n_images - 1)
            frame_age = (n_images - 1) - frame_idx
            frame_weight = np.exp(-temporal_decay * frame_age)
            point_weight = 1.0 / (1.0 + float(err))
            weight = float(frame_weight * point_weight)

            per_label[label]["views"].add(int(img_id))
            per_label[label]["obs"] += 1
            per_label[label]["weight"] += weight

        if len(per_label) >= 2:
            evidence[pid] = per_label

    return evidence


class UnionFind:
    def __init__(self):
        self.parent = {}
        self.rank = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
            return x
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True


def build_correspondence_alias_map(
    evidence,
    min_point_support=12,
    min_shared_views=6,
    min_weight_support=0.0,
    min_support_ratio=0.08,
    min_point_balance=0.20,
    min_obs_per_label_per_point=2,
):
    """Build alias map from correspondence co-support, then close transitively."""
    pair_stats = defaultdict(lambda: {
        "point_support": 0,
        "shared_views": set(),
        "weight_support": 0.0,
    })
    label_point_presence = defaultdict(int)

    labels_seen = set()
    for _, per_label in evidence.items():
        strong_labels = []
        for lbl, stats in per_label.items():
            li = int(lbl)
            if li == 0:
                continue
            if int(stats["obs"]) < min_obs_per_label_per_point:
                continue
            if len(stats["views"]) < 2:
                continue
            strong_labels.append(li)
            label_point_presence[li] += 1

        labels = sorted(strong_labels)
        labels_seen.update(labels)
        if len(labels) < 2:
            continue
        for a, b in combinations(labels, 2):
            la = per_label[a]
            lb = per_label[b]
            wa = float(la["weight"])
            wb = float(lb["weight"])
            weight_sum = wa + wb
            if weight_sum <= 0:
                continue
            # Reject pair evidence when one label is only a tiny noise tail on this point.
            balance = min(wa, wb) / weight_sum
            if balance < min_point_balance:
                continue

            pair = (a, b)
            pair_stats[pair]["point_support"] += 1
            pair_stats[pair]["shared_views"].update(la["views"] | lb["views"])
            pair_stats[pair]["weight_support"] += min(wa, wb)

    accepted_edges = []
    rejected_edges = []
    for (a, b), s in pair_stats.items():
        views = len(s["shared_views"])
        min_presence = max(1, min(label_point_presence.get(a, 0), label_point_presence.get(b, 0)))
        support_ratio = float(s["point_support"]) / float(min_presence)
        edge = {
            "a": int(a),
            "b": int(b),
            "point_support": int(s["point_support"]),
            "shared_view_count": int(views),
            "weight_support": float(s["weight_support"]),
            "support_ratio": float(support_ratio),
            "min_presence": int(min_presence),
        }
        if (
            s["point_support"] >= min_point_support
            and views >= min_shared_views
            and s["weight_support"] >= min_weight_support
            and support_ratio >= min_support_ratio
        ):
            accepted_edges.append(edge)
        else:
            reason = []
            if s["point_support"] < min_point_support:
                reason.append("insufficient_point_support")
            if views < min_shared_views:
                reason.append("insufficient_shared_views")
            if s["weight_support"] < min_weight_support:
                reason.append("insufficient_weight_support")
            if support_ratio < min_support_ratio:
                reason.append("insufficient_support_ratio")
            edge["reason"] = "+".join(reason)
            rejected_edges.append(edge)

    # Highest-evidence edges first for deterministic chain construction.
    accepted_edges.sort(key=lambda x: (x["point_support"], x["shared_view_count"], x["weight_support"]), reverse=True)

    uf = UnionFind()
    for lbl in labels_seen:
        uf.find(lbl)
    for edge in accepted_edges:
        uf.union(edge["a"], edge["b"])

    components = defaultdict(list)
    for lbl in sorted(labels_seen):
        components[uf.find(lbl)].append(lbl)

    alias_map = {}
    component_records = []
    for root, members in components.items():
        canonical = int(min(members))
        for m in members:
            alias_map[int(m)] = canonical
        component_records.append({
            "root": int(root),
            "canonical": canonical,
            "members": [int(x) for x in sorted(members)],
        })

    report = {
        "accepted_edges": accepted_edges,
        "rejected_edges": rejected_edges,
        "components": component_records,
        "thresholds": {
            "min_point_support": int(min_point_support),
            "min_shared_views": int(min_shared_views),
            "min_weight_support": float(min_weight_support),
            "min_support_ratio": float(min_support_ratio),
            "min_point_balance": float(min_point_balance),
            "min_obs_per_label_per_point": int(min_obs_per_label_per_point),
        },
        "label_point_presence": {str(int(k)): int(v) for k, v in sorted(label_point_presence.items())},
    }
    return alias_map, report


def apply_alias_map(labels, alias_map):
    """Apply canonical alias mapping to label array."""
    if not alias_map:
        return labels
    remapped = labels.copy()
    for src, dst in alias_map.items():
        if src == 0:
            continue
        remapped[labels == src] = dst
    return remapped


##### Label resolution strategies ###########################################################

def _weighted_label_counts(vote_items):
    counts = defaultdict(float)
    for label, weight in vote_items:
        counts[int(label)] += float(weight)
    return counts


def resolve_majority(vote_items, min_confidence=0.0, min_support=1):
    """Pick the highest weighted label, or background if support/confidence is too low."""
    if len(vote_items) < min_support:
        return 0
    counts = _weighted_label_counts(vote_items)
    winner_label, winner_weight = max(counts.items(), key=lambda kv: kv[1])
    total_weight = sum(counts.values())
    if total_weight <= 0:
        return 0
    if min_confidence > 0 and (winner_weight / total_weight) < min_confidence:
        return 0  # too ambiguous → background
    return winner_label


def resolve_probability(vote_items, min_confidence=0.0, min_support=1):
    """Sample one label proportional to weighted frequency."""
    if len(vote_items) < min_support:
        return 0
    counts = _weighted_label_counts(vote_items)
    labels, freqs = zip(*counts.items())
    probs = np.array(freqs, dtype=float)
    if probs.sum() <= 0:
        return 0
    probs /= probs.sum()
    return int(np.random.choice(labels, p=probs))


def resolve_correspondence(vote_items, min_confidence=0.0, min_support=1):
    """Majority vote ignoring background, with confidence gate."""
    if len(vote_items) < min_support:
        return 0
    fg = [(l, w) for l, w in vote_items if l != 0]
    if not fg:
        return 0
    counts = _weighted_label_counts(fg)
    winner_label, winner_weight = max(counts.items(), key=lambda kv: kv[1])
    total_weight = sum(w for _, w in vote_items)
    if total_weight <= 0:
        return 0
    if min_confidence > 0 and (winner_weight / total_weight) < min_confidence:
        return 0
    return winner_label


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
        ("label", "u2"),
    ]
    n = len(xyz)
    normals = np.zeros((n, 3), dtype=np.float32)
    labels_u16 = labels.astype(np.uint16)
    arr = np.empty(n, dtype=dtype)
    for i, row in enumerate(
        np.hstack([xyz, normals, rgb, labels_u16.reshape(-1, 1)])
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


# def prune_3d_outliers_dbscan(xyz, labels, eps=0.05, min_samples=5):
#     """DBSCAN-based spatial cleanup: for each label, keep only the largest
#     spatial cluster and reclassify scattered outlier points as background.
#     
#     Args:
#         xyz: Nx3 point coordinates.
#         labels: N-length label array.
#         eps: DBSCAN neighbourhood radius (tune to scene scale).
#         min_samples: min points to form a DBSCAN core point.
#     """
#     cleaned = labels.copy()
#     for lbl in np.unique(labels):
#         if lbl == 0:
#             continue
#         mask = (labels == lbl)
#         obj_xyz = xyz[mask]
#         if len(obj_xyz) < min_samples:
#             cleaned[mask] = 0
#             continue
#         clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(obj_xyz)
#         cluster_labels = clustering.labels_
#         if (cluster_labels == -1).all():
#             cleaned[mask] = 0
#             continue
#         # Keep only the largest cluster
#         counts = Counter(cl for cl in cluster_labels if cl != -1)
#         largest_cluster = counts.most_common(1)[0][0]
#         outlier_within = (cluster_labels != largest_cluster)
#         indices = np.where(mask)[0]
#         cleaned[indices[outlier_within]] = 0
#     return cleaned
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
        votes = collect_correspondence_votes(images, points, mask_dir, temporal_decay=args.temporal_decay)
    else:
        votes = collect_projection_votes(images, cameras, points, mask_dir, temporal_decay=args.temporal_decay)

    resolver = _RESOLVERS[algo]

    # Resolve labels ###########################################################
    pid_order = list(points.keys())
    xyz_arr = np.array([points[p][0] for p in pid_order], dtype=np.float32)
    rgb_arr = np.array([points[p][1] for p in pid_order], dtype=np.uint8)

    labels = np.zeros(len(pid_order), dtype=np.uint16)
    for i, pid in enumerate(pid_order):
        if pid in votes and votes[pid]:
            labels[i] = resolver(
                votes[pid],
                min_confidence=args.min_confidence,
                min_support=args.min_support,
            )

    labels_pre_merge = labels.copy()
    alias_map = {}
    merge_report = {
        "alias_merge_enabled": not args.disable_alias_merge,
        "accepted_edges": [],
        "rejected_edges": [],
        "components": [],
    }
    if not args.disable_alias_merge:
        evidence = collect_correspondence_label_evidence(
            images,
            points,
            mask_dir,
            temporal_decay=args.temporal_decay,
        )
        alias_map, merge_report = build_correspondence_alias_map(
            evidence,
            min_point_support=args.alias_min_point_support,
            min_shared_views=args.alias_min_shared_views,
            min_weight_support=args.alias_min_weight_support,
            min_support_ratio=args.alias_min_support_ratio,
            min_point_balance=args.alias_min_point_balance,
            min_obs_per_label_per_point=args.alias_min_obs_per_label_per_point,
        )
        labels = apply_alias_map(labels, alias_map)
        print(
            f"\nAlias merge: {len(merge_report['accepted_edges'])} edges accepted, "
            f"{len(merge_report['rejected_edges'])} rejected, "
            f"{len(merge_report['components'])} components"
        )

    unique, counts = np.unique(labels, return_counts=True)
    print(f"\nLabel distribution ({len(unique)} labels):")
    for lbl, cnt in zip(unique, counts):
        print(f"  label {lbl:3d}: {cnt:6d} pts ({100*cnt/len(labels):.1f}%)")

    # Create output folder ###########################################################
    out_dir = os.path.join(data, args.output_dir)
    obj_dir = os.path.join(out_dir, "object_clouds")
    os.makedirs(obj_dir, exist_ok=True)

    # Persist alias artifacts before pruning for post-hoc inspection.
    alias_map_path = os.path.join(out_dir, "alias_map.json")
    merge_report_path = os.path.join(out_dir, "merge_report.json")
    with open(alias_map_path, "w", encoding="utf-8") as f:
        json.dump({str(k): int(v) for k, v in sorted(alias_map.items())}, f, indent=2)

    pre_unique, pre_counts = np.unique(labels_pre_merge, return_counts=True)
    post_unique, post_counts = np.unique(labels, return_counts=True)
    merge_report["pre_merge_distribution"] = {str(int(l)): int(c) for l, c in zip(pre_unique, pre_counts)}
    merge_report["post_merge_distribution"] = {str(int(l)): int(c) for l, c in zip(post_unique, post_counts)}
    with open(merge_report_path, "w", encoding="utf-8") as f:
        json.dump(merge_report, f, indent=2)

    # Prune outliers ###########################################################
    labels = prune_3d_outliers(xyz_arr, labels, min_points=args.min_points)

    unique, counts = np.unique(labels, return_counts=True)
    pruned_count = int(np.count_nonzero((labels_pre_merge != 0) & (labels == 0)))
    print(f"\nPruned {pruned_count} points to background")
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
    p.add_argument("--min_confidence", type=float, default=0.3, help="Min vote agreement ratio to assign a label (0-1, 0=disabled)")
    p.add_argument("--min_support", type=int, default=3, help="Minimum number of per-point view votes before assigning a label")
    p.add_argument("--temporal_decay", type=float, default=0.02, help="Exponential decay for older frame votes")
    p.add_argument("--disable_alias_merge", action="store_true", help="Disable correspondence-based alias merging")
    p.add_argument("--alias_min_point_support", type=int, default=20, help="Minimum shared COLMAP points to accept alias edge")
    p.add_argument("--alias_min_shared_views", type=int, default=6, help="Minimum distinct views to accept alias edge")
    p.add_argument("--alias_min_weight_support", type=float, default=0.0, help="Minimum accumulated weighted co-support for alias edge")
    p.add_argument("--alias_min_support_ratio", type=float, default=0.12, help="Minimum pair support ratio over weaker-label point presence")
    p.add_argument("--alias_min_point_balance", type=float, default=0.25, help="Minimum per-point weight balance between paired labels")
    p.add_argument("--alias_min_obs_per_label_per_point", type=int, default=2, help="Minimum observations per label on a point for alias evidence")
    run_voting(p.parse_args())
