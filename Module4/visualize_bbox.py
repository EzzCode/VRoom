"""
Bounding Box Visualizer — check whether unprojected object points stay inside the bbox.

Shows a 3D scatter plot of all unprojected points for a given object label, overlaid with:
  - Red wireframe  : tight bbox  (obj_min / obj_max, before padding)
  - Blue wireframe : padded TSDF grid bbox  (grid_min / grid_max)
  - Grey points    : inside the tight bbox
  - Orange points  : leaking outside the tight bbox (these are the problem cases)

Usage:
    python visualize_bbox.py                        # browse all labels with ↑/↓
    python visualize_bbox.py --label 5              # start on label 5
    python visualize_bbox.py --bbox_clip 1.0 --padding 0.3

Controls:
  ↑ / ↓       : previous / next label
  Q / Escape  : quit

Rotate the 3D view with the mouse. Rotation angle is preserved when switching labels.
"""

import argparse
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from PIL import Image

from utils import compute_depth_trunc, unproject_to_3d

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--label",        type=int,   default=None,   help="Start on this label ID (default: first label found)")
parser.add_argument("--bbox_clip",    type=float, default=2.0,    help="Percentile clip for bbox (default 2.0)")
parser.add_argument("--padding",      type=float, default=0.22,   help="Bbox padding fraction (default 0.22)")
parser.add_argument("--depth_margin", type=float, default=1.1,    help="Depth trunc margin (default 1.1)")
parser.add_argument("--max_pts",      type=int,   default=50000,  help="Max points to plot (subsample for speed, default 50000)")
args = parser.parse_args()

BBOX_CLIP    = args.bbox_clip
PADDING      = args.padding
DEPTH_MARGIN = args.depth_margin
MAX_PTS      = args.max_pts

# ── Load data ─────────────────────────────────────────────────────────────────
input_dir = os.path.join(os.path.dirname(__file__), "inputs")

with open(os.path.join(input_dir, "cameras.json"), "r") as f:
    cameras = json.load(f)

num_depth_files = len(os.listdir(os.path.join(input_dir, "raw_depth")))
cameras = cameras[:num_depth_files]
num_cams = len(cameras)
print(f"Loaded {num_cams} cameras.")

depth_maps_raw  = []
semantic_maps   = []
intrinsics_list = []
extrinsics_list = []

for i, cam in enumerate(cameras):
    fx, fy   = cam["fx"], cam["fy"]
    W, H     = cam["width"], cam["height"]
    cx, cy   = W / 2.0, H / 2.0
    intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    R_c2w = np.array(cam["rotation"])
    R_w2c = R_c2w.T
    pos   = np.array(cam["position"])
    extrinsics = np.eye(4)
    extrinsics[:3, :3] = R_w2c
    extrinsics[:3, 3]  = -R_w2c @ pos

    depth_maps_raw.append(np.load(os.path.join(input_dir, "raw_depth", f"{i:05d}.npy")))
    sem = np.array(Image.open(os.path.join(input_dir, "semantic", f"{i:05d}.png")))
    semantic_maps.append(sem)
    intrinsics_list.append(intrinsics)
    extrinsics_list.append(extrinsics)

# ── Discover all labels ───────────────────────────────────────────────────────
all_labels = sorted({v for sem in semantic_maps for v in np.unique(sem)})
print(f"Labels found: {all_labels}")

if args.label is not None and args.label not in all_labels:
    sys.exit(f"Label {args.label} not found. Available: {all_labels}")

start_idx = all_labels.index(args.label) if args.label is not None else 0
state = {"label_idx": start_idx, "pts_cache": {}}

# ── Helpers ───────────────────────────────────────────────────────────────────
def _get_pts(label_id):
    """Unproject + bbox-compute for label_id. Results are cached."""
    if label_id in state["pts_cache"]:
        return state["pts_cache"][label_id]

    depth_trunc = compute_depth_trunc(depth_maps_raw, semantic_maps, label_id, margin=DEPTH_MARGIN)
    raw_pts = []
    for i in range(num_cams):
        d    = depth_maps_raw[i]
        mask = (semantic_maps[i] == label_id) & (d > 0) & (d < depth_trunc)
        if not np.any(mask):
            continue
        pts = unproject_to_3d(d, mask, intrinsics_list[i], extrinsics_list[i])
        if len(pts) > 0:
            raw_pts.append(pts)

    if not raw_pts:
        result = None
    else:
        all_pts  = np.vstack(raw_pts)
        obj_min  = np.percentile(all_pts, BBOX_CLIP,       axis=0)
        obj_max  = np.percentile(all_pts, 100 - BBOX_CLIP, axis=0)
        obj_size = obj_max - obj_min
        pad      = obj_size.max() * PADDING
        grid_min = obj_min - pad
        grid_max = obj_max + pad
        result   = (all_pts, obj_min, obj_max, grid_min, grid_max)

    state["pts_cache"][label_id] = result
    return result


def _box_segments(lo, hi):
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    corners = np.array([
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
    ])
    edges = [(0,1),(1,2),(2,3),(3,0), (4,5),(5,6),(6,7),(7,4), (0,4),(1,5),(2,6),(3,7)]
    return np.array([[corners[a], corners[b]] for a, b in edges])


# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(12, 9))
ax  = fig.add_subplot(111, projection="3d")
hint = fig.text(0.5, 0.01, "↑/↓ = change label   |   Q/Escape = quit",
                ha="center", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="#eeeeee", ec="none"))


def draw(label_id):
    elev, azim = ax.elev, ax.azim   # preserve current rotation
    ax.cla()

    data = _get_pts(label_id)
    if data is None:
        ax.set_title(f"Label {label_id} — no 3D points found")
        fig.canvas.draw_idle()
        return

    all_pts, obj_min, obj_max, grid_min, grid_max = data
    n_total   = len(all_pts)
    n_outside = int(np.sum(np.any((all_pts < obj_min) | (all_pts > obj_max), axis=1)))

    # Subsample for plotting
    pts = all_pts
    if len(pts) > MAX_PTS:
        idx = np.random.choice(len(pts), MAX_PTS, replace=False)
        pts = pts[idx]

    outside_mask = np.any((pts < obj_min) | (pts > obj_max), axis=1)
    inside_pts   = pts[~outside_mask]
    outside_pts  = pts[outside_mask]

    if len(inside_pts) > 0:
        ax.scatter(inside_pts[:, 0], inside_pts[:, 1], inside_pts[:, 2],
                   s=1, c="0.6", alpha=0.3, label=f"Inside ({len(inside_pts):,})")
    if len(outside_pts) > 0:
        ax.scatter(outside_pts[:, 0], outside_pts[:, 1], outside_pts[:, 2],
                   s=3, c="orangered", alpha=0.7, label=f"Outside ({len(outside_pts):,})")

    ax.add_collection3d(Line3DCollection(_box_segments(obj_min,  obj_max),
                                         colors="red",       linewidths=1.5, label="Tight bbox"))
    ax.add_collection3d(Line3DCollection(_box_segments(grid_min, grid_max),
                                         colors="steelblue", linewidths=1.0,
                                         linestyles="--",    label="Padded grid"))

    ax.set_xlim(grid_min[0], grid_max[0])
    ax.set_ylim(grid_min[1], grid_max[1])
    ax.set_zlim(grid_min[2], grid_max[2])
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.view_init(elev=elev, azim=azim)

    label_idx = all_labels.index(label_id)
    ax.set_title(
        f"Label {label_id}  [{label_idx + 1}/{len(all_labels)}]  —  "
        f"{n_outside:,} / {n_total:,} points leak outside tight bbox  "
        f"({100 * n_outside / n_total:.1f}%)\n"
        f"bbox_clip={BBOX_CLIP}  padding={PADDING}  depth_margin={DEPTH_MARGIN}"
    )
    ax.legend(loc="upper left", markerscale=4)
    print(f"  Label {label_id}: {n_outside:,}/{n_total:,} points outside "
          f"({100 * n_outside / n_total:.1f}%)")
    fig.canvas.draw_idle()


def on_key(event):
    li = state["label_idx"]
    if event.key == "up":
        state["label_idx"] = (li - 1) % len(all_labels)
    elif event.key == "down":
        state["label_idx"] = (li + 1) % len(all_labels)
    elif event.key in ("q", "escape"):
        plt.close(fig)
        return
    else:
        return
    draw(all_labels[state["label_idx"]])


fig.canvas.mpl_connect("key_press_event", on_key)
draw(all_labels[start_idx])

print("\nControls:  ↑/↓ = prev/next label   |   Q/Escape = quit")
plt.tight_layout()
plt.show()
