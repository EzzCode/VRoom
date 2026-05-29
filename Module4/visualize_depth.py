"""
Depth Map Visualizer — diagnose whether depth data is the source of mesh erosion.

4 panels per camera:
  Top-Left  : GS-rendered RGB
  Top-Right : Full depth map (colorized, closer = brighter)
  Bot-Left  : Depth masked to ONE object label (white = valid depth, black = zero/missing)
  Bot-Right : Per-pixel depth error map — shows WHERE depth is missing inside the label mask

Navigation:
  Left / Right arrows  → previous / next camera
  Ctrl + Left/Right    → jump 10 cameras
  Up / Down arrows     → cycle through object labels
  Q / Escape           → quit

Stats printed in the title:
  - How many cameras have ANY valid depth for the selected label
  - This camera's valid pixel count and % of the label's mask pixels
  - Min / max / mean depth for the selected label in this camera
"""

import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from PIL import Image

# ── Paths ──────────────────────────────────────────────────────────────────
input_dir    = os.path.join(os.path.dirname(__file__), "inputs")
renders_dir  = os.path.join(input_dir, "renders")
depth_dir    = os.path.join(input_dir, "raw_depth")
semantic_dir = os.path.join(input_dir, "semantic")

# ── Load file lists ────────────────────────────────────────────────────────
render_files  = sorted(f for f in os.listdir(renders_dir)  if f.endswith(".png"))
depth_files   = sorted(f for f in os.listdir(depth_dir)    if f.endswith(".npy"))
sem_files     = sorted(f for f in os.listdir(semantic_dir) if f.endswith(".png"))

n_frames = min(len(render_files), len(depth_files), len(sem_files))
if n_frames == 0:
    sys.exit("No frames found — check inputs/")
print(f"Loaded {n_frames} frames.")

# ── Cache semantic maps and discover labels ────────────────────────────────
print("Scanning semantic maps...")
sem_cache   = {}
all_labels  = set()
for i in range(n_frames):
    sem = np.array(Image.open(os.path.join(semantic_dir, sem_files[i])))
    sem_cache[i] = sem
    all_labels.update(np.unique(sem).tolist())
all_labels = sorted(all_labels)
print(f"Labels found: {all_labels}")

# Pre-compute per-label camera coverage (how many cameras see valid depth for each label)
print("Computing coverage stats...")
depth_cache = {}
for i in range(n_frames):
    depth_cache[i] = np.load(os.path.join(depth_dir, depth_files[i]))

label_coverage = {}
for lbl in all_labels:
    count = sum(
        1 for i in range(n_frames)
        if np.any((sem_cache[i] == lbl) & (depth_cache[i] > 0))
    )
    label_coverage[lbl] = count

# ── State ──────────────────────────────────────────────────────────────────
state = {"frame": 0, "label_idx": 0}

# ── Figure ─────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 9), facecolor="#1a1a1a")
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.08, wspace=0.05)

ax_rgb    = fig.add_subplot(gs[0, 0])
ax_depth  = fig.add_subplot(gs[0, 1])
ax_masked = fig.add_subplot(gs[1, 0])
ax_miss   = fig.add_subplot(gs[1, 1])

for ax in [ax_rgb, ax_depth, ax_masked, ax_miss]:
    ax.axis("off")
    ax.set_facecolor("#1a1a1a")

ax_rgb.set_title("RGB Render",                     color="white", fontsize=10, pad=3)
ax_depth.set_title("Full Depth (closer = brighter)", color="white", fontsize=10, pad=3)
ax_masked.set_title("Label Depth  (white = valid, black = MISSING)", color="white", fontsize=10, pad=3)
ax_miss.set_title("Missing-depth overlay on RGB  (red = no depth inside label)", color="white", fontsize=10, pad=3)

dummy = np.zeros((10, 10, 3), dtype=np.uint8)
im_rgb    = ax_rgb.imshow(dummy)
im_depth  = ax_depth.imshow(dummy)
im_masked = ax_masked.imshow(dummy, cmap="plasma")
im_miss   = ax_miss.imshow(dummy)

stat_text = fig.text(0.5, 0.01, "", ha="center", color="white", fontsize=9,
                     bbox=dict(boxstyle="round,pad=0.3", fc="#333333", ec="none"))

def draw(frame, label):
    rgb  = np.array(Image.open(os.path.join(renders_dir, render_files[frame])).convert("RGB"))
    d    = depth_cache[frame]
    sem  = sem_cache[frame]

    # ── Full depth panel ──────────────────────────────────────────────────
    valid_d = d[d > 0]
    if len(valid_d):
        lo, hi = np.percentile(valid_d, 1), np.percentile(valid_d, 99)
    else:
        lo, hi = 0.0, 1.0
    # Invert so closer = brighter
    depth_norm = np.clip((hi - d) / max(hi - lo, 1e-6), 0, 1)
    depth_norm[d == 0] = 0.0
    depth_rgb  = plt.cm.plasma(depth_norm)[:, :, :3]

    # ── Label mask ────────────────────────────────────────────────────────
    label_mask    = (sem == label)                     # every pixel labelled as this object
    depth_valid   = (d > 0)
    has_depth     = label_mask & depth_valid           # labelled + has depth
    missing_depth = label_mask & ~depth_valid          # labelled but NO depth  ← the problem

    n_label   = int(np.sum(label_mask))
    n_valid   = int(np.sum(has_depth))
    n_missing = int(np.sum(missing_depth))
    pct_valid = (n_valid / n_label * 100) if n_label > 0 else 0.0

    # Masked depth — show only label pixels
    masked_d = np.where(has_depth, d, 0.0)
    if n_valid > 0:
        lo_m = masked_d[has_depth].min()
        hi_m = masked_d[has_depth].max()
        mn_m = masked_d[has_depth].mean()
    else:
        lo_m = hi_m = mn_m = 0.0
    # Normalise masked depth to [0,1] for display
    span = max(hi_m - lo_m, 1e-6)
    masked_norm = np.where(has_depth, (masked_d - lo_m) / span, 0.0)
    masked_rgb  = plt.cm.plasma(masked_norm)[:, :, :3]
    masked_rgb[~label_mask] = [0.05, 0.05, 0.05]  # dim background

    # ── Missing-depth overlay ─────────────────────────────────────────────
    overlay = rgb.astype(np.float32) / 255.0
    # Dim areas outside label
    overlay[~label_mask] *= 0.25
    # Paint missing pixels red
    overlay[missing_depth] = [1.0, 0.1, 0.1]

    # ── Update images ─────────────────────────────────────────────────────
    im_rgb.set_data(rgb);         im_rgb.set_extent([0, rgb.shape[1], rgb.shape[0], 0])
    im_depth.set_data(depth_rgb); im_depth.set_extent([0, rgb.shape[1], rgb.shape[0], 0])
    im_masked.set_data(masked_rgb); im_masked.set_extent([0, rgb.shape[1], rgb.shape[0], 0])
    im_miss.set_data(np.clip(overlay, 0, 1)); im_miss.set_extent([0, rgb.shape[1], rgb.shape[0], 0])

    for ax in [ax_rgb, ax_depth, ax_masked, ax_miss]:
        ax.set_xlim(0, rgb.shape[1]); ax.set_ylim(rgb.shape[0], 0)

    # ── Title & stats ─────────────────────────────────────────────────────
    coverage = label_coverage.get(label, 0)
    fig.suptitle(
        f"Camera {frame}/{n_frames-1}   |   Label {label}   |   "
        f"Camera coverage: {coverage}/{n_frames} cams   |   "
        f"This cam: {n_valid:,} / {n_label:,} label-pixels have depth  ({pct_valid:.1f}%)",
        color="white", fontsize=11, y=0.99
    )

    depth_range = f"depth range [{lo_m:.3f} – {hi_m:.3f}], mean {mn_m:.3f}" if n_valid else "NO VALID DEPTH IN THIS CAMERA"
    stat_text.set_text(
        f"Missing (red): {n_missing:,} px  |  {depth_range}   |   "
        f"↑/↓ = change label   ←/→ = change camera"
    )

    fig.canvas.draw_idle()

def on_key(event):
    f   = state["frame"]
    li  = state["label_idx"]

    if event.key == "right":
        state["frame"] = min(f + 1, n_frames - 1)
    elif event.key == "left":
        state["frame"] = max(f - 1, 0)
    elif event.key == "ctrl+right":
        state["frame"] = min(f + 10, n_frames - 1)
    elif event.key == "ctrl+left":
        state["frame"] = max(f - 10, 0)
    elif event.key == "up":
        state["label_idx"] = (li - 1) % len(all_labels)
    elif event.key == "down":
        state["label_idx"] = (li + 1) % len(all_labels)
    elif event.key in ("q", "escape"):
        plt.close(fig); return

    draw(state["frame"], all_labels[state["label_idx"]])

fig.canvas.mpl_connect("key_press_event", on_key)
draw(0, all_labels[0])

print("\nControls:")
print("  ← / →         : previous / next camera")
print("  Ctrl + ← / →  : jump 10 cameras")
print("  ↑ / ↓         : cycle object labels")
print("  Q / Escape    : quit\n")
print("What to look for:")
print("  • Red pixels in bot-right = label pixels with ZERO depth → those voxels starved")
print("  • Low coverage count in title = chair rarely seen → expected mesh holes")
print("  • Depth range jumping wildly across cameras → corrupted depth extraction\n")

plt.show()
