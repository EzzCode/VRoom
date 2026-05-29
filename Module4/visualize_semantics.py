"""
Semantic Visualizer — helps diagnose whether mesh problems come from bad semantics.

Shows 3 panels per camera:
  Left  : GS-rendered RGB image
  Middle: Semantic labels colorized (each object gets a distinct color)
  Right : Semantic overlay blended onto RGB (50/50)

Navigation:
  Left / Right arrow keys  → previous / next camera
  Number keys 1-9          → jump to that object label and highlight it
  H                        → toggle label legend
  Q / Escape               → quit

The title bar shows which labels are visible in the current view and their pixel counts.
"""

import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from PIL import Image

# ── Paths ──────────────────────────────────────────────────────────────────
input_dir = os.path.join(os.path.dirname(__file__), "inputs")
renders_dir  = os.path.join(input_dir, "renders")
semantic_dir = os.path.join(input_dir, "semantic")

# ── Load all frames ────────────────────────────────────────────────────────
render_files   = sorted(f for f in os.listdir(renders_dir)  if f.endswith(".png"))
semantic_files = sorted(f for f in os.listdir(semantic_dir) if f.endswith(".png"))

if not render_files:
    sys.exit("No render images found in inputs/renders/")
if not semantic_files:
    sys.exit("No semantic images found in inputs/semantic/")

n_frames = min(len(render_files), len(semantic_files))
print(f"Found {n_frames} frames.")

# ── Discover all global labels so colors stay consistent across cameras ────
print("Scanning all semantic maps for labels...")
all_labels = set()
sem_cache  = {}  # cache as numpy arrays

for i in range(n_frames):
    sem = np.array(Image.open(os.path.join(semantic_dir, semantic_files[i])))
    sem_cache[i] = sem
    all_labels.update(np.unique(sem).tolist())

all_labels = sorted(all_labels)
n_labels   = len(all_labels)
label_to_idx = {lbl: idx for idx, lbl in enumerate(all_labels)}
print(f"Global labels ({n_labels}): {all_labels}")

# ── Build a fixed color palette (one color per label, label 0 = dark grey) ─
rng = np.random.RandomState(42)
palette = rng.randint(60, 230, size=(n_labels, 3)).astype(np.uint8)
if 0 in label_to_idx:
    palette[label_to_idx[0]] = [40, 40, 40]   # background → near-black

def sem_to_rgb(sem_img):
    """Convert a label image to an RGB color image using the global palette."""
    h, w = sem_img.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for lbl, idx in label_to_idx.items():
        mask = sem_img == lbl
        out[mask] = palette[idx]
    return out

# ── State ──────────────────────────────────────────────────────────────────
state = {"frame": 0, "highlight": None, "legend_on": True}

# ── Figure setup ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.patch.set_facecolor("#1a1a1a")
for ax in axes:
    ax.axis("off")
    ax.set_facecolor("#1a1a1a")

axes[0].set_title("RGB Render",       color="white", fontsize=11)
axes[1].set_title("Semantic Labels",  color="white", fontsize=11)
axes[2].set_title("Overlay (50/50)",  color="white", fontsize=11)

im_rgb     = axes[0].imshow(np.zeros((10, 10, 3), dtype=np.uint8))
im_sem     = axes[1].imshow(np.zeros((10, 10, 3), dtype=np.uint8))
im_overlay = axes[2].imshow(np.zeros((10, 10, 3), dtype=np.uint8))

legend_ax = fig.add_axes([0.0, 0.0, 0.12, 1.0])
legend_ax.set_facecolor("#111111")
legend_ax.axis("off")

status_text = fig.text(0.5, 0.01, "", ha="center", va="bottom",
                       color="white", fontsize=9,
                       bbox=dict(boxstyle="round,pad=0.3", fc="#333333", ec="none"))

def draw(frame_idx):
    rgb_img = np.array(Image.open(os.path.join(renders_dir, render_files[frame_idx])).convert("RGB"))
    sem_img = sem_cache[frame_idx]
    sem_rgb = sem_to_rgb(sem_img)

    # Highlight a single label if requested (dim everything else)
    if state["highlight"] is not None and state["highlight"] in label_to_idx:
        hl = state["highlight"]
        mask = sem_img == hl
        dimmed = (sem_rgb.astype(np.float32) * 0.15).astype(np.uint8)
        dimmed[mask] = palette[label_to_idx[hl]]
        sem_rgb = dimmed

    # Overlay blend
    alpha = 0.5
    overlay = np.clip(
        rgb_img.astype(np.float32) * (1 - alpha) + sem_rgb.astype(np.float32) * alpha,
        0, 255
    ).astype(np.uint8)

    im_rgb.set_data(rgb_img)
    im_rgb.set_extent([0, rgb_img.shape[1], rgb_img.shape[0], 0])
    im_sem.set_data(sem_rgb)
    im_sem.set_extent([0, sem_rgb.shape[1], sem_rgb.shape[0], 0])
    im_overlay.set_data(overlay)
    im_overlay.set_extent([0, overlay.shape[1], overlay.shape[0], 0])

    for ax in axes:
        ax.set_xlim(0, rgb_img.shape[1])
        ax.set_ylim(rgb_img.shape[0], 0)

    # Per-frame label info in status bar
    present = sorted(np.unique(sem_img).tolist())
    parts = []
    for lbl in present:
        count = int(np.sum(sem_img == lbl))
        parts.append(f"Label {lbl}: {count:,}px")
    status_text.set_text(f"Camera {frame_idx}/{n_frames-1}   |   " + "   ".join(parts))

    fig.suptitle(f"Camera {frame_idx} of {n_frames-1}  —  {len(present)} objects visible"
                 + (f"  [highlight: label {state['highlight']}]" if state["highlight"] is not None else ""),
                 color="white", fontsize=13, y=0.98)

    # Legend
    legend_ax.cla()
    legend_ax.set_facecolor("#111111")
    legend_ax.axis("off")
    if state["legend_on"]:
        patches = []
        for lbl in all_labels:
            col = palette[label_to_idx[lbl]] / 255.0
            in_view = lbl in present
            label_str = f"  {lbl}" + ("" if in_view else " (hidden)")
            alpha_val = 1.0 if in_view else 0.35
            patches.append(mpatches.Patch(color=(*col, alpha_val), label=label_str))
        legend_ax.legend(handles=patches, loc="upper left", fontsize=8,
                         framealpha=0.0, labelcolor="white",
                         handlelength=1.2, handleheight=1.2)
        legend_ax.set_title("Labels", color="white", fontsize=9, pad=4)

    fig.canvas.draw_idle()

def on_key(event):
    f = state["frame"]
    if event.key == "right":
        state["frame"] = min(f + 1, n_frames - 1)
    elif event.key == "left":
        state["frame"] = max(f - 1, 0)
    elif event.key == "ctrl+right":
        state["frame"] = min(f + 10, n_frames - 1)
    elif event.key == "ctrl+left":
        state["frame"] = max(f - 10, 0)
    elif event.key in [str(d) for d in range(10)]:
        # Number key: highlight that label ID
        target = int(event.key)
        if target in label_to_idx:
            state["highlight"] = None if state["highlight"] == target else target
        else:
            print(f"Label {target} does not exist. Available: {all_labels}")
    elif event.key == "h":
        state["legend_on"] = not state["legend_on"]
    elif event.key in ("q", "escape"):
        plt.close(fig)
        return
    draw(state["frame"])

fig.canvas.mpl_connect("key_press_event", on_key)

plt.tight_layout(rect=[0.12, 0.04, 1.0, 0.95])
draw(0)

print("\nControls:")
print("  ← / →          : previous / next camera")
print("  Ctrl + ← / →   : jump 10 cameras")
print("  0-9             : highlight that label ID (press again to clear)")
print("  H               : toggle legend")
print("  Q / Escape      : quit\n")

plt.show()
