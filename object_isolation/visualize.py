"""Diagnostic visualization for the object-isolation pipeline.

Produces a single multi-panel PNG (``diagnostic.png``) inside the object
output directory plus a separate 3D scatter (``cage_3d.png``). Run after
all phases up to ``calibrate`` have completed.

Panels
------
1. Reference real view (highlighted on its source frame).
2. Sample of 4 real cropped views from real_views/.
3. Zero123++ input canvas.
4. The 6 hallucinated tiles with az/el labels.
5. Metric cage: anchor cloud (raw vs kept) + visible AABB + full AABB.
6. Top-down camera diagram: real ref camera + 6 novel virtual cameras +
   object center, in world XY.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)
from PIL import Image


def _load_rgba(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGBA")
    return np.asarray(img)


def _read_simple_ply(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        header = b""
        while True:
            line = f.readline()
            header += line
            if line.strip() == b"end_header":
                break
        n = 0
        for ln in header.split(b"\n"):
            if ln.startswith(b"element vertex"):
                n = int(ln.split()[-1])
                break
        return np.frombuffer(f.read(n * 12), dtype=np.float32).reshape(n, 3).astype(np.float64)


def _draw_aabb(ax, aabb, color, label, lw=2.0):
    (xmin, ymin, zmin), (xmax, ymax, zmax) = aabb[0], aabb[1]
    corners = np.array([
        [xmin, ymin, zmin], [xmax, ymin, zmin], [xmax, ymax, zmin], [xmin, ymax, zmin],
        [xmin, ymin, zmax], [xmax, ymin, zmax], [xmax, ymax, zmax], [xmin, ymax, zmax],
    ])
    edges = [(0, 1), (1, 2), (2, 3), (3, 0),
             (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    first = True
    for a, b in edges:
        ax.plot(*zip(corners[a], corners[b]), color=color, lw=lw,
                label=label if first else None)
        first = False


def visualize(obj_dir: str, out_path: str | None = None) -> str:
    obj = Path(obj_dir)
    out_path = Path(out_path) if out_path else obj / "diagnostic.png"

    # ── load ─────────────────────────────────────────────────────────────
    summary = json.loads((obj / "extraction_summary.json").read_text(encoding="utf-8"))
    cage = json.loads((obj / "metric_cage.json").read_text(encoding="utf-8"))
    ref = json.loads((obj / "reference.json").read_text(encoding="utf-8"))
    z123_in_meta = json.loads((obj / "zero123_input.json").read_text(encoding="utf-8"))
    nv_meta = json.loads((obj / "novel_views" / "novel_views_meta.json").read_text(encoding="utf-8"))
    poses = json.loads((obj / "novel_views" / "poses.json").read_text(encoding="utf-8"))

    real_views = sorted((obj / "real_views").glob("*.png"))
    selected = ref.get("selected", {})
    ref_idx = selected.get("frame_index")
    ref_basename = selected.get("img_name", "")
    ref_score = selected.get("score", 0.0)
    if ref_score == 0.0:
        # score may live in all_scores keyed by frame_index
        for s in ref.get("all_scores", []):
            if s.get("frame_index") == ref_idx:
                ref_score = s.get("total", s.get("score", 0.0))
                break

    # find the actual reference real-view file (real_views/00170.png style)
    ref_view_path = None
    rel = selected.get("image_path")
    if rel:
        ref_view_path = obj / rel
    if ref_view_path is None or not ref_view_path.exists():
        ref_view_path = real_views[0] if real_views else None

    # ── figure layout ────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 14), constrained_layout=True)
    gs = fig.add_gridspec(3, 6)

    # Panel 1: Reference view
    ax = fig.add_subplot(gs[0, 0])
    if ref_view_path is not None and ref_view_path.exists():
        ax.imshow(_load_rgba(ref_view_path))
    ax.set_title(f"Reference\n{ref_basename}\nscore={ref_score:.3f}", fontsize=9)
    ax.axis("off")

    # Panel 2: Zero123 input canvas
    ax = fig.add_subplot(gs[0, 1])
    z123_path = obj / "zero123_input.png"
    if z123_path.exists():
        ax.imshow(np.asarray(Image.open(z123_path)))
    bbox = z123_in_meta.get("object_bbox")
    if bbox is not None:
        x0, y0, x1, y1 = bbox
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                               edgecolor="cyan", lw=1.5))
    ax.set_title(f"Zero123++ input\ncanvas={z123_in_meta.get('canvas_size')}", fontsize=9)
    ax.axis("off")

    # Panel 3: 4 sample real views
    sample_views = real_views[:4]
    for i, p in enumerate(sample_views):
        ax = fig.add_subplot(gs[0, 2 + i])
        ax.imshow(_load_rgba(p))
        ax.set_title(f"real_views/{p.name}", fontsize=8)
        ax.axis("off")

    # Panels 4: 6 novel tiles with az/el labels
    for i in range(6):
        r, c = divmod(i, 6)  # row 1, all 6 cols
        ax = fig.add_subplot(gs[1, i])
        tile_path = obj / "novel_views" / f"tile_{i}.png"
        if tile_path.exists():
            ax.imshow(_load_rgba(tile_path))
        az = poses[i]["tile_az_deg"]
        el = poses[i]["tile_el_deg"]
        ax.set_title(f"tile_{i}\naz={az:+.0f}°  el={el:+.0f}°", fontsize=9)
        ax.axis("off")

    # Panel 5: Top-down camera plot (world XZ — most rooms put up = ±Y)
    ax = fig.add_subplot(gs[2, 0:3])
    cams_world = []
    for p in poses:
        R = np.asarray(p["R_w2c"]); T = np.asarray(p["T_w2c"])
        C = -R.T @ T
        cams_world.append(C)
    cams_world = np.asarray(cams_world)
    center = np.asarray(cage["object_center_clean"])
    aabb_v = np.asarray(cage["aabb_visible"])
    aabb_f = np.asarray(cage["aabb_full"])

    # pick the two horizontal axes (perp to up)
    up = np.asarray(cage["object_up_world"])
    up_axis = int(np.argmax(np.abs(up)))
    horiz = [i for i in range(3) if i != up_axis]
    a0, a1 = horiz

    # AABB rectangles
    ax.add_patch(Rectangle(
        (aabb_v[0, a0], aabb_v[0, a1]),
        aabb_v[1, a0] - aabb_v[0, a0], aabb_v[1, a1] - aabb_v[0, a1],
        fill=False, edgecolor="tab:green", lw=2, label="AABB visible"))
    ax.add_patch(Rectangle(
        (aabb_f[0, a0], aabb_f[0, a1]),
        aabb_f[1, a0] - aabb_f[0, a0], aabb_f[1, a1] - aabb_f[0, a1],
        fill=False, edgecolor="tab:orange", lw=2, ls="--", label="AABB full (mirror-ext.)"))
    ax.scatter(center[a0], center[a1], color="red", s=80, marker="*",
               label="object center (clean)", zorder=5)
    ax.scatter(cams_world[:, a0], cams_world[:, a1], c="tab:blue",
               s=60, marker="^", label="virtual cams", zorder=4)
    for i, c in enumerate(cams_world):
        ax.annotate(str(i), (c[a0], c[a1]), fontsize=8,
                    textcoords="offset points", xytext=(5, 5))
        # camera→object direction
        ax.plot([c[a0], center[a0]], [c[a1], center[a1]],
                color="tab:blue", alpha=0.2, lw=0.7)

    ax.set_aspect("equal")
    ax.set_xlabel(f"world axis {a0}")
    ax.set_ylabel(f"world axis {a1}")
    ax.set_title("Top-down: virtual cameras + metric cage (perp to up)", fontsize=10)
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)

    # Panel 6: stats summary text
    ax = fig.add_subplot(gs[2, 3:6])
    ax.axis("off")
    extents_v = aabb_v[1] - aabb_v[0]
    extents_f = aabb_f[1] - aabb_f[0]
    text = (
        f"OBJECT ID: {summary['object_frame']['object_id']}\n"
        f"\n"
        f"-- Phase 1 (extract) --\n"
        f"  real views kept: {summary.get('n_real_views', '?')} / {summary.get('n_total_cameras', '?')}\n"
        f"  raw object_radius: {summary['object_frame']['object_radius']:.3f} m\n"
        f"  raw object_center: {np.round(summary['object_frame']['object_center'], 3).tolist()}\n"
        f"\n"
        f"-- Phase 2 (reference + Z123) --\n"
        f"  reference: idx={ref_idx}  {ref_basename}  score={ref_score:.3f}\n"
        f"  Z123 backend: {nv_meta.get('backend','?')}  steps={nv_meta.get('num_inference_steps','?')}  "
        f"guidance={nv_meta.get('guidance_scale','?')}\n"
        f"  Z123 tiles: {len(nv_meta.get('tiles', poses))}\n"
        f"\n"
        f"-- Phase 3.5 (metric cage) --\n"
        f"  DBSCAN: kept {cage['n_anchors_kept']}/{cage['n_anchors_total']} "
        f"({100.0*cage['n_anchors_kept']/cage['n_anchors_total']:.1f}%)\n"
        f"  eps={cage['dbscan_eps']:.4f}, n_clusters={cage['dbscan_stats']['n_clusters']}\n"
        f"  CLEAN center: {np.round(cage['object_center_clean'], 3).tolist()}\n"
        f"  CLEAN radius: {cage['object_radius_clean']:.3f} m  "
        f"(was {summary['object_frame']['object_radius']:.3f})\n"
        f"  AABB visible extents: {np.round(extents_v, 3).tolist()}\n"
        f"  AABB full   extents: {np.round(extents_f, 3).tolist()}\n"
    )
    ax.text(0, 1, text, fontsize=10, family="monospace", va="top",
            transform=ax.transAxes)

    fig.suptitle(f"Object-Isolation Pipeline — Diagnostic ({obj.name})",
                 fontsize=14, fontweight="bold")
    fig.savefig(out_path, dpi=110)
    plt.close(fig)

    # ── separate 3D scatter ─────────────────────────────────────────────
    pts = _read_simple_ply(obj / "object_anchors.ply")
    fig3 = plt.figure(figsize=(11, 9))
    ax3 = fig3.add_subplot(111, projection="3d")
    # all anchors, faint
    ax3.scatter(pts[::3, 0], pts[::3, 1], pts[::3, 2], c="lightgray", s=1.5,
                alpha=0.35, label=f"all anchors ({pts.shape[0]})")
    # NOTE: we don't have explicit kept indices saved; visualise using a radius
    # mask around clean center as a proxy for "near the cage" — purely visual.
    cclean = np.asarray(cage["object_center_clean"])
    rclean = float(cage["object_radius_clean"])
    near = np.linalg.norm(pts - cclean, axis=1) <= 1.05 * rclean
    ax3.scatter(pts[near, 0], pts[near, 1], pts[near, 2], c="tab:blue", s=3,
                alpha=0.7, label=f"within clean radius ({near.sum()})")
    _draw_aabb(ax3, aabb_v, "tab:green", "AABB visible")
    _draw_aabb(ax3, aabb_f, "tab:orange", "AABB full")
    ax3.scatter(*cclean, color="red", s=100, marker="*", label="center", zorder=10)
    # camera positions
    ax3.scatter(cams_world[:, 0], cams_world[:, 1], cams_world[:, 2],
                c="tab:purple", s=40, marker="^", label="virtual cams")
    ax3.set_xlabel("X"); ax3.set_ylabel("Y"); ax3.set_zlabel("Z")
    ax3.set_title("3D — anchors, metric cage, virtual cameras")
    ax3.legend(loc="upper left", fontsize=8)
    fig3.tight_layout()
    fig3.savefig(obj / "cage_3d.png", dpi=110)
    plt.close(fig3)

    print(f"[visualize] wrote {out_path}")
    print(f"[visualize] wrote {obj / 'cage_3d.png'}")
    return str(out_path)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("obj_dir")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    visualize(args.obj_dir, args.out)
