"""Phase 4 — assemble a COLMAP-format training dataset for the standalone
2DGS run.

Combines:
    * Real-photo crops (`real_views/00*.png` from extraction_real)
    * Zero123++ synthetic tiles (`novel_views/tile_*.png` from zero123_runner)

into a single self-contained dataset directory that vendored 3DGS / 2DGS
training scripts can ingest unchanged:

    <out>/<obj>/dataset/
        images/              # 512x512 PNGs (RGB-only; alpha is exported as masks)
        masks/               # optional — alpha as 8-bit single-channel PNGs
        sparse/0/
            cameras.txt
            images.txt
            points3D.txt     # seeded with metric-cage-cleaned anchors
        dataset_summary.json
        train_list.txt       # all images
        test_list.txt        # subset (every 8th real view)

Each image is exported as its own PINHOLE camera (intrinsics differ between
real-view crops because of letterboxing, and tiles use a fixed-FoV virtual
camera). All poses are world->camera rotation+translation matching COLMAP.

Usage::

    python object_isolation/build_dataset.py isolated_output_real/obj_9
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import List

import numpy as np

logger = logging.getLogger(__name__)


# ── pose / format helpers ──────────────────────────────────────────────────


def _R_to_quat(R: np.ndarray) -> tuple[float, float, float, float]:
    """3x3 rotation -> COLMAP quaternion (qw, qx, qy, qz)."""
    R = np.asarray(R, dtype=np.float64)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = 0.5 / np.sqrt(tr + 1.0)
        qw = 0.25 / s
        qx = (R[2, 1] - R[1, 2]) * s
        qy = (R[0, 2] - R[2, 0]) * s
        qz = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    q /= np.linalg.norm(q)
    return float(q[0]), float(q[1]), float(q[2]), float(q[3])


def _read_simple_ply_xyz_rgb(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (xyz (N,3) float32, rgb (N,3) uint8). Handles ascii or binary
    little-endian PLYs that include x/y/z and optionally red/green/blue.
    Falls back to grey if no colours present.
    """
    with open(path, "rb") as f:
        header = b""
        while True:
            line = f.readline()
            header += line
            if line.strip() == b"end_header":
                break
        head = header.decode("latin1", errors="replace")
        ascii_mode = "format ascii" in head
        # detect prop order
        props = []
        n_vertex = 0
        for line in head.splitlines():
            if line.startswith("element vertex"):
                n_vertex = int(line.split()[-1])
            elif line.startswith("property"):
                props.append(line.split()[-1])

        if ascii_mode:
            xs, ys, zs = [], [], []
            rs, gs, bs = [], [], []
            has_rgb = "red" in props
            for _ in range(n_vertex):
                parts = f.readline().decode("latin1").split()
                idx_x = props.index("x")
                xs.append(float(parts[idx_x]))
                ys.append(float(parts[props.index("y")]))
                zs.append(float(parts[props.index("z")]))
                if has_rgb:
                    rs.append(int(parts[props.index("red")]))
                    gs.append(int(parts[props.index("green")]))
                    bs.append(int(parts[props.index("blue")]))
            xyz = np.stack([np.array(xs), np.array(ys), np.array(zs)], axis=1).astype(np.float32)
            if has_rgb:
                rgb = np.stack([rs, gs, bs], axis=1).astype(np.uint8)
            else:
                rgb = np.full((n_vertex, 3), 180, dtype=np.uint8)
            return xyz, rgb
        else:
            # binary little-endian — assume float32 x,y,z (+optional uchar rgb)
            dt = []
            for p in props:
                if p in ("x", "y", "z", "nx", "ny", "nz"):
                    dt.append((p, "<f4"))
                elif p in ("red", "green", "blue", "alpha", "label"):
                    dt.append((p, "<u1"))
                else:
                    dt.append((p, "<f4"))
            arr = np.frombuffer(f.read(), dtype=np.dtype(dt), count=n_vertex)
            xyz = np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float32)
            if "red" in props:
                rgb = np.stack([arr["red"], arr["green"], arr["blue"]], axis=1).astype(np.uint8)
            else:
                rgb = np.full((n_vertex, 3), 180, dtype=np.uint8)
            return xyz, rgb


# ── manifest collection ────────────────────────────────────────────────────


def _gather_real_views(obj_dir: Path) -> List[dict]:
    meta_path = obj_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"missing {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    out = []
    for entry in meta:
        rel = entry["image_path"]
        if not (obj_dir / rel).exists():
            continue
        out.append({
            "source": "real",
            "src_image": str((obj_dir / rel).resolve()),
            "name": Path(rel).stem,           # 00003
            "width": int(entry["width"]),
            "height": int(entry["height"]),
            "K": np.asarray(entry["K"], dtype=np.float64),
            "R_w2c": np.asarray(entry["R_w2c"], dtype=np.float64),
            "T_w2c": np.asarray(entry["T_w2c"], dtype=np.float64),
            "extras": {
                "frame_index": entry.get("frame_index"),
                "img_name": entry.get("img_name"),
                "mean_alpha": entry.get("mean_alpha"),
            },
        })
    return out


def _gather_tiles(obj_dir: Path) -> List[dict]:
    poses_path = obj_dir / "novel_views" / "poses.json"
    if not poses_path.exists():
        logger.warning("no novel_views/poses.json — skipping tiles")
        return []
    poses = json.loads(poses_path.read_text(encoding="utf-8"))
    out = []
    for i, p in enumerate(poses):
        tile = obj_dir / "novel_views" / f"tile_{i}.png"
        if not tile.exists():
            continue
        out.append({
            "source": "tile",
            "src_image": str(tile.resolve()),
            "name": f"tile_{i}",
            "width": int(p["width"]),
            "height": int(p["height"]),
            "K": np.asarray(p["K"], dtype=np.float64),
            "R_w2c": np.asarray(p["R_w2c"], dtype=np.float64),
            "T_w2c": np.asarray(p["T_w2c"], dtype=np.float64),
            "extras": {
                "tile_az_deg": p.get("tile_az_deg"),
                "tile_el_deg": p.get("tile_el_deg"),
            },
        })
    return out


# ── COLMAP text writers ────────────────────────────────────────────────────


def _write_cameras_txt(path: Path, items: List[dict]) -> None:
    lines = [
        "# Camera list with one line of data per camera:",
        "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]",
        f"# Number of cameras: {len(items)}",
    ]
    for cam_id, it in enumerate(items, start=1):
        K = it["K"]
        fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
        lines.append(
            f"{cam_id} PINHOLE {it['width']} {it['height']} "
            f"{fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_images_txt(path: Path, items: List[dict]) -> None:
    lines = [
        "# Image list with two lines of data per image:",
        "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
        "#   POINTS2D[] as (X, Y, POINT3D_ID)",
        f"# Number of images: {len(items)}",
    ]
    for img_id, it in enumerate(items, start=1):
        qw, qx, qy, qz = _R_to_quat(it["R_w2c"])
        tx, ty, tz = (float(v) for v in it["T_w2c"])
        cam_id = img_id          # 1:1
        name = it["dst_image_name"]
        lines.append(
            f"{img_id} {qw:.9f} {qx:.9f} {qy:.9f} {qz:.9f} "
            f"{tx:.9f} {ty:.9f} {tz:.9f} {cam_id} {name}"
        )
        lines.append("")          # empty 2D points list
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_points3d_txt(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    lines = [
        "# 3D point list with one line of data per point:",
        "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)",
        f"# Number of points: {len(xyz)}",
    ]
    for i, (p, c) in enumerate(zip(xyz, rgb), start=1):
        lines.append(
            f"{i} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} "
            f"{int(c[0])} {int(c[1])} {int(c[2])} 0.0"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ── driver ────────────────────────────────────────────────────────────────


def build_dataset(
    obj_dir: str | Path,
    out_subdir: str = "dataset",
    test_every: int = 8,
    keep_alpha_as_mask: bool = True,
) -> dict:
    """Materialise a COLMAP-format dataset for vendored 2DGS training."""
    obj = Path(obj_dir)
    if not obj.is_dir():
        raise FileNotFoundError(obj)

    real_items = _gather_real_views(obj)
    tile_items = _gather_tiles(obj)
    all_items = real_items + tile_items
    if not all_items:
        raise RuntimeError("no images gathered — check obj_dir")

    # Assign final filenames (rename to source-stamped names for clarity).
    for it in all_items:
        prefix = "r" if it["source"] == "real" else "n"
        it["dst_image_name"] = f"{prefix}_{it['name']}.png"

    out_root = obj / out_subdir
    img_dir = out_root / "images"
    mask_dir = out_root / "masks"
    sparse_dir = out_root / "sparse" / "0"
    if out_root.exists():
        shutil.rmtree(out_root)
    img_dir.mkdir(parents=True)
    if keep_alpha_as_mask:
        mask_dir.mkdir(parents=True)
    sparse_dir.mkdir(parents=True)

    # Copy images; if RGBA, split alpha into masks/.
    import cv2
    for it in all_items:
        src = it["src_image"]
        bgra = cv2.imread(src, cv2.IMREAD_UNCHANGED)
        if bgra is None:
            raise RuntimeError(f"failed to read {src}")
        if bgra.ndim == 2:
            bgr = cv2.cvtColor(bgra, cv2.COLOR_GRAY2BGR)
            alpha = np.full(bgr.shape[:2], 255, np.uint8)
        elif bgra.shape[2] == 4:
            bgr = bgra[:, :, :3]
            alpha = bgra[:, :, 3]
        else:
            bgr = bgra
            alpha = np.full(bgr.shape[:2], 255, np.uint8)
        # Write RGB (compositing alpha onto white so background is neutral
        # for any consumer that ignores the mask).
        comp = bgr.astype(np.float32)
        a = alpha.astype(np.float32) / 255.0
        white = np.full_like(comp, 255.0)
        comp = comp * a[..., None] + white * (1.0 - a[..., None])
        cv2.imwrite(str(img_dir / it["dst_image_name"]), comp.astype(np.uint8))
        if keep_alpha_as_mask:
            cv2.imwrite(str(mask_dir / it["dst_image_name"]), alpha)

    # Sparse seed cloud — prefer cleaned anchors (post metric-cage).
    cage_path = obj / "metric_cage.json"
    anchors_path = obj / "object_anchors.ply"
    seed_xyz: np.ndarray
    seed_rgb: np.ndarray
    if anchors_path.exists():
        seed_xyz, seed_rgb = _read_simple_ply_xyz_rgb(anchors_path)
    else:
        raise FileNotFoundError(anchors_path)

    # If a cage AABB exists, drop anchors outside the FULL cage to remove
    # any residual outliers that survived the per-object DBSCAN sweep.
    if cage_path.exists():
        cage = json.loads(cage_path.read_text(encoding="utf-8"))
        aabb = np.asarray(cage["aabb_full"], dtype=np.float32)  # 2x3
        lo, hi = aabb[0], aabb[1]
        keep = np.all((seed_xyz >= lo) & (seed_xyz <= hi), axis=1)
        seed_xyz = seed_xyz[keep]
        seed_rgb = seed_rgb[keep]
        logger.info("Seed cloud: %d/%d anchors kept inside cage", len(seed_xyz), keep.size)

    _write_cameras_txt(sparse_dir / "cameras.txt", all_items)
    _write_images_txt(sparse_dir / "images.txt", all_items)
    _write_points3d_txt(sparse_dir / "points3D.txt", seed_xyz, seed_rgb)

    # Train / test split.
    train_names: list[str] = []
    test_names: list[str] = []
    for i, it in enumerate(all_items):
        if it["source"] == "real" and test_every > 0 and (i % test_every == 0):
            test_names.append(it["dst_image_name"])
        else:
            train_names.append(it["dst_image_name"])
    (out_root / "train_list.txt").write_text("\n".join(train_names) + "\n", encoding="utf-8")
    (out_root / "test_list.txt").write_text("\n".join(test_names) + "\n", encoding="utf-8")

    summary = {
        "obj_dir": str(obj.resolve()),
        "dataset_dir": str(out_root.resolve()),
        "n_real": len(real_items),
        "n_tiles": len(tile_items),
        "n_train": len(train_names),
        "n_test": len(test_names),
        "n_seed_points": int(len(seed_xyz)),
        "image_size": [int(all_items[0]["width"]), int(all_items[0]["height"])],
        "cage_aabb_full": (json.loads(cage_path.read_text())["aabb_full"]
                            if cage_path.exists() else None),
        "manifest": [
            {
                "name": it["dst_image_name"],
                "source": it["source"],
                "width": it["width"],
                "height": it["height"],
                **{k: v for k, v in it["extras"].items() if v is not None},
            }
            for it in all_items
        ],
    }
    (out_root / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    logger.info(
        "Dataset assembled at %s — real=%d tiles=%d train=%d test=%d seeds=%d",
        out_root, len(real_items), len(tile_items),
        len(train_names), len(test_names), len(seed_xyz),
    )
    return summary


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("obj_dir")
    p.add_argument("--out_subdir", default="dataset")
    p.add_argument("--test_every", type=int, default=8,
                   help="Hold out every Nth real view as test (0 disables).")
    p.add_argument("--no_masks", action="store_true",
                   help="Don't write masks/ alongside images/ (alpha discarded).")
    p.add_argument("-v", "--verbose", action="store_true")
    a = p.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if a.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    summary = build_dataset(
        a.obj_dir,
        out_subdir=a.out_subdir,
        test_every=a.test_every,
        keep_alpha_as_mask=not a.no_masks,
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "manifest"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
