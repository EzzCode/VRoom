"""Module4/ply_to_glb.py
---------------------
Convert PLY meshes (with per-vertex RGB colours) to self-contained GLB files.

The output GLB:
  - Uses an identity index buffer (ViroReact rejects un-indexed meshes)
  - Uses flat per-face normals
  - Bakes per-vertex colours into a compact per-face texture atlas
  - Embeds the atlas PNG as a base64 data URI inside the JSON chunk
    (avoids the ViroReact image.bufferView hang bug)

Dependencies: numpy, Pillow — both already used by Module4.

Usage
-----
As a module (called from extract_object_meshes.py):
    from ply_to_glb import build_glb_from_mesh

    glb_bytes = build_glb_from_mesh(verts_arr, triangles_arr, vertex_colors_float01)
    with open("object_001.glb", "wb") as f:
        f.write(glb_bytes)

As a standalone script (convert existing PLY files):
    python ply_to_glb.py objects/object_001.ply objects_glb/object_001.glb

    # Or convert an entire folder:
    python ply_to_glb.py objects/ objects_glb/
"""

from __future__ import annotations

import base64
import io
import json
import struct
import sys
from pathlib import Path

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pad4(n: int) -> int:
    """Round *n* up to the next multiple of 4."""
    return (n + 3) & ~3


def _png_bytes(rgba: np.ndarray) -> bytes:
    """Encode an (H, W, 4) uint8 array as a PNG byte string."""
    img = Image.fromarray(rgba, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Core builder – no external dependencies beyond numpy / Pillow
# ---------------------------------------------------------------------------

def build_glb_from_mesh(
    vertices: np.ndarray,
    triangles: np.ndarray,
    vertex_colors: np.ndarray,
) -> bytes:
    """Build a self-contained GLB binary from a coloured triangle mesh.

    Parameters
    ----------
    vertices : array-like, shape (V, 3), float
        World-space XYZ positions.
    triangles : array-like, shape (F, 3), int
        Face vertex indices.
    vertex_colors : array-like, shape (V, 3)
        Per-vertex RGB colours – float [0, 1] **or** uint8 [0, 255].

    Returns
    -------
    bytes
        Raw GLB file content ready to be written to disk.
    """
    vertices  = np.asarray(vertices,  dtype=np.float32)
    triangles = np.asarray(triangles, dtype=np.int64)

    # Normalise colours to uint8 [0, 255]
    vc = np.asarray(vertex_colors)
    if vc.dtype.kind == "f":
        vc_u8 = np.clip(vc * 255.0, 0, 255).astype(np.uint8)
    else:
        vc_u8 = np.clip(vc, 0, 255).astype(np.uint8)

    n_faces = len(triangles)
    if n_faces == 0:
        raise ValueError("Mesh has no faces.")

    # ------------------------------------------------------------------
    # 1. Texture atlas  (one solid-colour pixel per face)
    # ------------------------------------------------------------------
    atlas_sz = 1
    while atlas_sz * atlas_sz < n_faces:
        atlas_sz *= 2

    # Per-face average colour
    face_vc  = vc_u8[triangles]                                  # (F, 3, 3)
    face_avg = face_vc.mean(axis=1).clip(0, 255).astype(np.uint8)  # (F, 3)

    atlas = np.zeros((atlas_sz, atlas_sz, 4), dtype=np.uint8)
    atlas[:, :, 3] = 255                                         # fully opaque
    f_idx        = np.arange(n_faces)
    rows, cols   = np.divmod(f_idx, atlas_sz)
    atlas[rows, cols, :3] = face_avg

    png_data  = _png_bytes(atlas)
    image_uri = "data:image/png;base64," + base64.b64encode(png_data).decode("ascii")

    # ------------------------------------------------------------------
    # 2. De-indexed geometry  (no shared vertices → no index buffer)
    # ------------------------------------------------------------------
    n_out = n_faces * 3

    # Positions
    pos = vertices[triangles.reshape(-1)].astype(np.float32)     # (F*3, 3)

    # Flat normals — same normal for all three vertices of each face
    e1 = pos[1::3] - pos[0::3]
    e2 = pos[2::3] - pos[0::3]
    fn = np.cross(e1, e2).astype(np.float32)
    nlen = np.linalg.norm(fn, axis=1, keepdims=True)
    nlen[nlen == 0.0] = 1.0
    fn  /= nlen
    nrm  = np.repeat(fn, 3, axis=0).astype(np.float32)           # (F*3, 3)

    # UV coordinates — all three vertices of face i point to its atlas pixel
    u_coords = (cols + 0.5) / atlas_sz
    v_coords = (rows + 0.5) / atlas_sz
    uv_face  = np.column_stack([u_coords, v_coords]).astype(np.float32)  # (F, 2)
    uv       = np.repeat(uv_face, 3, axis=0)                     # (F*3, 2)

    # ------------------------------------------------------------------
    # 3. Binary buffer layout  (positions | normals | uvs | indices)
    # ------------------------------------------------------------------
    # Identity index buffer: each vertex referenced exactly once, in order.
    # ViroReact's GLTF loader rejects un-indexed primitives, so we add
    # indices = [0, 1, 2, ..., n_out-1].
    if n_out <= 65535:
        idx_dtype = np.uint16
        idx_component_type = 5123  # UNSIGNED_SHORT
    else:
        idx_dtype = np.uint32
        idx_component_type = 5125  # UNSIGNED_INT
    idx = np.arange(n_out, dtype=idx_dtype)

    pos_bytes = pos.tobytes()
    nrm_bytes = nrm.tobytes()
    uv_bytes  = uv.tobytes()
    idx_bytes = idx.tobytes()

    pos_off = 0
    pos_len = len(pos_bytes)
    nrm_off = _pad4(pos_off + pos_len)
    nrm_len = len(nrm_bytes)
    uv_off  = _pad4(nrm_off + nrm_len)
    uv_len  = len(uv_bytes)
    idx_off = _pad4(uv_off + uv_len)
    idx_len = len(idx_bytes)
    bin_len = _pad4(idx_off + idx_len)

    bin_buf = (
        pos_bytes
        + b"\x00" * (nrm_off - pos_off - pos_len)
        + nrm_bytes
        + b"\x00" * (uv_off  - nrm_off - nrm_len)
        + uv_bytes
        + b"\x00" * (idx_off - uv_off  - uv_len)
        + idx_bytes
        + b"\x00" * (bin_len - idx_off - idx_len)
    )

    # ------------------------------------------------------------------
    # 4. glTF JSON
    # ------------------------------------------------------------------
    pos_min = pos.min(axis=0).tolist()
    pos_max = pos.max(axis=0).tolist()

    gltf = {
        "asset":  {"version": "2.0", "generator": "VRoom Module4"},
        "scene":  0,
        "scenes": [{"nodes": [0]}],
        "nodes":  [{"mesh": 0}],
        "meshes": [{"primitives": [{
            "attributes": {"POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2},
            "indices": 3,
            "material": 0,
            "mode": 4,
        }]}],
        "materials": [{"pbrMetallicRoughness": {
            "baseColorTexture": {"index": 0},
            "metallicFactor": 0,
            "roughnessFactor": 1,
        }, "doubleSided": True}],
        "textures": [{"sampler": 0, "source": 0}],
        "images":  [{"uri": image_uri}],
        "samplers": [{"magFilter": 9728, "minFilter": 9728,
                      "wrapS": 33071, "wrapT": 33071}],
        "accessors": [
            {"bufferView": 0, "byteOffset": 0, "componentType": 5126,
             "count": n_out, "type": "VEC3",
             "min": pos_min, "max": pos_max},
            {"bufferView": 1, "byteOffset": 0, "componentType": 5126,
             "count": n_out, "type": "VEC3"},
            {"bufferView": 2, "byteOffset": 0, "componentType": 5126,
             "count": n_out, "type": "VEC2"},
            {"bufferView": 3, "byteOffset": 0, "componentType": idx_component_type,
             "count": n_out, "type": "SCALAR"},
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": pos_off,
             "byteLength": pos_len, "target": 34962},
            {"buffer": 0, "byteOffset": nrm_off,
             "byteLength": nrm_len, "target": 34962},
            {"buffer": 0, "byteOffset": uv_off,
             "byteLength": uv_len,  "target": 34962},
            {"buffer": 0, "byteOffset": idx_off,
             "byteLength": idx_len, "target": 34963},
        ],
        "buffers": [{"byteLength": bin_len}],
    }

    json_raw     = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    json_padded  = _pad4(len(json_raw))
    json_bytes   = json_raw + b" " * (json_padded - len(json_raw))

    # ------------------------------------------------------------------
    # 5. Assemble GLB
    # ------------------------------------------------------------------
    total = 12 + 8 + json_padded + 8 + bin_len

    glb = bytearray()
    glb += struct.pack("<III", 0x46546C67, 2, total)        # file header
    glb += struct.pack("<II",  json_padded, 0x4E4F534A)     # JSON chunk header
    glb += json_bytes
    glb += struct.pack("<II",  bin_len,     0x004E4942)     # BIN  chunk header
    glb += bin_buf

    return bytes(glb)


# ---------------------------------------------------------------------------
# PLY parser  (handles Module4's binary_little_endian format)
# ---------------------------------------------------------------------------

def _parse_ply(path: str | Path):
    """Parse a Module4 binary PLY file.

    Returns
    -------
    vertices      : (V, 3) float32
    triangles     : (F, 3) int64
    vertex_colors : (V, 3) uint8
    """
    with open(path, "rb") as f:
        header_lines: list[str] = []
        while True:
            line = f.readline().decode("ascii").strip()
            header_lines.append(line)
            if line == "end_header":
                break

        n_verts = n_faces = 0
        has_rgb = False
        for line in header_lines:
            parts = line.split()
            if line.startswith("element vertex"):
                n_verts = int(parts[2])
            elif line.startswith("element face"):
                n_faces = int(parts[2])
            elif parts[:3] in (["property", "uchar", "red"],
                               ["property", "uchar", "r"]):
                has_rgb = True

        if has_rgb:
            vdtype = np.dtype([
                ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                ("r", "u1"),  ("g", "u1"),  ("b", "u1"),
            ])
        else:
            vdtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4")])

        vdata = np.frombuffer(f.read(vdtype.itemsize * n_verts), dtype=vdtype)

        fdtype = np.dtype([
            ("cnt", "u1"), ("v0", "<i4"), ("v1", "<i4"), ("v2", "<i4"),
        ])
        fdata = np.frombuffer(f.read(fdtype.itemsize * n_faces), dtype=fdtype)

    vertices  = np.column_stack([vdata["x"], vdata["y"], vdata["z"]]).astype(np.float32)
    triangles = np.column_stack([fdata["v0"], fdata["v1"], fdata["v2"]]).astype(np.int64)

    if has_rgb:
        vc = np.column_stack([vdata["r"], vdata["g"], vdata["b"]]).astype(np.uint8)
    else:
        vc = np.full((n_verts, 3), 200, dtype=np.uint8)

    return vertices, triangles, vc


# ---------------------------------------------------------------------------
# Public convenience wrapper
# ---------------------------------------------------------------------------

def save_glb(vertices: np.ndarray, triangles: np.ndarray, vertex_colors: np.ndarray,
             output_path: str) -> None:
    """Build a GLB from mesh arrays and write it to *output_path*."""
    glb_bytes = build_glb_from_mesh(vertices, triangles, vertex_colors)
    with open(output_path, "wb") as f:
        f.write(glb_bytes)


def ply_to_glb(ply_path: str | Path, glb_path: str | Path) -> None:
    """Convert a single Module4 binary PLY file to a GLB file."""
    ply_path = Path(ply_path)
    glb_path = Path(glb_path)

    vertices, triangles, vertex_colors = _parse_ply(ply_path)
    print(f"  Loaded {len(vertices):,} vertices, {len(triangles):,} faces")

    glb_bytes = build_glb_from_mesh(vertices, triangles, vertex_colors)

    glb_path.parent.mkdir(parents=True, exist_ok=True)
    glb_path.write_bytes(glb_bytes)
    print(f"  → {glb_path}  ({len(glb_bytes) / 1024:.1f} KB)")


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) not in (3,):
        print("Usage:")
        print("  python ply_to_glb.py <input.ply>  <output.glb>")
        print("  python ply_to_glb.py <input_dir/> <output_dir/>")
        sys.exit(1)

    src = Path(sys.argv[1])
    dst = Path(sys.argv[2])

    if src.is_dir():
        ply_files = sorted(src.glob("*.ply"))
        if not ply_files:
            print(f"No .ply files found in {src}")
            sys.exit(1)
        dst.mkdir(parents=True, exist_ok=True)
        for ply_file in ply_files:
            glb_file = dst / (ply_file.stem + ".glb")
            print(f"Converting {ply_file.name} …")
            ply_to_glb(ply_file, glb_file)
        print(f"\nDone — {len(ply_files)} file(s) converted.")
    else:
        print(f"Converting {src.name} …")
        ply_to_glb(src, dst)
        print("Done.")
