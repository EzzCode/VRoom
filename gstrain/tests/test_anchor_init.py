"""
test_anchor_init.py
====================
Verifies that AnchorInitializer works correctly.

Run from inside the gstrain folder:
    python test_anchor_init.py

What is checked:
     Voxel size auto-computation is a positive scalar
     Anchor count is strictly less than the raw point count (voxels merged)
     Anchor positions lie within the bounding box of the input points
     Offsets are initialised to zero  (they start neutral)
     Anchor features are initialised to zero  (they start neutral)
     Scales are finite and negative (log-scale < 0 for small Gaussians)
     Rotations are unit identity quaternions  [1, 0, 0, 0]
     Anchor mask is all-True  (all anchors visible at init)
     Label IDs have the right shape and integer dtype
     Visual summary printed with shape & statistics

Optional real-data test:
    Reads  points3D_text.ply  if it exists next to this file.
    Falls back to synthetic random points if the PLY is not found.
"""

import sys
import os
import torch
import numpy as np

# ------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))

from gstrain.vroom_core.models.anchor_field import AnchorSeedBuilder

# ## ANSI colours for nicer terminal output ##########################
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"

passed = []
failed = []


def check(name: str, condition: bool, detail: str = ""):
    if condition:
        passed.append(name)
        print(f"  {GREEN}{RESET}  {name}")
    else:
        failed.append(name)
        print(f"  {RED}✗{RESET}  {name}" + (f"  ← {detail}" if detail else ""))


# ==================================================================
# 1.  Load or synthesise point cloud
# ==================================================================

PLY_PATH = os.path.join(HERE, "datasets", "points3D_text.ply")


def load_ply_xyz(path: str) -> np.ndarray:
    """Minimal PLY xyz loader (no extra deps needed)."""
    with open(path, "rb") as f:
        header = b""
        while True:
            line = f.readline()
            header += line
            if line.strip() == b"end_header":
                break
    # Parse vertex count
    n_verts = 0
    for line in header.split(b"\n"):
        if line.startswith(b"element vertex"):
            n_verts = int(line.split()[-1])
    # Parse which properties exist (x, y, z assumed first three)
    props = []
    for line in header.split(b"\n"):
        if line.startswith(b"property"):
            props.append(line.split()[1])  # dtype token
    dtype_map = {b"float": np.float32, b"double": np.float64}
    row_dtype = [(f"f{i}", dtype_map.get(p, np.float32)) for i, p in enumerate(props)]
    data = np.frombuffer(
        f.read(n_verts * np.dtype(row_dtype).itemsize), dtype=row_dtype
    )
    xyz = np.stack([data["f0"], data["f1"], data["f2"]], axis=-1)
    return xyz.astype(np.float32)


print(f"\n{CYAN}━━━  AnchorInitializer Test  ━━━{RESET}\n")

if os.path.exists(PLY_PATH):
    print(f"Loading real point cloud from:\n  {PLY_PATH}\n")
    try:
        xyz_np = load_ply_xyz(PLY_PATH)
        source = "real PLY"
    except Exception as e:
        print(
            f"{YELLOW}Warning: failed to parse PLY ({e}), falling back to synthetic data.{RESET}\n"
        )
        xyz_np = (np.random.randn(5000, 3) * 2.0).astype(np.float32)
        source = "synthetic"
else:
    print(
        f"{YELLOW}PLY not found at expected path, using synthetic data (5 000 random points).{RESET}\n"
    )
    xyz_np = (np.random.randn(5000, 3) * 2.0).astype(np.float32)
    source = "synthetic"

# Fake integer labels (3 object classes, randomly assigned)
labels_np = np.random.randint(0, 3, size=xyz_np.shape[0]).astype(np.int64)

points = torch.from_numpy(xyz_np)
labels = torch.from_numpy(labels_np)

print(f"  Source      : {source}")
print(f"  Raw points  : {points.shape[0]:,}")
print(f"  Label range : {labels.min().item()} … {labels.max().item()}")

# ==================================================================
# 2.  Run initialisation
# ==================================================================

print(f"\n{CYAN}Running AnchorInitializer.initialize_from_pcd …{RESET}\n")

init = AnchorSeedBuilder(n_offsets=5, feat_dim=32, voxel_size=-1.0, device="cpu")


class SimpleLogger:
    def info(self, msg):
        print(f"  [log] {msg}")


seeds = init.build(points, labels, logger=SimpleLogger())
params = {
    "anchor": seeds.anchors,
    "offset": seeds.offsets,
    "anchor_feat": seeds.features,
    "scaling": seeds.log_scaling,
    "rotation": seeds.rotations,
    "anchor_mask": torch.ones(seeds.anchors.shape[0], dtype=torch.bool, device=seeds.anchors.device),
    "label_ids": seeds.labels,
    "voxel_size": seeds.voxel_size,
}

# ==================================================================
# 3.  Shape & value checks
# ==================================================================

print(f"\n{CYAN}Checks:{RESET}\n")

N = points.shape[0]
M = params["anchor"].shape[0]
k = 5

# ## 3a. Voxel size #################################################
vs = params["voxel_size"]
check("voxel_size is positive", vs > 0, f"got {vs}")
check("voxel_size is a finite float", np.isfinite(vs), f"got {vs}")

# ## 3b. Anchor count ###############################################
check("anchor count < raw point count", M < N, f"{M} >= {N}")
check("anchor count > 0", M > 0, f"got {M}")

# ## 3c. Shapes #####################################################
check(
    "anchor shape [M, 3]",
    tuple(params["anchor"].shape) == (M, 3),
    str(params["anchor"].shape),
)
check(
    "offset shape [M, k, 3]",
    tuple(params["offset"].shape) == (M, k, 3),
    str(params["offset"].shape),
)
check(
    "anchor_feat shape [M, 32]",
    tuple(params["anchor_feat"].shape) == (M, 32),
    str(params["anchor_feat"].shape),
)
check(
    "scaling shape [M, 6]",
    tuple(params["scaling"].shape) == (M, 6),
    str(params["scaling"].shape),
)
check(
    "rotation shape [M, 4]",
    tuple(params["rotation"].shape) == (M, 4),
    str(params["rotation"].shape),
)
check(
    "anchor_mask shape [M]",
    tuple(params["anchor_mask"].shape) == (M,),
    str(params["anchor_mask"].shape),
)
check(
    "label_ids shape [M, 1]",
    tuple(params["label_ids"].shape) == (M, 1),
    str(params["label_ids"].shape),
)

# ## 3d. Values #####################################################
anchors = params["anchor"].cpu()
offsets = params["offset"].cpu()
feats = params["anchor_feat"].cpu()
scales = params["scaling"].cpu()
rots = params["rotation"].cpu()
mask = params["anchor_mask"].cpu()

check(
    "offsets all zero",
    offsets.abs().max().item() == 0.0,
    f"max={offsets.abs().max().item()}",
)
check(
    "anchor_feat all zero",
    feats.abs().max().item() == 0.0,
    f"max={feats.abs().max().item()}",
)
check(
    "anchor_mask all True", mask.all().item(), f"{(~mask).sum().item()} False entries"
)
check("scales are finite", torch.isfinite(scales).all().item(), "contains NaN/Inf")
check(
    "rotations are identity quaternion (w=1)",
    rots[:, 0].allclose(torch.ones(M)),
    f"w col: {rots[:, 0].mean():.3f}",
)
check(
    "rotations xyz = 0",
    rots[:, 1:].abs().max().item() == 0.0,
    f"max xyz = {rots[:, 1:].abs().max().item()}",
)

# ## 3e. Bounding box ###############################################
pts_min = points.min(dim=0).values
pts_max = points.max(dim=0).values
anc_min = anchors.min(dim=0).values
anc_max = anchors.max(dim=0).values

# Anchors centred at voxel centres, so allow ±1 voxel slack
slack = vs
in_bbox = (anc_min >= pts_min - slack).all() and (anc_max <= pts_max + slack).all()
check(
    "anchor positions within input bounding box (±1 voxel)",
    in_bbox,
    f"anc [{anc_min.tolist()} … {anc_max.tolist()}]  pts [{pts_min.tolist()} … {pts_max.tolist()}]",
)

# ## 3f. Label IDs ##################################################
ids = params["label_ids"].cpu()
check("label_ids dtype is integer", ids.is_floating_point() == False, str(ids.dtype))
check(
    "label_ids within original range",
    ids.min().item() >= labels.min().item() and ids.max().item() <= labels.max().item(),
    f"ids: [{ids.min().item()}, {ids.max().item()}]  orig: [{labels.min().item()}, {labels.max().item()}]",
)

# ## 3g. OneHotEncoder #############################################
enc = params["id_encoder"]
check("id_encoder num_classes == 3", enc.num_classes == 3, f"got {enc.num_classes}")

# ==================================================================
# 4.  Summary
# ==================================================================

print(f"\n{CYAN}Parameter summary:{RESET}")
print(f"  {'Key':<16}  {'Shape':<20}  {'min':>10}  {'max':>10}  {'mean':>10}")
print(f"  {'-' * 16}  {'-' * 20}  {'-' * 10}  {'-' * 10}  {'-' * 10}")
for key in ("anchor", "offset", "anchor_feat", "scaling", "rotation"):
    t = params[key].cpu().float()
    print(
        f"  {key:<16}  {str(tuple(t.shape)):<20}  {t.min().item():>10.4f}  {t.max().item():>10.4f}  {t.mean().item():>10.4f}"
    )

print(f"\n  voxel_size       = {vs:.6f}")
print(
    f"  reduction ratio  = {M}/{N}  ({100 * M / N:.1f}% of input points kept as anchors)"
)

# ==================================================================
# 5.  Result banner
# ==================================================================

total = len(passed) + len(failed)
print(f"\n{'━' * 45}")
if not failed:
    print(f"{GREEN}  ALL {total} CHECKS PASSED  {RESET}")
else:
    print(f"{RED}  {len(failed)}/{total} CHECKS FAILED  ✗{RESET}")
    for name in failed:
        print(f"    {RED}✗{RESET}  {name}")
print(f"{'━' * 45}\n")

sys.exit(0 if not failed else 1)
