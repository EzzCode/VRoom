"""
refine_all_objects.py  —  Centralized VRoom object refinement + mesh extraction runner.

For every object ID (auto-discovered or explicitly listed) this script will:
  1. Run object_refiner  →  <output_root>/obj_<ID>/06_model/
  2. Run extract_refined_mesh.py  →  <output_root>/obj_<ID>/07_mesh/

Usage example
-------------
python refine_all_objects.py \\
    --model_path  test_30k_prototype_output/training/gs_model/2026-06-05_14-50-07 \\
    --scene_dir   test_30k_prototype_output \\
    --iterations  1200 \\
    --object_ids  1 2 4

Flags
-----
--debug       Forward --debug to object_refiner (saves per-step debug artefacts).
--skip_refine Skip refinement and use existing <output_root>/obj_<ID>/06_model/.
--skip_mesh   Run refinement only; skip mesh extraction.
--dry_run     Print commands without executing them.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(step_name: str, cmd: list, *, conda_env: Optional[str] = None, dry_run: bool = False):
    """Run *cmd*, optionally wrapped in `conda run -n <env>`."""
    if cmd and cmd[0] == "python":
        cmd = ["python", "-u"] + cmd[1:]

    if conda_env:
        cmd = ["conda", "run", "--no-capture-output", "-n", conda_env] + cmd

    print(f"\n{'=' * 60}")
    print(f"  {step_name}")
    print(f"{'=' * 60}")
    if conda_env:
        print(f"  [env: {conda_env}]")
    print(" ".join(str(c) for c in cmd))

    if dry_run:
        print("  [DRY RUN — skipped]")
        return

    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"Step '{step_name}' failed (exit code {result.returncode})")


def _discover_object_ids(model_path: str, scene_dir: str,
                         ignore_labels: list, conda_env: str) -> List[int]:
    """
    Discover unique object label IDs via a small subprocess so it runs inside
    the target conda environment (where plyfile is guaranteed to be installed).
    """
    python_code = f"""\
import json, sys
from pathlib import Path
import numpy as np

ids = set()
ignore = {set(ignore_labels)}

# 1. Latest checkpoint anchor_cloud.ply
chkpt_dir = Path(r"{model_path}") / "checkpoints"
if chkpt_dir.exists():
    try:
        from plyfile import PlyData
        iters = sorted(
            [d for d in chkpt_dir.iterdir() if d.is_dir() and d.name.startswith("iter_")],
            key=lambda x: int(x.name.split("_")[1])
        )
        if iters:
            ply = PlyData.read(str(iters[-1] / "anchor_cloud.ply"))
            labels = ply.elements[0]["label"]
            ids.update(int(l) for l in np.unique(labels))
    except Exception:
        pass

# 2. points3D_labeled.ply in labeled_output
if not ids:
    labeled_ply = Path(r"{scene_dir}") / "labeled_output" / "points3D_labeled.ply"
    if labeled_ply.exists():
        try:
            from plyfile import PlyData
            ply = PlyData.read(str(labeled_ply))
            if "label" in [p.name for p in ply.elements[0].properties]:
                ids.update(int(l) for l in np.unique(ply.elements[0]["label"]))
        except Exception:
            pass

# 3. Exported GLB meshes
if not ids:
    for mesh_dir in [
        Path(r"{scene_dir}") / "mesh_objects" / "glb",
        Path(r"{model_path}").parent / "mesh_objects" / "glb",
    ]:
        if mesh_dir.exists():
            for f in mesh_dir.glob("object_*.glb"):
                try:
                    ids.add(int(f.stem.split("_")[1]))
                except Exception:
                    pass
            if ids:
                break

print(json.dumps(sorted(ids - ignore)))
"""
    tmp = Path("_discover_ids_tmp.py")
    tmp.write_text(python_code, encoding="utf-8")
    try:
        cmd = ["conda", "run", "--no-capture-output", "-n", conda_env,
               "python", str(tmp)]
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if r.returncode == 0:
            return json.loads(r.stdout.strip())
        print(f"[discover] subprocess error:\n{r.stderr}")
    except Exception as e:
        print(f"[discover] failed: {e}")
    finally:
        tmp.unlink(missing_ok=True)
    return []


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Centralized VRoom object refinement + mesh extraction runner.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Required ──────────────────────────────────────────────────────────────
    p.add_argument("--model_path", required=True,
                   help="Trained GS model dir (contains checkpoints/, cameras.json, config.json)")
    p.add_argument("--scene_dir", required=True,
                   help="Scene root dir (contains images/, labeled_output/, …)")

    # ── Output ────────────────────────────────────────────────────────────────
    p.add_argument("--output_root", default=None,
                   help="Root output dir for all refined objects "
                        "(default: <scene_dir>/refined_objects)")

    # ── Object selection ──────────────────────────────────────────────────────
    p.add_argument("--object_ids", nargs="+", type=int, default=None,
                   help="Explicit object IDs to process (skips auto-discovery)")
    p.add_argument("--ignore_labels", nargs="+", type=int, default=[0, 255],
                   help="Label IDs to skip during auto-discovery")

    # ── Refinement options ────────────────────────────────────────────────────
    p.add_argument("--iterations", type=int, default=1200,
                   help="Refinement training iterations per object")
    p.add_argument("--reuse_sv3d", action="store_true",
                   help="Re-use previously generated SV3D novel views")
    p.add_argument("--debug", action="store_true",
                   help="Enable object_refiner debug mode (saves per-step artefacts "
                        "and pipeline summary JSON)")
    p.add_argument("--skip_refine", action="store_true",
                   help="Skip refinement and use existing 06_model outputs")

    # ── Mesh options ──────────────────────────────────────────────────────────
    p.add_argument("--skip_mesh", action="store_true",
                   help="Skip mesh extraction after refinement")
    p.add_argument("--mesh_resolution", type=int, default=128,
                   help="TSDF grid resolution for mesh extraction")

    # ── Environment ───────────────────────────────────────────────────────────
    p.add_argument("--conda_env", default="pipeline",
                   help="Conda environment name")
    p.add_argument("--dry_run", action="store_true",
                   help="Print commands without executing them")

    args = p.parse_args()

    if args.skip_refine and args.skip_mesh:
        print("ERROR: both --skip_refine and --skip_mesh are set; nothing to do.")
        sys.exit(2)

    script_dir = Path(__file__).resolve().parent

    # ── Resolve output root ───────────────────────────────────────────────────
    output_root = Path(args.output_root) if args.output_root else \
                  Path(args.scene_dir) / "refined_objects"
    output_root = output_root.resolve()

    # ── Discover object IDs ───────────────────────────────────────────────────
    if args.object_ids is not None:
        object_ids = args.object_ids
        print(f"Using explicitly specified object IDs: {object_ids}")
    else:
        print("Auto-discovering object IDs …")
        object_ids = _discover_object_ids(
            args.model_path, args.scene_dir, args.ignore_labels, args.conda_env
        )
        if not object_ids:
            print("ERROR: No object IDs found. Use --object_ids to specify them manually.")
            sys.exit(1)
        print(f"Discovered: {object_ids}")

    # ── Resolve shared paths (computed once, reused per object) ───────────────
    # Checkpoint PLY (VRoom layout: checkpoints/iter_N/anchor_cloud.ply)
    ply_path = None
    chkpt_dir = Path(args.model_path) / "checkpoints"
    if chkpt_dir.exists():
        iters = sorted(
            [d for d in chkpt_dir.iterdir() if d.is_dir() and d.name.startswith("iter_")],
            key=lambda x: int(x.name.split("_")[1]),
        )
        if iters:
            candidate = iters[-1] / "anchor_cloud.ply"
            if candidate.exists():
                ply_path = candidate

    # Tracked id-map directory
    scene_path = Path(args.scene_dir)
    tracked_id_map_dir = None
    for cand in [
        scene_path / "labeled_output" / "tracked" / "id_maps",
        scene_path / "tracked" / "id_maps",
        scene_path / "object_mask",
    ]:
        if cand.exists() and any(cand.iterdir()):
            tracked_id_map_dir = cand
            break

    # Cameras JSON (needed for mesh extraction)
    cameras_json = Path(args.model_path) / "cameras.json"

    # ── Per-object loop ───────────────────────────────────────────────────────
    failed = []

    for obj_id in object_ids:
        print(f"\n{'#' * 60}")
        print(f"#  Object {obj_id}")
        print(f"{'#' * 60}")

        obj_model_dir = output_root / f"obj_{obj_id}" / "06_model"
        obj_mesh_dir  = output_root / f"obj_{obj_id}" / "07_mesh"

        # ── Step 1: Refinement ────────────────────────────────────────────────
        if args.skip_refine:
            print(f"  --skip_refine set, skipping refinement for object {obj_id}.")
            if not obj_model_dir.exists():
                print(f"  [ERROR] Missing refined model directory: {obj_model_dir}")
                failed.append((obj_id, "refinement (skipped but missing 06_model)"))
                continue
        else:
            refine_cmd = [
                "python", "-m", "object_refiner",
                "--model_path",  str(args.model_path),
                "--scene_dir",   str(args.scene_dir),
                "--output_root", str(output_root),
                "--object_id",   str(obj_id),
                "--iterations",  str(args.iterations),
            ]
            if ply_path is not None:
                refine_cmd += ["--ply_path", str(ply_path)]
            if tracked_id_map_dir is not None:
                refine_cmd += ["--tracked_id_map_dir", str(tracked_id_map_dir)]
            if args.reuse_sv3d:
                refine_cmd.append("--reuse_sv3d")
            if args.debug:
                refine_cmd.append("--debug")

            try:
                _run(f"Refine object {obj_id}", refine_cmd,
                     conda_env=args.conda_env, dry_run=args.dry_run)
            except RuntimeError as e:
                print(f"[ERROR] {e}")
                failed.append((obj_id, "refinement"))
                continue  # still try the next object

        # ── Step 2: Mesh extraction ───────────────────────────────────────────
        if args.skip_mesh:
            print(f"  --skip_mesh set, skipping mesh extraction for object {obj_id}.")
            continue

        if not cameras_json.exists():
            print(f"  [WARN] cameras.json not found at {cameras_json} — skipping mesh.")
            continue

        mesh_cmd = [
            "python", str(script_dir / "extract_refined_mesh.py"),
            "--refined_model_dir", str(obj_model_dir),
            "--cameras_json",      str(cameras_json),
            "--object_id",         str(obj_id),
            "--output_dir",        str(obj_mesh_dir),
            "--resolution",        str(args.mesh_resolution),
        ]

        try:
            _run(f"Extract mesh for object {obj_id}", mesh_cmd,
                 conda_env=args.conda_env, dry_run=args.dry_run)
        except RuntimeError as e:
            print(f"[ERROR] {e}")
            failed.append((obj_id, "mesh"))
            # Don't stop — continue with remaining objects

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  ALL OBJECTS PROCESSED")
    print(f"{'=' * 60}")
    if failed:
        print("  Failed steps:")
        for obj_id, step in failed:
            print(f"    object {obj_id} — {step}")
        sys.exit(1)
    else:
        print("  All objects refined and meshed successfully.")


if __name__ == "__main__":
    main()
