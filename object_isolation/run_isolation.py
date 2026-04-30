"""CLI entry point for the object-isolation pipeline.

Phases (selectable via --phase):
    extract   : Phase 1   — per-object dataset assembly
    reference : Phase 2.1 — pick best reference frame
    align     : Phase 3   — build pose alignment trace from reference frame
                            (writes obj_<id>/novel_views/poses.json + pose_trace.json)
    all       : run extract -> reference -> align (stops there until later
                phases are implemented)

Phases 2.3 (Zero123++ inference), 3.5 (metric calibration), 4 (standalone
training), and 5 (reintegration) are wired in as stubs that print a TODO
message — implement them in their dedicated modules.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

_VROOM_ROOT = Path(__file__).resolve().parent.parent
if str(_VROOM_ROOT) not in sys.path:
    sys.path.insert(0, str(_VROOM_ROOT))

from object_isolation import extraction
from object_isolation import extraction_real
from object_isolation import reference_picker
from object_isolation import pose_alignment
from object_isolation import zero123_input_prep
from object_isolation import zero123_runner
from object_isolation import metric_cage as metric_cage_mod
from object_isolation import build_dataset as build_dataset_mod


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="object_isolation",
        description="Object-centric isolation pipeline (Zero123++ healing).",
    )
    p.add_argument("--model_path", required=True, help="Trained ObjectGS room model folder")
    p.add_argument("--object_id", type=int, required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument(
        "--phase",
        choices=["extract", "reference", "align", "prep", "generate", "calibrate", "dataset", "train", "reintegrate", "all"],
        default="all",
    )
    p.add_argument("--iteration", type=int, default=-1, help="-1 = latest")
    # Phase 1 knobs
    p.add_argument("--render_size", type=int, default=512)
    p.add_argument("--crop_pad_frac", type=float, default=0.15)
    p.add_argument("--full_render_scale", type=float, default=1.0)
    p.add_argument("--min_visible_pixels", type=int, default=256)
    p.add_argument("--alpha_thr", type=float, default=0.05)
    # Phase 1 — real-mask mode (recommended): use actual photos + DEVA masks
    p.add_argument("--use_real_masks", action="store_true", default=True,
                   help="Use real photos + DEVA masks instead of ObjectGS renders for Phase 1 (default ON).")
    p.add_argument("--no_real_masks", dest="use_real_masks", action="store_false",
                   help="Disable real-mask mode and fall back to ObjectGS renders.")
    p.add_argument("--scene_dir", default=None,
                   help="Scene directory containing images_all/ + object_mask/ (required when --use_real_masks).")
    p.add_argument("--images_subdir", default="images_all")
    p.add_argument("--mask_subdir", default="object_mask")
    p.add_argument("--deva_label", type=int, default=None,
                   help="DEVA mask label for the target object (auto-discovered if omitted).")
    p.add_argument("--vote_ply", default=None,
                   help="Path to vote.py's points3D_labeled.ply for robust label discovery. "
                        "If omitted, looks under <scene_dir>/vote_output/ then <scene_dir>/output/.")
    # Phase 2.1 knobs
    p.add_argument("--min_area_frac", type=float, default=0.06)
    p.add_argument("--min_complete_frac", type=float, default=0.90,
                   help="Hard filter: drop reference candidates where less than this fraction of "
                        "object anchors project inside the original image.")
    p.add_argument("--w_center", type=float, default=0.10)
    p.add_argument("--w_area", type=float, default=0.15)
    p.add_argument("--w_clip", type=float, default=0.10)
    p.add_argument("--w_front", type=float, default=0.15)
    p.add_argument("--w_light", type=float, default=0.10)
    p.add_argument("--w_complete", type=float, default=0.40,
                   help="Weight for the whole-object visibility (anchor completeness) score.")
    # Phase 3 knobs
    p.add_argument("--azimuth_sign", type=int, default=-1, choices=[-1, 1])
    p.add_argument("--elevation_sign", type=int, default=1, choices=[-1, 1])
    p.add_argument(
        "--r_v_synth",
        type=float,
        default=None,
        help="Override Zero123++ synthetic distance. Default = "
             "1.5 * object_radius (Zero123++ training convention).",
    )
    # Phase 2.2 knobs (input prep)
    p.add_argument("--canvas_size", type=int, default=320, help="Zero123++ input canvas side")
    p.add_argument("--margin_frac", type=float, default=0.10, help="White margin around object on canvas")
    # Phase 2.3 knobs (Zero123++ runner)
    p.add_argument("--z123_backend", default="plus_v12", choices=["plus_v12"])
    p.add_argument("--z123_steps", type=int, default=75)
    p.add_argument("--z123_guidance", type=float, default=4.0)
    p.add_argument("--z123_seed", type=int, default=42)
    p.add_argument("--z123_device", default="cuda")
    p.add_argument("--z123_dtype", default="float16", choices=["float16", "float32", "bfloat16"])
    p.add_argument("--z123_white_thr", type=int, default=12,
                   help="Chroma-key tolerance against auto-detected bg colour (Zero123++ v1.2 outputs ~grey).")
    # Phase 3.5 knobs (metric cage / DBSCAN cleanup)
    p.add_argument("--cage_eps", type=float, default=None,
                   help="DBSCAN eps for anchor cleanup; auto-estimated if omitted.")
    p.add_argument("--cage_min_samples", type=int, default=8)
    p.add_argument("--cage_radius_percentile", type=float, default=99.0)
    # Phase 4 knobs (dataset assembly)
    p.add_argument("--dataset_subdir", default="dataset",
                   help="Subdirectory inside obj_<id>/ where dataset is written.")
    p.add_argument("--dataset_test_every", type=int, default=8,
                   help="Hold out every Nth real view as test (0 disables).")
    p.add_argument("--dataset_no_masks", action="store_true",
                   help="Skip writing masks/ alongside images/.")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main() -> int:
    args = _build_argparser().parse_args()
    _setup_logging(args.verbose)

    out_root = Path(args.output_dir) / f"obj_{int(args.object_id)}"
    out_root.mkdir(parents=True, exist_ok=True)

    do_extract = args.phase in ("extract", "all")
    do_reference = args.phase in ("reference", "all")
    do_align = args.phase in ("align", "all")
    do_prep = args.phase in ("prep", "all")
    do_generate = args.phase in ("generate", "all")
    do_calibrate = args.phase in ("calibrate", "all")
    do_dataset = args.phase in ("dataset", "all")

    if do_extract:
        if args.use_real_masks:
            if not args.scene_dir:
                raise SystemExit(
                    "--scene_dir is required when --use_real_masks is on "
                    "(point at e.g. data/replica/office_0). Use --no_real_masks to fall back to ObjectGS renders."
                )
            extraction_real.extract_from_real_masks(
                model_path=args.model_path,
                object_id=int(args.object_id),
                output_dir=args.output_dir,
                scene_dir=args.scene_dir,
                images_subdir=args.images_subdir,
                mask_subdir=args.mask_subdir,
                deva_label=args.deva_label,
                vote_ply=args.vote_ply,
                render_size=args.render_size,
                crop_pad_frac=args.crop_pad_frac,
                min_visible_pixels=args.min_visible_pixels,
                alpha_thr=args.alpha_thr,
                iteration=args.iteration,
            )
        else:
            extraction.extract(
                model_path=args.model_path,
                object_id=int(args.object_id),
                output_dir=args.output_dir,
                render_size=args.render_size,
                crop_pad_frac=args.crop_pad_frac,
                full_render_scale=args.full_render_scale,
                min_visible_pixels=args.min_visible_pixels,
                alpha_thr=args.alpha_thr,
                iteration=args.iteration,
            )

    if do_reference:
        weights = reference_picker.ScoreWeights(
            center=args.w_center, area=args.w_area, clip=args.w_clip,
            front=args.w_front, light=args.w_light, complete=args.w_complete,
        )
        reference_picker.pick_reference(
            obj_dir=str(out_root),
            weights=weights,
            min_area_frac=args.min_area_frac,
            min_complete_frac=args.min_complete_frac,
        )

    if do_align:
        _run_align_phase(out_root, args)

    if do_prep:
        zero123_input_prep.prepare_zero123_input(
            obj_dir=str(out_root),
            canvas_size=args.canvas_size,
            margin_frac=args.margin_frac,
            alpha_thr=args.alpha_thr,
        )

    if do_generate:
        zero123_runner.generate_novel_views(
            obj_dir=str(out_root),
            backend=args.z123_backend,
            num_inference_steps=args.z123_steps,
            guidance_scale=args.z123_guidance,
            seed=args.z123_seed,
            device=args.z123_device,
            dtype=args.z123_dtype,
            white_thr=args.z123_white_thr,
        )

    if do_calibrate:
        metric_cage_mod.build_metric_cage(
            obj_dir=str(out_root),
            eps=args.cage_eps,
            min_samples=args.cage_min_samples,
            radius_percentile=args.cage_radius_percentile,
        )

    if do_dataset:
        build_dataset_mod.build_dataset(
            obj_dir=str(out_root),
            out_subdir=args.dataset_subdir,
            test_every=args.dataset_test_every,
            keep_alpha_as_mask=not args.dataset_no_masks,
        )

    if args.phase in ("train", "reintegrate"):
        print(
            f"[object_isolation] Phase '{args.phase}' is not implemented yet.\n"
            f"  TODO: wire up the corresponding module under object_isolation/."
        )
        return 0

    return 0


def _run_align_phase(out_root: Path, args) -> None:
    """Phase 3 wrapper: load reference + extraction summaries and build the
    full pose trace (W -> O -> V) for all 6 Zero123++ tiles. Persists
    ``novel_views/poses.json`` and ``novel_views/pose_trace.json``."""
    ref_path = out_root / "reference.json"
    summary_path = out_root / "extraction_summary.json"
    if not ref_path.exists():
        raise FileNotFoundError(
            f"reference.json missing at {ref_path}; run --phase reference first."
        )
    with open(ref_path, "r", encoding="utf-8") as f:
        ref = json.load(f)
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    obj_frame = summary["object_frame"]
    selected = ref["selected"]

    trace = pose_alignment.build_full_pose_chain(
        object_center=obj_frame["object_center"],
        object_up_world=obj_frame["object_up_world"],
        object_radius=obj_frame["object_radius"],
        reference_R_w2c=selected["R_w2c"],
        reference_T_w2c=selected["T_w2c"],
        tile_schedule=pose_alignment.zero123plus_v12_tile_schedule(),
        azimuth_sign=int(args.azimuth_sign),
        elevation_sign=int(args.elevation_sign),
        r_v_synth=args.r_v_synth,
    )

    novel_dir = out_root / "novel_views"
    novel_dir.mkdir(parents=True, exist_ok=True)

    # Compact per-tile pose record + native Zero123++ K (one K per tile, but
    # they're all identical for now since image_size is fixed)
    K_v = pose_alignment.zero123_native_K(
        image_size=int(summary["render_size"]), fov_deg=49.1,
    )
    poses = []
    for t in trace.tiles:
        poses.append({
            "tile_az_deg": t["tile_az_deg"],
            "tile_el_deg": t["tile_el_deg"],
            "R_w2c": t["R_w2c"],
            "T_w2c": t["T_w2c"],
            "K": K_v.tolist(),
            "width": int(summary["render_size"]),
            "height": int(summary["render_size"]),
        })
    with open(novel_dir / "poses.json", "w", encoding="utf-8") as f:
        json.dump(poses, f, indent=2)
    with open(novel_dir / "pose_trace.json", "w", encoding="utf-8") as f:
        json.dump(trace.to_dict(), f, indent=2)


if __name__ == "__main__":
    sys.exit(main())
