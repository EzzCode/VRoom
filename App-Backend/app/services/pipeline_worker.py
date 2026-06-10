"""
Background worker that executes ``full_pipeline_runner.py`` as a subprocess.

The function acquires the global GPU lock before running so that at most
one pipeline is active on the GPU at any time.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import List

from app.config import settings
from app.models.job import JobStatus
from app.services import s3 as s3_service
from app.services.job_store import JobStore

logger = logging.getLogger(__name__)


def _build_cli_args(job: "JobStore", job_id: str) -> List[str]:
    """Translate validated Pydantic params into CLI arguments for the runner."""
    entry = job.get_job(job_id)
    if entry is None:
        raise ValueError(f"Job {job_id} not found")

    params = entry.params
    work_dir = entry.work_dir

    import os
    is_modal = os.environ.get("RUNNING_IN_MODAL") == "1"

    args: List[str] = [
        "python",
        "-u",
        str(settings.vroom_project_root / "full_pipeline_runner.py"),
        "--data_path",
        str(work_dir),
        "--out_base_dir",
        str(work_dir / "output"),
        # Conda environments
        "--pipeline_env",
        settings.pipeline_conda_env,
        "--masks_env",
        settings.masks_conda_env,
    ]
    if is_modal:
        args.extend(["--sam_ckpt", "/app/jobs_data/sam3.pt"])

    # COLMAP
    args.extend(["--camera_model", params.colmap.camera_model])
    args.extend(["--matcher_type", params.colmap.matcher_type])
    if params.colmap.force_colmap:
        args.append("--force_colmap")

    # Masks & Tracking
    # If the caller supplied a top-level sam_prompt, parse it into individual
    # prompts and override the nested defaults.  Ignore empty / whitespace.
    text_prompts = params.masks_tracking.text_prompts  # defaults
    if params.sam_prompt and params.sam_prompt.strip():
        parsed = [p.strip() for p in params.sam_prompt.split(",") if p.strip()]
        if parsed:
            text_prompts = parsed
    args.extend(["--text_prompts"] + text_prompts)
    args.extend(["--min_mask_area", str(params.masks_tracking.min_mask_area)])
    args.extend(["--max_area_ratio", str(params.masks_tracking.max_area_ratio)])
    args.extend(["--border_threshold", str(params.masks_tracking.border_threshold)])
    args.extend(["--merge_thresh", str(params.masks_tracking.merge_thresh)])
    args.extend(["--proximity_gap", str(params.masks_tracking.proximity_gap)])
    args.extend(
        [
            "--proximity_color_thresh",
            str(params.masks_tracking.proximity_color_thresh),
        ]
    )
    if params.masks_tracking.no_split_disconnected:
        args.append("--no_split_disconnected")

    # Tracker weights
    args.extend(["--iou_w", str(params.masks_tracking.iou_w)])
    args.extend(["--color_w", str(params.masks_tracking.color_w)])
    args.extend(["--texture_w", str(params.masks_tracking.texture_w)])
    args.extend(["--bbox_w", str(params.masks_tracking.bbox_w)])
    args.extend(["--match_threshold", str(params.masks_tracking.match_threshold)])
    args.extend(["--patience", str(params.masks_tracking.patience)])
    args.extend(
        ["--smoothing_factor", str(params.masks_tracking.smoothing_factor)]
    )
    args.extend(["--reid_threshold", str(params.masks_tracking.reid_threshold)])
    if params.masks_tracking.disable_motion_comp:
        args.append("--disable_motion_comp")
    args.extend(
        ["--consensus_window", str(params.masks_tracking.consensus_window)]
    )
    args.extend(
        [
            "--consensus_tie_margin",
            str(params.masks_tracking.consensus_tie_margin),
        ]
    )
    if params.masks_tracking.use_opencv:
        args.append("--use_opencv")

    # Voting
    args.extend(["--min_points", str(params.masks_tracking.min_points)])
    if params.masks_tracking.disable_alias_merge:
        args.append("--disable_alias_merge")
    args.extend(
        ["--alias_iou_thresh", str(params.masks_tracking.alias_iou_thresh)]
    )
    args.extend(
        [
            "--alias_min_covisibility",
            str(params.masks_tracking.alias_min_covisibility),
        ]
    )

    # Training
    if params.training.small_run:
        args.append("--small_run")
    if params.training.num_iterations is not None:
        args.extend(["--num_iterations", str(params.training.num_iterations)])

    # Skip stages
    if params.skip.skip_colmap:
        args.append("--skip_colmap")
    if params.skip.skip_masks:
        args.append("--skip_masks")
    if params.skip.skip_tracking:
        args.append("--skip_tracking")
    if params.skip.skip_voting:
        args.append("--skip_voting")
    if params.skip.skip_training:
        args.append("--skip_training")
    if params.skip.skip_mesh_gen:
        args.append("--skip_mesh_gen")

    return args


def _zip_directory(source_dir: Path, zip_path: Path) -> None:
    """Create a zip archive of *source_dir* at *zip_path*."""
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in source_dir.rglob("*"):
            if file_path.is_file():
                zf.write(file_path, file_path.relative_to(source_dir))


def _upload_results(job_id: str, work_dir: Path) -> None:
    """Upload pipeline outputs to S3 (if enabled)."""
    if not settings.s3_enabled:
        logger.info("S3 disabled — results remain on local disk at %s", work_dir)
        return

    output_dir = work_dir / "output"
    s3_prefix = f"jobs/{job_id}"

    # Upload the splat .ply (latest checkpoint)
    training_dir = output_dir / "training" / "gs_model"
    if training_dir.exists():
        s3_service.upload_directory_to_s3(training_dir, f"{s3_prefix}/training")

    # Upload individual mesh files
    mesh_dir = output_dir / "mesh_objects"
    if mesh_dir.exists():
        s3_service.upload_directory_to_s3(mesh_dir, f"{s3_prefix}/meshes")

        # Create and upload bulk zip
        zip_path = output_dir / "meshes.zip"
        _zip_directory(mesh_dir, zip_path)
        s3_service.upload_file_to_s3(zip_path, f"{s3_prefix}/meshes.zip")


async def run_pipeline(
    job_id: str,
    job_store: JobStore,
    gpu_lock: asyncio.Lock,
) -> None:
    """Acquire the GPU lock, run the pipeline subprocess, upload results."""

    entry = job_store.get_job(job_id)
    if entry is None:
        logger.error("Job %s not found — aborting worker.", job_id)
        return

    import os
    is_modal = os.environ.get("RUNNING_IN_MODAL") == "1"

    if is_modal:
        # We don't acquire the local GPU lock for Modal execution because Modal manages containers
        # We immediately spawn the detached background function!
        job_store.update_job(job_id, status=JobStatus.PROCESSING, current_stage="initializing")
        cli_args = _build_cli_args(job_store, job_id)
        from modal_app import orchestrate_pipeline_modal
        logger.info("Spawning detached Modal background task for job %s", job_id)
        orchestrate_pipeline_modal.spawn(job_id, cli_args, str(entry.work_dir))
        return

    # Local execution fallback
    async with gpu_lock:
        job_store.update_job(job_id, status=JobStatus.PROCESSING, current_stage="initializing")

        try:
            cli_args = _build_cli_args(job_store, job_id)
            log_file = entry.work_dir / "pipeline.log"

            logger.info("Starting pipeline for job %s: %s", job_id, " ".join(cli_args))
            job_store.update_job(job_id, current_stage="running_pipeline")

            def _run_subprocess() -> int:
                env = os.environ.copy()
                env["PYTHONPATH"] = str(settings.vroom_project_root)
                with open(log_file, "w") as lf:
                    result = subprocess.run(
                        cli_args,
                        stdout=lf,
                        stderr=subprocess.STDOUT,
                        cwd=str(settings.vroom_project_root),
                        env=env,
                    )
                return result.returncode

            returncode = await asyncio.get_event_loop().run_in_executor(
                None, _run_subprocess
            )

            if returncode != 0:
                error_tail = ""
                if log_file.exists():
                    try:
                        error_tail = log_file.read_text(errors="replace")[-5000:]
                    except Exception:
                        pass
                raise RuntimeError(
                    f"Pipeline exited with code {returncode}. Tail: {error_tail}"
                )

            job_store.update_job(job_id, current_stage="uploading_results")
            _upload_results(job_id, entry.work_dir)

            job_store.update_job(job_id, status=JobStatus.COMPLETED, current_stage="done")
            logger.info("Job %s completed successfully.", job_id)

        except Exception as exc:
            logger.exception("Job %s failed.", job_id)
            job_store.update_job(
                job_id,
                status=JobStatus.FAILED,
                current_stage="failed",
                error_message=str(exc)[:1000],
            )
        finally:
            images_dir = entry.work_dir / "images"
            if entry.status in (JobStatus.COMPLETED, JobStatus.FAILED):
                if images_dir.exists():
                    shutil.rmtree(images_dir, ignore_errors=True)
                logger.info("Cleaned up images directory %s to conserve space.", images_dir)


def run_pipeline_modal_logic(job_id: str, cli_args: list[str], work_dir: str):
    """
    This function runs completely detached in the Modal background.
    It downloads SAM3 weights, spawns the GPU pipeline, reloads the
    shared volume to read outputs, uploads results to S3, and updates
    job status throughout.
    """
    from modal_app import run_pipeline_on_gpu, jobs_volume
    import os
    from huggingface_hub import hf_hub_download

    job_store = JobStore()

    try:
        # 1. Download SAM3 Weights (if not already on the volume)
        sam_ckpt_path = Path("/app/jobs_data/sam3.pt")
        if not sam_ckpt_path.exists():
            job_store.update_job(job_id, current_stage="queued")
            logger.info("Downloading SAM3 from HuggingFace to %s...", sam_ckpt_path)
            hf_hub_download(
                repo_id="facebook/sam3",
                filename="sam3.pt",
                local_dir=str(sam_ckpt_path.parent),
                token=os.environ.get("HF_TOKEN"),
            )
            logger.info("SAM3 download complete. Committing volume...")
            jobs_volume.commit()

        # 2. Commit volume so GPU container can see S3 downloads + SAM3 weights
        logger.info("Committing Modal Volume to sync S3 downloads and SAM3 weights...")
        jobs_volume.commit()

        # 3. Run GPU Pipeline
        job_store.update_job(job_id, current_stage="queued")
        logger.info("Executing pipeline on Modal A10G GPU for job %s", job_id)
        result = run_pipeline_on_gpu.remote(job_id, cli_args, work_dir)

        # 4. Reload volume to see GPU outputs (GPU container committed before returning)
        logger.info("Reloading Modal Volume to sync GPU outputs...")
        jobs_volume.reload()

        # 5. Check GPU result
        if result.get("returncode", 1) != 0:
            log_tail = result.get("log_tail", "")
            raise RuntimeError(
                f"Pipeline exited with code {result['returncode']}. Tail: {log_tail}"
            )

        # 6. Upload Results to S3
        job_store.update_job(job_id, current_stage="mesh-extraction")
        work_dir_path = Path(work_dir)
        _upload_results(job_id, work_dir_path)

        # 7. Mark Done
        job_store.update_job(job_id, status=JobStatus.COMPLETED, current_stage="done")
        logger.info("Modal pipeline completed successfully for job %s", job_id)

    except Exception as e:
        logger.exception("Modal pipeline execution failed for job %s", job_id)
        job_store.update_job(
            job_id,
            status=JobStatus.FAILED,
            current_stage="failed",
            error_message=str(e)[:1000],
        )
    finally:
        entry = job_store.get_job(job_id)
        work_dir_path = Path(work_dir)
        images_dir = work_dir_path / "images"
        if entry and entry.status in (JobStatus.COMPLETED, JobStatus.FAILED):
            if images_dir.exists():
                shutil.rmtree(images_dir, ignore_errors=True)
                logger.info("Cleaned up images directory %s to conserve space.", images_dir)
        jobs_volume.commit()

