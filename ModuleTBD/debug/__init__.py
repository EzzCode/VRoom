"""ModuleTBD debug-artifact subpackage.

Mirrors ``object_isolation.debug`` but is adapted to ModuleTBD's API
(``ObjectFrame`` instead of ``WorldLocal``/``LocalSV3D``, ``azimuth_deg``
field names, no ``scope.*_W`` suffixes, etc.).

Nothing here is needed for a production training run. Importing this
package or calling any of its functions only writes files when the user
explicitly opts in via ``--debug`` (or invokes one of the standalone
``python -m ModuleTBD.debug.<sub>`` CLIs).
"""
from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def generate_all_debug_artifacts(
    *,
    obj_dir,
    scope,
    frame,
    gaussians=None,
    pipe_config=None,
    images_dir=None,
    extraction_manifest=None,
    scores_manifest=None,
    halluc_manifest=None,
    training_summary=None,
    model_path=None,
    object_id=None,
    n_compare_views=8,
    do_compare_renders=True,
):
    """Run every available debug artifact for the current pipeline state.

    Each phase is wrapped in its own try/except so that a single failure
    does not block the others.
    """
    obj_dir = Path(obj_dir)
    obj_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    # ── Scope (00_scope_debug) ────────────────────────────────────────────
    try:
        from .debug_scope import generate_debug_artifacts as _scope
        results["scope"] = _scope(
            scope=scope, frame=frame,
            gaussians=gaussians, pipe_config=pipe_config,
            debug_dir=obj_dir / "00_scope" / "debug",
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("debug_scope failed: %s", exc)

    # ── Extraction (01_extraction_debug) ──────────────────────────────────
    if extraction_manifest is not None and images_dir is not None:
        try:
            from .debug_extraction import generate_debug_artifacts as _extr
            results["extraction"] = _extr(
                manifest=extraction_manifest,
                scope=scope, gaussians=gaussians, pipe_config=pipe_config,
                images_dir=Path(images_dir),
                debug_dir=obj_dir / "01_extraction" / "debug",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("debug_extraction failed: %s", exc)

    # ── Frame scoring (02_frame_scoring_debug) ────────────────────────────
    if scores_manifest is not None:
        try:
            from .debug_frame_scoring import generate_debug_artifacts as _fs
            results["frame_scoring"] = _fs(
                scores=scores_manifest,
                debug_dir=obj_dir / "02_frame_scoring" / "debug",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("debug_frame_scoring failed: %s", exc)

    # ── Novel views (03_novel_views_debug) ────────────────────────────────
    if halluc_manifest is not None:
        try:
            from .debug_novel_views import generate_debug_artifacts as _nv
            results["novel_views"] = _nv(
                manifest=halluc_manifest,
                scope_cameras=scope.cameras if scope is not None else [],
                debug_dir=obj_dir / "03_novel_views" / "debug",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("debug_novel_views failed: %s", exc)

    # ── Supervision & training (04_supervision_debug) ─────────────────────
    try:
        from .debug_supervision import generate_debug_artifacts as _sv
        results["supervision"] = _sv(
            obj_dir=obj_dir,
            training_summary=training_summary,
            debug_dir=obj_dir / "04_supervision" / "debug",
            scope=scope, frame=frame,
            gaussians=gaussians, pipe_config=pipe_config,
            n_compare_views=n_compare_views,
            do_compare_renders=bool(do_compare_renders),
            object_id=object_id,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("debug_supervision failed: %s", exc)

    # ── Audit (05_supervision_audit + 06_projection_audit) ───────────────
    try:
        from .debug_audit import generate_debug_artifacts as _audit
        audit_results = _audit(
            obj_dir=obj_dir,
            scope=scope,
            frame=frame,
            model_path=model_path,
            object_id=object_id,
        )
        results["supervision_audit"] = audit_results.get("supervision")
        results["projection_audit"] = audit_results.get("projection")
    except Exception as exc:  # noqa: BLE001
        logger.warning("debug_audit failed: %s", exc)

    return results


__all__ = ["generate_all_debug_artifacts"]
