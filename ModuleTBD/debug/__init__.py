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
import shutil
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _reset_debug_tree(obj_dir, debug_root):
    for path in [
        debug_root,
        obj_dir / "00_scope" / "debug",
        obj_dir / "01_extraction" / "debug",
        obj_dir / "02_frame_scoring" / "debug",
        obj_dir / "03_novel_views" / "debug",
        obj_dir / "04_supervision" / "debug",
        obj_dir / "07_compare",
    ]:
        if path.exists():
            shutil.rmtree(path)
    for path in [
        obj_dir / "00_scope",
        obj_dir / "02_frame_scoring",
        obj_dir / "04_supervision",
    ]:
        if path.exists():
            try:
                path.rmdir()
            except OSError:
                pass


def generate_all_debug_artifacts(
    *,
    obj_dir,
    scope,
    frame,
    gaussians=None,
    trained_gaussians=None,
    pipe_config=None,
    images_dir=None,
    extraction_manifest=None,
    scores_manifest=None,
    halluc_manifest=None,
    training_summary=None,
    model_path=None,
    object_id=None,
):
    """Run every available debug artifact for the current pipeline state.

    Each phase is wrapped in its own try/except so that a single failure
    does not block the others.
    """
    obj_dir = Path(obj_dir)
    debug_root = obj_dir / "debug"
    _reset_debug_tree(obj_dir, debug_root)
    debug_root.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}

    # ── Scope ──────────────────────────────────────────────────────────────
    try:
        from .debug_scope import generate_debug_artifacts as _scope
        results["scope"] = _scope(
            scope=scope, frame=frame,
            gaussians=gaussians, pipe_config=pipe_config,
            debug_dir=debug_root / "scope",
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("debug_scope failed: %s", exc)

    # ── Extraction ────────────────────────────────────────────────────────
    if extraction_manifest is not None and images_dir is not None:
        try:
            from .debug_extraction import generate_debug_artifacts as _extr
            results["extraction"] = _extr(
                manifest=extraction_manifest,
                scope=scope, gaussians=gaussians, pipe_config=pipe_config,
                images_dir=Path(images_dir),
                debug_dir=debug_root / "extraction",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("debug_extraction failed: %s", exc)

    # ── Frame scoring ─────────────────────────────────────────────────────
    if scores_manifest is not None:
        try:
            from .debug_frame_scoring import generate_debug_artifacts as _fs
            results["frame_scoring"] = _fs(
                scores=scores_manifest,
                debug_dir=debug_root / "frame_scoring",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("debug_frame_scoring failed: %s", exc)

    # ── Novel views ───────────────────────────────────────────────────────
    if halluc_manifest is not None:
        try:
            from .debug_novel_views import generate_debug_artifacts as _nv
            results["novel_views"] = _nv(
                manifest=halluc_manifest,
                scope_cameras=scope.cameras if scope is not None else [],
                debug_dir=debug_root / "novel_views",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("debug_novel_views failed: %s", exc)

    # ── Before/after comparisons ──────────────────────────────────────────
    if trained_gaussians is not None:
        try:
            from .debug_compare import generate_debug_artifacts as _compare
            results["compare"] = _compare(
                scope=scope,
                frame=frame,
                parent_gaussians=gaussians,
                trained_gaussians=trained_gaussians,
                pipe_config=pipe_config,
                object_id=object_id,
                halluc_manifest=halluc_manifest,
                debug_dir=debug_root / "compare",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("debug_compare failed: %s", exc)

    # ── Projection audit ──────────────────────────────────────────────────
    try:
        from .debug_audit import generate_debug_artifacts as _audit
        audit_results = _audit(
            obj_dir=obj_dir,
            scope=scope,
            frame=frame,
            model_path=model_path,
            object_id=object_id,
        )
        results["projection_audit"] = audit_results.get("projection")
    except Exception as exc:  # noqa: BLE001
        logger.warning("debug_audit failed: %s", exc)

    return results


__all__ = ["generate_all_debug_artifacts"]
