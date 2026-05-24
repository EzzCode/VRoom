"""Configuration loader for VRoom – loads the flat JSON config directly."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple


def load_vroom_config(
    config_path: Path | str,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Load a VRoom JSON config and return its four top-level sections.

    Returns:
        cfg            – the full raw dict (saved alongside checkpoints)
        model_params   – ``cfg["experiment"]`` merged with ``cfg["model"]``
        optim_params   – ``cfg["optimization"]`` (flat, used directly by Optimizer)
        pipeline_params – ``cfg["pipeline"]``
    """
    with open(config_path, "r", encoding="utf-8") as handle:
        cfg = json.load(handle)

    model_params = {
        **cfg.get("experiment", {}),
        "model_config": {
            "name": cfg.get("model", {}).get("name", "GaussianModel"),
            "kwargs": {k: v for k, v in cfg.get("model", {}).items() if k != "name"},
        },
    }
    optim_params = cfg.get("optimization", {})
    pipeline_params = cfg.get("pipeline", {})

    return cfg, model_params, optim_params, pipeline_params
