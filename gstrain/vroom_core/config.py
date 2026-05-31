from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple


def load_vroom_config(
    config_path: Path | str,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Load a VRoom JSON config and return its four top-level sections
    """
    with open(config_path, "r", encoding="utf-8") as handle:
        cfg = json.load(handle)

    experiment = cfg.setdefault("experiment", {})
    model = cfg.setdefault("model", {})

    model_params = {
        **experiment,
        "model_config": {
            "name": model.get("name", "GaussianModel"),
            "kwargs": {k: v for k, v in model.items() if k != "name"},
        },
    }
    optim_params = cfg.get("optimization", {})
    pipeline_params = cfg.get("pipeline", {})

    return cfg, model_params, optim_params, pipeline_params
