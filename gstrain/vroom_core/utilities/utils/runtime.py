from __future__ import annotations

import os
import random
from typing import Callable

import numpy as np
import torch


def ensure_directory(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def seed_everything(seed: int = 0, quiet: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if not quiet:
        print(f"Seeded RNGs with {seed}")


def exponential_lr_schedule(
    lr_init: float,
    lr_final: float,
    lr_delay_steps: int = 0,
    lr_delay_mult: float = 1.0,
    max_steps: int = 1_000_000,
    warmup_steps: int = 0,
) -> Callable[[int], float]:
    def schedule(step: int) -> float:
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            return 0.0
        if warmup_steps > 0 and step < warmup_steps:
            alpha = step / warmup_steps
            return lr_init * (lr_delay_mult + (1.0 - lr_delay_mult) * alpha)
        if lr_delay_steps > 0:
            delay = lr_delay_mult + (1.0 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay = 1.0
        t = np.clip(step / max_steps, 0, 1)
        interpolated = np.exp(np.log(max(lr_init, 1e-20)) * (1.0 - t) + np.log(max(lr_final, 1e-20)) * t)
        if lr_init == 0.0 and lr_final == 0.0:
            return 0.0
        return float(delay * interpolated)

    return schedule


get_expon_lr_func = exponential_lr_schedule

