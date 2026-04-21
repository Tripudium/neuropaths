"""Reproducibility: seed numpy, torch, and Python's random module.

TODO: decide whether to also enforce deterministic cuBLAS / cuDNN kernels
(``torch.use_deterministic_algorithms(True)``). This can harm throughput
on Blythe so we leave it opt-in via a config flag.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, *, deterministic: bool = False) -> None:
    """Seed Python, numpy, torch (CPU + CUDA + MPS)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # TODO: consider torch.use_deterministic_algorithms(True, warn_only=True)
