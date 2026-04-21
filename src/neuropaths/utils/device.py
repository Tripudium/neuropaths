"""Device selection for CUDA / MPS / CPU.

The legacy code in `Neural_operator/FNO_1.py` hardcodes::

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

with no CUDA path, which breaks on the target Warwick Blythe GPU cluster
(see ../../../scrtp/ docs). This module centralises the choice so the CLI
entry points and tests pick up CUDA automatically and can be overridden
from a config.
"""

from __future__ import annotations

from typing import Literal

import torch

DeviceStr = Literal["auto", "cuda", "mps", "cpu"]


def get_device(preference: DeviceStr = "auto") -> torch.device:
    """Return a torch.device respecting a preference string.

    Order when preference == "auto":
        1. CUDA if available (Blythe GPU nodes, workstations with NVIDIA)
        2. MPS if available (local Apple Silicon development)
        3. CPU fallback

    Explicit choices ("cuda" / "mps" / "cpu") raise a RuntimeError if the
    requested backend is unavailable, so experiment configs fail loudly
    rather than silently dropping to CPU on a cluster submission.
    """
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
        return torch.device("cuda")

    if preference == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but torch.backends.mps.is_available() is False.")
        return torch.device("mps")

    if preference == "cpu":
        return torch.device("cpu")

    raise ValueError(f"Unknown device preference: {preference!r}")


def describe_device(device: torch.device) -> str:
    """Human-readable description for logs."""
    if device.type == "cuda":
        idx = device.index if device.index is not None else 0
        name = torch.cuda.get_device_name(idx)
        return f"cuda:{idx} ({name})"
    return str(device)
