"""FNO2D builder.

Thin wrapper around ``neuralop.models.FNO`` that takes a ModelConfig
and returns a configured 2D FNO. We delegate the architecture itself
to neuraloperator so that:

    * spectral convolutions use complex weights of shape
      (in, out, modes, modes) and rfftn (the legacy `Neural_operator/
      FNO_1.py` used a single real weight of shape (1, out, m, m), which
      is part of why its relative L2 plateaued around 0.4-0.5);
    * the per-block W_l skip connection is included by default;
    * the (x, y) grid positional embedding is regenerated at the
      evaluation resolution, which is what makes the dissertation's
      zero-shot super-resolution study (16 -> 32 -> 63) well-defined.

Translation from ModelConfig to neuralop's FNO:

    modes              -> n_modes = (modes, modes)
    width              -> hidden_channels
    depth              -> n_layers
    projection_hidden  -> projection_channel_ratio = projection_hidden / width
    in_channels        -> in_channels (data-only; grid embedding adds 2 inside)
    out_channels       -> out_channels
    activation         -> non_linearity (F.gelu / F.relu)
"""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from neuralop.models import FNO

from neuropaths.config import ModelConfig

_ACTIVATIONS = {"gelu": F.gelu, "relu": F.relu}


def build_fno(cfg: ModelConfig) -> nn.Module:
    """Build a 2D FNO from `cfg`."""
    if cfg.activation not in _ACTIVATIONS:
        raise ValueError(f"Unsupported activation: {cfg.activation!r}")
    return FNO(
        n_modes=(cfg.modes, cfg.modes),
        in_channels=cfg.in_channels,
        out_channels=cfg.out_channels,
        hidden_channels=cfg.width,
        n_layers=cfg.depth,
        projection_channel_ratio=cfg.projection_hidden / cfg.width,
        positional_embedding="grid",
        non_linearity=_ACTIVATIONS[cfg.activation],
        domain_padding=cfg.domain_padding,
    )


# Keep the historical name as an alias so callers (CLI, eval) don't churn.
FNO2D = build_fno
