"""2D Fourier Neural Operator.

Architecture (Li et al. 2021; dissertation eq. Psi_FNO):

    u -> R(u) -> L_1 -> L_2 -> ... -> L_L -> Q -> output,
    L_l(v)(x) = sigma( W_l v(x) + b_l + K(v)(x; gamma_l) ).

R is a pointwise lift (in_channels -> width), Q a pointwise projection
(width -> projection_hidden -> out_channels), and each K is an FFT-based
spectral convolution that keeps only the first `modes` x `modes` Fourier
coefficients (the truncation that gives discretisation invariance).

Dissertation uses: modes=12, width=128 for the square experiment, and
modes=16, width=256 for the curved experiment; both with L=4 and GELU.

Notes on the legacy FourierLayer2D:

    * It uses `torch.fft.fftn` with a *single* real-valued weight
      `torch.randn(1, out_channels, modes1, modes2)`. The original
      Li et al. FNO uses *complex* weights of shape
      `(in_channels, out_channels, modes1, modes2)` and `rfftn`;
      the real-weighted / fftn version is a (probably unintended)
      simplification. Flag this when porting -- it may explain part
      of the 40-50% relative L2 error ceiling.
    * The skip path `W_l v(x) + b_l` is absent in the legacy code;
      the layer reduces to `sigma(K(v))`. Add it back.

TODO:
    * Port FNO2D + FourierLayer2D from Neural_operator/FNO_1.py.
    * Fix the Fourier-weight dtype + shape (see note above).
    * Add the pointwise W_l skip connection inside each Fourier block.
    * Optional: add `SpectralConv2d` alias matching the upstream
      neuraloperator library naming so external users recognise it.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from neuropaths.config import ModelConfig


class FourierLayer2D(nn.Module):
    """Single spectral convolution block."""

    def __init__(self, in_channels: int, out_channels: int, modes: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        raise NotImplementedError(
            "TODO: port from Neural_operator/FNO_1.py FourierLayer2D. "
            "Use complex weights of shape (in, out, modes, modes) and "
            "rfftn along (-2, -1); add the pointwise W_l skip."
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class FNO2D(nn.Module):
    """Full FNO2D as specified by the ModelConfig."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        # Lifting R: pointwise linear in_channels -> width.
        # Fourier blocks L_1 ... L_{cfg.depth}.
        # Projection Q: width -> projection_hidden -> out_channels.
        raise NotImplementedError(
            "TODO: port from Neural_operator/FNO_1.py FNO2D, driven by "
            "`cfg` rather than positional ints."
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C_in, G, G] -> [B, C_out, G, G]."""
        raise NotImplementedError
