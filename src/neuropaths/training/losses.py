"""Loss functions.

The dissertation (eq. 10-11) trains and evaluates with the *relative*
L2 loss::

    L2(y_hat, y) = (1 / N) * sum_i || y_hat_i - y_i ||_2 / || y_i ||_2.

The legacy `FNO_1.py` trains with plain `nn.MSELoss`, while
`FNO_1_Test.py` evaluates with the relative L2. That inconsistency is
almost certainly the wrong thing; bring training and evaluation onto
the same metric.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RelativeL2Loss(nn.Module):
    """Mean per-sample relative L2 (dissertation eq. 11)."""

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """y_hat, y: [B, C, G, G] (C is usually 1 here)."""
        # TODO: flatten per-sample, take ratio of L2 norms, mean over B.
        raise NotImplementedError(
            "TODO: implement per-sample ||y_hat - y||_2 / ||y||_2, mean over batch."
        )
