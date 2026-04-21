"""Random trigonometric-polynomial boundary parametrisations.

Dissertation Algorithm 1 (section 4.1):

    phi(y) = (sum_k c_{1,k} sin(k*pi*y)) / phi_norm
    psi(y) = (sum_k c_{2,k} sin(k*pi*y)) / psi_norm + 1

with N_1, N_2 drawn uniformly from [1, N_max=8], c_{i,k} ~ U[-1,1], and
normalising constants phi_norm = 3*max|phi_temp| + mean|phi_temp|.
Regenerate if min_y |phi(y) - psi(y)| < epsilon to avoid degenerate
near-touching domains.

Constraints:
    phi(0) = phi(1) = 0, psi(0) = psi(1) = 1 (required by the top/
    bottom periodic BC; the sin-basis automatically enforces these).

Returned callables operate on ``autograd.numpy`` arrays because the FD
solvers need second derivatives of phi, psi via ``autograd.elementwise_grad``.

TODO: port from Neural_operator/Trig_polynomial_boundary.py
``generate_trig_functions``. The legacy file has the hardcoded overrides
``N1 = 3; N2 = 3`` which kill the N_max=8 sweep described in the
dissertation -- drop those when porting and expose `n_max` as a kwarg.
"""

from __future__ import annotations

from typing import Callable

import autograd.numpy as anp  # noqa: F401 -- used via type; see above

BoundaryFn = Callable[["anp.ndarray"], "anp.ndarray"]


def identity_boundaries() -> tuple[BoundaryFn, BoundaryFn]:
    """Return phi=0, psi=1 (square domain, no coordinate transform).

    Used by the "square" experiment family in the dissertation.
    """

    def phi(y):
        # Lambda in legacy code: `phi = lambda y: 0.0`. Promote to a
        # function so autograd's elementwise_grad handles vectorised y.
        return 0.0 * y

    def psi(y):
        return 0.0 * y + 1.0

    return phi, psi


def random_trig_boundaries(
    *,
    n_max: int = 8,
    eps: float = 0.1,
    rng: "anp.random.Generator | None" = None,
    max_retries: int = 20,
) -> tuple[BoundaryFn, BoundaryFn]:
    """Sample (phi, psi) via Algorithm 1.

    TODO: port from Neural_operator/Trig_polynomial_boundary.py.
    """
    raise NotImplementedError(
        "TODO: port from Neural_operator/Trig_polynomial_boundary.py "
        "generate_trig_functions (dissertation Algorithm 1)"
    )
