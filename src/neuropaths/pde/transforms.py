"""Coordinate transform Omega <-> [0,1]^2 and its autograd derivatives.

From dissertation eq. 30:

    f(x, y) = (x - phi(y)) / (psi(y) - phi(y)),
    f^{-1}(hat_x, y) = phi(y) + hat_x * (psi(y) - phi(y)).

Theorem 4.1 transforms the steady advection-diffusion PDE

    Delta q + b . grad q = 0

into the following PDE on [0,1]^2 (eq. 33; note `f_x = 1`, `f_{xx} = 0`,
`f_{xy} = 0` eliminate several cross-terms):

    Delta u + (f_y)^2 d_{hat_x hat_x} u + 2 f_y d_{hat_x y} u
           + (f_{yy} + b_1 + b_2 f_y) d_{hat_x} u + d_y u = 0.

The FD solvers need f_y and f_{yy} evaluated on the grid. Both come from
``autograd.elementwise_grad`` acting on the (autograd-numpy) callables
phi, psi. ``autograd`` means the HIPS Autograd package, NOT torch.autograd;
torch tensors passed through here silently fail.
"""

from __future__ import annotations

from typing import Callable

import autograd.numpy as anp
from autograd import elementwise_grad

BoundaryFn = Callable[[anp.ndarray], anp.ndarray]


def forward_map(phi: BoundaryFn, psi: BoundaryFn) -> Callable[..., anp.ndarray]:
    """Return f(x, y) = (x - phi(y)) / (psi(y) - phi(y))."""

    def f(x, y):
        return (x - phi(y)) / (psi(y) - phi(y))

    return f


def inverse_map(phi: BoundaryFn, psi: BoundaryFn) -> Callable[..., anp.ndarray]:
    """Return f^{-1}(hat_x, y) = phi(y) + hat_x * (psi(y) - phi(y))."""

    def f_inv(hat_x, y):
        return phi(y) + hat_x * (psi(y) - phi(y))

    return f_inv


def coord_transform_derivatives(
    phi: BoundaryFn,
    psi: BoundaryFn,
    X: anp.ndarray,
    Y: anp.ndarray,
) -> tuple[anp.ndarray, anp.ndarray]:
    """Return (f_y, f_{yy}) evaluated at each point of the (X, Y) meshgrid.

    Uses ``autograd.elementwise_grad`` on a flattened (X, Y) pair and
    reshapes back. X and Y must be autograd-numpy arrays of identical
    shape.

    Notes on identity boundaries
    ----------------------------
    For phi=0, psi=1 the map reduces to f(x,y) = x so f_y = f_yy = 0.
    autograd handles this cleanly: the returned arrays are exact zeros
    and the caller's mixed-derivative stencil terms vanish, which is
    exactly what the transformed PDE reduces to (plain Delta q on the
    square).
    """
    f = forward_map(phi, psi)

    df_dy = elementwise_grad(f, argnum=1)
    df_dy2 = elementwise_grad(lambda x, y: df_dy(x, y), argnum=1)

    X_flat = X.flatten()
    Y_flat = Y.flatten()

    f_y_flat = df_dy(X_flat, Y_flat)
    f_yy_flat = df_dy2(X_flat, Y_flat)

    return f_y_flat.reshape(X.shape), f_yy_flat.reshape(X.shape)
