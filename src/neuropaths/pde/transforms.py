"""Coordinate transform Omega <-> [0,1]^2 and its autograd derivatives.

From dissertation eq. 30:

    f(x, y) = (x - phi(y)) / (psi(y) - phi(y)),
    f^{-1}(hat_x, y) = phi(y) + hat_x * (psi(y) - phi(y)).

Theorem 4.1 shows that the steady advection-diffusion PDE

    Delta q + b . grad q = 0

becomes (eq. 33):

    Delta u + (f_y)^2 d_{hat_x hat_x} u + 2 f_y d_{hat_x y} u
           + (f_{yy} + b_1 + b_2 f_y) d_{hat_x} u + d_y u = 0

on [0,1]^2, exploiting f_x = 1 and f_{xx} = 0.

The FD solvers rely on `f_y` and `f_{yy}` evaluated on the grid; both
are obtained by ``autograd.elementwise_grad``. This module will expose
a single ``coord_transform_derivatives(phi, psi, X, Y)`` helper so the
forward and backward solvers stop duplicating it.

TODO: extract from Neural_operator/Finite_difference_forward_comittor.py
and .../Finite_difference_backwards_committor.py (the private
``f_derivatives`` function is literally identical in both files).
"""

from __future__ import annotations

from typing import Callable

import autograd.numpy as anp

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
    """Return (f_y, f_{yy}) evaluated at each point of the meshgrid.

    TODO: port from the private ``f_derivatives`` in the FD solvers.
    Use ``autograd.elementwise_grad`` on a flattened (X, Y) pair and
    reshape back; do NOT feed torch tensors through.
    """
    raise NotImplementedError(
        "TODO: extract from Neural_operator/Finite_difference_*_committor.py "
        "(private f_derivatives helper)"
    )
