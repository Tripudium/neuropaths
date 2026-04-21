"""Finite-difference solvers for the (transformed) committor PDEs.

The solvers operate on a uniform (N+2) x (N+2) grid on [0,1]^2 (the
transformed domain) and use:

    * second-order central differences for Laplacian and f-derivative terms,
    * upwind differences for the advection b . grad u (stability under
      irregular turbulence; dissertation eq. 35),
    * Dirichlet BCs in x (q^+: 0 at x=0, 1 at x=1; q^-: 1 at x=0, 0 at x=1),
    * periodic BCs in y via modular indexing,
    * scipy.sparse.csr_matrix + spsolve for the linear system.

The derivatives of f(x, y) = (x - phi(y)) / (psi(y) - phi(y)) are taken
with autograd (NOT torch.autograd) because phi, psi are Python callables
over autograd.numpy arrays. Keep inputs as ``autograd.numpy`` throughout
these call chains.

TODO: port from:
    * Neural_operator/Finite_difference_forward_comittor.py
      -> solve_forward_committor(...)
    * Neural_operator/Finite_difference_backwards_committor.py
      -> solve_backward_committor(...)

TODO: unify the two solvers; they share ~95% of their code and only
differ in the sign of the advection term and the Dirichlet data. A
single ``solve_committor(direction: {'forward', 'backward'})`` would
eliminate the drift between the two copies (the backward solver has an
extra y-upwinding block that the forward one lacks -- looks accidental).

TODO: verify eq. 33 discretisation matches what the FD routines
actually assemble. In particular the mixed derivative 2 f_y d_{xy} u is
discretised with a full 4-point stencil; boundary treatment near i=1
and i=N silently drops the stencil rather than using one-sided
differences -- document whether this is OK given Dirichlet BCs in x.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

# A boundary callable maps y -> scalar (in autograd.numpy).
BoundaryFn = Callable[[np.ndarray], np.ndarray]
# A velocity component maps (x, y) meshgrids -> values.
VelocityComponent = Callable[[np.ndarray, np.ndarray], np.ndarray]


def solve_forward_committor(
    phi: BoundaryFn,
    psi: BoundaryFn,
    bx: VelocityComponent,
    by: VelocityComponent,
    n_interior: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve for q^+ on the transformed square [0,1]^2.

    Parameters
    ----------
    phi, psi
        Left / right boundary parametrisations y -> R (autograd.numpy).
    bx, by
        Components of the turbulent velocity field on the ORIGINAL
        (curved) domain; the solver evaluates them at x' = phi(y) + x * (psi(y) - phi(y)).
    n_interior
        Number of interior grid points per axis, N. Total grid is (N+2) x (N+2).

    Returns
    -------
    q_plus, X, Y
        Solution and meshgrid in transformed coordinates.
    """
    raise NotImplementedError(
        "TODO: port from Neural_operator/Finite_difference_forward_comittor.py "
        "solve_forward_committor_2D"
    )


def solve_backward_committor(
    phi: BoundaryFn,
    psi: BoundaryFn,
    bx: VelocityComponent,
    by: VelocityComponent,
    n_interior: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve for q^- on the transformed square [0,1]^2.

    Uses the reverse drift -b (valid because rho_steady = 1/|Omega| is
    constant; dissertation eq. 18 + Section 3.3).
    """
    raise NotImplementedError(
        "TODO: port from Neural_operator/Finite_difference_backwards_committor.py "
        "solve_backwards_committor_2D"
    )


def reactive_density(q_plus: np.ndarray, q_minus: np.ndarray) -> np.ndarray:
    """Reactive trajectory density rho_react on the discrete grid.

    Strictly speaking the dissertation (eq. 18) gives::

        rho_react = q^- * q^+ / |Omega|.

    The legacy `training_data_1_generator.py` drops the 1/|Omega| factor
    (on the square domain |Omega| = 1 so it is a no-op, but on curved
    domains this is a bug). The implementation here should compute
    |Omega| from phi, psi (numerical integration of psi - phi over y in
    [0,1]) when the curved-domain experiment is enabled.

    TODO: add `domain_area` kwarg once the curved-domain experiment
    is wired up end-to-end.
    """
    return q_plus * q_minus
