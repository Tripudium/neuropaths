"""Finite-difference solvers for the (transformed) committor PDEs.

The solvers assemble the transformed advection-diffusion PDE (Theorem 4.1,
dissertation eq. 33) on a uniform (N+2) x (N+2) grid on [0,1]^2 and solve
the sparse linear system with scipy.sparse.linalg.spsolve:

    * Second-order central differences for the Laplacian and the
      (f_y)^2 d_{xx} and 2 f_y d_{xy} stencil terms.
    * Upwind differences for the advection (f_yy + b_1 + b_2 f_y) d_x u
      + d_y u (dissertation eq. 35; stabilises under irregular b).
    * Dirichlet BCs in x (q^+: 0 at x=0, 1 at x=1; q^-: 1 at x=0, 0 at x=1).
    * Periodic BCs in y via modular indexing with period N+1 (the grid
      is (N+2) points, of which index 0 and index N+1 coincide under the
      period, so the modular arithmetic uses mod (N+1)).

Background on direction signs: the forward and backward Kolmogorov
generators differ only in the sign of the drift b, so we implement a
single solver parameterised by ``direction in {'forward', 'backward'}``.
This kills the drift between the two legacy files (the backward solver
had an extra y-upwinding block that the forward one lacked — that was
accidental and is not reproduced here).

``autograd`` (HIPS) is used for the f_y, f_{yy} derivatives and hence
phi, psi must be autograd-numpy callables, not torch. Velocity
components bx, by are plain numpy-friendly callables.
"""

from __future__ import annotations

from typing import Callable, Literal

import autograd.numpy as anp
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

from neuropaths.pde.transforms import coord_transform_derivatives

BoundaryFn = Callable[[anp.ndarray], anp.ndarray]
VelocityComponent = Callable[[np.ndarray, np.ndarray], np.ndarray]
Direction = Literal["forward", "backward"]


def _assemble_and_solve(
    phi: BoundaryFn,
    psi: BoundaryFn,
    bx: VelocityComponent,
    by: VelocityComponent,
    n_interior: int,
    direction: Direction,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Core FD assembler shared by forward and backward committor solves.

    Returns (u, X, Y) where u is the full (N+2, N+2) grid solution in
    transformed coordinates and X, Y are the meshgrid arrays.
    """
    if direction not in ("forward", "backward"):
        raise ValueError(f"direction must be 'forward' or 'backward', got {direction!r}")

    N = int(n_interior)
    if N < 1:
        raise ValueError(f"n_interior must be >= 1, got {N}")

    dx = 1.0 / (N + 1)
    dy = 1.0 / (N + 1)

    # Grid in transformed coordinates.
    x = anp.linspace(0.0, 1.0, N + 2)
    y = anp.linspace(0.0, 1.0, N + 2)
    X, Y = anp.meshgrid(x, y, indexing="ij")

    # Coordinate-transform derivatives (identity boundaries -> exact zeros).
    f_y_values, f_yy_values = coord_transform_derivatives(phi, psi, X, Y)

    # Velocity field in original coordinates:
    #   x' = phi(y) + x * (psi(y) - phi(y)).
    X_flat = np.asarray(X).flatten()
    Y_flat = np.asarray(Y).flatten()
    phi_Y = np.asarray(phi(Y_flat))
    psi_Y = np.asarray(psi(Y_flat))
    Xdash_flat = phi_Y + X_flat * (psi_Y - phi_Y)
    Xdash = Xdash_flat.reshape(np.asarray(X).shape)
    bx_values = np.asarray(bx(Xdash, np.asarray(Y)))
    by_values = np.asarray(by(Xdash, np.asarray(Y)))

    # Enforce periodicity in y at the ghost column j=N+1 (identified with j=0).
    bx_values[:, N + 1] = bx_values[:, 0]
    by_values[:, N + 1] = by_values[:, 0]

    # Sign flip for backward committor (reverse drift, rho_steady constant).
    # The backward generator is Delta + b^R . grad with b^R = -b (eq. 21).
    sign = 1.0 if direction == "forward" else -1.0

    # Dirichlet BCs on x:
    #   forward : q^+(0) = 0, q^+(1) = 1
    #   backward: q^-(0) = 1, q^-(1) = 0
    u_left = 0.0 if direction == "forward" else 1.0
    u_right = 1.0 if direction == "forward" else 0.0

    # Convert autograd output to plain numpy for the sparse assembly.
    f_y_values = np.asarray(f_y_values)
    f_yy_values = np.asarray(f_yy_values)

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    B = np.zeros(N * (N + 2))

    # i indexes interior x points (1..N); j indexes all y points (0..N+1),
    # with periodic wrap mod (N+1) since indices 0 and N+1 coincide.
    for i in range(1, N + 1):
        for j in range(N + 2):
            idx = (i - 1) * (N + 2) + j

            f_y = f_y_values[i, j]
            f_yy = f_yy_values[i, j]

            # ==== Laplacian + (f_y)^2 d_xx + 2 f_y d_xy terms ====
            # Centre point.
            rows.append(idx)
            cols.append(idx)
            data.append(-2.0 * (1.0 + f_y * f_y) / (dx * dx) - 2.0 / (dy * dy))

            # x-neighbours. At i=1 and i=N the Dirichlet BC puts the
            # neighbour's value on the RHS B rather than into A.
            if i > 1:
                rows.append(idx)
                cols.append(idx - (N + 2))
                data.append((1.0 + f_y * f_y) / (dx * dx))
            else:  # i == 1: left boundary contributes u_left.
                B[idx] -= (1.0 + f_y * f_y) / (dx * dx) * u_left
            if i < N:
                rows.append(idx)
                cols.append(idx + (N + 2))
                data.append((1.0 + f_y * f_y) / (dx * dx))
            else:  # i == N: right boundary contributes u_right.
                B[idx] -= (1.0 + f_y * f_y) / (dx * dx) * u_right

            # y-neighbours (periodic, period N+1).
            rows.append(idx)
            cols.append((i - 1) * (N + 2) + (j - 1) % (N + 1))
            data.append(1.0 / (dy * dy))
            rows.append(idx)
            cols.append((i - 1) * (N + 2) + (j + 1) % (N + 1))
            data.append(1.0 / (dy * dy))

            # ==== Advection  (f_yy + b_1 + b_2 f_y) d_x u + d_y u ====
            # Using face-average upwinding in both x and y so the scheme
            # stays positive-coefficient under irregular b (eq. 35).
            bx_ij = sign * bx_values[i, j]
            by_ij = sign * by_values[i, j]
            # Effective x-advection coefficient carries the transform
            # residue f_yy + b2 * f_y (the "+ b1" piece is bx_ij itself).
            ax_ij = bx_ij + f_yy + sign * by_values[i, j] * f_y

            jp = (j + 1) % (N + 2)
            jm = (j - 1) % (N + 2)

            # Face averages for the "effective" x-advection ax. Index
            # range is safe: i ranges 1..N, so i-1 in {0..N-1} and i+1
            # in {2..N+1} are all valid (the transformed grid has N+2
            # points per axis including the Dirichlet boundary columns).
            ax_left = 0.5 * (
                ax_ij + (sign * bx_values[i - 1, j] + f_yy_values[i - 1, j]
                          + sign * by_values[i - 1, j] * f_y_values[i - 1, j])
            )
            ax_right = 0.5 * (
                ax_ij + (sign * bx_values[i + 1, j]
                          + f_yy_values[i + 1, j]
                          + sign * by_values[i + 1, j] * f_y_values[i + 1, j])
            )
            by_up = 0.5 * (by_ij + sign * by_values[i, jp])
            by_down = 0.5 * (by_ij + sign * by_values[i, jm])

            P = idx
            E = idx + (N + 2)
            W = idx - (N + 2)
            Nn = (i - 1) * (N + 2) + (j + 1) % (N + 1)
            S_ = (i - 1) * (N + 2) + (j - 1) % (N + 1)

            # --- x flux, right face ---
            if ax_right >= 0:
                rows.append(P)
                cols.append(P)
                data.append(ax_right / dx)
            else:
                if i < N:
                    rows.append(P)
                    cols.append(E)
                    data.append(ax_right / dx)
                else:
                    B[idx] -= (ax_right / dx) * u_right

            # --- x flux, left face ---
            if ax_left >= 0:
                if i > 1:
                    rows.append(P)
                    cols.append(W)
                    data.append(-ax_left / dx)
                else:
                    B[idx] -= (-ax_left / dx) * u_left
            else:
                rows.append(P)
                cols.append(P)
                data.append(-ax_left / dx)

            # --- y flux, top face (periodic) ---
            if by_up >= 0:
                rows.append(P)
                cols.append(P)
                data.append(by_up / dy)
            else:
                rows.append(P)
                cols.append(Nn)
                data.append(by_up / dy)

            # --- y flux, bottom face (periodic) ---
            if by_down >= 0:
                rows.append(P)
                cols.append(S_)
                data.append(-by_down / dy)
            else:
                rows.append(P)
                cols.append(P)
                data.append(-by_down / dy)

            # ==== Mixed second derivative 2 f_y d_{xy} u ====
            # Central-difference cross stencil (factor 2/(4 dx dy) = 1/(2 dx dy)).
            mixed = (2.0 * f_y) / (4.0 * dx * dy)
            if i > 1:
                rows.append(idx)
                cols.append((i - 2) * (N + 2) + (j - 1) % (N + 1))
                data.append(mixed)
                rows.append(idx)
                cols.append((i - 2) * (N + 2) + (j + 1) % (N + 1))
                data.append(-mixed)
            if i < N:
                rows.append(idx)
                cols.append((i) * (N + 2) + (j - 1) % (N + 1))
                data.append(-mixed)
                rows.append(idx)
                cols.append((i) * (N + 2) + (j + 1) % (N + 1))
                data.append(mixed)
            # At i == 1 and i == N the mixed-derivative stencil terms
            # couple to Dirichlet boundary values whose partial y-
            # derivatives are zero (u_left, u_right are constants), so
    # the stencil contributes nothing at those boundaries.

    A = csr_matrix((data, (rows, cols)), shape=(N * (N + 2), N * (N + 2)))
    U = spsolve(A, B)

    # Reshape and prepend / append the Dirichlet boundary rows in x.
    # np.asarray narrows spsolve's ndarray | csc_array return type for type-checkers.
    u_inner = np.asarray(U).reshape((N, N + 2))
    u_full = np.vstack(
        (
            np.full(N + 2, u_left),
            u_inner,
            np.full(N + 2, u_right),
        )
    )
    return u_full, np.asarray(X), np.asarray(Y)


def solve_forward_committor(
    phi: BoundaryFn,
    psi: BoundaryFn,
    bx: VelocityComponent,
    by: VelocityComponent,
    n_interior: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve for q^+ on the transformed square [0,1]^2.

    Dirichlet: q^+(0, y) = 0 (on set A), q^+(1, y) = 1 (on set B).
    Periodic: q^+(x, 0) = q^+(x, 1).
    """
    return _assemble_and_solve(phi, psi, bx, by, n_interior, "forward")


def solve_backward_committor(
    phi: BoundaryFn,
    psi: BoundaryFn,
    bx: VelocityComponent,
    by: VelocityComponent,
    n_interior: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve for q^- on the transformed square [0,1]^2.

    Dirichlet: q^-(0, y) = 1 (on set A), q^-(1, y) = 0 (on set B).
    Uses reverse drift -b (valid when rho_steady is constant; see
    dissertation §3.3 / eq. 21 with rho = 1/|Omega|).
    """
    return _assemble_and_solve(phi, psi, bx, by, n_interior, "backward")


def reactive_density(
    q_plus: np.ndarray,
    q_minus: np.ndarray,
    *,
    domain_area: float = 1.0,
) -> np.ndarray:
    """Reactive trajectory density rho_react on the discrete grid.

    Dissertation eq. 18 with rho_steady = 1/|Omega|:

        rho_react = q^- * rho_steady * q^+ = q^+ * q^- / |Omega|.

    On the unit square |Omega| = 1 so this is the elementwise product;
    for curved domains the caller should pass ``domain_area`` computed
    as the numerical integral of (psi(y) - phi(y)) dy over [0, 1].
    """
    return q_plus * q_minus / float(domain_area)
