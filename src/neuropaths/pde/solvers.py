"""Finite-difference solvers for the (transformed) committor PDEs.

The committor q^+ for the SDE dX = b(X) dt + sqrt(2) dW with viscosity
nu = 1 satisfies the boundary-value problem

    Delta q^+ + b . grad q^+ = 0  on Omega,
    q^+ = 0 on the "A" boundary, q^+ = 1 on the "B" boundary.

q^- satisfies the same equation with reverse drift -b (valid when the
invariant density is constant, dissertation eq. 21). Both committors
take values in [0, 1] by the discrete maximum principle.

The solvers assemble the PDE on the unit square [0, 1]^2 in
"transformed" coordinates (x_hat, y) where x_hat = (x - phi(y)) /
(psi(y) - phi(y)). For a square domain (phi=0, psi=1) the transform is
the identity and the equation reduces to Delta q + b . grad q = 0 on
the square. For curved domains the transformed PDE picks up extra
terms (Theorem 4.1 / eq. 33):

    (1 + f_y^2) u_{x_hat x_hat} + 2 f_y u_{x_hat y} + u_{yy}
    + (f_yy + b_1 + b_2 f_y) u_{x_hat} + b_2 u_y = 0.

Discretisation
--------------
* Nx interior x-points (Dirichlet boundaries excluded), so dx = 1/(Nx+1).
* Ny *periodic* y-points (NO duplicate at j=Ny), so dy = 1/Ny. Enforcing
  periodicity by index modulus avoids the rank-deficient duplicated-
  boundary system that the previous version produced.
* Diffusion stencils: standard 5-point central differences for u_xx,
  u_yy and (1+f_y^2) u_xx.
* Mixed derivative 2 f_y u_xy: 4-point central cross stencil
  (m_coef = 2 f_y / (4 dx dy)). For square (f_y = 0) this contributes
  nothing; for curved domains the M-matrix property is mildly violated
  but at production resolution (cell Pe << 1) the discrete solution
  still respects the maximum principle to numerical precision.
* Advection: first-order pointwise upwind. For ax = (b_1 + f_yy + b_2 f_y)
  and a_y = b_2:
      ax >= 0:  backward diff (uses (i, j) and (i-1, j))
      ax <  0:  forward  diff (uses (i, j) and (i+1, j))
  Same for a_y. This gives an M-matrix at cell Pe = |b| dx <= 1 and
  preserves the discrete maximum principle.

The output is reshaped to (Nx+2) x (Ny+1): Dirichlet boundary rows are
prepended/appended in x, and the j=0 column is duplicated as j=Ny so
the array shape matches the dissertation convention `fine_grid` x
`fine_grid` (the duplicated column is purely for downstream
plotting/subsampling and does not enter the linear system).

``autograd`` (HIPS) is used for f_y, f_yy and hence phi, psi must be
autograd-numpy callables, not torch. Velocity components bx, by are
plain numpy-friendly callables.
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

    Parameters
    ----------
    phi, psi
        Domain boundaries. Identity (phi=0, psi=1) for the square case.
    bx, by
        Velocity components in the *original* (curved) coordinates. The
        solver evaluates them at points x' = phi(y) + x_hat (psi(y) - phi(y)).
    n_interior
        Number of interior x-points (Dirichlet boundaries excluded).
        With ``n_interior = fine_grid - 2`` (so e.g. 309 for fine_grid=311)
        the returned arrays have shape (fine_grid, fine_grid).
    direction
        ``"forward"`` solves for q^+ (Dirichlet 0 on x=0, 1 on x=1).
        ``"backward"`` solves for q^- (drift -> -b, Dirichlet flipped).

    Returns
    -------
    u : ndarray, shape (Nx+2, Ny+1)
        Full grid solution including Dirichlet boundary rows in x and
        the duplicated periodic column in y.
    X, Y : ndarray, shape (Nx+2, Ny+1)
        Coordinate meshgrid (in the transformed coordinates) for plotting
        and subsampling.
    """
    if direction not in ("forward", "backward"):
        raise ValueError(f"direction must be 'forward' or 'backward', got {direction!r}")

    Nx = int(n_interior)
    if Nx < 1:
        raise ValueError(f"n_interior must be >= 1, got {Nx}")
    Ny = Nx + 1  # periodic y has one less point than full x (Nx + 2)

    dx = 1.0 / (Nx + 1)
    dy = 1.0 / Ny

    sign = 1.0 if direction == "forward" else -1.0
    u_left = 0.0 if direction == "forward" else 1.0
    u_right = 1.0 if direction == "forward" else 0.0

    # ---- Coordinate-transform derivatives at all (Nx+2) x-points x Ny y-points
    x_full = anp.linspace(0.0, 1.0, Nx + 2)
    y_per = anp.linspace(0.0, 1.0, Ny, endpoint=False)
    X_full, Y_full = anp.meshgrid(x_full, y_per, indexing="ij")
    f_y, f_yy = coord_transform_derivatives(phi, psi, X_full, Y_full)
    f_y = np.asarray(f_y)
    f_yy = np.asarray(f_yy)

    # ---- Velocity field in the original (curved) coordinates
    X_full_np = np.asarray(X_full)
    Y_full_np = np.asarray(Y_full)
    phi_Y = np.asarray(phi(Y_full_np))
    psi_Y = np.asarray(psi(Y_full_np))
    Xdash = phi_Y + X_full_np * (psi_Y - phi_Y)
    bxv = sign * np.asarray(bx(Xdash, Y_full_np))
    byv = sign * np.asarray(by(Xdash, Y_full_np))

    # ---- Sparse-matrix assembly (Nx * Ny unknowns)
    # Indexing: P = i * Ny + j, where i in [0, Nx) is the *interior* x-index
    # (i_full = i + 1 for the full grid that includes Dirichlet boundaries),
    # and j in [0, Ny) is the periodic y-index.
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    B = np.zeros(Nx * Ny)

    for i in range(Nx):
        i_full = i + 1
        E_ok = i + 1 < Nx
        W_ok = i - 1 >= 0
        for j in range(Ny):
            P = i * Ny + j
            E = (i + 1) * Ny + j
            W = (i - 1) * Ny + j
            jp = (j + 1) % Ny
            jm = (j - 1) % Ny
            Nn = i * Ny + jp
            S = i * Ny + jm

            fy = f_y[i_full, j]
            fyy = f_yy[i_full, j]
            cx = 1.0 + fy * fy  # coefficient of u_xx in the transformed PDE

            # Diagonal accumulator (advection adds to it as we go).
            diag = -2.0 * cx / (dx * dx) - 2.0 / (dy * dy)

            # ---- Diffusion off-diagonals in x ----
            if E_ok:
                rows.append(P); cols.append(E); data.append(cx / (dx * dx))
            else:
                B[P] -= cx / (dx * dx) * u_right
            if W_ok:
                rows.append(P); cols.append(W); data.append(cx / (dx * dx))
            else:
                B[P] -= cx / (dx * dx) * u_left

            # ---- Diffusion off-diagonals in y (periodic) ----
            rows.append(P); cols.append(Nn); data.append(1.0 / (dy * dy))
            rows.append(P); cols.append(S);  data.append(1.0 / (dy * dy))

            # ---- Mixed derivative 2 f_y u_xy ----
            # Vanishes for the square domain (f_y = 0). For curved
            # domains the central cross stencil mildly violates the
            # M-matrix property but is acceptable at production
            # resolution (cell Pe << 1).
            if fy != 0.0:
                m = 2.0 * fy / (4.0 * dx * dy)
                if E_ok:
                    rows.append(P); cols.append((i + 1) * Ny + jp); data.append(+m)
                    rows.append(P); cols.append((i + 1) * Ny + jm); data.append(-m)
                # else: u at i=Nx is constant = u_right, so u_xy contrib = 0.
                if W_ok:
                    rows.append(P); cols.append((i - 1) * Ny + jp); data.append(-m)
                    rows.append(P); cols.append((i - 1) * Ny + jm); data.append(+m)
                # else: u at i=-1 is constant = u_left, so u_xy contrib = 0.

            # ---- Advection (first-order upwind, pointwise b) ----
            ax = bxv[i_full, j] + fyy + byv[i_full, j] * fy
            ay = byv[i_full, j]

            # x-advection: ax * u_x
            if ax >= 0.0:  # backward difference
                diag += ax / dx
                if W_ok:
                    rows.append(P); cols.append(W); data.append(-ax / dx)
                else:
                    B[P] -= (-ax / dx) * u_left
            else:  # forward difference
                diag += -ax / dx
                if E_ok:
                    rows.append(P); cols.append(E); data.append(ax / dx)
                else:
                    B[P] -= (ax / dx) * u_right

            # y-advection: ay * u_y (periodic, never hits a Dirichlet face)
            if ay >= 0.0:
                diag += ay / dy
                rows.append(P); cols.append(S); data.append(-ay / dy)
            else:
                diag += -ay / dy
                rows.append(P); cols.append(Nn); data.append(ay / dy)

            rows.append(P); cols.append(P); data.append(diag)

    A = csr_matrix((data, (rows, cols)), shape=(Nx * Ny, Nx * Ny))
    U = np.asarray(spsolve(A, B))
    u_inner = U.reshape((Nx, Ny))

    # Stitch boundaries: Dirichlet rows in x, duplicated j=0 column in y.
    u_with_xbcs = np.vstack(
        [np.full(Ny, u_left), u_inner, np.full(Ny, u_right)]
    )
    u_full = np.hstack([u_with_xbcs, u_with_xbcs[:, [0]]])

    # Output meshgrid (transformed coordinates).
    x_out = np.linspace(0.0, 1.0, Nx + 2)
    y_out = np.linspace(0.0, 1.0, Ny + 1)
    X_out, Y_out = np.meshgrid(x_out, y_out, indexing="ij")

    return u_full, X_out, Y_out


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
    Reverse drift: rho_steady is constant, so the backward generator
    simplifies to Delta + (-b) . grad (dissertation eq. 21).
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

    Both q^+ and q^- are in [0, 1] and so rho_react is in [0, 1/|Omega|]
    (i.e., [0, 1] on the unit square). Tests/callers may assert this.
    """
    return q_plus * q_minus / float(domain_area)
