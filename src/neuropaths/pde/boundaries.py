"""Random trigonometric-polynomial boundary parametrisations.

Dissertation Algorithm 1 (section 4.1):

    phi_temp(y) = sum_{k=1..N_1} c_{1,k} sin(k*pi*y),
    psi_temp(y) = sum_{k=1..N_2} c_{2,k} sin(k*pi*y),
    phi(y) = phi_temp(y) / phi_norm,
    psi(y) = psi_temp(y) / psi_norm + 1,

with N_1, N_2 ~ Uniform{1,...,N_max}, c_{i,k} ~ U[-1,1], and normalising
constants phi_norm = 3*max|phi_temp| + mean|phi_temp|. Regenerate if
min_y |phi(y) - psi(y)| < epsilon to avoid degenerate near-touching
domains.

The sin-basis automatically enforces phi(0)=phi(1)=0, psi(0)=psi(1)=1
which the top/bottom y-periodic BC requires.

Returned callables operate on ``autograd.numpy`` arrays because the FD
solvers need second derivatives of phi, psi via
``autograd.elementwise_grad``; torch tensors silently fail.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import autograd.numpy as anp
import numpy as np

BoundaryFn = Callable[[anp.ndarray], anp.ndarray]


@dataclass
class Boundary:
    """A pair of left/right boundary callables with their metadata.

    Carrying the random coefficients around makes it possible to log /
    reproduce a specific domain from a CSV later on.
    """

    phi: BoundaryFn
    psi: BoundaryFn
    # Optional: the raw trig coefficients, useful for provenance.
    phi_coeffs: np.ndarray | None = None
    psi_coeffs: np.ndarray | None = None


def identity_boundaries() -> tuple[BoundaryFn, BoundaryFn]:
    """Return (phi=0, psi=1) — the square-domain identity transform.

    Used by the square-domain experiment (dissertation §5.1 and notes
    §2). Because f(x,y) = (x - 0)/(1 - 0) = x, Theorem 4.1's transformed
    PDE collapses to the plain advection-diffusion PDE on [0,1]^2.
    """

    def phi(y):
        # Must be a function of y (not a plain 0.0) so autograd's
        # elementwise_grad handles a vectorised y; `0.0 * y` keeps
        # the output shape matched.
        return 0.0 * y

    def psi(y):
        return 0.0 * y + 1.0

    return phi, psi


def _trig_polynomial(coeffs: np.ndarray) -> BoundaryFn:
    """Return y -> sum_k c_k sin(k pi y), autograd-compatible.

    Works on `y` of arbitrary shape (scalar, 1D, 2D meshgrid, ...). The
    leading mode axis is reshaped to ``(N, 1, 1, ...)`` so it broadcasts
    against any `y` and the sum reduces it to `y.shape`.
    """
    coeffs = np.asarray(coeffs, dtype=float)
    N = coeffs.shape[0]
    k_arr = np.arange(1, N + 1, dtype=float)
    c_arr = coeffs

    def f(y):
        y_arr = anp.atleast_1d(y)
        new_shape = (N,) + (1,) * y_arr.ndim
        k = k_arr.reshape(new_shape)
        c = c_arr.reshape(new_shape)
        return anp.sum(c * anp.sin(k * anp.pi * y_arr), axis=0)

    return f


def random_trig_boundaries(
    *,
    n_max: int = 8,
    eps: float = 0.1,
    rng: np.random.Generator | None = None,
    max_retries: int = 20,
    n_probe: int = 512,
) -> tuple[Boundary, Boundary]:
    """Sample (phi, psi) via dissertation Algorithm 1.

    Parameters
    ----------
    n_max
        Upper bound for the trig-polynomial order N_1, N_2 ~ U{1,...,n_max}.
    eps
        Minimum allowed gap inf_y |psi(y) - phi(y)|; if the random draw
        violates this, resample up to `max_retries` times.
    rng
        Explicit numpy Generator. Never touches the global seed.
    max_retries
        Cap on resampling attempts; raises if the gap constraint keeps
        failing (very unlikely for n_max <= 16).
    n_probe
        Number of y-points to check the gap on.

    Returns
    -------
    (Boundary, Boundary)
        Two Boundary records. The first has phi normalised from the
        sin-sum; the second has psi normalised similarly plus a ``+1``
        offset. Callers typically destructure as ``phi_b.phi``, ``psi_b.psi``
        but we return the dataclass pair so the RNG coefficients are
        available for provenance.
    """
    if rng is None:
        rng = np.random.default_rng()

    y_probe = np.linspace(0.0, 1.0, n_probe)

    for _ in range(max_retries):
        n1 = int(rng.integers(1, n_max + 1))
        n2 = int(rng.integers(1, n_max + 1))
        c1 = rng.uniform(-1.0, 1.0, size=n1)
        c2 = rng.uniform(-1.0, 1.0, size=n2)

        phi_temp = _trig_polynomial(c1)
        psi_temp = _trig_polynomial(c2)

        # Normalising constants (dissertation eq. 28).
        phi_vals = np.asarray(phi_temp(y_probe))
        psi_vals = np.asarray(psi_temp(y_probe))
        phi_norm = 3.0 * np.max(np.abs(phi_vals)) + np.mean(np.abs(phi_vals))
        psi_norm = 3.0 * np.max(np.abs(psi_vals)) + np.mean(np.abs(psi_vals))

        # Avoid a pathological all-zero draw (phi_norm == 0).
        if phi_norm == 0.0 or psi_norm == 0.0:
            continue

        # Capture by default-arg to pin the numeric constants.
        def phi(y, _f=phi_temp, _n=phi_norm):
            return _f(y) / _n

        def psi(y, _f=psi_temp, _n=psi_norm):
            return _f(y) / _n + 1.0

        gap = np.asarray(psi(y_probe)) - np.asarray(phi(y_probe))
        if np.min(np.abs(gap)) >= eps:
            return (
                Boundary(phi=phi, psi=psi, phi_coeffs=c1, psi_coeffs=c2),
                Boundary(phi=phi, psi=psi, phi_coeffs=c1, psi_coeffs=c2),
            )

    raise RuntimeError(
        f"random_trig_boundaries: could not satisfy min-gap eps={eps} "
        f"within {max_retries} retries (n_max={n_max})."
    )


def generate_boundary_pair(
    rng: np.random.Generator,
    *,
    kind: str = "square",
    n_max: int = 8,
    eps: float = 0.1,
) -> tuple[BoundaryFn, BoundaryFn]:
    """Convenience wrapper returning bare (phi, psi) callables.

    Parameters
    ----------
    rng
        Required; reproducibility is the whole point.
    kind
        "square" -> identity; "curved" -> random trig polynomials.
    n_max, eps
        Forwarded to random_trig_boundaries for "curved" only.
    """
    if kind == "square":
        return identity_boundaries()
    if kind == "curved":
        b, _ = random_trig_boundaries(n_max=n_max, eps=eps, rng=rng)
        return b.phi, b.psi
    raise ValueError(f"Unknown boundary kind {kind!r}; expected 'square' or 'curved'.")
