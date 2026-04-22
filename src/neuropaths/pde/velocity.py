"""Synthetic 2D turbulent velocity fields (dissertation Algorithm 2).

Pipeline:

    1. Centre the N_grid x N_grid domain at the origin; FFT grid.
    2. Populate complex Fourier coefficients i.i.d. N(0, 1) + i N(0, 1)
       on modes with |k| in the band [k_min * k_fund, k_max * k_fund],
       where k_fund = 2*pi / L_domain; scale by |k|^{-5/3} (Kolmogorov).
    3. Project onto the divergence-free subspace:
           P(u) = u - k (k . u) / |k|^2.
    4. IFFT back to physical space.
    5. Rescale to target RMS speed U = nu * Re / L (dissertation §4.2.2
       eq. 24).
    6. Wrap each component in scipy.interpolate.RegularGridInterpolator
       so the FD solvers can evaluate b at arbitrary (x, y).

Notes-chapter §2.4 specialisation: cap the upper wavenumber via the
config knob `velocity_kmax` (8 for laminar, 16 for turbulent). This
replaces the legacy hardcoded `kfmax = 30 * kfmin`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.fftpack import ifft2
from scipy.interpolate import RegularGridInterpolator


# A velocity component maps meshgrids (or arrays of query points) of shape
# (M,) or (H, W) to arrays of the same shape. The RegularGridInterpolator
# wrapper below accepts either broadcastable (x, y) arrays or an (N, 2)
# point array; the FD solvers use the former.
VelocityComponent = Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass
class TurbulentField:
    """Evaluable turbulent drift b: R^2 -> R^2 plus diagnostic arrays."""

    bx: VelocityComponent
    by: VelocityComponent

    # Raw physical-space arrays on the fine FFT grid; kept for
    # divergence tests, spectral plots, and FNO input channels.
    ux_grid: np.ndarray
    uy_grid: np.ndarray
    x_axis: np.ndarray
    y_axis: np.ndarray

    # Diagnostics: realised RMS speed and the target U.
    rms_target: float
    rms_realised: float


def _build_interpolator(
    u_grid: np.ndarray,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
) -> VelocityComponent:
    """Wrap a 2D array in a RegularGridInterpolator with the
    broadcast-friendly (x, y) calling convention the FD solvers use.
    """
    interp = RegularGridInterpolator(
        (x_axis, y_axis),
        u_grid,
        method="cubic",
        bounds_error=False,
        fill_value=None,  # type: ignore[arg-type]  # scipy accepts None ("extrapolate"); stub says float
    )

    def eval_fn(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x)
        y_arr = np.asarray(y)
        pts = np.column_stack((x_arr.ravel(), y_arr.ravel()))
        return interp(pts).reshape(x_arr.shape)

    return eval_fn


def turbulent_velocity_field(
    *,
    reynolds: float = 10.0,
    char_length: float = 0.2,
    viscosity: float = 1.0,
    k_min: int = 1,
    k_max: int = 8,
    eps_1: float = 0.0,
    eps_2: float = 0.0,
    n_grid: int = 311,
    rng: np.random.Generator | None = None,
) -> TurbulentField:
    """Generate a divergence-free, k^{-5/3}-scaled synthetic velocity field.

    Parameters
    ----------
    reynolds, char_length, viscosity
        Sets the target RMS speed U = nu * Re / L (eq. 24).
    k_min, k_max
        Mode-band multiples of the domain fundamental k_fund = 2*pi/L_domain.
        `k_max=8` -> laminar (Section 2 first test), `k_max=16` -> turbulent.
    eps_1, eps_2
        Epsilon-neighbourhood padding; used when Omega is a curved domain
        and the velocity needs to be evaluated slightly outside [0,1]^2.
        Square domain uses eps_1 = eps_2 = 0.
    n_grid
        FFT / FD grid resolution. Dissertation fixes 311.
    rng
        Explicit numpy Generator; never touches the global seed.
    """
    if rng is None:
        rng = np.random.default_rng()

    N_x = N_y = int(n_grid)
    L_domain = 1.0 + 2.0 * max(eps_1, eps_2)

    # ---- Fourier-space grid -------------------------------------------------
    dx = L_domain / N_x
    dy = L_domain / N_y
    k_xv = 2.0 * np.pi * np.fft.fftfreq(N_x, d=dx)
    k_yv = 2.0 * np.pi * np.fft.fftfreq(N_y, d=dy)
    kx, ky = np.meshgrid(k_xv, k_yv, indexing="ij")
    k2 = kx * kx + ky * ky
    k_mag = np.sqrt(k2)

    # Band-pass in |k| using the domain-fundamental as the base unit.
    k_fund = 2.0 * np.pi / L_domain
    band_lo = k_min * k_fund
    band_hi = k_max * k_fund
    init_modes = (k_mag >= band_lo) & (k_mag <= band_hi)

    # ---- random Fourier coefficients ---------------------------------------
    n_active = int(np.count_nonzero(init_modes))
    if n_active == 0:
        raise ValueError(
            f"turbulent_velocity_field: no modes in band [{band_lo:.3f}, {band_hi:.3f}]. "
            f"Try increasing n_grid or widening [k_min, k_max]."
        )

    # Initial amplitude is rescaled away by the RMS normalisation below, so
    # its absolute value is irrelevant; keep it modest to avoid FP range issues.
    amplitude = 1.0
    ux_hat = np.zeros((N_x, N_y), dtype=complex)
    uy_hat = np.zeros((N_x, N_y), dtype=complex)
    ux_hat[init_modes] = amplitude * (
        rng.standard_normal(n_active) + 1j * rng.standard_normal(n_active)
    )
    uy_hat[init_modes] = amplitude * (
        rng.standard_normal(n_active) + 1j * rng.standard_normal(n_active)
    )

    # k^{-5/3} scaling on the active band.
    scaling = np.zeros_like(k_mag)
    scaling[init_modes] = k_mag[init_modes] ** (-5.0 / 3.0)
    ux_hat *= scaling
    uy_hat *= scaling

    # ---- divergence-free projection ----------------------------------------
    k2_safe = k2.copy()
    k2_safe[0, 0] = 1.0  # avoid 1/0 at the zero mode; scalar potential is zero there anyway.
    phi_hat = (1j * kx * ux_hat + 1j * ky * uy_hat) / k2_safe
    phi_hat[0, 0] = 0.0
    ux_hat = ux_hat - 1j * kx * phi_hat
    uy_hat = uy_hat - 1j * ky * phi_hat

    # ---- back to physical space --------------------------------------------
    ux_phys = np.real(ifft2(ux_hat))
    uy_phys = np.real(ifft2(uy_hat))

    # ---- rescale to target RMS = U = nu*Re/L (dissertation eq. 24) ---------
    rms_target = (viscosity * reynolds) / char_length
    rms_current = np.sqrt(np.mean(ux_phys * ux_phys + uy_phys * uy_phys))
    if rms_current < 1e-30:
        raise RuntimeError(
            "turbulent_velocity_field: realised RMS is zero; bad mode band or rng draw."
        )
    ux_phys *= rms_target / rms_current
    uy_phys *= rms_target / rms_current
    rms_realised = np.sqrt(np.mean(ux_phys * ux_phys + uy_phys * uy_phys))

    # ---- build interpolators -----------------------------------------------
    # Physical-space axes the IFFT corresponds to. For the square case with
    # eps_1 = eps_2 = 0 this is just [0, 1] on both axes; for curved domains
    # we pad slightly so the FD solver can evaluate b outside [0,1]^2 during
    # the coordinate transform x' = phi(y) + x*(psi(y) - phi(y)).
    pad = max(eps_1, eps_2) + 0.01 if max(eps_1, eps_2) > 0 else 0.0
    x_axis = np.linspace(-pad, 1.0 + pad, N_x)
    y_axis = np.linspace(0.0, 1.0, N_y)

    bx = _build_interpolator(ux_phys, x_axis, y_axis)
    by = _build_interpolator(uy_phys, x_axis, y_axis)

    return TurbulentField(
        bx=bx,
        by=by,
        ux_grid=ux_phys,
        uy_grid=uy_phys,
        x_axis=x_axis,
        y_axis=y_axis,
        rms_target=rms_target,
        rms_realised=rms_realised,
    )
