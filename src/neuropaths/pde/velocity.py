"""Synthetic 2D turbulent velocity fields.

Follows dissertation Algorithm 2 (section 4.2.2):

    1. Centre the 311x311 domain at the origin; FFT grid.
    2. Populate Fourier coefficients uniformly at random on modes with
       k in [k_min, k_max], scaled by |k|^{-5/3} (Kolmogorov's law).
    3. Project onto the divergence-free subspace:
           u' = u - k_x (k . u) / |k|^2,
           v' = v - k_y (k . u) / |k|^2.
    4. IFFT back to physical space.
    5. Rescale to target RMS speed U = nu * Re / L (eq. 24).
    6. Wrap in scipy.interpolate.RegularGridInterpolator so the rest
       of the pipeline can evaluate b at arbitrary (x, y).

TODO: port from Neural_operator/turbulent_velocity_field.py. Notes on
what to clean up while porting:

    * `N_x = N_y = 311` is hardcoded -- expose as a kwarg defaulting
      to `PDEConfig.fine_grid`.
    * `kfmax = 30 * kfmin` is hardcoded; dissertation picks `kfmax`
      based on the Nyquist limit and has a separate "laminar vs
      turbulent" knob (notes chapter uses k_max=8 for laminar and 16
      for turbulent -- this doesn't match the legacy code).
    * Amplitude `1e4` and the `Reynolds = 8.0` call-site default in
      `training_data_1_generator.py` both differ from the dissertation's
      `Re = 10`; pick one and document.
    * Return type is `(interp_ux, interp_uy)` today; consider wrapping
      in a small `TurbulentField` dataclass that also carries the raw
      physical-space grid for diagnostics (spectral plots, divergence
      check).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class TurbulentField:
    """Evaluable turbulent drift b: R^2 -> R^2 plus diagnostic arrays."""

    bx: Callable[[np.ndarray], np.ndarray]
    by: Callable[[np.ndarray], np.ndarray]

    # Raw physical-space arrays on the fine FFT grid; kept for
    # divergence tests, spectral plots, and inclusion as FNO inputs.
    ux_grid: np.ndarray
    uy_grid: np.ndarray
    x_axis: np.ndarray
    y_axis: np.ndarray


def turbulent_velocity_field(
    *,
    reynolds: float,
    char_length: float,
    viscosity: float,
    eps_1: float = 0.0,
    eps_2: float = 0.0,
    n_grid: int = 311,
    rng: np.random.Generator | None = None,
) -> TurbulentField:
    """Generate a divergence-free, k^{-5/3}-scaled synthetic velocity field.

    TODO: port from Neural_operator/turbulent_velocity_field.py
    ``turbulent_velocity_field(eps_1, eps_2, Reynolds, L, nu)``.
    Switch the RNG from ``np.random.randn`` to the passed ``rng`` so
    seeding is reproducible.
    """
    raise NotImplementedError(
        "TODO: port from Neural_operator/turbulent_velocity_field.py"
    )
