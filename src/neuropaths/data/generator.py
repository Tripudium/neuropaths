"""Offline dataset generation: solve committor PDEs, subsample, write CSV.

Pipeline (mirrors the legacy `training_data_1_generator.py` but driven
entirely by ExperimentConfig and a passed-in RNG):

    for sid in range(num_solutions):
        1. Draw phi, psi (square -> identity; curved -> Algorithm 1).
        2. Draw b via turbulent_velocity_field on the 311x311 FFT grid.
        3. Build btilde_x, btilde_y (b evaluated on the transformed grid).
        4. Solve q^+ and q^- on a (fine_grid) x (fine_grid) grid.
        5. rho = q^+ * q^- / |Omega|   (square: |Omega|=1).
        6. Subsample to coarse_grid via linspace index selection
           (dissertation: fine_grid = (coarse_grid - 1) * step + 1, so
           the subsample is exact — e.g. 311 -> 32 takes every 10th).
        7. Append rows (sid, x, y, b1, b2, rho) (+ finv for curved).

The CSV schema matches the legacy format so existing notebooks continue
to read training sets. Switching to Parquet/Zarr is a Phase 3 concern
once dataset sizes exceed a few GB.
"""

from __future__ import annotations

from pathlib import Path

import autograd.numpy as anp
import numpy as np
import pandas as pd
from tqdm import tqdm

from neuropaths.config import DataConfig, PDEConfig
from neuropaths.pde.boundaries import generate_boundary_pair
from neuropaths.pde.solvers import (
    reactive_density,
    solve_backward_committor,
    solve_forward_committor,
)
from neuropaths.pde.transforms import inverse_map
from neuropaths.pde.velocity import turbulent_velocity_field


def _domain_area(phi, psi, n_probe: int = 1024) -> float:
    """Numerical |Omega| = integral_0^1 (psi(y) - phi(y)) dy (trapz)."""
    y = np.linspace(0.0, 1.0, n_probe)
    gap = np.asarray(psi(y)) - np.asarray(phi(y))
    return float(np.trapezoid(gap, y))


def _subsample_indices(n_fine: int, n_coarse: int) -> np.ndarray:
    """Linspace index selection including both boundaries.

    Matches legacy: `np.linspace(0, n_fine - 1, n_coarse, dtype=int)`.
    For n_fine=311, n_coarse=32 this is exactly `arange(0, 311, 10)` plus
    the endpoint — a clean stride-10 subsample.
    """
    return np.linspace(0, n_fine - 1, n_coarse, dtype=int)


def generate_dataset(
    pde_cfg: PDEConfig,
    data_cfg: DataConfig,
    *,
    split: str = "train",
    output_path: str | Path | None = None,
    rng: np.random.Generator | None = None,
) -> Path:
    """Generate a single CSV of (solution_id, x, y, b1, b2, rho[, finv]).

    Parameters
    ----------
    pde_cfg, data_cfg
        Config slices. Everything numerical (fine grid, Reynolds,
        k_max, n_train_solutions, output path) is driven from here.
    split
        "train" or "test" -- selects num_{train,test}_solutions and
        the matching output path.
    output_path
        Explicit override; falls back to data_cfg.{train,test}_csv.
    rng
        Explicit Generator. If None, builds one from data_cfg.seed with
        a ``split`` offset so train and test draws don't collide.
    """
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")

    raw_path = output_path if output_path is not None else (
        data_cfg.train_csv if split == "train" else data_cfg.test_csv
    )
    out_path = Path(raw_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if rng is None:
        # Offset so train/test draws are independent.
        base_seed = data_cfg.seed + (0 if split == "train" else 1)
        rng = np.random.default_rng(base_seed)

    num_solutions = (
        data_cfg.num_train_solutions if split == "train" else data_cfg.num_test_solutions
    )

    fine_grid = pde_cfg.fine_grid
    coarse_grid = pde_cfg.coarse_grid
    n_interior = fine_grid - 2  # fine FD grid has n_interior + 2 points per axis
    coarse_idx = _subsample_indices(fine_grid, coarse_grid)

    records: list[list[float]] = []
    columns = ["solution_id", "x", "y", "b1", "b2", "rho"]
    if data_cfg.include_finv_column:
        columns.append("finv")

    for sid in tqdm(range(num_solutions), desc=f"generate[{split}]", unit="pde"):
        # 1. Boundaries.
        phi, psi = generate_boundary_pair(
            rng,
            kind=pde_cfg.domain,
            n_max=pde_cfg.boundary_n_max,
            eps=pde_cfg.boundary_eps,
        )

        # 2. Velocity field.
        field = turbulent_velocity_field(
            reynolds=pde_cfg.reynolds,
            char_length=pde_cfg.char_length,
            viscosity=pde_cfg.viscosity,
            k_min=pde_cfg.velocity_kmin,
            k_max=pde_cfg.velocity_kmax,
            eps_1=pde_cfg.eps_1,
            eps_2=pde_cfg.eps_2,
            n_grid=fine_grid,
            rng=rng,
        )

        # 3. Wrap as (x, y)-meshgrid evaluators (the solvers call them
        # with 2D arrays representing the transformed grid).
        bx = field.bx
        by = field.by

        # 4. Solve committor PDEs.
        q_plus, X, Y = solve_forward_committor(phi, psi, bx, by, n_interior)
        q_minus, _, _ = solve_backward_committor(phi, psi, bx, by, n_interior)

        # 5. Reactive density with the correct |Omega| factor.
        omega = 1.0 if pde_cfg.domain == "square" else _domain_area(phi, psi)
        rho = reactive_density(q_plus, q_minus, domain_area=omega)

        # 6. Subsample to the coarse training grid.
        ii, jj = np.meshgrid(coarse_idx, coarse_idx, indexing="ij")
        x_c = np.asarray(X)[ii, jj]
        y_c = np.asarray(Y)[ii, jj]
        rho_c = rho[ii, jj]

        # 7. Original-coordinate evaluations of b on the coarse grid.
        #    For the square domain this is just b(x, y); for curved it
        #    is b(f^{-1}(x, y), y).
        if pde_cfg.domain == "square":
            bx_c = bx(x_c, y_c)
            by_c = by(x_c, y_c)
        else:
            f_inv = inverse_map(phi, psi)
            xprime = np.asarray(f_inv(anp.asarray(x_c), anp.asarray(y_c)))
            bx_c = bx(xprime, y_c)
            by_c = by(xprime, y_c)

        # Optional finv channel (curved domain only).
        finv_c = None
        if data_cfg.include_finv_column:
            # f^{-1}(x, y) evaluated on the transformed coarse grid -- this
            # carries the "shape of the boundary" information that the
            # dissertation feeds as an extra FNO input channel.
            f_inv = inverse_map(phi, psi)
            finv_c = np.asarray(f_inv(anp.asarray(x_c), anp.asarray(y_c)))

        for i in range(coarse_grid):
            for j in range(coarse_grid):
                row = [
                    sid,
                    float(x_c[i, j]),
                    float(y_c[i, j]),
                    float(bx_c[i, j]),
                    float(by_c[i, j]),
                    float(rho_c[i, j]),
                ]
                if finv_c is not None:
                    row.append(float(finv_c[i, j]))
                records.append(row)

    df = pd.DataFrame(records, columns=columns)
    df.to_csv(out_path, index=False)
    return out_path
