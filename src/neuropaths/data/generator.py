"""Offline dataset generation: solve PDEs, subsample, write CSV.

Pipeline (replaces Neural_operator/training_data_1_generator.py):

    for sid in range(num_solutions):
        1. Sample phi, psi (square -> identity; curved -> Algorithm 1).
        2. Sample b via turbulent_velocity_field(...) on the fine FFT grid.
        3. Build btilde_x, btilde_y (evaluate b on the transformed grid).
        4. Solve q^+ and q^- on an (n_fine+2) x (n_fine+2) grid.
        5. rho = q^+ * q^- (divide by |Omega| when curved).
        6. Subsample to `coarse_grid` with linspace index selection
           (the legacy code uses `(grid_size - 1) * 5 + 1` for n_fine=311
           and grid_size=63; make this explicit).
        7. Append rows (sid, x, y, b1, b2, rho) -- plus `finv` for curved.

Output is a single CSV grouped by solution_id; this matches the legacy
schema so existing notebooks continue to read training sets. Consider
switching to a columnar format (Parquet / Zarr) once dataset sizes grow
beyond a few GB; CSV at 5000 x 32^2 ~= 5M rows is already 200+ MB.

TODO: port from Neural_operator/training_data_1_generator.py
``generate_training_1(num_solutions, grid_size)``, but:

    * Replace the hardcoded output filename with `DataConfig.train_csv`.
    * Remove the module-level ``main()`` call -- entry point lives in
      neuropaths.cli.generate.
    * Add a ``--split train|test`` flag so the `train_csv` /
      `test_csv` split is driven by config rather than comment-toggling
      two different ``np.random.seed`` lines.
    * Respect `PDEConfig.domain`: when "curved", sample (phi, psi) from
      ``random_trig_boundaries`` and append the `finv` column so
      curved-domain experiments have that extra FNO input channel.
"""

from __future__ import annotations

from pathlib import Path

from neuropaths.config import DataConfig, PDEConfig


def generate_dataset(
    pde_cfg: PDEConfig,
    data_cfg: DataConfig,
    *,
    split: str = "train",
    output_path: str | Path | None = None,
) -> Path:
    """Generate a single CSV of (solution_id, x, y, b1, b2, rho[, finv]).

    Returns the path written.
    """
    raise NotImplementedError(
        "TODO: port from Neural_operator/training_data_1_generator.py "
        "generate_training_1. Ensure phi/psi come from pde_cfg.domain, "
        "number of solutions from data_cfg.num_{train,test}_solutions, "
        "seed from data_cfg.seed, and CSV path from output_path (or "
        "data_cfg.train_csv/test_csv)."
    )
