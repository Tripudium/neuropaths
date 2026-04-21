"""Visualisations used in the dissertation (figs 5, 8-10).

Panels:
    * |b| (viridis) on the left,
    * ground-truth rho_react (coolwarm) in the middle,
    * predicted rho_react (coolwarm, same colour norm) on the right.

Keep the per-sample colour normalisation (shared vmin/vmax between
ground truth and prediction) so the eye isn't misled -- the legacy
`FNO_1_Test.plot_prediction_comparison` already does this; preserve it.

TODO: port from Neural_operator/FNO_1_Test.py:
    * plot_prediction_comparison -> plot_prediction_comparison
    * plot_error_distribution     -> plot_error_distribution
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def plot_prediction_comparison(
    velocity_mag: list[np.ndarray],
    ground_truth: list[np.ndarray],
    predictions: list[np.ndarray],
    output_path: str | Path,
) -> None:
    """Triple-column (|b|, truth, prediction) figure per sample."""
    raise NotImplementedError(
        "TODO: port from Neural_operator/FNO_1_Test.py plot_prediction_comparison. "
        "Remove reliance on live torch model; take pre-computed numpy arrays."
    )


def plot_error_distribution(
    predictions: np.ndarray,
    targets: np.ndarray,
    output_path: str | Path,
) -> None:
    """Histogram of (pred - target) plus pred-vs-actual scatter."""
    raise NotImplementedError(
        "TODO: port from Neural_operator/FNO_1_Test.py plot_error_distribution."
    )
