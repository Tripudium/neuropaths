"""Evaluation metrics.

The dissertation reports relative L2 as the headline metric (Tables 1/2)
but also implicitly uses qualitative comparison of predicted rho_react
vs ground truth (figs 8-10). This module exposes both the aggregate
relative L2 AND the per-sample array, because the ``resolution study''
plots need per-sample statistics (confidence bands over seeds).

TODO: extract from Neural_operator/FNO_1_Test.py evaluate_model / relative_l2_test_error.
"""

from __future__ import annotations

import numpy as np


def relative_l2_error(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """||y_pred - y_true||_2 / (||y_true||_2 + eps) (aggregate, no batching).

    TODO: also add ``per_sample_relative_l2`` that reduces over spatial
    dims only, returning shape [B]. That's the version the resolution
    study actually wants.
    """
    num = float(np.linalg.norm(y_pred - y_true))
    den = float(np.linalg.norm(y_true)) + eps
    return num / den


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    raise NotImplementedError("TODO: sklearn.metrics.mean_squared_error wrapper")


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    raise NotImplementedError("TODO: sklearn.metrics.mean_absolute_error wrapper")


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    raise NotImplementedError("TODO: sklearn.metrics.r2_score wrapper")
