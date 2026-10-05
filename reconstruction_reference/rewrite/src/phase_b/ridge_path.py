"""Batched weighted Ridge path, equivalent to Ridge(solver='svd')."""

from __future__ import annotations

import numpy as np


def weighted_ridge_path(
    x_train: np.ndarray,
    y_train: np.ndarray,
    weights: np.ndarray,
    x_test: np.ndarray,
    alphas: tuple[float, ...] | list[float],
) -> tuple[dict[float, np.ndarray], dict[float, np.ndarray], dict[float, float]]:
    x_train = np.asarray(x_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    x_test = np.asarray(x_test, dtype=np.float64)
    x_mean = np.sum(weights[:, None] * x_train, axis=0) / weights.sum()
    y_mean = float(np.sum(weights * y_train) / weights.sum())
    weighted_x = (x_train - x_mean) * np.sqrt(weights)[:, None]
    weighted_y = (y_train - y_mean) * np.sqrt(weights)
    u, singular, vt = np.linalg.svd(weighted_x, full_matrices=False)
    projection = u.T @ weighted_y
    predictions, coefficients, intercepts = {}, {}, {}
    for alpha_value in alphas:
        alpha = float(alpha_value)
        coefficient = vt.T @ ((singular / (singular**2 + alpha)) * projection)
        intercept = float(y_mean - x_mean @ coefficient)
        coefficients[alpha] = coefficient
        intercepts[alpha] = intercept
        predictions[alpha] = intercept + x_test @ coefficient
    return predictions, coefficients, intercepts

