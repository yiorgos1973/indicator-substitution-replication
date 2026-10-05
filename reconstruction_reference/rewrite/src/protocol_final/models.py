"""Exact Ridge, weighted OLS, and Lasso fitting conventions."""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso, Ridge
from threadpoolctl import threadpool_limits


RIDGE_GRID = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)
LASSO_GRID = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)


@dataclass(frozen=True)
class ModelResult:
    prediction: np.ndarray
    coefficients: np.ndarray
    intercept: float
    metadata: dict[str, object]


def fit_ridge(
    x_train: np.ndarray,
    y_train: np.ndarray,
    weights: np.ndarray,
    x_test: np.ndarray,
    alpha: float,
) -> ModelResult:
    estimator = Ridge(
        alpha=float(alpha),
        solver="svd",
        fit_intercept=True,
        positive=False,
    )
    with threadpool_limits(limits=1):
        estimator.fit(x_train, y_train, sample_weight=weights)
        prediction = estimator.predict(x_test)
    return ModelResult(
        np.asarray(prediction, dtype=np.float64),
        np.asarray(estimator.coef_, dtype=np.float64),
        float(estimator.intercept_),
        {
            "estimator": "ridge",
            "alpha": float(alpha),
            "solver": "svd",
            "fit_intercept": True,
            "positive": False,
            "objective": "sum(weight*residual^2)+alpha*sum(beta^2)",
            "attempts": 1,
            "converged": True,
        },
    )


def fit_weighted_ols(
    x_train: np.ndarray,
    y_train: np.ndarray,
    weights: np.ndarray,
    x_test: np.ndarray,
) -> ModelResult:
    augmented = np.column_stack([np.ones(len(x_train)), x_train])
    weighted_x = augmented * np.sqrt(weights)[:, None]
    weighted_y = y_train * np.sqrt(weights)
    with threadpool_limits(limits=1):
        coefficients, _, rank, singular_values = np.linalg.lstsq(
            weighted_x, weighted_y, rcond=1e-12
        )
    prediction = np.column_stack([np.ones(len(x_test)), x_test]) @ coefficients
    return ModelResult(
        np.asarray(prediction, dtype=np.float64),
        np.asarray(coefficients[1:], dtype=np.float64),
        float(coefficients[0]),
        {
            "estimator": "ordinary_weighted_least_squares",
            "rcond": 1e-12,
            "fit_intercept": True,
            "numerical_rank": int(rank),
            "column_count_with_intercept": int(weighted_x.shape[1]),
            "rank_deficient": bool(rank < weighted_x.shape[1]),
            "singular_values": [float(value) for value in singular_values],
            "objective": "sum(weight*residual^2)",
            "attempts": 1,
            "converged": True,
        },
    )


def _lasso_attempt(
    x_train: np.ndarray,
    y_train: np.ndarray,
    weights: np.ndarray,
    x_test: np.ndarray,
    alpha: float,
    max_iter: int,
) -> tuple[ModelResult, bool]:
    estimator = Lasso(
        alpha=float(alpha),
        fit_intercept=True,
        selection="cyclic",
        positive=False,
        warm_start=False,
        precompute=False,
        tol=1e-8,
        max_iter=max_iter,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        with threadpool_limits(limits=1):
            estimator.fit(x_train, y_train, sample_weight=weights)
            prediction = estimator.predict(x_test)
    warned = any(issubclass(item.category, ConvergenceWarning) for item in caught)
    result = ModelResult(
        np.asarray(prediction, dtype=np.float64),
        np.asarray(estimator.coef_, dtype=np.float64),
        float(estimator.intercept_),
        {
            "estimator": "lasso",
            "alpha": float(alpha),
            "fit_intercept": True,
            "selection": "cyclic",
            "positive": False,
            "warm_start": False,
            "precompute": False,
            "tolerance": 1e-8,
            "max_iter": max_iter,
            "iterations": int(estimator.n_iter_),
            "dual_gap": float(estimator.dual_gap_),
            "convergence_warning": warned,
            "objective": "sum(weight*residual^2)/(2*N)+alpha*sum(abs(beta))",
        },
    )
    return result, warned


def fit_lasso(
    x_train: np.ndarray,
    y_train: np.ndarray,
    weights: np.ndarray,
    x_test: np.ndarray,
    alpha: float,
) -> ModelResult:
    attempts = []
    first, warned = _lasso_attempt(
        x_train, y_train, weights, x_test, alpha, max_iter=100000
    )
    attempts.append(first.metadata)
    if warned:
        second, warned_again = _lasso_attempt(
            x_train, y_train, weights, x_test, alpha, max_iter=1000000
        )
        attempts.append(second.metadata)
        selected = second
        converged = not warned_again
    else:
        selected = first
        converged = True
    metadata = dict(selected.metadata)
    metadata.update(
        {
            "attempts": len(attempts),
            "attempt_records": attempts,
            "retry_used": len(attempts) == 2,
            "converged": converged,
        }
    )
    return ModelResult(
        selected.prediction,
        selected.coefficients,
        selected.intercept,
        metadata,
    )
