"""Nested country validation and deterministic alpha selection."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .core import (
    StructuralInfeasibility,
    deterministic_inner_folds,
    equal_country_fit_weights,
    fit_preprocessor,
)
from .evaluation import score_metrics
from .models import LASSO_GRID, RIDGE_GRID, fit_lasso, fit_ridge, fit_weighted_ols


def select_alpha(candidate_scores: Sequence[dict[str, object]]) -> tuple[float, float]:
    valid = [row for row in candidate_scores if row.get("valid") and row.get("completed_folds") == 5]
    if not valid:
        raise StructuralInfeasibility("No alpha completed all five inner folds")
    minimum = min(float(row["score"]) for row in valid)
    tolerance = 1e-10 + 1e-8 * max(1.0, abs(minimum))
    tied = [row for row in valid if float(row["score"]) <= minimum + tolerance]
    return max(float(row["alpha"]) for row in tied), minimum


def _fit_once(
    training: pd.DataFrame,
    test: pd.DataFrame,
    features: Sequence[str],
    context_features: Sequence[str],
    target: str,
    estimator: str,
    alpha: float | None,
) -> tuple[np.ndarray, dict[str, object], list[dict[str, object]]]:
    state = fit_preprocessor(training, features, context_features, target)
    if state.target_constant:
        metadata = {"estimator": estimator, "alpha": None, "target_constant": True, "converged": True}
        return state.inverse_target(np.zeros(len(test))), metadata, state.ledger_rows()
    x_train = state.transform_features(training)
    x_test = state.transform_features(test)
    y_train = state.transform_target(training[target])
    weights = state.training_weights
    if estimator == "ridge":
        fitted = fit_ridge(x_train, y_train, weights, x_test, float(alpha))
    elif estimator == "lasso":
        fitted = fit_lasso(x_train, y_train, weights, x_test, float(alpha))
    elif estimator == "ols":
        fitted = fit_weighted_ols(x_train, y_train, weights, x_test)
    else:
        raise ValueError(f"Unknown estimator: {estimator}")
    if not fitted.metadata["converged"] or not np.isfinite(fitted.prediction).all():
        raise StructuralInfeasibility(f"{estimator} did not produce a valid finite fit")
    metadata = dict(fitted.metadata)
    metadata["target_constant"] = False
    return state.inverse_target(fitted.prediction), metadata, state.ledger_rows()


def tune_alpha(
    outer_training: pd.DataFrame,
    features: Sequence[str],
    context_features: Sequence[str],
    target: str,
    estimator: str,
) -> tuple[float | None, list[dict[str, object]], dict[str, int]]:
    allocation = deterministic_inner_folds(outer_training["iso3"])
    grid = RIDGE_GRID if estimator == "ridge" else LASSO_GRID
    rows: list[dict[str, object]] = []
    for alpha in grid:
        fold_predictions = []
        failures = []
        for fold in range(5):
            validation = outer_training[outer_training["iso3"].map(allocation) == fold]
            training = outer_training[outer_training["iso3"].map(allocation) != fold]
            try:
                prediction, metadata, _ = _fit_once(
                    training, validation, features, context_features, target, estimator, alpha
                )
                fold_predictions.append(
                    pd.DataFrame({"iso3": validation["iso3"].to_numpy(), "observed": validation[target].to_numpy(), "predicted": prediction})
                )
                if not metadata["converged"]:
                    failures.append(f"fold_{fold}_nonconvergence")
            except (StructuralInfeasibility, ValueError) as error:
                failures.append(f"fold_{fold}:{error}")
        completed = len(fold_predictions)
        valid = completed == 5 and not failures
        score = None
        if valid:
            combined = pd.concat(fold_predictions, ignore_index=True)
            score = score_metrics(combined["observed"], combined["predicted"], combined["iso3"])["mse"]
        rows.append({"alpha": alpha, "completed_folds": completed, "valid": valid, "score": score, "failures": failures})
    selected, _ = select_alpha(rows)
    return selected, rows, allocation


def fit_outer_configuration(
    frame: pd.DataFrame,
    outer_country: str,
    model_id: str,
    estimator: str,
    features: Sequence[str],
    context_features: Sequence[str],
    target: str = "target",
) -> dict[str, object]:
    ordered = frame.sort_values(["iso3", "window_start", "window_end"], kind="mergesort")
    training = ordered[ordered["iso3"] != outer_country]
    test = ordered[ordered["iso3"] == outer_country]
    if test.empty:
        raise ValueError("Outer country is absent")
    if model_id == "M0":
        weights = equal_country_fit_weights(training["iso3"])
        value = float(np.sum(weights * training[target].to_numpy(dtype=float)) / weights.sum())
        return {"prediction": np.repeat(value, len(test)), "selected_alpha": None, "tuning": [], "allocation": {}, "metadata": {"estimator": "weighted_mean", "converged": True}, "preprocessing": []}
    selected = None
    tuning: list[dict[str, object]] = []
    allocation: dict[str, int] = {}
    if estimator in {"ridge", "lasso"}:
        selected, tuning, allocation = tune_alpha(training, features, context_features, target, estimator)
    prediction, metadata, preprocessing = _fit_once(training, test, features, context_features, target, estimator, selected)
    if metadata.get("target_constant"):
        selected = None
    return {"prediction": prediction, "selected_alpha": selected, "tuning": tuning, "allocation": allocation, "metadata": metadata, "preprocessing": preprocessing}
