"""Detailed nested-LOCO execution for one finite-manifest job."""

from __future__ import annotations

import hashlib
from typing import Sequence

import numpy as np
import pandas as pd

from protocol_final.core import (
    StructuralInfeasibility,
    deterministic_inner_folds,
    equal_country_fit_weights,
    fit_preprocessor,
)
from protocol_final.evaluation import score_metrics
from protocol_final.models import LASSO_GRID, RIDGE_GRID, fit_lasso, fit_weighted_ols
from protocol_final.tuning import select_alpha

from .matrices import FEATURES
from .plan import Job
from .ridge_path import weighted_ridge_path


def model_features(model_id: str) -> tuple[str, ...]:
    return {
        "M0": (),
        "M1": ("source_indicator_score",),
        "M2": FEATURES,
        "M3": ("source_indicator_score",) + FEATURES,
    }[model_id]


def _training_id(countries: Sequence[str]) -> str:
    return hashlib.sha256("|".join(sorted(set(countries))).encode("utf-8")).hexdigest()


def _ridge_tune(training: pd.DataFrame, features: tuple[str, ...]) -> tuple[float | None, list[dict[str, object]], dict[str, int]]:
    allocation = deterministic_inner_folds(training["iso3"])
    predictions = {alpha: [] for alpha in RIDGE_GRID}
    rows = []
    for fold in range(5):
        validation = training[training["iso3"].map(allocation) == fold]
        inner_training = training[training["iso3"].map(allocation) != fold]
        state = fit_preprocessor(inner_training, features, FEATURES, "target_score")
        if state.target_constant:
            path = {alpha: np.repeat(state.target_mean, len(validation)) for alpha in RIDGE_GRID}
        else:
            standardized, _, _ = weighted_ridge_path(
                state.transform_features(inner_training), state.transform_target(inner_training["target_score"]),
                state.training_weights, state.transform_features(validation), RIDGE_GRID,
            )
            path = {alpha: state.inverse_target(values) for alpha, values in standardized.items()}
        for alpha in RIDGE_GRID:
            predictions[alpha].append(pd.DataFrame({"iso3": validation["iso3"].to_numpy(), "observed": validation["target_score"].to_numpy(), "predicted": path[alpha]}))
    for alpha in RIDGE_GRID:
        combined = pd.concat(predictions[alpha], ignore_index=True)
        score = score_metrics(combined["observed"], combined["predicted"], combined["iso3"])["mse"]
        rows.append({"alpha": alpha, "completed_folds": 5, "valid": True, "score": score, "failures": []})
    selected, _ = select_alpha(rows)
    return selected, rows, allocation


def _lasso_tune(training: pd.DataFrame, features: tuple[str, ...]) -> tuple[float, list[dict[str, object]], dict[str, int]]:
    allocation = deterministic_inner_folds(training["iso3"])
    candidate_rows = []
    for alpha in LASSO_GRID:
        predictions, failures = [], []
        for fold in range(5):
            validation = training[training["iso3"].map(allocation) == fold]
            inner_training = training[training["iso3"].map(allocation) != fold]
            try:
                state = fit_preprocessor(inner_training, features, FEATURES, "target_score")
                if state.target_constant:
                    predicted = np.repeat(state.target_mean, len(validation))
                else:
                    fitted = fit_lasso(state.transform_features(inner_training), state.transform_target(inner_training["target_score"]), state.training_weights, state.transform_features(validation), alpha)
                    if not fitted.metadata["converged"]:
                        raise StructuralInfeasibility("persistent Lasso nonconvergence")
                    predicted = state.inverse_target(fitted.prediction)
                predictions.append(pd.DataFrame({"iso3": validation["iso3"].to_numpy(), "observed": validation["target_score"].to_numpy(), "predicted": predicted}))
            except (StructuralInfeasibility, ValueError) as error:
                failures.append(f"fold_{fold}:{error}")
        valid = len(predictions) == 5 and not failures
        score = None
        if valid:
            combined = pd.concat(predictions, ignore_index=True)
            score = score_metrics(combined["observed"], combined["predicted"], combined["iso3"])["mse"]
        candidate_rows.append({"alpha": alpha, "completed_folds": len(predictions), "valid": valid, "score": score, "failures": failures})
    selected, _ = select_alpha(candidate_rows)
    return selected, candidate_rows, allocation


def run_job(matrix: pd.DataFrame, job: Job) -> dict[str, object]:
    frame = matrix.rename(columns={"target_score": "target_score"}).copy()
    frame = frame.sort_values(["iso3", "window_start", "window_end"], kind="mergesort")
    features = model_features(job.model_id)
    ledgers: dict[str, list[dict[str, object]]] = {name: [] for name in ("predictions", "splits", "preprocessing", "tuning", "fits", "failures")}
    planned_cells = set(frame["sample_cell_id"].astype(str))
    for outer_country in sorted(frame["iso3"].unique()):
        training = frame[frame["iso3"] != outer_country]
        test = frame[frame["iso3"] == outer_country]
        fit_id = f"{job.job_id}__outer_{outer_country}"
        try:
            selected_alpha = None
            allocation: dict[str, int] = {}
            tuning_rows: list[dict[str, object]] = []
            if job.model_id == "M0":
                weights = equal_country_fit_weights(training["iso3"])
                prediction = np.repeat(float(np.sum(weights * training["target_score"]) / weights.sum()), len(test))
                target_state = fit_preprocessor(training, (), (), "target_score")
                coefficients, metadata, preprocess_rows = [], {"estimator": "weighted_mean", "converged": True}, []
            else:
                if job.estimator == "ridge":
                    selected_alpha, tuning_rows, allocation = _ridge_tune(training, features)
                elif job.estimator == "lasso":
                    selected_alpha, tuning_rows, allocation = _lasso_tune(training, features)
                target_state = fit_preprocessor(training, features, FEATURES, "target_score")
                preprocess_rows = target_state.ledger_rows()
                if target_state.target_constant:
                    prediction = np.repeat(target_state.target_mean, len(test))
                    selected_alpha, coefficients = None, [0.0] * len(features)
                    metadata = {"estimator": job.estimator, "target_constant": True, "converged": True}
                elif job.estimator == "ridge":
                    path, coefficients_by_alpha, intercepts = weighted_ridge_path(target_state.transform_features(training), target_state.transform_target(training["target_score"]), target_state.training_weights, target_state.transform_features(test), [selected_alpha])
                    prediction = target_state.inverse_target(path[selected_alpha])
                    coefficients = coefficients_by_alpha[selected_alpha].tolist()
                    metadata = {"estimator": "ridge", "alpha": selected_alpha, "solver": "batched_weighted_svd_equivalent", "intercept": intercepts[selected_alpha], "converged": True}
                elif job.estimator == "ols":
                    fitted = fit_weighted_ols(target_state.transform_features(training), target_state.transform_target(training["target_score"]), target_state.training_weights, target_state.transform_features(test))
                    prediction = target_state.inverse_target(fitted.prediction)
                    coefficients, metadata = fitted.coefficients.tolist(), fitted.metadata
                elif job.estimator == "lasso":
                    fitted = fit_lasso(target_state.transform_features(training), target_state.transform_target(training["target_score"]), target_state.training_weights, target_state.transform_features(test), selected_alpha)
                    if not fitted.metadata["converged"]:
                        raise StructuralInfeasibility("persistent Lasso nonconvergence")
                    prediction = target_state.inverse_target(fitted.prediction)
                    coefficients, metadata = fitted.coefficients.tolist(), fitted.metadata
                else:
                    raise ValueError(f"Unknown estimator {job.estimator}")
            training_id = _training_id(training["iso3"].tolist())
            for country, fold in sorted(allocation.items()):
                ledgers["splits"].append({"fit_id": fit_id, "outer_country": outer_country, "training_country_set_id": training_id, "inner_country": country, "inner_fold": fold})
            for row in tuning_rows:
                ledgers["tuning"].append({"fit_id": fit_id, **row, "selected": row["alpha"] == selected_alpha})
            for row in preprocess_rows:
                ledgers["preprocessing"].append({"fit_id": fit_id, **row})
            ledgers["preprocessing"].append({"fit_id": fit_id, "feature": "__target__", "imputable": False, "median": None, "weighted_mean": target_state.target_mean, "weighted_scale": target_state.target_scale, "constant": target_state.target_constant})
            ledgers["fits"].append({"fit_id": fit_id, "outer_country": outer_country, "training_country_set_id": training_id, "selected_alpha": selected_alpha, "features": list(features), "coefficients_standardized": coefficients, "metadata": metadata})
            for (_, row), predicted in zip(test.iterrows(), prediction):
                scale = target_state.target_scale
                ledgers["predictions"].append({"fit_id": fit_id, "cell_id": row["sample_cell_id"], "iso3": row["iso3"], "window_start": row["window_start"], "window_end": row["window_end"], "observed": row["target_score"], "predicted": float(predicted), "raw_source_indicator": row["source_indicator_score"], "error_original": float(predicted - row["target_score"]), "error_scaled_by_outer_training_sd": None if target_state.target_constant else float((predicted - row["target_score"]) / scale), "status": "complete", "failure_reason": None})
        except (StructuralInfeasibility, ValueError, np.linalg.LinAlgError) as error:
            for _, row in test.iterrows():
                ledgers["failures"].append({"fit_id": fit_id, "outer_country": outer_country, "failure_type": type(error).__name__, "detail": str(error), "cell_id": row["sample_cell_id"], "iso3": row["iso3"], "window_start": row["window_start"], "window_end": row["window_end"], "observed": row["target_score"], "raw_source_indicator": row["source_indicator_score"]})
    predicted_cells = {str(row["cell_id"]) for row in ledgers["predictions"] if row["status"] == "complete"}
    complete = predicted_cells == planned_cells and not ledgers["failures"]
    metrics = None
    if complete:
        predictions = pd.DataFrame(ledgers["predictions"])
        metrics = score_metrics(predictions["observed"], predictions["predicted"], predictions["iso3"])
    return {"job_id": job.job_id, "complete": complete, "planned_cell_count": len(planned_cells), "predicted_cell_count": len(predicted_cells), "metrics": metrics, **ledgers}
