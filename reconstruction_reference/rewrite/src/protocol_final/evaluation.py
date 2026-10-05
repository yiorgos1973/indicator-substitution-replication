"""Original-unit score, calibration, rank, ordering, and tail summaries."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, rankdata, spearmanr

from .core import (
    equal_country_evaluation_weights,
    weighted_mean_and_scale,
)


def score_metrics(
    observed: np.ndarray,
    predicted: np.ndarray,
    countries: np.ndarray,
    baseline_mse: float | None = None,
) -> dict[str, float | None]:
    observed = np.asarray(observed, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    weights = equal_country_evaluation_weights(countries)
    errors = predicted - observed
    mse = float(np.sum(weights * errors**2))
    return {
        "mae": float(np.sum(weights * np.abs(errors))),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "bias_pred_minus_observed": float(np.sum(weights * errors)),
        "mse_improvement_vs_m0": (
            None
            if baseline_mse is None or baseline_mse == 0
            else float(1.0 - mse / baseline_mse)
        ),
    }


def calibration_summary(
    observed: np.ndarray, predicted: np.ndarray, countries: np.ndarray
) -> dict[str, float | str | int | None]:
    observed = np.asarray(observed, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    weights = equal_country_evaluation_weights(countries)
    _, _, constant = weighted_mean_and_scale(predicted, weights)
    result: dict[str, float | str | int | None] = {
        "country_count": len(set(map(str, countries))),
        "bias_pred_minus_observed": float(np.sum(weights * (predicted - observed))),
        "intercept": None,
        "slope": None,
        "undefined_reason": None,
    }
    if result["country_count"] < 3:
        result["undefined_reason"] = "fewer_than_three_countries"
        return result
    if constant:
        result["undefined_reason"] = "constant_prediction"
        return result
    design = np.column_stack([np.ones(len(predicted)), predicted])
    coefficients = np.linalg.lstsq(
        design * np.sqrt(weights)[:, None], observed * np.sqrt(weights), rcond=1e-12
    )[0]
    result["intercept"], result["slope"] = map(float, coefficients)
    return result


def strict_pair_reversals(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float | int | None]:
    numerator = denominator = 0
    for left, right in itertools.combinations(range(len(observed)), 2):
        target_order = np.sign(observed[left] - observed[right])
        prediction_order = np.sign(predicted[left] - predicted[right])
        if target_order == 0 or prediction_order == 0:
            continue
        denominator += 1
        numerator += int(target_order != prediction_order)
    return {
        "reversal_numerator": numerator,
        "reversal_denominator": denominator,
        "reversal_fraction": None if denominator == 0 else numerator / denominator,
    }


def period_order_summary(frame: pd.DataFrame) -> dict[str, object]:
    observed = frame["observed"].to_numpy(dtype=np.float64)
    predicted = frame["predicted"].to_numpy(dtype=np.float64)
    count = len(frame)
    target_rank = rankdata(-observed, method="average")
    prediction_rank = rankdata(-predicted, method="average")
    rank_loss = None if count < 2 else float(np.mean(np.abs(target_rank - prediction_rank) / (count - 1)))
    variable = count > 1 and np.ptp(observed) > 0 and np.ptp(predicted) > 0
    spearman = spearmanr(observed, predicted).statistic if variable else np.nan
    kendall = kendalltau(observed, predicted, variant="b").statistic if variable else np.nan
    return {
        "country_count": count,
        "rank_loss": rank_loss,
        "spearman": None if not np.isfinite(spearman) else float(spearman),
        "kendall_tau_b": None if not np.isfinite(kendall) else float(kendall),
        **strict_pair_reversals(observed, predicted),
    }


def rank_loss_summary(frame: pd.DataFrame, period: str = "window_start") -> dict[str, object]:
    """Average normalized rank error within country, then across countries."""
    rows = []
    excluded_periods = []
    period_losses = []
    for period_value, group in frame.groupby(period, sort=True):
        if len(group) < 2:
            excluded_periods.append(period_value)
            continue
        target_rank = rankdata(-group["observed"].to_numpy(dtype=float), method="average")
        prediction_rank = rankdata(-group["predicted"].to_numpy(dtype=float), method="average")
        losses = np.abs(target_rank - prediction_rank) / (len(group) - 1)
        period_losses.append({"period": period_value, "country_count": len(group), "rank_loss": float(np.mean(losses))})
        rows.extend({"iso3": iso3, "period": period_value, "rank_loss": float(loss)} for iso3, loss in zip(group["iso3"], losses))
    if not rows:
        overall = None
        country_losses = []
    else:
        cell_losses = pd.DataFrame(rows)
        country_frame = cell_losses.groupby("iso3", sort=True)["rank_loss"].agg(["mean", "count"]).reset_index()
        overall = float(country_frame["mean"].mean())
        country_losses = country_frame.rename(columns={"mean": "rank_loss", "count": "period_count"}).to_dict("records")
    return {"overall_rank_loss": overall, "country_losses": country_losses, "period_losses": period_losses, "excluded_one_country_periods": excluded_periods}


def tail_group_summary(frame: pd.DataFrame, tail: str) -> dict[str, object]:
    observed = frame["observed"].to_numpy(dtype=np.float64)
    predicted = frame["predicted"].to_numpy(dtype=np.float64)
    quantile = 10 if tail == "low" else 90
    observed_threshold = float(np.percentile(observed, quantile, method="linear"))
    predicted_threshold = float(np.percentile(predicted, quantile, method="linear"))
    if tail == "low":
        target_mask, prediction_mask = observed <= observed_threshold, predicted <= predicted_threshold
    elif tail == "high":
        target_mask, prediction_mask = observed >= observed_threshold, predicted >= predicted_threshold
    else:
        raise ValueError("tail must be 'low' or 'high'")
    target_count = int(target_mask.sum())
    prediction_count = int(prediction_mask.sum())
    intersection = int(np.sum(target_mask & prediction_mask))
    union = int(np.sum(target_mask | prediction_mask))
    return {
        "tail": tail,
        "target_threshold": observed_threshold,
        "prediction_threshold": predicted_threshold,
        "target_count": target_count,
        "prediction_count": prediction_count,
        "intersection_count": intersection,
        "target_recall": intersection / target_count if target_count else None,
        "jaccard": intersection / union if union else None,
        "small_group": target_count < 3 or prediction_count < 3,
        "nonselective_tie": target_count == len(frame) or prediction_count == len(frame),
    }
