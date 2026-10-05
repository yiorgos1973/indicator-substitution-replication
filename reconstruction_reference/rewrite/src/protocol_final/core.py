"""Coverage, grouped splits, weights, and training-only transformations."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


SEED_LABEL = 20260930
CONSTANT_RELATIVE_TOLERANCE = 1e-12


class StructuralInfeasibility(RuntimeError):
    """The declared schema cannot be fitted on the supplied training subset."""


def classify_history(requested_years: int, observed_years: int) -> tuple[bool, int]:
    if requested_years <= 0:
        raise ValueError("requested_years must be positive")
    if observed_years < 0 or observed_years > requested_years:
        raise ValueError("observed_years must lie between zero and requested_years")
    minimum = (4 * requested_years + 4) // 5
    return 5 * observed_years >= 4 * requested_years, minimum


def usable_historical_mean(frame: pd.DataFrame) -> dict[str, object]:
    """Apply the integer 80% rule to one country-feature requested-year ledger."""

    required = {"requested_year", "value"}
    if not required.issubset(frame.columns):
        raise ValueError(f"History ledger requires {sorted(required)}")
    if frame["requested_year"].duplicated().any():
        raise ValueError("Duplicate country-feature-year rows require source review")
    requested = int(len(frame))
    observed_values = pd.to_numeric(frame["value"], errors="coerce").dropna()
    observed = int(len(observed_values))
    usable, minimum = classify_history(requested, observed)
    return {
        "requested_years": requested,
        "observed_years": observed,
        "minimum_observed_years": minimum,
        "usable": usable,
        "historical_mean": float(observed_values.mean()) if usable else np.nan,
    }


def inner_hash(iso3: str) -> str:
    value = f"NIQ-HLO-inner|{SEED_LABEL}|{iso3}".encode("utf-8")
    return sha256(value).hexdigest()


def deterministic_inner_folds(countries: Iterable[str]) -> dict[str, int]:
    unique = sorted({str(country) for country in countries})
    if len(unique) < 6:
        raise StructuralInfeasibility(
            "After outer holdout, at least six training countries are required"
        )
    ordered = sorted(unique, key=lambda country: (inner_hash(country), country))
    return {country: position % 5 for position, country in enumerate(ordered)}


def outer_countries(countries: Iterable[str]) -> list[str]:
    unique = sorted({str(country) for country in countries})
    if len(unique) < 7:
        raise StructuralInfeasibility(
            "A scenario needs at least seven countries for the fixed nested design"
        )
    return unique


def _country_row_counts(countries: Sequence[object]) -> tuple[np.ndarray, int, int]:
    values = np.asarray([str(country) for country in countries], dtype=object)
    unique, counts = np.unique(values, return_counts=True)
    mapping = dict(zip(unique, counts))
    row_counts = np.asarray([mapping[value] for value in values], dtype=float)
    return row_counts, len(values), len(unique)


def equal_country_fit_weights(countries: Sequence[object]) -> np.ndarray:
    row_counts, rows, country_count = _country_row_counts(countries)
    if rows == 0:
        raise ValueError("Cannot weight an empty fitting subset")
    return rows / (country_count * row_counts)


def equal_country_evaluation_weights(countries: Sequence[object]) -> np.ndarray:
    row_counts, rows, country_count = _country_row_counts(countries)
    if rows == 0:
        raise ValueError("Cannot weight an empty evaluation subset")
    return 1.0 / (country_count * row_counts)


def weighted_mean_and_scale(
    values: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float, bool]:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    total = float(weights.sum())
    mean = float(np.sum(weights * values) / total)
    variance = float(np.sum(weights * np.square(values - mean)) / total)
    variance = max(0.0, variance)
    scale = float(np.sqrt(variance))
    constant = scale <= CONSTANT_RELATIVE_TOLERANCE * max(1.0, abs(mean))
    return mean, scale, constant


@dataclass(frozen=True)
class FeatureTransform:
    feature: str
    imputable: bool
    median: float | None
    mean: float
    scale: float
    constant: bool


@dataclass(frozen=True)
class TrainingPreprocessor:
    features: tuple[str, ...]
    transforms: tuple[FeatureTransform, ...]
    target_mean: float
    target_scale: float
    target_constant: bool
    training_weights: np.ndarray

    def transform_features(self, frame: pd.DataFrame) -> np.ndarray:
        columns = []
        for state in self.transforms:
            values = pd.to_numeric(frame[state.feature], errors="coerce").to_numpy(dtype=np.float64)
            if state.imputable:
                values = np.where(np.isnan(values), state.median, values)
            elif np.isnan(values).any():
                raise StructuralInfeasibility(
                    f"Required non-imputable predictor is missing: {state.feature}"
                )
            if state.constant:
                columns.append(np.zeros(len(frame), dtype=np.float64))
            else:
                columns.append((values - state.mean) / state.scale)
        return np.column_stack(columns)

    def transform_target(self, values: Sequence[float]) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if self.target_constant:
            return np.zeros(len(array), dtype=np.float64)
        return (array - self.target_mean) / self.target_scale

    def inverse_target(self, standardized: Sequence[float]) -> np.ndarray:
        array = np.asarray(standardized, dtype=np.float64)
        if self.target_constant:
            return np.repeat(self.target_mean, len(array))
        return self.target_mean + self.target_scale * array

    def ledger_rows(self) -> list[dict[str, object]]:
        return [
            {
                "feature": state.feature,
                "imputable": state.imputable,
                "median": state.median,
                "weighted_mean": state.mean,
                "weighted_scale": state.scale,
                "constant": state.constant,
            }
            for state in self.transforms
        ]


def fit_preprocessor(
    training: pd.DataFrame,
    features: Sequence[str],
    context_features: Sequence[str],
    target: str,
    country: str = "iso3",
) -> TrainingPreprocessor:
    weights = equal_country_fit_weights(training[country].tolist())
    contexts = set(context_features)
    transforms = []
    for feature in features:
        raw = pd.to_numeric(training[feature], errors="coerce").to_numpy(dtype=np.float64)
        imputable = feature in contexts
        median: float | None = None
        if imputable:
            observed = raw[np.isfinite(raw)]
            if len(observed) == 0:
                raise StructuralInfeasibility(
                    f"Context feature has no observed training values: {feature}"
                )
            median = float(np.median(observed))
            completed = np.where(np.isnan(raw), median, raw)
        else:
            if not np.isfinite(raw).all():
                raise StructuralInfeasibility(
                    f"Required non-imputable predictor is missing: {feature}"
                )
            completed = raw
        mean, scale, constant = weighted_mean_and_scale(completed, weights)
        transforms.append(
            FeatureTransform(feature, imputable, median, mean, scale, constant)
        )

    target_values = pd.to_numeric(training[target], errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(target_values).all():
        raise StructuralInfeasibility("Training target is missing or non-finite")
    target_mean, target_scale, target_constant = weighted_mean_and_scale(
        target_values, weights
    )
    return TrainingPreprocessor(
        tuple(features),
        tuple(transforms),
        target_mean,
        target_scale,
        target_constant,
        weights,
    )
