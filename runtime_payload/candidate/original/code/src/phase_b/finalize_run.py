"""Derive immutable Phase B evaluation ledgers from an assembled run.

This module only reads saved ledgers.  It contains no estimator, tuning, or
model-selection code and must be run only after checkpoint assembly finishes.
"""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
import platform
from pathlib import Path

import numpy as np
import pandas as pd


PROTOCOL_HASH = "13fb577533c28acc88613fcf84549784e130bd658139ea7bbbfc7f77a3b8e773"
IDENTITY_COLUMNS = (
    "job_id",
    "scenario_id",
    "direction",
    "construction_role",
    "model_id",
    "estimator",
    "deleted_country",
)
CORE_FILES = (
    "execution_plan.csv",
    "matrix_manifest.csv",
    "preflight_status.json",
    "gate_results.json",
    "PREDICTIONS.csv",
    "SPLITS.csv",
    "PREPROCESSING.csv",
    "TUNING.csv",
    "FITS.json",
    "FAILURES.csv",
    "RUN_STATUS.json",
)
METRIC_COLUMNS = (*IDENTITY_COLUMNS, "mae", "mse", "rmse", "bias_pred_minus_observed", "mse_improvement_vs_m0")
CALIBRATION_COLUMNS = (*IDENTITY_COLUMNS, "scope", "country_count", "bias_pred_minus_observed", "intercept", "slope", "undefined_reason")
RANK_COLUMNS = (*IDENTITY_COLUMNS, "cell_id", "iso3", "window_start", "period_country_count", "target_rank", "prediction_rank", "normalized_absolute_displacement")
TAIL_COLUMNS = (*IDENTITY_COLUMNS, "window_start", "tail", "target_threshold", "prediction_threshold", "target_count", "prediction_count", "intersection_count", "target_recall", "jaccard", "small_group", "nonselective_tie")
PAIRED_COLUMNS = ("scenario_id", "direction", "construction_role", "deleted_country", "estimator", "M0_to_M3_complete", "planned_cell_count")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _normalize_deleted(value: object) -> str | None:
    if pd.isna(value) or str(value).strip() in ("", "None", "nan"):
        return None
    return str(value)


def _plan_hash(plan: pd.DataFrame) -> str:
    payload = []
    for row in plan.itertuples(index=False):
        payload.append(
            {
                "job_id": str(row.job_id),
                "scenario_id": str(row.scenario_id),
                "direction": str(row.direction),
                "construction_role": str(row.construction_role),
                "model_id": str(row.model_id),
                "estimator": str(row.estimator),
                "deleted_country": _normalize_deleted(row.deleted_country),
            }
        )
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _identity(frame: pd.DataFrame) -> dict[str, object]:
    row = frame.iloc[0]
    return {column: (_normalize_deleted(row[column]) if column == "deleted_country" else row[column]) for column in IDENTITY_COLUMNS}


def _country_weights(countries: pd.Series) -> np.ndarray:
    countries = countries.astype(str)
    counts = countries.value_counts()
    return countries.map(lambda country: 1 / (len(counts) * counts[country])).to_numpy(dtype=float)


def _score(frame: pd.DataFrame) -> dict[str, float]:
    weights = _country_weights(frame["iso3"])
    errors = frame["predicted"].to_numpy(dtype=float) - frame["observed"].to_numpy(dtype=float)
    mse = float(np.sum(weights * errors**2))
    return {
        "mae": float(np.sum(weights * np.abs(errors))),
        "mse": mse,
        "rmse": math.sqrt(mse),
        "bias_pred_minus_observed": float(np.sum(weights * errors)),
    }


def _calibration(frame: pd.DataFrame) -> dict[str, object]:
    weights = _country_weights(frame["iso3"])
    observed = frame["observed"].to_numpy(dtype=float)
    predicted = frame["predicted"].to_numpy(dtype=float)
    country_count = int(frame["iso3"].nunique())
    mean = float(np.sum(weights * predicted) / weights.sum())
    scale = math.sqrt(float(np.sum(weights * np.square(predicted - mean)) / weights.sum()))
    result = {
        "country_count": country_count,
        "bias_pred_minus_observed": float(np.sum(weights * (predicted - observed))),
        "intercept": np.nan,
        "slope": np.nan,
        "undefined_reason": np.nan,
    }
    if country_count < 3:
        result["undefined_reason"] = "fewer_than_three_countries"
    elif scale <= 1e-12 * max(1.0, abs(mean)):
        result["undefined_reason"] = "constant_prediction"
    else:
        design = np.column_stack([np.ones(len(predicted)), predicted])
        coefficients = np.linalg.lstsq(
            design * np.sqrt(weights)[:, None],
            observed * np.sqrt(weights),
            rcond=1e-12,
        )[0]
        result["intercept"], result["slope"] = map(float, coefficients)
    return result


def _tail(frame: pd.DataFrame, tail: str) -> dict[str, object]:
    observed = frame["observed"].to_numpy(dtype=float)
    predicted = frame["predicted"].to_numpy(dtype=float)
    percentile = 10 if tail == "low" else 90
    target_threshold = float(np.percentile(observed, percentile, method="linear"))
    prediction_threshold = float(np.percentile(predicted, percentile, method="linear"))
    if tail == "low":
        target, prediction = observed <= target_threshold, predicted <= prediction_threshold
    else:
        target, prediction = observed >= target_threshold, predicted >= prediction_threshold
    intersection = int(np.sum(target & prediction))
    union = int(np.sum(target | prediction))
    target_count = int(target.sum())
    prediction_count = int(prediction.sum())
    return {
        "tail": tail,
        "target_threshold": target_threshold,
        "prediction_threshold": prediction_threshold,
        "target_count": target_count,
        "prediction_count": prediction_count,
        "intersection_count": intersection,
        "target_recall": intersection / target_count if target_count else np.nan,
        "jaccard": intersection / union if union else np.nan,
        "small_group": target_count < 3 or prediction_count < 3,
        "nonselective_tie": target_count == len(frame) or prediction_count == len(frame),
    }


def _validate_core(run: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict, str]:
    if (run / "VALIDATION_STATUS.json").exists():
        raise ValueError("VALIDATION_STATUS.json must remain external to the immutable run directory")
    missing = [name for name in CORE_FILES if not (run / name).is_file()]
    if missing:
        raise ValueError(f"Missing assembled run files: {missing}")
    plan = _read_csv(run / "execution_plan.csv")
    predictions = _read_csv(run / "PREDICTIONS.csv")
    status = _read_json(run / "RUN_STATUS.json")
    preflight = _read_json(run / "preflight_status.json")
    gate = _read_json(run / "gate_results.json")
    if plan.empty or not plan["job_id"].is_unique:
        raise ValueError("Execution plan is empty or has duplicate job IDs")
    if gate.get("passed") is not True or preflight.get("gate_passed") is not True:
        raise ValueError("Approval/measurement gate is not passed")
    hashes = {gate.get("protocol_hash"), preflight.get("protocol_hash"), status.get("protocol_hash")}
    if hashes != {PROTOCOL_HASH}:
        raise ValueError(f"Protocol hash mismatch: {hashes}")
    current_plan_hash = _plan_hash(plan)
    if preflight.get("plan_hash") != current_plan_hash or status.get("plan_hash") != current_plan_hash:
        raise ValueError("Execution plan hash mismatch")
    summaries = status.get("job_summaries", [])
    summary_ids = [str(row.get("job_id")) for row in summaries]
    plan_ids = set(plan["job_id"].astype(str))
    if set(summary_ids) != plan_ids or len(summary_ids) != len(plan_ids) or status.get("jobs_complete") != len(plan_ids):
        raise ValueError("Run assembly is incomplete; every planned job must have one saved summary")
    if predictions.empty:
        raise ValueError("No saved predictions")
    required = {*IDENTITY_COLUMNS, "cell_id", "iso3", "window_start", "observed", "predicted"}
    if required - set(predictions.columns):
        raise ValueError(f"Prediction ledger lacks {sorted(required - set(predictions.columns))}")
    if not set(predictions["job_id"].astype(str)) <= plan_ids:
        raise ValueError("Prediction ledger contains a job absent from the execution plan")
    plan_identity = plan.set_index("job_id")
    for row in predictions.itertuples(index=False):
        expected = plan_identity.loc[str(row.job_id)]
        for column in ("scenario_id", "direction", "construction_role", "model_id", "estimator"):
            if str(getattr(row, column)) != str(expected[column]):
                raise ValueError(f"Prediction identity mismatch for {row.job_id}: {column}")
        if _normalize_deleted(row.deleted_country) != _normalize_deleted(expected["deleted_country"]):
            raise ValueError(f"Prediction identity mismatch for {row.job_id}: deleted_country")
    if predictions.duplicated(["job_id", "cell_id"]).any():
        raise ValueError("Prediction ledger contains duplicate job/cell rows")
    if not np.isfinite(predictions[["observed", "predicted"]].to_numpy(dtype=float)).all():
        raise ValueError("Saved observed/predicted values must be finite")
    return plan, predictions, status, current_plan_hash


def _metrics(plan: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    prediction_groups = {str(job_id): group for job_id, group in predictions.groupby("job_id", sort=False)}
    scores = {job_id: _score(group) for job_id, group in prediction_groups.items()}
    rows = []
    for row in plan.itertuples(index=False):
        identity = {
            column: (_normalize_deleted(getattr(row, column)) if column == "deleted_country" else getattr(row, column))
            for column in IDENTITY_COLUMNS
        }
        rows.append({**identity, **scores.get(str(row.job_id), {"mae": np.nan, "mse": np.nan, "rmse": np.nan, "bias_pred_minus_observed": np.nan})})
    metrics = pd.DataFrame(rows)
    baselines = {}
    group_columns = ["scenario_id", "direction", "construction_role", "deleted_country"]
    for _, row in metrics[metrics["model_id"].eq("M0")].iterrows():
        key = tuple(_normalize_deleted(row[column]) if column == "deleted_country" else row[column] for column in group_columns)
        baselines[key] = (row["job_id"], row["mse"])
    improvements = []
    for _, row in metrics.iterrows():
        key = tuple(_normalize_deleted(row[column]) if column == "deleted_country" else row[column] for column in group_columns)
        baseline_record = baselines.get(key)
        if baseline_record is None or pd.isna(row["mse"]):
            improvements.append(np.nan)
            continue
        baseline_job, baseline = baseline_record
        baseline_group = prediction_groups.get(str(baseline_job))
        model_group = prediction_groups.get(str(row["job_id"]))
        matched = baseline_group is not None and model_group is not None and set(baseline_group["cell_id"]) == set(model_group["cell_id"])
        improvements.append(np.nan if not matched or baseline == 0 or pd.isna(baseline) else float(1 - row["mse"] / baseline))
    metrics["mse_improvement_vs_m0"] = improvements
    return metrics.loc[:, METRIC_COLUMNS]


def _reporting_ledgers(plan: pd.DataFrame, predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    calibration_rows, rank_rows, tail_rows = [], [], []
    prediction_groups = {str(job_id): group for job_id, group in predictions.groupby("job_id", sort=False)}
    for plan_row in plan.itertuples(index=False):
        identity = {
            column: (_normalize_deleted(getattr(plan_row, column)) if column == "deleted_country" else getattr(plan_row, column))
            for column in IDENTITY_COLUMNS
        }
        group = prediction_groups.get(str(plan_row.job_id))
        if group is None:
            calibration_rows.append({**identity, "scope": "pooled_mixing_periods", "country_count": 0, "bias_pred_minus_observed": np.nan, "intercept": np.nan, "slope": np.nan, "undefined_reason": "no_valid_predictions"})
            continue
        calibration_rows.append({**identity, "scope": "pooled_mixing_periods", **_calibration(group)})
        for window, period in group.groupby("window_start", sort=True):
            if period["iso3"].nunique() >= 3:
                calibration_rows.append({**identity, "scope": f"period_{window}", **_calibration(period)})
            if identity["model_id"] == "M0":
                continue
            target_rank = period["observed"].rank(ascending=False, method="average").to_numpy(dtype=float)
            prediction_rank = period["predicted"].rank(ascending=False, method="average").to_numpy(dtype=float)
            count = len(period)
            for (_, row), left, right in zip(period.iterrows(), target_rank, prediction_rank):
                rank_rows.append(
                    {
                        **identity,
                        "cell_id": row["cell_id"],
                        "iso3": row["iso3"],
                        "window_start": window,
                        "period_country_count": count,
                        "target_rank": left,
                        "prediction_rank": right,
                        "normalized_absolute_displacement": np.nan if count < 2 else abs(left - right) / (count - 1),
                    }
                )
            for tail in ("low", "high"):
                tail_rows.append({**identity, "window_start": window, **_tail(period, tail)})
    return (
        pd.DataFrame(calibration_rows, columns=CALIBRATION_COLUMNS),
        pd.DataFrame(rank_rows, columns=RANK_COLUMNS),
        pd.DataFrame(tail_rows, columns=TAIL_COLUMNS),
    )


def _paired(plan: pd.DataFrame, status: dict) -> pd.DataFrame:
    plan = plan.copy()
    plan["deleted_country"] = plan["deleted_country"].map(_normalize_deleted)
    summaries = {str(row["job_id"]): row for row in status["job_summaries"]}
    rows = []
    group_columns = ["scenario_id", "direction", "construction_role", "deleted_country"]
    for key, jobs in plan.groupby(group_columns, dropna=False, sort=True):
        reference = jobs[jobs["model_id"].eq("M0")]
        for estimator in sorted(jobs.loc[~jobs["model_id"].eq("M0"), "estimator"].unique()):
            required = []
            if len(reference) == 1:
                required.append(str(reference.iloc[0]["job_id"]))
            for model in ("M1", "M2", "M3"):
                match = jobs[jobs["model_id"].eq(model) & jobs["estimator"].eq(estimator)]
                if len(match) == 1:
                    required.append(str(match.iloc[0]["job_id"]))
            records = [summaries[job_id] for job_id in required if job_id in summaries]
            counts = {row.get("planned_cell_count") for row in records}
            complete = len(required) == 4 and len(records) == 4 and all(row.get("complete") is True for row in records) and len(counts) == 1
            rows.append(
                {
                    "scenario_id": key[0],
                    "direction": key[1],
                    "construction_role": key[2],
                    "deleted_country": _normalize_deleted(key[3]),
                    "estimator": estimator,
                    "M0_to_M3_complete": complete,
                    "planned_cell_count": next(iter(counts)) if len(counts) == 1 else np.nan,
                }
            )
    return pd.DataFrame(rows, columns=PAIRED_COLUMNS)


def _package_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "not-installed"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finalize_run(run_directory: str | Path, run_id: str) -> dict[str, object]:
    """Write deterministic evaluation ledgers and an immutable top-level manifest."""
    run = Path(run_directory)
    plan, predictions, status, current_plan_hash = _validate_core(run)
    metrics = _metrics(plan, predictions)
    calibration, ranks, tails = _reporting_ledgers(plan, predictions)
    paired = _paired(plan, status)
    metrics.to_csv(run / "METRICS.csv", index=False, lineterminator="\n")
    calibration.to_csv(run / "CALIBRATION.csv", index=False, lineterminator="\n")
    ranks.to_csv(run / "RANK_LEDGER.csv", index=False, lineterminator="\n")
    tails.to_csv(run / "TAIL_GROUPS.csv", index=False, lineterminator="\n")
    paired.to_csv(run / "PAIRED_COMPLETENESS.csv", index=False, lineterminator="\n")
    settings = {
        "run_id": run_id,
        "protocol_hash": PROTOCOL_HASH,
        "plan_hash": current_plan_hash,
        "finalizer_one_thread": True,
        "environment_scope": "finalization_only",
        "fit_environment_provenance": "not_reconstructed_from_analytical_ledgers",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": _package_version("scipy"),
        "scikit_learn": _package_version("scikit-learn"),
        "job_count": len(plan),
        "finalizer": "phase_b.finalize_run",
        "finalizer_empirical_fitting_performed": False,
    }
    (run / "RUN_SETTINGS.json").write_text(json.dumps(settings, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    files = {path.name: _sha256(path) for path in sorted(run.iterdir()) if path.is_file() and path.name != "MANIFEST.json"}
    manifest = {"protocol_hash": PROTOCOL_HASH, "plan_hash": current_plan_hash, "files": files}
    (run / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "run_id": run_id,
        "protocol_hash": PROTOCOL_HASH,
        "plan_hash": current_plan_hash,
        "job_count": len(plan),
        "prediction_rows": len(predictions),
        "metric_rows": len(metrics),
        "paired_rows": len(paired),
        "manifest_file_count": len(files),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Finalize a completed assembled Phase B run without fitting models.")
    parser.add_argument("--run", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    summary = finalize_run(args.run, args.run_id)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
