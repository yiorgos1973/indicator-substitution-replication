from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


PROTOCOL_HASH = "13fb577533c28acc88613fcf84549784e130bd658139ea7bbbfc7f77a3b8e773"
FEATURES = (
    "SE.PRM.ENRR",
    "SE.SEC.ENRR",
    "SE.XPD.TOTL.GD.ZS",
    "SH.DYN.MORT",
    "SP.DYN.TFRT.IN",
    "SP.URB.TOTL.IN.ZS",
    "IT.NET.USER.ZS",
)
RIDGE_GRID = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)
LASSO_GRID = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
IDENTITY_COLUMNS = (
    "job_id",
    "scenario_id",
    "direction",
    "construction_role",
    "model_id",
    "estimator",
    "deleted_country",
)
REQUIRED_FILES = {
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
    "PAIRED_COMPLETENESS.csv",
    "METRICS.csv",
    "CALIBRATION.csv",
    "RANK_LEDGER.csv",
    "TAIL_GROUPS.csv",
    "RUN_SETTINGS.json",
    "MANIFEST.json",
}


@dataclass
class ValidationResult:
    check_id: str
    passed: bool
    detail: str


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series
    return series.astype(str).str.lower().map({"true": True, "false": False})


def normalize_deleted(value):
    return None if pd.isna(value) or str(value).strip() in ("", "None", "nan") else str(value)


def job_record(row) -> dict:
    return {
        "job_id": str(row.job_id),
        "scenario_id": str(row.scenario_id),
        "direction": str(row.direction),
        "construction_role": str(row.construction_role),
        "model_id": str(row.model_id),
        "estimator": str(row.estimator),
        "deleted_country": normalize_deleted(row.deleted_country),
    }


def plan_hash(plan: pd.DataFrame) -> str:
    payload = [job_record(row) for row in plan.itertuples(index=False)]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def job_id_for(record: dict) -> str:
    return "__".join(
        [
            record["scenario_id"],
            record["direction"],
            record["construction_role"],
            record["model_id"],
            record["estimator"],
            record["deleted_country"] or "none",
        ]
    )


def cell_hash(cell_ids) -> str:
    payload = "\n".join(sorted(map(str, cell_ids)))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def training_set_id(countries) -> str:
    payload = "|".join(sorted(set(map(str, countries))))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def deterministic_folds(countries) -> dict[str, int]:
    unique = sorted(set(map(str, countries)))
    ordered = sorted(unique, key=lambda country: (hashlib.sha256(f"NIQ-HLO-inner|20260930|{country}".encode()).hexdigest(), country))
    return {country: position % 5 for position, country in enumerate(ordered)}


def country_weights(countries, fitting: bool) -> np.ndarray:
    values = np.asarray(list(map(str, countries)), dtype=object)
    unique, counts = np.unique(values, return_counts=True)
    count_map = dict(zip(unique, counts))
    row_counts = np.asarray([count_map[value] for value in values], dtype=float)
    if fitting:
        return len(values) / (len(unique) * row_counts)
    return 1 / (len(unique) * row_counts)


def weighted_target_stats(frame: pd.DataFrame) -> tuple[float, float, bool]:
    weights = country_weights(frame.iso3, fitting=True)
    values = frame.observed.to_numpy(float)
    mean = float(np.sum(weights * values) / weights.sum())
    variance = float(np.sum(weights * np.square(values - mean)) / weights.sum())
    scale = math.sqrt(max(0.0, variance))
    constant = scale <= 1e-12 * max(1.0, abs(mean))
    return mean, scale, constant


def score_metrics(frame: pd.DataFrame) -> dict[str, float]:
    weights = country_weights(frame.iso3, fitting=False)
    errors = frame.predicted.to_numpy(float) - frame.observed.to_numpy(float)
    mse = float(np.sum(weights * errors**2))
    return {
        "mae": float(np.sum(weights * np.abs(errors))),
        "mse": mse,
        "rmse": math.sqrt(mse),
        "bias_pred_minus_observed": float(np.sum(weights * errors)),
    }


def calibration(frame: pd.DataFrame) -> dict:
    weights = country_weights(frame.iso3, fitting=False)
    observed = frame.observed.to_numpy(float)
    predicted = frame.predicted.to_numpy(float)
    country_count = frame.iso3.nunique()
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
        coefficients = np.linalg.lstsq(design * np.sqrt(weights)[:, None], observed * np.sqrt(weights), rcond=1e-12)[0]
        result["intercept"], result["slope"] = map(float, coefficients)
    return result


def tail_summary(frame: pd.DataFrame, tail: str) -> dict:
    observed = frame.observed.to_numpy(float)
    predicted = frame.predicted.to_numpy(float)
    quantile = 10 if tail == "low" else 90
    observed_threshold = float(np.percentile(observed, quantile, method="linear"))
    predicted_threshold = float(np.percentile(predicted, quantile, method="linear"))
    if tail == "low":
        target_mask, prediction_mask = observed <= observed_threshold, predicted <= predicted_threshold
    else:
        target_mask, prediction_mask = observed >= observed_threshold, predicted >= predicted_threshold
    target_count = int(target_mask.sum())
    prediction_count = int(prediction_mask.sum())
    intersection = int((target_mask & prediction_mask).sum())
    union = int((target_mask | prediction_mask).sum())
    return {
        "tail": tail,
        "target_threshold": observed_threshold,
        "prediction_threshold": predicted_threshold,
        "target_count": target_count,
        "prediction_count": prediction_count,
        "intersection_count": intersection,
        "target_recall": intersection / target_count if target_count else np.nan,
        "jaccard": intersection / union if union else np.nan,
        "small_group": target_count < 3 or prediction_count < 3,
        "nonselective_tie": target_count == len(frame) or prediction_count == len(frame),
    }


def close(left, right, tolerance=1e-10) -> bool:
    if pd.isna(left) and pd.isna(right):
        return True
    try:
        return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)
    except (TypeError, ValueError):
        return str(left) == str(right)


class EmpiricalValidator:
    def __init__(self, run_directory: str | Path, strict_protocol: bool = True) -> None:
        self.run = Path(run_directory)
        self.strict_protocol = strict_protocol
        self.results: list[ValidationResult] = []
        self.tables: dict[str, pd.DataFrame] = {}
        self.jsons: dict[str, object] = {}

    def add(self, check_id: str, passed: bool, detail: str) -> None:
        self.results.append(ValidationResult(check_id, bool(passed), detail))

    def load(self) -> bool:
        present = {path.name for path in self.run.iterdir() if path.is_file()} if self.run.is_dir() else set()
        missing = REQUIRED_FILES - present
        if missing:
            self.add("REQUIRED_ARTIFACTS", False, f"missing={sorted(missing)}")
            return False
        self.add("REQUIRED_ARTIFACTS", True, f"files={len(REQUIRED_FILES)}")
        for name in (
            "execution_plan.csv", "matrix_manifest.csv", "PREDICTIONS.csv", "SPLITS.csv",
            "PREPROCESSING.csv", "TUNING.csv", "FAILURES.csv", "PAIRED_COMPLETENESS.csv",
            "METRICS.csv", "CALIBRATION.csv", "RANK_LEDGER.csv", "TAIL_GROUPS.csv",
        ):
            self.tables[name] = read_csv(self.run / name)
        for name in ("preflight_status.json", "gate_results.json", "FITS.json", "RUN_STATUS.json", "RUN_SETTINGS.json", "MANIFEST.json"):
            self.jsons[name] = json.loads((self.run / name).read_text(encoding="utf-8"))
        return True

    def validate_hashes(self) -> None:
        manifest = self.jsons["MANIFEST.json"]
        actual_files = {path.name for path in self.run.iterdir() if path.is_file() and path.name != "MANIFEST.json"}
        declared = set(manifest.get("files", {}))
        hashes_match = actual_files == declared and all(sha256(self.run / name) == digest for name, digest in manifest.get("files", {}).items())
        current_plan_hash = plan_hash(self.tables["execution_plan.csv"])
        bound_hashes = {
            str(self.jsons["preflight_status.json"].get("plan_hash")),
            str(self.jsons["RUN_STATUS.json"].get("plan_hash")),
            str(self.jsons["RUN_SETTINGS.json"].get("plan_hash")),
            str(manifest.get("plan_hash")),
        }
        protocol_hashes = {
            str(self.jsons["preflight_status.json"].get("protocol_hash")),
            str(self.jsons["RUN_STATUS.json"].get("protocol_hash")),
            str(self.jsons["RUN_SETTINGS.json"].get("protocol_hash")),
            str(manifest.get("protocol_hash")),
        }
        protocol_ok = protocol_hashes == {PROTOCOL_HASH} if self.strict_protocol else len(protocol_hashes) == 1
        self.add("HASH_INTEGRITY", hashes_match and bound_hashes == {current_plan_hash} and protocol_ok, f"files={len(actual_files)}, plan_hash={current_plan_hash}, protocol_hashes={sorted(protocol_hashes)}")

    def validate_plan(self) -> None:
        plan = self.tables["execution_plan.csv"].copy()
        plan["deleted_country"] = plan.deleted_country.map(normalize_deleted)
        unique = plan.job_id.is_unique
        reconstructed = plan.apply(lambda row: job_id_for(row.to_dict()), axis=1)
        ids_exact = reconstructed.eq(plan.job_id).all()
        valid_models = set(plan.model_id) <= {"M0", "M1", "M2", "M3"}
        valid_estimators = set(plan.estimator) <= {"reference", "ridge", "ols", "lasso"}
        structure = True
        detail = f"jobs={len(plan)}"
        if self.strict_protocol:
            matrices = self.tables["matrix_manifest.csv"]
            expected = []
            for direction in ("HLO_to_P", "P_to_HLO"):
                expected.append(("PRIMARY", direction, "primary_5y", "M0", "reference", None))
                for estimator in ("ridge", "ols", "lasso"):
                    for model in ("M1", "M2", "M3"):
                        expected.append(("PRIMARY", direction, "primary_5y", model, estimator, None))
            for row in matrices.itertuples(index=False):
                if row.scenario_id == "PRIMARY":
                    continue
                for model in ("M0", "M1", "M2", "M3"):
                    expected.append((row.scenario_id, row.target_direction, row.construction_role, model, "reference" if model == "M0" else "ridge", None))
            predictions = self.tables["PREDICTIONS.csv"]
            primary_countries = sorted(predictions.loc[(predictions.scenario_id == "PRIMARY") & (predictions.direction == "HLO_to_P") & (predictions.model_id == "M0"), "iso3"].unique())
            for deleted in primary_countries:
                for model in ("M0", "M1", "M2", "M3"):
                    expected.append(("PRIMARY_DELETE", "HLO_to_P", "primary_5y", model, "reference" if model == "M0" else "ridge", deleted))
            observed = set(zip(plan.scenario_id, plan.direction, plan.construction_role, plan.model_id, plan.estimator, plan.deleted_country))
            structure = observed == set(expected) and len(plan) == len(expected) == 252
            detail += f", expected={len(expected)}, primary_countries={len(primary_countries)}"
        else:
            groups = plan.groupby(["scenario_id", "direction", "construction_role", "deleted_country"], dropna=False)
            structure = all(set(group.model_id) == {"M0", "M1", "M2", "M3"} for _, group in groups)
        self.add("PLAN_COVERAGE", unique and ids_exact and valid_models and valid_estimators and structure, detail)

    def validate_completeness(self) -> None:
        plan_ids = set(self.tables["execution_plan.csv"].job_id)
        status = self.jsons["RUN_STATUS.json"]
        summaries = {row["job_id"]: row for row in status.get("job_summaries", [])}
        predictions = self.tables["PREDICTIONS.csv"]
        all_complete = set(summaries) == plan_ids and status.get("jobs_complete") == len(plan_ids)
        for job_id in plan_ids:
            row = summaries.get(job_id, {})
            observed_count = predictions.loc[predictions.job_id == job_id, "cell_id"].nunique()
            all_complete &= row.get("complete") is True and row.get("planned_cell_count") == row.get("predicted_cell_count") == observed_count
        failures_empty = self.tables["FAILURES.csv"].empty
        paired = self.tables["PAIRED_COMPLETENESS.csv"]
        paired_ok = not paired.empty and as_bool(paired.M0_to_M3_complete).all()
        self.add("JOB_COMPLETENESS", all_complete and failures_empty and paired_ok, f"status_jobs={len(summaries)}, plan_jobs={len(plan_ids)}, failures={len(self.tables['FAILURES.csv'])}, paired_rows={len(paired)}")

    def validate_predictions(self) -> None:
        plan = self.tables["execution_plan.csv"].set_index("job_id")
        predictions = self.tables["PREDICTIONS.csv"].copy()
        predictions["deleted_country"] = predictions.deleted_country.map(normalize_deleted)
        finite = np.isfinite(predictions[["observed", "predicted", "error_original"]].to_numpy(float)).all()
        errors = np.allclose(predictions.error_original, predictions.predicted - predictions.observed, atol=1e-12, rtol=1e-12)
        unique = not predictions.duplicated(["job_id", "cell_id"]).any()
        metadata = True
        fit_outer = True
        for row in predictions.itertuples(index=False):
            if row.job_id not in plan.index:
                metadata = False
                continue
            expected = plan.loc[row.job_id]
            metadata &= all(str(getattr(row, column)) == str(expected[column]) for column in ("scenario_id", "direction", "construction_role", "model_id", "estimator"))
            metadata &= normalize_deleted(row.deleted_country) == normalize_deleted(expected.deleted_country)
            fit_outer &= row.fit_id == f"{row.job_id}__outer_{row.iso3}"

        paired = True
        observed_consistent = True
        for _, group in predictions.groupby(["scenario_id", "direction", "construction_role", "deleted_country"], dropna=False):
            sets = group.groupby("job_id").cell_id.apply(set)
            paired &= len({frozenset(values) for values in sets}) == 1
            observed_counts = group.groupby("cell_id")["observed"].nunique(dropna=False)
            observed_consistent &= observed_counts.eq(1).all()

        matrix_hashes = True
        matrix_manifest = self.tables["matrix_manifest.csv"]
        for row in matrix_manifest.itertuples(index=False):
            part = predictions[(predictions.scenario_id == row.scenario_id) & (predictions.direction == row.target_direction) & (predictions.construction_role == row.construction_role)]
            if part.empty:
                matrix_hashes = False
                continue
            ids = part.groupby("job_id").cell_id.apply(list)
            matrix_hashes &= all(cell_hash(values) == row.cell_identity_sha256 for values in ids)
        self.add("PREDICTION_IDENTITIES", finite and errors and unique and metadata and fit_outer and paired and observed_consistent and matrix_hashes, f"rows={len(predictions)}, jobs={predictions.job_id.nunique()}, paired={paired}, matrix_hashes={matrix_hashes}")

    def validate_leakage(self) -> None:
        fits = self.jsons["FITS.json"]
        expected_features = {
            "M0": [],
            "M1": ["source_indicator_score"],
            "M2": list(FEATURES),
            "M3": ["source_indicator_score", *FEATURES],
        }
        fit_ok = True
        for fit in fits:
            job_id = fit["fit_id"].rsplit("__outer_", 1)[0]
            model = job_id.split("__")[3]
            fit_ok &= fit.get("features") == expected_features[model]
            fit_ok &= len(fit.get("coefficients_standardized", [])) == len(expected_features[model])
        preprocessing = self.tables["PREPROCESSING.csv"]
        allowed = {"__target__", "source_indicator_score", *FEATURES}
        preprocess_ok = set(preprocessing.feature) <= allowed
        prediction_columns_ok = not ({"p_score", "hlo_score", *FEATURES} & set(self.tables["PREDICTIONS.csv"].columns))
        self.add("NO_LEAKAGE", fit_ok and preprocess_ok and prediction_columns_ok, f"fits={len(fits)}, preprocessing_features={sorted(set(preprocessing.feature))}")

    def validate_splits_and_weights(self) -> None:
        predictions = self.tables["PREDICTIONS.csv"]
        splits = self.tables["SPLITS.csv"]
        fits = {row["fit_id"]: row for row in self.jsons["FITS.json"]}
        split_ok = True
        weight_ok = True
        scaled_error_ok = True
        for job_id, job_predictions in predictions.groupby("job_id"):
            estimator = str(job_predictions.estimator.iloc[0])
            model = str(job_predictions.model_id.iloc[0])
            for outer, test_rows in job_predictions.groupby("iso3"):
                fit_id = f"{job_id}__outer_{outer}"
                training = job_predictions[job_predictions.iso3 != outer]
                countries = sorted(training.iso3.unique())
                expected_id = training_set_id(countries)
                fit = fits.get(fit_id, {})
                weight_ok &= fit.get("training_country_set_id") == expected_id
                split_rows = splits[splits.fit_id == fit_id]
                if model != "M0" and estimator in ("ridge", "lasso"):
                    expected_folds = deterministic_folds(countries)
                    observed_folds = dict(zip(split_rows.inner_country.astype(str), split_rows.inner_fold.astype(int)))
                    split_ok &= observed_folds == expected_folds
                    split_ok &= split_rows.outer_country.astype(str).eq(str(outer)).all()
                    split_ok &= split_rows.training_country_set_id.eq(expected_id).all()
                else:
                    split_ok &= split_rows.empty
                target = self.tables["PREPROCESSING.csv"]
                target = target[(target.fit_id == fit_id) & (target.feature == "__target__")]
                if len(target) != 1:
                    weight_ok = False
                    continue
                mean, scale, constant = weighted_target_stats(training)
                state = target.iloc[0]
                weight_ok &= close(state.weighted_mean, mean) and close(state.weighted_scale, scale) and bool(state.constant) == constant
                if model == "M0":
                    weight_ok &= np.allclose(test_rows.predicted, mean, atol=1e-10, rtol=1e-10)
                if constant:
                    scaled_error_ok &= test_rows.error_scaled_by_outer_training_sd.isna().all()
                else:
                    scaled_error_ok &= np.allclose(test_rows.error_scaled_by_outer_training_sd, test_rows.error_original / scale, atol=1e-10, rtol=1e-10)
        self.add("WEIGHTS_AND_SPLITS", split_ok and weight_ok and scaled_error_ok, f"split_rows={len(splits)}, fits={len(fits)}, split_ok={split_ok}, weight_ok={weight_ok}")

    def validate_preprocessing(self) -> None:
        preprocessing = self.tables["PREPROCESSING.csv"].copy()
        plan = self.tables["execution_plan.csv"].set_index("job_id")
        schema_ok = True
        values_ok = True
        signatures: dict[tuple, set[tuple]] = {}
        for fit_id, group in preprocessing.groupby("fit_id"):
            job_id, outer = fit_id.rsplit("__outer_", 1)
            if job_id not in plan.index:
                schema_ok = False
                continue
            job = plan.loc[job_id]
            features = {"M0": [], "M1": ["source_indicator_score"], "M2": list(FEATURES), "M3": ["source_indicator_score", *FEATURES]}[job.model_id]
            schema_ok &= set(group.feature) == {"__target__", *features}
            for row in group.itertuples(index=False):
                values_ok &= np.isfinite(float(row.weighted_mean)) and np.isfinite(float(row.weighted_scale)) and float(row.weighted_scale) >= 0
                if row.feature in FEATURES:
                    values_ok &= bool(row.imputable) and np.isfinite(float(row.median))
                elif row.feature != "__target__":
                    values_ok &= not bool(row.imputable) and pd.isna(row.median)
                constant_expected = float(row.weighted_scale) <= 1e-12 * max(1.0, abs(float(row.weighted_mean)))
                values_ok &= bool(row.constant) == constant_expected
                key = (job.scenario_id, job.direction, job.construction_role, normalize_deleted(job.deleted_country), outer, row.feature)
                signature = (None if pd.isna(row.median) else float(row.median), float(row.weighted_mean), float(row.weighted_scale), bool(row.constant), bool(row.imputable))
                signatures.setdefault(key, set()).add(signature)
        consistent = all(len(values) == 1 for values in signatures.values())
        self.add("PREPROCESSING_CONVENTIONS", schema_ok and values_ok and consistent, f"rows={len(preprocessing)}, schemas={schema_ok}, values={values_ok}, cross_model_consistency={consistent}")

    def validate_tuning(self) -> None:
        tuning = self.tables["TUNING.csv"].copy()
        if not tuning.empty:
            tuning["selected"] = as_bool(tuning.selected)
            tuning["valid"] = as_bool(tuning.valid)
        fit_records = {row["fit_id"]: row for row in self.jsons["FITS.json"]}
        plan = self.tables["execution_plan.csv"].set_index("job_id")
        tuning_ok = True
        expected_tuned_fits = 0
        for fit_id, fit in fit_records.items():
            job_id = fit_id.rsplit("__outer_", 1)[0]
            if job_id not in plan.index:
                tuning_ok = False
                continue
            job = plan.loc[job_id]
            rows = tuning[tuning.fit_id == fit_id]
            if job.model_id == "M0" or job.estimator == "ols":
                tuning_ok &= rows.empty and fit.get("selected_alpha") is None
                continue
            expected_tuned_fits += 1
            grid = RIDGE_GRID if job.estimator == "ridge" else LASSO_GRID
            tuning_ok &= len(rows) == len(grid) and set(rows.alpha.astype(float)) == set(grid)
            valid_rows = rows[rows.valid]
            invalid_rows = rows[~rows.valid]
            tuning_ok &= not valid_rows.empty
            tuning_ok &= valid_rows.completed_folds.eq(5).all() and np.isfinite(valid_rows.score.astype(float)).all()
            if not invalid_rows.empty:
                tuning_ok &= invalid_rows.completed_folds.lt(5).all() and invalid_rows.score.isna().all()
            target = self.tables["PREPROCESSING.csv"]
            target = target[(target.fit_id == fit_id) & (target.feature == "__target__")]
            constant = len(target) == 1 and bool(target.iloc[0].constant)
            if constant:
                tuning_ok &= not rows.selected.any() and fit.get("selected_alpha") is None
            else:
                minimum = float(valid_rows.score.min())
                tolerance = 1e-10 + 1e-8 * max(1.0, abs(minimum))
                expected_alpha = float(valid_rows.loc[valid_rows.score <= minimum + tolerance, "alpha"].max())
                selected = rows[rows.selected]
                tuning_ok &= len(selected) == 1 and close(selected.iloc[0].alpha, expected_alpha) and close(fit.get("selected_alpha"), expected_alpha)
        known_fit_ids = set(fit_records)
        tuning_ok &= set(tuning.fit_id) <= known_fit_ids
        self.add("TUNING_CONVENTIONS", tuning_ok, f"rows={len(tuning)}, tuned_fits={expected_tuned_fits}")

    def validate_metrics(self) -> None:
        predictions = self.tables["PREDICTIONS.csv"]
        metric_table = self.tables["METRICS.csv"]
        metric_columns = {*IDENTITY_COLUMNS, "mae", "mse", "rmse", "bias_pred_minus_observed", "mse_improvement_vs_m0"}
        metric_schema_ok = set(metric_table.columns) == metric_columns
        metric_schema_ok &= metric_table.job_id.is_unique and set(metric_table.job_id) == set(predictions.job_id)
        metrics = metric_table.set_index("job_id")
        calibration_ledger = self.tables["CALIBRATION.csv"]
        rank_ledger = self.tables["RANK_LEDGER.csv"]
        tails = self.tables["TAIL_GROUPS.csv"]
        score_ok = metric_schema_ok
        expected_scores = {job_id: score_metrics(group) for job_id, group in predictions.groupby("job_id")}
        baselines = {}
        for job_id, group in predictions[predictions.model_id == "M0"].groupby("job_id"):
            row = group.iloc[0]
            key = (row.scenario_id, row.direction, row.construction_role, normalize_deleted(row.deleted_country))
            baselines[key] = expected_scores[job_id]["mse"]
        calibration_ok = rank_ok = tail_ok = True
        for job_id, group in predictions.groupby("job_id"):
            expected = expected_scores[job_id]
            if job_id not in metrics.index:
                score_ok = False
            else:
                row = metrics.loc[job_id]
                score_ok &= all(close(row[key], value) for key, value in expected.items())
                identity = group.iloc[0]
                baseline_key = (identity.scenario_id, identity.direction, identity.construction_role, normalize_deleted(identity.deleted_country))
                baseline = baselines.get(baseline_key)
                expected_improvement = np.nan if baseline is None or baseline == 0 else 1 - expected["mse"] / baseline
                score_ok &= close(row["mse_improvement_vs_m0"], expected_improvement)
            scopes = [("pooled_mixing_periods", group)] + [(f"period_{window}", period) for window, period in group.groupby("window_start") if period.iso3.nunique() >= 3]
            for scope, frame in scopes:
                actual = calibration_ledger[(calibration_ledger.job_id == job_id) & (calibration_ledger.scope == scope)]
                expected_calibration = calibration(frame)
                calibration_ok &= len(actual) == 1
                if len(actual) == 1:
                    calibration_ok &= all(close(actual.iloc[0][key], value) for key, value in expected_calibration.items())
            if group.model_id.iloc[0] == "M0":
                rank_ok &= rank_ledger[rank_ledger.job_id == job_id].empty
                tail_ok &= tails[tails.job_id == job_id].empty
                continue
            for window, period in group.groupby("window_start"):
                target_rank = period.observed.rank(ascending=False, method="average").to_numpy()
                prediction_rank = period.predicted.rank(ascending=False, method="average").to_numpy()
                count = len(period)
                expected_rank = pd.DataFrame(
                    {
                        "cell_id": period.cell_id.to_numpy(),
                        "target_rank": target_rank,
                        "prediction_rank": prediction_rank,
                        "normalized_absolute_displacement": np.nan if count < 2 else np.abs(target_rank - prediction_rank) / (count - 1),
                    }
                ).set_index("cell_id")
                actual_rank = rank_ledger[(rank_ledger.job_id == job_id) & (rank_ledger.window_start == window)].set_index("cell_id")
                rank_ok &= set(actual_rank.index) == set(expected_rank.index)
                if set(actual_rank.index) == set(expected_rank.index):
                    rank_ok &= all(close(actual_rank.loc[cell, column], expected_rank.loc[cell, column]) for cell in expected_rank.index for column in expected_rank.columns)
                for tail in ("low", "high"):
                    actual_tail = tails[(tails.job_id == job_id) & (tails.window_start == window) & (tails["tail"] == tail)]
                    expected_tail = tail_summary(period, tail)
                    tail_ok &= len(actual_tail) == 1
                    if len(actual_tail) == 1:
                        tail_ok &= all(close(actual_tail.iloc[0][key], value) for key, value in expected_tail.items())
        self.add("RECOMPUTED_METRICS", score_ok and calibration_ok and rank_ok and tail_ok, f"jobs={predictions.job_id.nunique()}, score={score_ok}, calibration={calibration_ok}, rank={rank_ok}, tails={tail_ok}")

    def validate_deletions(self) -> None:
        plan = self.tables["execution_plan.csv"].copy()
        plan["deleted_country"] = plan.deleted_country.map(normalize_deleted)
        deletion_plan = plan[plan.scenario_id == "PRIMARY_DELETE"]
        predictions = self.tables["PREDICTIONS.csv"]
        primary = predictions[(predictions.scenario_id == "PRIMARY") & (predictions.direction == "HLO_to_P") & (predictions.model_id == "M0")]
        countries = set(primary.iso3)
        declared = set(deletion_plan.deleted_country.dropna())
        coverage_ok = declared == countries if self.strict_protocol else bool(declared) and declared <= countries
        coverage_ok &= deletion_plan.groupby("deleted_country").model_id.apply(lambda values: set(values) == {"M0", "M1", "M2", "M3"}).all()
        for deleted, jobs in deletion_plan.groupby("deleted_country"):
            parts = [set(predictions.loc[predictions.job_id == job_id, "cell_id"]) for job_id in jobs.job_id]
            expected = set(primary.loc[primary.iso3 != deleted, "cell_id"])
            coverage_ok &= all(values == expected for values in parts)
            coverage_ok &= not predictions[predictions.job_id.isin(jobs.job_id)].iso3.eq(deleted).any()
        self.add("DELETION_COVERAGE", coverage_ok, f"declared_deletions={len(declared)}, primary_countries={len(countries)}")

    def validate(self) -> pd.DataFrame:
        if not self.load():
            return self.frame()
        self.validate_hashes()
        self.validate_plan()
        self.validate_completeness()
        self.validate_predictions()
        self.validate_leakage()
        self.validate_splits_and_weights()
        self.validate_preprocessing()
        self.validate_tuning()
        self.validate_metrics()
        self.validate_deletions()
        return self.frame()

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame([{"check_id": row.check_id, "status": "PASS" if row.passed else "FAIL", "detail": row.detail} for row in self.results])


def validate_run(run_directory: str | Path, strict_protocol: bool = True) -> pd.DataFrame:
    return EmpiricalValidator(run_directory, strict_protocol).validate()


def write_validation_outputs(
    run_directory: str | Path,
    output_directory: str | Path,
    strict_protocol: bool = True,
) -> tuple[pd.DataFrame, dict[str, object]]:
    run = Path(run_directory)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    results = validate_run(run, strict_protocol=strict_protocol)
    results.to_csv(output / "validation_results.csv", index=False)
    passed = int(results.status.eq("PASS").sum())
    failed = int(results.status.eq("FAIL").sum())
    report = [
        "# Independent Phase B empirical-ledger validation",
        "",
        f"Result: **{passed}/{len(results)} PASS; {failed} FAIL**.",
        "",
        "This validator checks completeness and numerical integrity only. It does not interpret substantive empirical performance.",
        "",
        "| Check | Status | Detail |",
        "|---|---|---|",
    ]
    report.extend(f"| {row.check_id} | {row.status} | {str(row.detail).replace('|', '/')} |" for row in results.itertuples())
    (output / "VALIDATION_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    manifest = run / "MANIFEST.json"
    validation_status = {
        "protocol_hash": PROTOCOL_HASH,
        "status": "PASS" if failed == 0 else "FAIL",
        "material_failures": failed,
        "validated_manifest_hash": sha256(manifest) if manifest.is_file() else None,
        "completed_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    (output / "VALIDATION_STATUS.json").write_text(
        json.dumps(validation_status, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return results, validation_status


def main() -> int:
    parser = argparse.ArgumentParser(description="Independently validate saved Phase B empirical ledgers.")
    parser.add_argument("--run-directory", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--fixture-mode", action="store_true", help="Validate a bounded synthetic fixture rather than the full 252-job protocol plan.")
    args = parser.parse_args()
    results, _ = write_validation_outputs(
        args.run_directory,
        args.output_directory,
        strict_protocol=not args.fixture_mode,
    )
    passed = int(results.status.eq("PASS").sum())
    failed = int(results.status.eq("FAIL").sum())
    print(f"Empirical ledger validation: {passed}/{len(results)} PASS")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
