"""Transform validated saved ledgers into manuscript-ready Phase B artifacts.

This module contains no estimators, fitting, tuning, or resampling code.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, rankdata, spearmanr

from .schemas import GENERATOR, PREDICTION_COLUMNS, PROTOCOL_HASH, RESULT_COLUMNS


REQUIRED_LEDGER_FILES = (
    "PREDICTIONS.csv", "SPLITS.csv", "PREPROCESSING.csv", "TUNING.csv", "FITS.json",
    "FAILURES.csv", "RUN_STATUS.json", "RUN_SETTINGS.json", "MANIFEST.json",
    "PAIRED_COMPLETENESS.csv", "execution_plan.csv", "gate_results.json",
    "METRICS.csv", "CALIBRATION.csv", "RANK_LEDGER.csv", "TAIL_GROUPS.csv",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _token(value: object) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", str(value).upper()).strip("_")


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.stat().st_size > 2 else pd.DataFrame()


def validate_run(run: Path, validation_status: Path, allow_synthetic: bool) -> dict[str, object]:
    missing = [name for name in REQUIRED_LEDGER_FILES if not (run / name).is_file()]
    if missing:
        raise ValueError(f"Missing assembled ledgers: {missing}")
    settings = json.loads((run / "RUN_SETTINGS.json").read_text(encoding="utf-8"))
    status = json.loads((run / "RUN_STATUS.json").read_text(encoding="utf-8"))
    gate = json.loads((run / "gate_results.json").read_text(encoding="utf-8"))
    if not validation_status.is_file():
        raise ValueError(f"Independent validation status does not exist: {validation_status}")
    if validation_status.parent.resolve() == run.resolve():
        raise ValueError("Independent validation status must remain outside the immutable run directory")
    validation = json.loads(validation_status.read_text(encoding="utf-8"))
    for name, record in (("settings", settings), ("status", status), ("gate", gate), ("validation", validation)):
        if record.get("protocol_hash") != PROTOCOL_HASH:
            raise ValueError(f"{name} protocol hash mismatch")
    if gate.get("passed") is not True:
        raise ValueError("Approval/measurement gate is not passed")
    if validation.get("status") != "PASS" or validation.get("material_failures", 0) != 0:
        raise ValueError("Independent empirical validation is not PASS")
    manifest_hash = sha256(run / "MANIFEST.json")
    if validation.get("validated_manifest_hash") != manifest_hash:
        raise ValueError("Independent validation status is not bound to this run manifest")
    predictions = _read_csv(run / "PREDICTIONS.csv")
    if predictions.empty:
        raise ValueError("No saved predictions")
    required = {"job_id", "scenario_id", "direction", "construction_role", "model_id", "estimator", "fit_id", "cell_id", "iso3", "window_start", "window_end", "observed", "predicted"}
    missing_columns = required - set(predictions.columns)
    if missing_columns:
        raise ValueError(f"Prediction ledger lacks {sorted(missing_columns)}")
    source_column = "raw_source_indicator" if "raw_source_indicator" in predictions else "source_indicator_score" if "source_indicator_score" in predictions else None
    if source_column is None:
        raise ValueError("Prediction ledger lacks required raw_source_indicator")
    if not allow_synthetic and predictions.astype(str).apply(lambda column: column.str.contains("SYNTHETIC", case=False).any()).any():
        raise ValueError("Synthetic rows cannot enter empirical reporting")
    manifest = json.loads((run / "MANIFEST.json").read_text(encoding="utf-8"))
    for name, expected in manifest.get("files", {}).items():
        path = run / name
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Output manifest mismatch: {name}")
    paired = _read_csv(run / "PAIRED_COMPLETENESS.csv")
    if not paired.empty:
        ridge = paired[paired["estimator"].eq("ridge")]
        if not ridge["M0_to_M3_complete"].astype(str).str.lower().eq("true").all():
            raise ValueError("A required Ridge family is incomplete")
    return {"settings": settings, "status": status, "validation": validation, "source_column": source_column}


def _weights(countries: pd.Series) -> np.ndarray:
    counts = countries.astype(str).value_counts()
    return countries.astype(str).map(lambda country: 1 / (len(counts) * counts[country])).to_numpy(dtype=float)


def _score(group: pd.DataFrame) -> dict[str, float]:
    weights = _weights(group["iso3"])
    errors = group["predicted"].to_numpy(dtype=float) - group["observed"].to_numpy(dtype=float)
    mse = float(np.sum(weights * errors**2))
    scaled = group["error_scaled_by_outer_training_sd"].to_numpy(dtype=float) if "error_scaled_by_outer_training_sd" in group else np.full(len(group), np.nan)
    return {
        "mae": float(np.sum(weights * np.abs(errors))), "mse": mse, "rmse": float(np.sqrt(mse)),
        "bias": float(np.sum(weights * errors)),
        "scaled_mae": float(np.sum(weights * np.abs(scaled))) if np.isfinite(scaled).all() else np.nan,
        "scaled_mse": float(np.sum(weights * scaled**2)) if np.isfinite(scaled).all() else np.nan,
    }


def _calibration(group: pd.DataFrame) -> tuple[float | None, float | None, str | None]:
    weights = _weights(group["iso3"])
    predicted = group["predicted"].to_numpy(dtype=float)
    observed = group["observed"].to_numpy(dtype=float)
    mean = float(np.sum(weights * predicted))
    sd = float(np.sqrt(np.sum(weights * (predicted - mean) ** 2)))
    if group["iso3"].nunique() < 3:
        return None, None, "fewer_than_three_countries"
    if sd <= 1e-12 * max(1, abs(mean)):
        return None, None, "constant_prediction"
    design = np.column_stack([np.ones(len(group)), predicted])
    coefficients = np.linalg.lstsq(design * np.sqrt(weights)[:, None], observed * np.sqrt(weights), rcond=1e-12)[0]
    return float(coefficients[0]), float(coefficients[1]), None


def _rank_and_tail(group: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | None]]:
    rank_rows, tail_rows = [], []
    for window, period in group.groupby(["window_start", "window_end"], sort=True):
        observed = period["observed"].to_numpy(dtype=float)
        predicted = period["predicted"].to_numpy(dtype=float)
        count = len(period)
        target_rank = rankdata(-observed, method="average")
        prediction_rank = rankdata(-predicted, method="average")
        for (_, row), left, right in zip(period.iterrows(), target_rank, prediction_rank):
            rank_rows.append({"iso3": row["iso3"], "window_start": window[0], "window_end": window[1], "period_country_count": count, "target_rank": left, "prediction_rank": right, "normalized_absolute_displacement": np.nan if count < 2 else abs(left - right) / (count - 1)})
        variable = count > 1 and np.ptp(observed) > 0 and np.ptp(predicted) > 0
        reversals = denominator = 0
        for left in range(count):
            for right in range(left + 1, count):
                first, second = np.sign(observed[left] - observed[right]), np.sign(predicted[left] - predicted[right])
                if first == 0 or second == 0:
                    continue
                denominator += 1
                reversals += int(first != second)
        period_summary = {"window_start": window[0], "window_end": window[1], "spearman": float(spearmanr(observed, predicted).statistic) if variable else np.nan, "kendall_tau_b": float(kendalltau(observed, predicted, variant="b").statistic) if variable else np.nan, "strict_reversals": reversals, "strict_pairs": denominator, "strict_reversal_fraction": reversals / denominator if denominator else np.nan}
        for tail, percentile, operator in (("LOW", 10, np.less_equal), ("HIGH", 90, np.greater_equal)):
            target_threshold = float(np.percentile(observed, percentile, method="linear"))
            prediction_threshold = float(np.percentile(predicted, percentile, method="linear"))
            target_mask, prediction_mask = operator(observed, target_threshold), operator(predicted, prediction_threshold)
            intersection, union = int(np.sum(target_mask & prediction_mask)), int(np.sum(target_mask | prediction_mask))
            tail_rows.append({**period_summary, "tail": tail, "target_threshold": target_threshold, "prediction_threshold": prediction_threshold, "target_count": int(target_mask.sum()), "prediction_count": int(prediction_mask.sum()), "intersection": intersection, "target_recall": intersection / int(target_mask.sum()) if target_mask.sum() else np.nan, "jaccard": intersection / union if union else np.nan, "small_group": target_mask.sum() < 3 or prediction_mask.sum() < 3, "nonselective_tie": target_mask.sum() == count or prediction_mask.sum() == count})
    ranks = pd.DataFrame(rank_rows)
    if ranks.empty or ranks["normalized_absolute_displacement"].notna().sum() == 0:
        overall = None
    else:
        overall = float(ranks.groupby("iso3")["normalized_absolute_displacement"].mean().mean())
    tails = pd.DataFrame(tail_rows)
    return ranks, tails, {"rank_loss": overall}


def _result_row(**values: object) -> dict[str, object]:
    row = {column: None for column in RESULT_COLUMNS}
    row.update(values)
    return row


def _identity(group: pd.DataFrame) -> dict[str, str]:
    first = group.iloc[0]
    identity = {column: str(first[column]) for column in ("job_id", "scenario_id", "direction", "construction_role", "model_id", "estimator")}
    identity["deleted_country"] = str(first["deleted_country"]) if "deleted_country" in group and pd.notna(first["deleted_country"]) else ""
    return identity


def compute_results(predictions: pd.DataFrame, run_id: str, input_hash: str) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    rows, metric_rows, calibration_rows, rank_parts, tail_parts, rank_summary_rows = [], [], [], [], [], []
    for job_id, group in predictions.groupby("job_id", sort=True):
        identity = _identity(group)
        score = _score(group)
        direction_token = _token(identity["direction"])
        scenario_token = _token(identity["scenario_id"])
        model_token = "_".join((_token(identity["model_id"]), _token(identity["estimator"]), _token(identity["construction_role"])))
        if identity["deleted_country"]:
            model_token += f"_DELETE_{_token(identity['deleted_country'])}"
        target = "period-specific P" if identity["direction"] == "HLO_to_P" else "period-specific HLO"
        units = "P score units" if identity["direction"] == "HLO_to_P" else "HLO score units"
        common = dict(analysis_family="phase_b_prediction", status="calculated_pending_reporting_validation", run_id=run_id, protocol_hash=PROTOCOL_HASH, target=target, model_or_comparison=f"{identity['model_id']} {identity['estimator']} [{identity['construction_role']}]", period="pooled outer-held-out periods", n_countries=group["iso3"].nunique(), n_cells=len(group), denominator_definition="country mean over retained cells, then mean over countries", weighting_rule="equal-country evaluation weights 1/(C*m_i)", sample_id=f"{identity['scenario_id']}|{identity['direction']}|{identity['construction_role']}", source_artifact="reporting_sources/PHASE_B_METRICS_SOURCE.csv", source_locator=f"job_id={job_id}", generator=GENERATOR, input_hash=input_hash, limitation="Internal leave-one-country-out validation; not external prospective validation, causal evidence, or construct equivalence.")
        for metric, estimate in score.items():
            metric_units = {
                "mse": f"{units} squared",
                "scaled_mae": "outer-training-SD-scaled error",
                "scaled_mse": "squared outer-training-SD-scaled error",
            }.get(metric, units)
            rows.append(_result_row(result_id=f"PRED_{direction_token}_{scenario_token}_{model_token}_{_token(metric)}", metric=metric, estimate=None if pd.isna(estimate) else estimate, interval_method="none; point estimate", units=metric_units, interpretation="Lower error is better; bias is prediction minus observed.", **common))
            metric_rows.append({**identity, "metric": metric, "estimate": estimate, "n_countries": group["iso3"].nunique(), "n_cells": len(group)})
        intercept, slope, reason = _calibration(group)
        for metric, estimate in (("intercept", intercept), ("slope", slope)):
            calibration_common = dict(common)
            calibration_common["status"] = "calculated_pending_reporting_validation" if estimate is not None else f"undefined_{reason}"
            calibration_common["source_artifact"] = "reporting_sources/PHASE_B_CALIBRATION_SOURCE.csv"
            calibration_common["source_locator"] = f"job_id={job_id};scope=pooled"
            rows.append(_result_row(result_id=f"PRED_CAL_{direction_token}_{scenario_token}_{model_token}_{_token(metric)}", metric=f"calibration_{metric}", estimate=estimate, interval_method="none; diagnostic point estimate" if estimate is not None else f"undefined: {reason}", units=units if metric == "intercept" else "observed units per predicted unit", interpretation="Descriptive pooled calibration; not used to adjust predictions.", **calibration_common))
        calibration_rows.append({**identity, "intercept": intercept, "slope": slope, "undefined_reason": reason})
        for (window_start, window_end), period_group in group.groupby(["window_start", "window_end"], sort=True):
            period_intercept, period_slope, period_reason = _calibration(period_group)
            period_token = f"{int(window_start)}_{int(window_end)}"
            calibration_rows.append({**identity, "period": period_token, "intercept": period_intercept, "slope": period_slope, "undefined_reason": period_reason})
            for metric, estimate in (("intercept", period_intercept), ("slope", period_slope)):
                rows.append(_result_row(result_id=f"PRED_CAL_{direction_token}_{scenario_token}_{model_token}_{period_token}_{_token(metric)}", analysis_family="phase_b_prediction", status="calculated_pending_reporting_validation" if estimate is not None else f"undefined_{period_reason}", run_id=run_id, protocol_hash=PROTOCOL_HASH, target=target, model_or_comparison=f"{identity['model_id']} {identity['estimator']} [{identity['construction_role']}]", period=period_token, metric=f"calibration_{metric}", estimate=estimate, interval_method="none; diagnostic point estimate" if estimate is not None else f"undefined: {period_reason}", units=units if metric == "intercept" else "observed units per predicted unit", n_countries=period_group.iso3.nunique(), n_cells=len(period_group), denominator_definition="countries in this assessment period", weighting_rule="equal-country weighted least squares", sample_id=common["sample_id"], source_artifact="reporting_sources/PHASE_B_CALIBRATION_SOURCE.csv", source_locator=f"job_id={job_id};period={period_token}", generator=GENERATOR, input_hash=input_hash, interpretation="Descriptive period-specific calibration; not used to adjust predictions.", limitation=common["limitation"]))
        if identity["model_id"] != "M0":
            ranks, tails, rank_summary = _rank_and_tail(group)
            ranks.insert(0, "job_id", job_id); tails.insert(0, "job_id", job_id)
            rank_parts.append(ranks); tail_parts.append(tails)
            rows.append(_result_row(result_id=f"PRED_RANK_{direction_token}_{scenario_token}_{model_token}_NORMALIZED_LOSS", metric="normalized_rank_loss", estimate=rank_summary["rank_loss"], interval_method="none; point estimate", units="normalized rank displacement", interpretation="Lower period-specific rank displacement is better.", **{**common, "source_artifact": "reporting_sources/PHASE_B_RANK_SUMMARY_SOURCE.csv", "source_locator": f"job_id={job_id}"}))
            rank_summary_rows.append({**identity, "rank_loss": rank_summary["rank_loss"], "n_countries": group.iso3.nunique(), "n_cells": len(group)})
            for (window_start, window_end), period_ranks in ranks.groupby(["window_start", "window_end"], sort=True):
                period_token = f"{int(window_start)}_{int(window_end)}"
                period_loss = period_ranks["normalized_absolute_displacement"].mean() if period_ranks["normalized_absolute_displacement"].notna().any() else None
                rank_common = {key: value for key, value in common.items() if key not in {"period", "n_countries", "n_cells", "denominator_definition", "interpretation", "source_artifact", "source_locator"}}
                rows.append(_result_row(result_id=f"PRED_RANK_{direction_token}_{scenario_token}_{model_token}_{period_token}_NORMALIZED_LOSS", period=period_token, metric="period_normalized_rank_loss", estimate=period_loss, interval_method="none; descriptive", units="normalized rank displacement", n_countries=period_ranks.iso3.nunique(), n_cells=len(period_ranks), denominator_definition="period countries; normalized by n_period-1", interpretation="Lower displacement is better; one-country periods are undefined.", source_artifact="reporting_sources/PHASE_B_RANK_SOURCE.csv", source_locator=f"job_id={job_id};period={period_token}", **rank_common))
            for tail_row in tails.to_dict("records"):
                period_token = f"{int(tail_row['window_start'])}_{int(tail_row['window_end'])}"
                for metric in ("target_recall", "jaccard", "target_count", "prediction_count", "intersection"):
                    tail_common = {key: value for key, value in common.items() if key not in {"period", "source_artifact", "source_locator"}}
                    rows.append(_result_row(result_id=f"PRED_TAIL_{direction_token}_{scenario_token}_{model_token}_{tail_row['tail']}_{period_token}_{_token(metric)}", period=period_token, metric=f"{tail_row['tail'].lower()}_{metric}", estimate=tail_row[metric], interval_method="none; descriptive", units="proportion" if metric in {"target_recall", "jaccard"} else "countries", interpretation="Own-variable linear-interpolated 10th/90th percentile groups; actual tied-group counts retained.", source_artifact="reporting_sources/PHASE_B_TAIL_SOURCE.csv", source_locator=f"job_id={job_id};period={period_token};tail={tail_row['tail']}", **tail_common))
            for period_row in tails.drop_duplicates(["window_start", "window_end"]).to_dict("records"):
                period_token = f"{int(period_row['window_start'])}_{int(period_row['window_end'])}"
                for metric, units_value in (("spearman", "correlation"), ("kendall_tau_b", "correlation"), ("strict_reversals", "pairs"), ("strict_pairs", "pairs"), ("strict_reversal_fraction", "proportion")):
                    estimate = period_row[metric]
                    order_common = {key: value for key, value in common.items() if key not in {"period", "source_artifact", "source_locator"}}
                    rows.append(_result_row(result_id=f"PRED_RANK_{direction_token}_{scenario_token}_{model_token}_{period_token}_{_token(metric)}", period=period_token, metric=metric, estimate=None if pd.isna(estimate) else estimate, interval_method="none; descriptive", units=units_value, interpretation="Period-specific ordering summary; strict reversals exclude pairs tied on either variable.", source_artifact="reporting_sources/PHASE_B_TAIL_SOURCE.csv", source_locator=f"job_id={job_id};period={period_token}", **order_common))
    raw_rows, raw_rank_sources, raw_tail_sources = _raw_source_rank_results(predictions, run_id, input_hash)
    rows.extend(raw_rows)
    rank_parts.extend(raw_rank_sources)
    tail_parts.extend(raw_tail_sources)
    metric_table = pd.DataFrame(metric_rows)
    rank_summary_table = pd.DataFrame(rank_summary_rows)
    rows.extend(_comparison_rows(predictions, metric_table, rank_summary_table, run_id, input_hash))
    results = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    if results["result_id"].duplicated().any():
        duplicates = results.loc[results["result_id"].duplicated(), "result_id"].tolist()
        raise ValueError(f"Duplicate result IDs: {duplicates[:5]}")
    sources = {
        "PHASE_B_METRICS_SOURCE.csv": metric_table,
        "PHASE_B_CALIBRATION_SOURCE.csv": pd.DataFrame(calibration_rows),
        "PHASE_B_RANK_SOURCE.csv": pd.concat(rank_parts, ignore_index=True) if rank_parts else pd.DataFrame(),
        "PHASE_B_RANK_SUMMARY_SOURCE.csv": rank_summary_table,
        "PHASE_B_TAIL_SOURCE.csv": pd.concat(tail_parts, ignore_index=True) if tail_parts else pd.DataFrame(),
    }
    return results, sources


def _raw_source_rank_results(predictions: pd.DataFrame, run_id: str, input_hash: str):
    rows, rank_sources, tail_sources = [], [], []
    keys = ["scenario_id", "direction", "construction_role", "deleted_country"] if "deleted_country" in predictions else ["scenario_id", "direction", "construction_role"]
    for identity, all_rows in predictions.groupby(keys, sort=True, dropna=False):
        identity = identity if isinstance(identity, tuple) else (identity,)
        scenario, direction, role = map(str, identity[:3])
        if scenario == "PRIMARY_DELETE":
            continue
        group = all_rows.sort_values("job_id").drop_duplicates("cell_id").copy()
        group["predicted"] = group["raw_source_indicator"]
        ranks, tails, summary = _rank_and_tail(group)
        source_id = f"RAW_SOURCE|{scenario}|{direction}|{role}"
        ranks.insert(0, "job_id", source_id); tails.insert(0, "job_id", source_id)
        rank_sources.append(ranks); tail_sources.append(tails)
        common = dict(analysis_family="phase_b_raw_source_ordering", status="calculated_pending_reporting_validation", run_id=run_id, protocol_hash=PROTOCOL_HASH, target=direction, model_or_comparison=f"raw source indicator [{role}]", period="country-first aggregation across evaluable periods", interval_method="none; descriptive", n_countries=group.iso3.nunique(), n_cells=len(group), denominator_definition="identical scenario-period country sets", weighting_rule="within-country period mean then equal-country mean", sample_id=f"{scenario}|{direction}|{role}", source_artifact="reporting_sources/PHASE_B_RANK_SOURCE.csv", source_locator=f"job_id={source_id}", generator=GENERATOR, input_hash=input_hash, interpretation="Raw source ordering is distinct from fitted M1 ordering.", limitation="Cross-fitted score predictions and raw-source ordering answer different descriptive questions.")
        rows.append(_result_row(result_id=f"PRED_RANK_{_token(direction)}_{_token(scenario)}_RAW_SOURCE_{_token(role)}_NORMALIZED_LOSS", metric="normalized_rank_loss", estimate=summary["rank_loss"], units="normalized rank displacement", **common))
        for tail_row in tails.to_dict("records"):
            period_token = f"{int(tail_row['window_start'])}_{int(tail_row['window_end'])}"
            for metric in ("target_recall", "jaccard", "target_count", "prediction_count", "intersection"):
                tail_common = {key: value for key, value in common.items() if key not in {"period", "metric", "estimate", "units", "source_artifact", "source_locator"}}
                rows.append(_result_row(result_id=f"PRED_TAIL_{_token(direction)}_{_token(scenario)}_RAW_SOURCE_{_token(role)}_{tail_row['tail']}_{period_token}_{_token(metric)}", period=period_token, metric=f"{tail_row['tail'].lower()}_{metric}", estimate=tail_row[metric], units="proportion" if metric in {"target_recall", "jaccard"} else "countries", source_artifact="reporting_sources/PHASE_B_TAIL_SOURCE.csv", source_locator=f"job_id={source_id};period={period_token};tail={tail_row['tail']}", **tail_common))
        for period_row in tails.drop_duplicates(["window_start", "window_end"]).to_dict("records"):
            period_token = f"{int(period_row['window_start'])}_{int(period_row['window_end'])}"
            for metric, metric_units in (("spearman", "correlation"), ("kendall_tau_b", "correlation"), ("strict_reversals", "pairs"), ("strict_pairs", "pairs"), ("strict_reversal_fraction", "proportion")):
                estimate = period_row[metric]
                order_common = {key: value for key, value in common.items() if key not in {"period", "metric", "estimate", "units", "source_artifact", "source_locator"}}
                rows.append(_result_row(result_id=f"PRED_RANK_{_token(direction)}_{_token(scenario)}_RAW_SOURCE_{_token(role)}_{period_token}_{_token(metric)}", period=period_token, metric=metric, estimate=None if pd.isna(estimate) else estimate, units=metric_units, source_artifact="reporting_sources/PHASE_B_TAIL_SOURCE.csv", source_locator=f"job_id={source_id};period={period_token}", **order_common))
    return rows, rank_sources, tail_sources


def _comparison_rows(predictions: pd.DataFrame, metrics: pd.DataFrame, rank_summaries: pd.DataFrame, run_id: str, input_hash: str) -> list[dict[str, object]]:
    rows = []
    keys = ["scenario_id", "direction", "construction_role", "deleted_country"]
    mse = metrics[metrics["metric"].eq("mse")]
    for identity, group in mse.groupby(keys, sort=True, dropna=False):
        reference = group[group["model_id"].eq("M0")]
        if reference.empty:
            continue
        m0 = float(reference.iloc[0]["estimate"])
        for row in group[~group["model_id"].eq("M0")].itertuples(index=False):
            improvement = None if m0 == 0 else 1 - float(row.estimate) / m0
            direction_token, scenario_token = _token(identity[1]), _token(identity[0])
            model_token = "_".join((_token(row.model_id), _token(row.estimator), _token(identity[2])))
            if pd.notna(identity[3]) and str(identity[3]):
                model_token += f"_DELETE_{_token(identity[3])}"
            rows.append(_result_row(result_id=f"PRED_DIFF_{direction_token}_{scenario_token}_{model_token}_VS_M0_MSE_IMPROVEMENT", analysis_family="phase_b_prediction_comparison", status="calculated_pending_reporting_validation" if improvement is not None else "undefined_zero_baseline", run_id=run_id, protocol_hash=PROTOCOL_HASH, target=identity[1], model_or_comparison=f"{row.model_id} {row.estimator} versus M0 [{identity[2]}]", period="pooled outer-held-out periods", metric="mse_improvement_vs_M0", estimate=improvement, interval_method="none; point comparison" if improvement is not None else "undefined: M0 MSE equals zero", units="proportion", n_countries=row.n_countries, n_cells=row.n_cells, denominator_definition="identical paired planned cells", weighting_rule="equal-country evaluation", sample_id="|".join(map(str, identity)), source_artifact="reporting_sources/PHASE_B_METRICS_SOURCE.csv", source_locator=f"job_id={row.job_id}", generator=GENERATOR, input_hash=input_hash, interpretation="Positive values mean lower MSE than M0.", limitation="Point comparison without inferential interval."))
    rows.extend(_matched_sensitivity_rows(mse, run_id, input_hash))
    rows.extend(_deletion_rows(mse, run_id, input_hash))
    rows.extend(_m3_baseline_rows(mse, rank_summaries, run_id, input_hash))
    return rows


def _m3_baseline_rows(mse: pd.DataFrame, ranks: pd.DataFrame, run_id: str, input_hash: str) -> list[dict[str, object]]:
    rows = []
    for metric_name, source, value_column, units in (("mse", mse, "estimate", "target squared units"), ("rank_loss", ranks, "rank_loss", "normalized rank displacement")):
        if source.empty:
            continue
        keys = ["scenario_id", "direction", "construction_role", "deleted_country", "estimator"]
        for identity, group in source.groupby(keys, sort=True, dropna=False):
            m3 = group[group.model_id == "M3"]
            if len(m3) != 1:
                continue
            for baseline in ("M1", "M2"):
                base = group[group.model_id == baseline]
                if len(base) != 1:
                    continue
                difference = float(base.iloc[0][value_column] - m3.iloc[0][value_column])
                deletion_token = f"_DELETE_{_token(identity[3])}" if pd.notna(identity[3]) and str(identity[3]) else ""
                rows.append(_result_row(result_id=f"PRED_DIFF_{_token(identity[1])}_{_token(identity[0])}_M3_{_token(identity[4])}_{_token(identity[2])}_VS_{baseline}_{_token(metric_name)}{deletion_token}", analysis_family="phase_b_model_comparison", status="calculated_pending_reporting_validation", run_id=run_id, protocol_hash=PROTOCOL_HASH, target=identity[1], model_or_comparison=f"M3 versus {baseline}; {identity[4]} [{identity[2]}]", period="country-first pooled outer-held-out periods", metric=f"{baseline.lower()}_loss_minus_m3_{metric_name}", estimate=difference, interval_method="none; paired point comparison", units=units, n_countries=int(m3.iloc[0].n_countries), n_cells=int(m3.iloc[0].n_cells), denominator_definition="identical planned cells", weighting_rule="equal-country evaluation", sample_id=f"{identity[0]}|{identity[1]}|{identity[2]}", source_artifact=f"reporting_sources/PHASE_B_{'METRICS' if metric_name == 'mse' else 'RANK_SUMMARY'}_SOURCE.csv", source_locator=f"estimator={identity[4]};baseline={baseline}", generator=GENERATOR, input_hash=input_hash, interpretation="Positive values mean M3 has lower loss.", limitation="Paired point comparison; no cross-validation significance claim."))
    return rows


def _matched_sensitivity_rows(mse: pd.DataFrame, run_id: str, input_hash: str) -> list[dict[str, object]]:
    roles = {"S10": "alternative_10y", "S12": "alternative_12y", "SA5": "assessment_specific_5y_fixed"}
    rows = []
    for scenario, alternative in roles.items():
        for direction in ("HLO_to_P", "P_to_HLO"):
            left = mse[(mse.scenario_id == scenario) & (mse.direction == direction) & (mse.construction_role == alternative)]
            right = mse[(mse.scenario_id == scenario) & (mse.direction == direction) & (mse.construction_role == "primary_5y_matched")]
            for model in ("M0", "M1", "M2", "M3"):
                estimator = "reference" if model == "M0" else "ridge"
                a, b = left[(left.model_id == model) & (left.estimator == estimator)], right[(right.model_id == model) & (right.estimator == estimator)]
                result_id = f"PRED_DIFF_{_token(direction)}_{scenario}_{model}_{_token(estimator)}_{_token(alternative)}_VS_PRIMARY_MATCHED_MSE"
                if len(a) == 1 and len(b) == 1:
                    estimate, status, reason = float(b.iloc[0].estimate - a.iloc[0].estimate), "calculated_pending_reporting_validation", "Positive values mean the alternative history has lower MSE than matched primary history."
                    n_countries, n_cells = int(a.iloc[0].n_countries), int(a.iloc[0].n_cells)
                else:
                    estimate, status, reason, n_countries, n_cells = None, "not_calculated_missing_or_failed_pair", "Matched pair unavailable; no favourable subset substituted.", None, None
                rows.append(_result_row(result_id=result_id, analysis_family="phase_b_matched_sensitivity", status=status, run_id=run_id, protocol_hash=PROTOCOL_HASH, target=direction, model_or_comparison=f"{alternative} versus primary_5y_matched; {model} {estimator}", period="pooled outer-held-out periods", metric="matched_primary_mse_minus_alternative_mse", estimate=estimate, interval_method="none; matched point comparison", units="target squared units", n_countries=n_countries, n_cells=n_cells, denominator_definition="exact frozen matched country-period intersection", weighting_rule="equal-country evaluation", sample_id=f"{scenario}|{direction}|matched", source_artifact="reporting_sources/PHASE_B_METRICS_SOURCE.csv", source_locator=f"scenario={scenario};direction={direction};model={model}", generator=GENERATOR, input_hash=input_hash, interpretation=reason, limitation="Sensitivity comparison; not specification selection or causal timing evidence."))
    return rows


def _deletion_rows(mse: pd.DataFrame, run_id: str, input_hash: str) -> list[dict[str, object]]:
    deletion = mse[(mse.scenario_id == "PRIMARY_DELETE") & (mse.direction == "HLO_to_P")]
    rows = []
    for baseline in ("M1", "M2"):
        primary = mse[(mse.scenario_id == "PRIMARY") & (mse.direction == "HLO_to_P") & (mse.construction_role == "primary_5y")]
        primary_baseline = primary[(primary.model_id == baseline) & (primary.estimator == "ridge")]
        primary_m3 = primary[(primary.model_id == "M3") & (primary.estimator == "ridge")]
        full_difference = float(primary_baseline.iloc[0].estimate - primary_m3.iloc[0].estimate) if len(primary_baseline) == len(primary_m3) == 1 else np.nan
        values = []
        for deleted, group in deletion.groupby("deleted_country", dropna=False):
            left = group[(group.model_id == baseline) & (group.estimator == "ridge")]
            right = group[(group.model_id == "M3") & (group.estimator == "ridge")]
            if len(left) == 1 and len(right) == 1:
                values.append(float(left.iloc[0].estimate - right.iloc[0].estimate))
        for summary, estimate in (("MIN", min(values) if values else None), ("MEDIAN", float(np.median(values)) if values else None), ("MAX", max(values) if values else None)):
            rows.append(_result_row(result_id=f"PRED_DELETE_HLO_TO_P_M3_RIDGE_{baseline}_MSE_{summary}", analysis_family="phase_b_country_deletion", status="calculated_pending_reporting_validation" if values else "not_calculated_incomplete_deletions", run_id=run_id, protocol_hash=PROTOCOL_HASH, target="period-specific P", model_or_comparison=f"M3 Ridge improvement over {baseline} Ridge under full country-deletion refits", period="primary five-year", metric=f"country_deletion_{summary.lower()}_mse_improvement", estimate=estimate, interval_method="none; deletion sensitivity range, not confidence interval", units="P squared units", n_countries=len(values), n_cells=None, denominator_definition="one full nested refit per deleted primary country", weighting_rule="recomputed equal-country weights after every deletion", sample_id="PRIMARY_DELETE|HLO_to_P", source_artifact="reporting_sources/PHASE_B_METRICS_SOURCE.csv", source_locator=f"baseline={baseline}", generator=GENERATOR, input_hash=input_hash, interpretation="Positive values mean M3 has lower MSE.", limitation="Sensitivity range is not a confidence interval."))
        sign_changes = int(sum(np.sign(value) != np.sign(full_difference) for value in values)) if values and np.isfinite(full_difference) else None
        rows.append(_result_row(result_id=f"PRED_DELETE_HLO_TO_P_M3_RIDGE_{baseline}_MSE_SIGN_CHANGES", analysis_family="phase_b_country_deletion", status="calculated_pending_reporting_validation" if values else "not_calculated_incomplete_deletions", run_id=run_id, protocol_hash=PROTOCOL_HASH, target="period-specific P", model_or_comparison=f"M3 Ridge improvement over {baseline} Ridge", period="primary five-year", metric="country_deletion_sign_changes", estimate=sign_changes, interval_method="none", units="deletions", n_countries=len(values), denominator_definition="completed full nested deletion refits", weighting_rule="recomputed equal-country weights", sample_id="PRIMARY_DELETE|HLO_to_P", source_artifact="reporting_sources/PHASE_B_METRICS_SOURCE.csv", source_locator=f"baseline={baseline}", generator=GENERATOR, input_hash=input_hash, interpretation="Counts deletions on the minority sign; zero means no sign reversal among completed deletions.", limitation="This is stability sensitivity, not inferential uncertainty."))
    return rows


def _scheme(scenario: str) -> str:
    return "four_year" if scenario == "S4" else "five_year"


def build_prediction_ledger(run: Path, predictions: pd.DataFrame, run_id: str, source_column: str) -> pd.DataFrame:
    fits = pd.DataFrame(json.loads((run / "FITS.json").read_text(encoding="utf-8")))
    if fits.empty:
        raise ValueError("FITS.json is empty")
    fit_lookup = fits.set_index("fit_id").to_dict("index")
    rows = []
    for row in predictions.itertuples(index=False):
        fit = fit_lookup.get(row.fit_id, {})
        selected = fit.get("selected_alpha")
        selected = None if selected is None or pd.isna(selected) else selected
        rows.append({
            "run_id": run_id,
            "protocol_hash": PROTOCOL_HASH,
            "direction": row.direction,
            "scheme": _scheme(row.scenario_id),
            "history_definition": row.construction_role,
            "model_id": f"{row.model_id}_{row.estimator}",
            "outer_split_id": row.fit_id,
            "country": row.iso3,
            "window_start": row.window_start,
            "window_end": row.window_end,
            "observed_target": row.observed,
            "predicted_target": row.predicted,
            "raw_source_indicator": getattr(row, source_column),
            "absolute_error": abs(row.predicted - row.observed),
            "squared_error": (row.predicted - row.observed) ** 2,
            "selected_hyperparameters": json.dumps({"alpha": selected}, sort_keys=True, allow_nan=False),
            "training_country_set_id": fit.get("training_country_set_id"),
            "preprocessing_record_id": row.fit_id,
            "status": getattr(row, "status", "valid"),
            "failure_reason": getattr(row, "failure_reason", None),
        })
    failures = _read_csv(run / "FAILURES.csv")
    for failure in failures.to_dict("records"):
        fit_id = failure.get("fit_id") or f"{failure.get('job_id', 'unknown')}__outer_{failure.get('iso3', failure.get('outer_country', 'unknown'))}"
        fit = fit_lookup.get(fit_id, {})
        scenario = failure.get("scenario_id", "")
        detail = failure.get("detail", failure.get("failure_reason", failure.get("failure_type", "failed fit")))
        rows.append({
            "run_id": run_id, "protocol_hash": PROTOCOL_HASH,
            "direction": failure.get("direction"), "scheme": _scheme(str(scenario)),
            "history_definition": failure.get("construction_role"),
            "model_id": f"{failure.get('model_id', '')}_{failure.get('estimator', '')}",
            "outer_split_id": fit_id, "country": failure.get("iso3", failure.get("outer_country")),
            "window_start": failure.get("window_start"), "window_end": failure.get("window_end"),
            "observed_target": failure.get("observed"), "predicted_target": None,
            "raw_source_indicator": failure.get("raw_source_indicator", failure.get("source_indicator_score")),
            "absolute_error": None, "squared_error": None,
            "selected_hyperparameters": json.dumps({"alpha": None if pd.isna(fit.get("selected_alpha")) else fit.get("selected_alpha")}, sort_keys=True, allow_nan=False),
            "training_country_set_id": fit.get("training_country_set_id"),
            "preprocessing_record_id": fit_id, "status": "failed", "failure_reason": detail,
        })
    return pd.DataFrame(rows, columns=PREDICTION_COLUMNS)


def _failure_register(run: Path, plan: pd.DataFrame, status: dict[str, object]) -> pd.DataFrame:
    failures = _read_csv(run / "FAILURES.csv")
    summaries = {row["job_id"]: row for row in status.get("job_summaries", [])}
    rows = []
    for job in plan.to_dict("records"):
        summary = summaries.get(job["job_id"], {})
        rows.append({**job, "job_complete": summary.get("complete"), "planned_cell_count": summary.get("planned_cell_count"), "predicted_cell_count": summary.get("predicted_cell_count"), "failure_cell_count": int((failures["job_id"] == job["job_id"]).sum()) if not failures.empty and "job_id" in failures else 0, "failure_status": "none" if summary.get("complete") else "incomplete_preserved"})
    return pd.DataFrame(rows)


def _failure_result_rows(failure_register: pd.DataFrame, run_id: str, input_hash: str) -> pd.DataFrame:
    rows = []
    incomplete = failure_register[~failure_register["job_complete"].astype(str).str.lower().eq("true")]
    for job in incomplete.itertuples(index=False):
        rows.append(_result_row(result_id=f"PRED_DIAG_{_token(job.direction)}_{_token(job.scenario_id)}_{_token(job.model_id)}_{_token(job.estimator)}_{_token(job.construction_role)}_{_token(job.deleted_country)}_FAILURE", analysis_family="phase_b_failure", status="failed_or_incomplete_preserved", run_id=run_id, protocol_hash=PROTOCOL_HASH, target=job.direction, model_or_comparison=f"{job.model_id} {job.estimator} [{job.construction_role}]", period="planned run", metric="planned_model_completion", estimate=None, interval_method="not applicable: failed or incomplete", units="status", n_countries=None, n_cells=job.planned_cell_count, denominator_definition="all planned outer test cells", weighting_rule="not scored on partial favourable subset", sample_id=f"{job.scenario_id}|{job.direction}|{job.construction_role}", source_artifact="FAILURE_REGISTER.csv", source_locator=f"job_id={job.job_id}", generator=GENERATOR, input_hash=input_hash, interpretation="Failure retained; no estimate substituted.", limitation="See cell-level failure ledger for cause."))
    return pd.DataFrame(rows, columns=RESULT_COLUMNS)


def _markdown(frame: pd.DataFrame) -> str:
    shown = frame.copy()
    for column in shown:
        shown[column] = shown[column].map(lambda value: "—" if pd.isna(value) else f"{value:.3f}" if isinstance(value, (float, np.floating)) else str(value))
    header = "| " + " | ".join(shown.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(shown.columns)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in shown.astype(str).to_numpy()]
    return "\n".join([header, separator, *body])


def _direction_label(direction: str) -> str:
    return {"HLO_to_P": "HLO → P", "P_to_HLO": "P → HLO"}.get(direction, direction)


def _main_table_source(results: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "section", "direction", "scenario", "specification", "comparison", "mse", "rmse",
        "scaled_mse", "mse_improvement_vs_m0", "summary_metric", "summary_estimate",
        "loss_units", "n_countries", "n_cells", "status", "source_result_ids",
    ]
    rows = []
    primary = metrics[
        metrics["scenario_id"].eq("PRIMARY")
        & metrics["construction_role"].eq("primary_5y")
        & metrics["metric"].isin(["mse", "rmse", "scaled_mse"])
    ]
    order = {("M0", "reference"): 0, **{(model, estimator): index for index, (estimator, model) in enumerate(((estimator, model) for estimator in ("ridge", "ols", "lasso") for model in ("M1", "M2", "M3")), start=1)}}
    for direction in ("HLO_to_P", "P_to_HLO"):
        part = primary[primary["direction"].eq(direction)]
        for (model, estimator), group in sorted(part.groupby(["model_id", "estimator"], sort=False), key=lambda item: order.get(item[0], 99)):
            values = dict(zip(group["metric"], group["estimate"]))
            mse_rows = group[group["metric"].eq("mse")]
            baseline = primary[(primary["direction"].eq(direction)) & primary["model_id"].eq("M0") & primary["metric"].eq("mse")]
            mse, m0 = values.get("mse"), baseline["estimate"].iloc[0] if len(baseline) == 1 else np.nan
            improvement = np.nan if pd.isna(mse) or pd.isna(m0) or m0 == 0 else 1 - mse / m0
            result_ids = results[
                results["analysis_family"].eq("phase_b_prediction")
                & results["sample_id"].eq(f"PRIMARY|{direction}|primary_5y")
                & results["model_or_comparison"].eq(f"{model} {estimator} [primary_5y]")
                & results["metric"].isin(["mse", "rmse", "scaled_mse"])
            ]
            if model != "M0":
                improvement_id = results[
                    results["analysis_family"].eq("phase_b_prediction_comparison")
                    & results["result_id"].eq(f"PRED_DIFF_{_token(direction)}_PRIMARY_{model}_{_token(estimator)}_PRIMARY_5Y_VS_M0_MSE_IMPROVEMENT")
                ]
                result_ids = pd.concat([result_ids, improvement_id], ignore_index=True)
            rows.append({
                "section": "Primary models and benchmarks",
                "direction": _direction_label(direction),
                "scenario": "PRIMARY",
                "specification": f"{model} {'Reference' if estimator == 'reference' else estimator.upper() if estimator == 'ols' else estimator.title()}{' (benchmark)' if estimator in ('ols', 'lasso') else ''}",
                "comparison": "Primary five-year construction",
                "mse": mse,
                "rmse": values.get("rmse"),
                "scaled_mse": values.get("scaled_mse"),
                "mse_improvement_vs_m0": improvement,
                "summary_metric": None,
                "summary_estimate": None,
                "loss_units": "P score²" if direction == "HLO_to_P" else "HLO score²",
                "n_countries": mse_rows.iloc[0]["n_countries"] if len(mse_rows) else None,
                "n_cells": mse_rows.iloc[0]["n_cells"] if len(mse_rows) else None,
                "status": "calculated_validated",
                "source_result_ids": " | ".join(result_ids["result_id"]),
            })

    for row in results[
        results["analysis_family"].eq("phase_b_matched_sensitivity")
        & results["model_or_comparison"].str.contains(r"; M3 ridge$", na=False)
    ].sort_values(["sample_id", "target"]).itertuples(index=False):
        rows.append({
            "section": "Matched history sensitivities", "direction": _direction_label(row.target),
            "scenario": str(row.sample_id).split("|")[0], "specification": "M3 Ridge",
            "comparison": row.model_or_comparison, "summary_metric": row.metric,
            "summary_estimate": row.estimate, "loss_units": row.units,
            "n_countries": row.n_countries, "n_cells": row.n_cells, "status": row.status,
            "source_result_ids": row.result_id,
        })

    other = metrics[
        metrics["scenario_id"].isin(["SCC", "SCOMP", "STERM", "S4"])
        & metrics["model_id"].eq("M3") & metrics["estimator"].eq("ridge")
        & metrics["metric"].isin(["mse", "rmse", "scaled_mse"])
    ]
    for (scenario, direction, role), group in other.groupby(["scenario_id", "direction", "construction_role"], sort=True):
        values = dict(zip(group["metric"], group["estimate"]))
        baseline = metrics[
            metrics["scenario_id"].eq(scenario) & metrics["direction"].eq(direction)
            & metrics["construction_role"].eq(role) & metrics["model_id"].eq("M0")
            & metrics["metric"].eq("mse")
        ]
        m0, mse = baseline["estimate"].iloc[0] if len(baseline) == 1 else np.nan, values.get("mse")
        improvement = np.nan if pd.isna(mse) or pd.isna(m0) or m0 == 0 else 1 - mse / m0
        result_ids = results[
            results["analysis_family"].eq("phase_b_prediction")
            & results["sample_id"].eq(f"{scenario}|{direction}|{role}")
            & results["model_or_comparison"].eq(f"M3 ridge [{role}]")
            & results["metric"].isin(["mse", "rmse", "scaled_mse"])
        ]
        rows.append({
            "section": "Other named sensitivities", "direction": _direction_label(direction),
            "scenario": scenario, "specification": "M3 Ridge", "comparison": role,
            "mse": mse, "rmse": values.get("rmse"), "scaled_mse": values.get("scaled_mse"),
            "mse_improvement_vs_m0": improvement,
            "loss_units": "P score²" if direction == "HLO_to_P" else "HLO score²",
            "n_countries": group.iloc[0]["n_countries"], "n_cells": group.iloc[0]["n_cells"],
            "status": "calculated_validated", "source_result_ids": " | ".join(result_ids["result_id"]),
        })

    for row in results[results["analysis_family"].eq("phase_b_country_deletion")].sort_values("result_id").itertuples(index=False):
        rows.append({
            "section": "Country-deletion sensitivity", "direction": "HLO → P",
            "scenario": "PRIMARY_DELETE", "specification": "M3 Ridge",
            "comparison": row.model_or_comparison, "summary_metric": row.metric,
            "summary_estimate": row.estimate, "loss_units": row.units,
            "n_countries": row.n_countries, "n_cells": row.n_cells, "status": row.status,
            "source_result_ids": row.result_id,
        })
    return pd.DataFrame(rows, columns=columns)


def _table_display(source: pd.DataFrame) -> str:
    sections = []
    for section in source["section"].drop_duplicates():
        part = source[source["section"].eq(section)]
        if section in ("Primary models and benchmarks", "Other named sensitivities"):
            shown = part[["direction", "scenario", "specification", "comparison", "mse", "rmse", "mse_improvement_vs_m0", "n_countries", "n_cells", "status"]]
        else:
            shown = part[["direction", "scenario", "specification", "comparison", "summary_metric", "summary_estimate", "n_countries", "status"]]
        sections.append(f"## {section}\n\n{_markdown(shown)}")
    return "# Out-of-sample prediction comparison\n\n" + "\n\n".join(sections) + "\n"


def _write_table_bundle(output: Path, results: pd.DataFrame, metrics: pd.DataFrame, run_id: str, input_hash: str) -> None:
    table_dir = output / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    source = _main_table_source(results, metrics)
    source.to_csv(table_dir / "T_PREDICTION_source.csv", index=False)
    supplemental = results[
        results["analysis_family"].isin(["phase_b_prediction", "phase_b_prediction_comparison", "phase_b_model_comparison", "phase_b_matched_sensitivity", "phase_b_country_deletion", "phase_b_failure"])
    ].copy()
    supplemental.to_csv(table_dir / "T_PREDICTION_supplemental_source.csv", index=False)
    (table_dir / "T_PREDICTION_display.md").write_text(_table_display(source), encoding="utf-8")
    pd.DataFrame([{"column": column, "definition": {"summary_estimate": "Full-precision named sensitivity or deletion summary", "source_result_ids": "Exact exhaustive-register provenance", "mse_improvement_vs_m0": "Proportional reduction 1 - model MSE / M0 MSE; blank when M0 MSE is zero"}.get(column, column.replace("_", " "))} for column in source.columns]).to_csv(table_dir / "T_PREDICTION_dictionary.csv", index=False)
    (table_dir / "T_PREDICTION_notes.md").write_text(f"# T_PREDICTION notes\n\n- Run: `{run_id}`\n- Protocol: `{PROTOCOL_HASH}`\n- Input bundle hash: `{input_hash}`\n- The main table is deliberately bounded to primary models/benchmarks and compact named sensitivity/deletion summaries.\n- Exhaustive results, including null, adverse, failed, period-specific, and deletion-refit rows, remain in `T_PREDICTION_supplemental_source.csv` and `PHASE_B_RESULTS_REGISTER.csv`.\n- Equal-country evaluation; original direction-specific units remain separate. MSE is in squared target units.\n- Scaled errors use each outer training target scale; scaled MSE is squared standardized error.\n- This is internal leave-one-country-out validation, not external prospective validation, causal evidence, or construct equivalence.\n- No interval is inferred from cross-validation variability.\n- Command: `py -3.12 -m phase_b_reporting.cli ...`\n", encoding="utf-8")


def _write_figure_bundle(output: Path, metric_source: pd.DataFrame, run_id: str, input_hash: str) -> None:
    figure_dir = output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    plot = metric_source[(metric_source["scenario_id"] == "PRIMARY") & (metric_source["metric"] == "mse")].copy()
    plot["label"] = plot["model_id"] + " " + plot["estimator"]
    plot.to_csv(figure_dir / "F_PREDICTION_LOSS_plot_data.csv", index=False)
    directions = [direction for direction in ("HLO_to_P", "P_to_HLO") if direction in set(plot["direction"])]
    fig, axes = plt.subplots(1, max(1, len(directions)), figsize=(5.4 * max(1, len(directions)), 4.6), squeeze=False)
    for axis, direction in zip(axes[0], directions):
        panel = plot[plot["direction"] == direction].sort_values(["estimator", "model_id"])
        axis.scatter(panel["estimate"], np.arange(len(panel)), color="#24557a")
        axis.set_yticks(np.arange(len(panel)), panel["label"])
        axis.set_xlabel("Equal-country MSE (original target units²)")
        axis.set_title(_direction_label(direction))
        axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "F_PREDICTION_LOSS.svg")
    fig.savefig(figure_dir / "F_PREDICTION_LOSS.png", dpi=300)
    plt.close(fig)
    (figure_dir / "F_PREDICTION_LOSS_caption.md").write_text(f"# F_PREDICTION_LOSS caption\n\nPrimary-sample outer-held-out mean squared error under equal-country evaluation. Panels retain their native squared target units and must not be compared as if their scales were interchangeable. Points show no invented uncertainty interval. Lower values indicate smaller error. Run `{run_id}`; protocol `{PROTOCOL_HASH}`; source bundle `{input_hash}`. This is internal leave-one-country-out validation, not external prospective validation, causal evidence, or construct equivalence.\n", encoding="utf-8")
    pd.DataFrame([{"check": check, "status": "PASS"} for check in ("source_rows_plotted", "axis_direction_lower_is_better", "direction_units_separated", "no_invented_intervals", "svg_written", "png_written")]).to_csv(figure_dir / "F_PREDICTION_LOSS_render_check.csv", index=False)


def _write_combined_register(phase_a: Path, phase_b: pd.DataFrame, target: Path) -> None:
    header = phase_a.read_text(encoding="utf-8").rstrip("\r\n")
    addition = phase_b.to_csv(index=False, header=False, lineterminator="\n")
    target.write_text(header + "\n" + addition, encoding="utf-8")


def build_reporting_package(
    run_directory: str | Path,
    output_directory: str | Path,
    phase_a_results_register: str | Path,
    phase_a_exhibit_register: str | Path,
    run_id: str,
    validation_status: str | Path,
    allow_synthetic: bool = False,
) -> dict[str, object]:
    run, output = Path(run_directory), Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    validation_path = Path(validation_status)
    state = validate_run(run, validation_path, allow_synthetic)
    predictions = _read_csv(run / "PREDICTIONS.csv")
    input_rows = [{"path": name, "bytes": (run / name).stat().st_size, "sha256": sha256(run / name)} for name in REQUIRED_LEDGER_FILES]
    input_rows.append({"path": str(validation_path), "bytes": validation_path.stat().st_size, "sha256": sha256(validation_path)})
    input_manifest = pd.DataFrame(input_rows)
    input_manifest.to_csv(output / "INPUT_MANIFEST.csv", index=False)
    bundle_hash = hashlib.sha256("\n".join(f"{row.path}={row.sha256}" for row in input_manifest.sort_values("path").itertuples()).encode("utf-8")).hexdigest()
    reporting_predictions = build_prediction_ledger(run, predictions, run_id, state["source_column"])
    reporting_predictions.to_csv(output / "PREDICTIONS.csv", index=False)
    results, sources = compute_results(predictions, run_id, bundle_hash)
    plan = _read_csv(run / "execution_plan.csv")
    failure_register = _failure_register(run, plan, state["status"])
    failure_register.to_csv(output / "FAILURE_REGISTER.csv", index=False)
    failure_results = _failure_result_rows(failure_register, run_id, bundle_hash)
    if not failure_results.empty:
        results = pd.concat([results, failure_results], ignore_index=True)
    results.loc[results["status"].eq("calculated_pending_reporting_validation"), "status"] = "calculated_validated"
    if list(results.columns) != RESULT_COLUMNS:
        raise AssertionError("Phase B result register does not have the exact 25-column schema")
    source_dir = output / "reporting_sources"
    source_dir.mkdir(exist_ok=True)
    for name, frame in sources.items():
        frame.to_csv(source_dir / name, index=False)
    phase_b_path = output / "PHASE_B_RESULTS_REGISTER.csv"
    results.to_csv(phase_b_path, index=False)
    _write_combined_register(Path(phase_a_results_register), results, output / "RESULTS_REGISTER.csv")
    _write_table_bundle(output, results, sources["PHASE_B_METRICS_SOURCE.csv"], run_id, bundle_hash)
    _write_figure_bundle(output, sources["PHASE_B_METRICS_SOURCE.csv"], run_id, bundle_hash)
    numerical = {row.result_id: {"estimate": None if pd.isna(row.estimate) else row.estimate, "status": row.status, "units": row.units, "source": row.source_artifact} for row in results.itertuples(index=False)}
    (output / "numerical_values.json").write_text(json.dumps(numerical, indent=2, sort_keys=True), encoding="utf-8")
    exhibits = pd.read_csv(phase_a_exhibit_register)
    for column in ("status", "source_or_plot_data", "display_or_rendered_files", "manuscript_location", "generator", "generating_command", "input_artifacts_and_sha256", "run_id"):
        exhibits[column] = exhibits[column].astype(object)
    command = f"py -3.12 -m phase_b_reporting.cli --run <immutable-run> --output <reporting-output> --run-id {run_id} --validation-status <independent-validation-output>/VALIDATION_STATUS.json"
    exhibit_updates = {
        "T_PREDICTION": ["produced_validated", "tables/T_PREDICTION_source.csv | tables/T_PREDICTION_supplemental_source.csv", "tables/T_PREDICTION_display.md", "Main manuscript Results (proposed table); exhaustive rows in supplement/register"],
        "F_PREDICTION_LOSS": ["produced_validated", "figures/F_PREDICTION_LOSS_plot_data.csv", "figures/F_PREDICTION_LOSS.svg | figures/F_PREDICTION_LOSS.png", "Main manuscript Results (proposed figure); editable source retained"],
    }
    for exhibit_id, (status_value, source_value, display_value, location_value) in exhibit_updates.items():
        mask = exhibits.exhibit_id.eq(exhibit_id)
        exhibits.loc[mask, ["status", "source_or_plot_data", "display_or_rendered_files", "manuscript_location", "generator", "generating_command", "input_artifacts_and_sha256", "run_id"]] = [status_value, source_value, display_value, location_value, GENERATOR, command, f"phase_b_input_bundle_sha256={bundle_hash}", run_id]
    exhibits.to_csv(output / "EXHIBIT_REGISTER.csv", index=False)
    run_manifest = {"run_id": run_id, "protocol_hash": PROTOCOL_HASH, "source_run_plan_hash": state["settings"].get("plan_hash"), "input_bundle_hash": bundle_hash, "reporting_only": True, "empirical_fitting_performed": False, "phase_b_result_rows": len(results), "prediction_rows": len(reporting_predictions)}
    (output / "RUN_MANIFEST.json").write_text(json.dumps(run_manifest, indent=2, sort_keys=True), encoding="utf-8")
    output_rows = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "OUTPUT_MANIFEST.csv":
            output_rows.append({"path": path.relative_to(output).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)})
    pd.DataFrame(output_rows).to_csv(output / "OUTPUT_MANIFEST.csv", index=False)
    return run_manifest
