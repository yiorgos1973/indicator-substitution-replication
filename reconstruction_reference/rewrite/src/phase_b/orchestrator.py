"""Preflight, resumable execution, checkpointing, and final ledger assembly."""

from __future__ import annotations

from dataclasses import asdict
import json
import hashlib
import platform
from pathlib import Path
from typing import Iterable

import pandas as pd
import numpy as np
import scipy
import sklearn
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits

from .engine import run_job
from protocol_final.evaluation import calibration_summary, score_metrics, tail_group_summary
from .gate import EXPECTED_PROTOCOL_HASH, GateResult, evaluate_gate
from .matrices import MatrixBundle, build_modelling_matrices
from .plan import Job, build_execution_plan, plan_hash


LEDGERS = ("predictions", "splits", "preprocessing", "tuning", "fits", "failures")


def build_preflight(
    repository_root: str | Path,
    protocol_path: str | Path,
    approval_path: str | Path,
    measurement_path: str | Path,
    sample_manifest: str | Path,
    feature_manifest: str | Path,
) -> tuple[GateResult, MatrixBundle, list[Job]]:
    gate = evaluate_gate(repository_root, protocol_path, approval_path, measurement_path)
    bundle = build_modelling_matrices(sample_manifest, feature_manifest)
    jobs = build_execution_plan(bundle)
    return gate, bundle, jobs


def write_preflight(output_directory: str | Path, gate: GateResult, bundle: MatrixBundle, jobs: list[Job]) -> None:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    (output / "gate_results.json").write_text(json.dumps({"passed": gate.passed, "protocol_hash": gate.protocol_hash, "checks": gate.checks}, indent=2, sort_keys=True), encoding="utf-8")
    bundle.manifest.to_csv(output / "matrix_manifest.csv", index=False)
    pd.DataFrame([asdict(job) for job in jobs]).to_csv(output / "execution_plan.csv", index=False)
    (output / "preflight_status.json").write_text(json.dumps({"protocol_hash": EXPECTED_PROTOCOL_HASH, "gate_passed": gate.passed, "matrix_count": len(bundle.matrices), "job_count": len(jobs), "plan_hash": plan_hash(jobs), "empirical_models_fitted": False}, indent=2, sort_keys=True), encoding="utf-8")


def _matrix_for_job(bundle: MatrixBundle, job: Job) -> pd.DataFrame:
    if job.scenario_id == "PRIMARY_DELETE":
        matrix = bundle.matrices[("PRIMARY", job.direction, job.construction_role)]
        return matrix[matrix["iso3"] != job.deleted_country].reset_index(drop=True)
    return bundle.matrices[(job.scenario_id, job.direction, job.construction_role)]


def _write_job(output: Path, job: Job, result: dict[str, object], current_plan_hash: str) -> None:
    directory = output / "checkpoints" / job.job_id
    directory.mkdir(parents=True, exist_ok=True)
    for ledger in LEDGERS:
        rows = result[ledger]
        if ledger == "fits":
            (directory / "fits.json").write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
        else:
            pd.DataFrame(rows).to_csv(directory / f"{ledger}.csv", index=False)
    summary = {key: value for key, value in result.items() if key not in LEDGERS}
    summary.update({"protocol_hash": EXPECTED_PROTOCOL_HASH, "plan_hash": current_plan_hash, "status": "complete"})
    temporary = directory / "checkpoint.tmp"
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(directory / "checkpoint.json")


def _checkpoint_valid(path: Path, current_plan_hash: str) -> bool:
    if not path.is_file():
        return False
    record = json.loads(path.read_text(encoding="utf-8"))
    return record.get("status") == "complete" and record.get("protocol_hash") == EXPECTED_PROTOCOL_HASH and record.get("plan_hash") == current_plan_hash


def execute_plan(bundle: MatrixBundle, jobs: list[Job], output_directory: str | Path, gate: GateResult) -> None:
    gate.require()
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    current_plan_hash = plan_hash(jobs)
    with threadpool_limits(limits=1):
        for job in jobs:
            checkpoint = output / "checkpoints" / job.job_id / "checkpoint.json"
            if _checkpoint_valid(checkpoint, current_plan_hash):
                continue
            result = run_job(_matrix_for_job(bundle, job), job)
            _write_job(output, job, result, current_plan_hash)
    assemble_ledgers(output, jobs, current_plan_hash)


def assemble_ledgers(output_directory: str | Path, jobs: Iterable[Job], current_plan_hash: str) -> None:
    output = Path(output_directory)
    tables = {ledger: [] for ledger in LEDGERS if ledger != "fits"}
    fits, summaries = [], []
    jobs = list(jobs)
    for job in jobs:
        directory = output / "checkpoints" / job.job_id
        checkpoint = directory / "checkpoint.json"
        if not _checkpoint_valid(checkpoint, current_plan_hash):
            continue
        summaries.append(json.loads(checkpoint.read_text(encoding="utf-8")))
        fits.extend(json.loads((directory / "fits.json").read_text(encoding="utf-8")))
        for ledger in tables:
            path = directory / f"{ledger}.csv"
            if path.stat().st_size > 2:
                table = pd.read_csv(path)
                for column, value in reversed([
                    ("job_id", job.job_id), ("scenario_id", job.scenario_id), ("direction", job.direction),
                    ("construction_role", job.construction_role), ("model_id", job.model_id),
                    ("estimator", job.estimator), ("deleted_country", job.deleted_country),
                ]):
                    table.insert(0, column, value)
                tables[ledger].append(table)
    combined = {}
    for ledger, parts in tables.items():
        combined[ledger] = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        combined[ledger].to_csv(output / f"{ledger.upper()}.csv", index=False)
    (output / "FITS.json").write_text(json.dumps(fits, indent=2, sort_keys=True), encoding="utf-8")
    (output / "RUN_STATUS.json").write_text(json.dumps({"protocol_hash": EXPECTED_PROTOCOL_HASH, "plan_hash": current_plan_hash, "jobs_complete": len(summaries), "job_summaries": summaries}, indent=2, sort_keys=True), encoding="utf-8")
    status_by_job = {row["job_id"]: row for row in summaries}
    paired_rows = []
    identity_groups = {}
    for job in jobs:
        key = (job.scenario_id, job.direction, job.construction_role, job.deleted_country)
        identity_groups.setdefault(key, []).append(job)
    for key, group_jobs in sorted(identity_groups.items(), key=lambda item: tuple(str(value) for value in item[0])):
        reference = next((job for job in group_jobs if job.model_id == "M0"), None)
        for estimator in sorted({job.estimator for job in group_jobs if job.model_id != "M0"}):
            required = [reference] + [next((job for job in group_jobs if job.estimator == estimator and job.model_id == model), None) for model in ("M1", "M2", "M3")]
            complete = all(job is not None and status_by_job.get(job.job_id, {}).get("complete") is True for job in required)
            counts = {status_by_job[job.job_id]["planned_cell_count"] for job in required if job is not None and job.job_id in status_by_job}
            paired_rows.append({"scenario_id": key[0], "direction": key[1], "construction_role": key[2], "deleted_country": key[3], "estimator": estimator, "M0_to_M3_complete": complete and len(counts) == 1, "planned_cell_count": next(iter(counts)) if len(counts) == 1 else None})
    pd.DataFrame(paired_rows).to_csv(output / "PAIRED_COMPLETENESS.csv", index=False)
    _write_reporting_ledgers(output, combined["predictions"])
    settings = {
        "protocol_hash": EXPECTED_PROTOCOL_HASH,
        "plan_hash": current_plan_hash,
        "one_thread": True,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "job_count": len(jobs),
    }
    (output / "RUN_SETTINGS.json").write_text(json.dumps(settings, indent=2, sort_keys=True), encoding="utf-8")
    manifest = {}
    for path in sorted(output.iterdir()):
        if path.is_file() and path.name != "MANIFEST.json":
            manifest[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (output / "MANIFEST.json").write_text(json.dumps({"protocol_hash": EXPECTED_PROTOCOL_HASH, "plan_hash": current_plan_hash, "files": manifest}, indent=2, sort_keys=True), encoding="utf-8")


def _write_reporting_ledgers(output: Path, predictions: pd.DataFrame) -> None:
    metric_rows, calibration_rows, rank_rows, tail_rows = [], [], [], []
    if predictions.empty:
        for name in ("METRICS.csv", "CALIBRATION.csv", "RANK_LEDGER.csv", "TAIL_GROUPS.csv"):
            pd.DataFrame().to_csv(output / name, index=False)
        return
    for job_id, group in predictions.groupby("job_id", sort=True):
        identity = {column: group.iloc[0][column] for column in ("job_id", "scenario_id", "direction", "construction_role", "model_id", "estimator", "deleted_country")}
        metric_rows.append({**identity, **score_metrics(group["observed"], group["predicted"], group["iso3"])})
        calibration_rows.append({**identity, "scope": "pooled_mixing_periods", **calibration_summary(group["observed"].to_numpy(), group["predicted"].to_numpy(), group["iso3"].to_numpy())})
        for window_start, period in group.groupby("window_start", sort=True):
            if period["iso3"].nunique() >= 3:
                calibration_rows.append({**identity, "scope": f"period_{window_start}", **calibration_summary(period["observed"].to_numpy(), period["predicted"].to_numpy(), period["iso3"].to_numpy())})
            if identity["model_id"] == "M0":
                continue
            observed_rank = rankdata(-period["observed"].to_numpy(dtype=float), method="average")
            predicted_rank = rankdata(-period["predicted"].to_numpy(dtype=float), method="average")
            count = len(period)
            for (_, row), target_rank, prediction_rank in zip(period.iterrows(), observed_rank, predicted_rank):
                rank_rows.append({**identity, "cell_id": row["cell_id"], "iso3": row["iso3"], "window_start": window_start, "period_country_count": count, "target_rank": target_rank, "prediction_rank": prediction_rank, "normalized_absolute_displacement": None if count < 2 else abs(target_rank - prediction_rank) / (count - 1)})
            for tail in ("low", "high"):
                tail_rows.append({**identity, "window_start": window_start, **tail_group_summary(period, tail)})
    pd.DataFrame(metric_rows).to_csv(output / "METRICS.csv", index=False)
    pd.DataFrame(calibration_rows).to_csv(output / "CALIBRATION.csv", index=False)
    pd.DataFrame(rank_rows).to_csv(output / "RANK_LEDGER.csv", index=False)
    pd.DataFrame(tail_rows).to_csv(output / "TAIL_GROUPS.csv", index=False)
