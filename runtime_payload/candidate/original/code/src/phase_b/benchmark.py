"""Synthetic timing probe and transparent Phase B duration projection."""

from __future__ import annotations

import json
from pathlib import Path
import time

from protocol_final.synthetic import make_synthetic_fixture

from .engine import run_job
from .matrices import FEATURES
from .plan import Job


def synthetic_matrix():
    frame = make_synthetic_fixture().rename(columns={"target": "target_score", "source_score": "source_indicator_score"})
    frame["sample_cell_id"] = frame["iso3"] + "_" + frame["window_start"].astype(str) + "_" + frame["window_end"].astype(str)
    frame["p_score"] = frame["target_score"]
    frame["hlo_score"] = frame["source_indicator_score"]
    frame["usable_feature_count"] = frame[list(FEATURES)].notna().sum(axis=1)
    frame["missing_feature_count"] = 7 - frame["usable_feature_count"]
    return frame


def run_benchmark(output_path: str | Path) -> dict[str, object]:
    matrix = synthetic_matrix()
    timings = {}
    for estimator in ("ridge", "ols", "lasso"):
        job = Job(f"benchmark_{estimator}", "SYNTHETIC", "HLO_to_P", "synthetic", "M3", estimator)
        started = time.perf_counter()
        result = run_job(matrix, job)
        timings[estimator] = {"seconds": time.perf_counter() - started, "complete": result["complete"], "countries": matrix["iso3"].nunique(), "rows": len(matrix)}
    # Exact protocol fit counts: batched Ridge path operations, individual Lasso fits,
    # and direct OLS fits. The multiplier projection is a scheduling estimate.
    # 1,368 primary + 11,952 sensitivity + 25,308 deletion operations.
    ridge_operations = 38628
    lasso_fits = 8208
    ols_fits = 228
    raw_seconds = (
        timings["ridge"]["seconds"] / (8 * 6) * ridge_operations
        + timings["lasso"]["seconds"] / (8 * 36) * lasso_fits
        + timings["ols"]["seconds"] / 8 * ols_fits
    )
    projection = {
        "batched_ridge_path_or_refit_operations": ridge_operations,
        "lasso_individual_fits": lasso_fits,
        "weighted_ols_outer_fits": ols_fits,
        "raw_linear_projection_hours": raw_seconds / 3600,
        "estimated_wall_hours_single_thread": {
            "lower": raw_seconds * 1.5 / 3600,
            "central": raw_seconds * 2.5 / 3600,
            "upper": raw_seconds * 5.0 / 3600,
        },
        "basis": "Synthetic 8-country M3 timings scaled by declared job/country/fold counts. Multipliers allow for larger empirical training frames, checkpoint I/O, and convergence variability.",
    }
    record = {"synthetic_only": True, "empirical_models_fitted": False, "timings": timings, "projection": projection}
    Path(output_path).write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    return record
