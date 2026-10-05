"""Small synthetic execution used only to validate the numerical protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
from threadpoolctl import threadpool_limits

from .evaluation import calibration_summary, period_order_summary, score_metrics, tail_group_summary
from .runtime import environment_record
from .synthetic import CONTEXT_FEATURES, make_synthetic_fixture
from .tuning import fit_outer_configuration


def run_synthetic_scenario(output_directory: str | Path) -> dict[str, object]:
    """Run a bounded synthetic LOCO exercise; never loads empirical data."""
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    frame = make_synthetic_fixture()
    predictions = []
    tuning = []
    preprocess = []
    fits = []
    configurations = [
        ("M0", "ridge", ()),
        ("M1", "ridge", ("source_score",)),
        ("M2", "ridge", CONTEXT_FEATURES),
        ("M3", "ridge", ("source_score",) + CONTEXT_FEATURES),
        ("M3", "ols", ("source_score",) + CONTEXT_FEATURES),
        ("M3", "lasso", ("source_score",) + CONTEXT_FEATURES),
    ]
    with threadpool_limits(limits=1):
        for country in sorted(frame["iso3"].unique()):
            held_out = frame[frame["iso3"] == country]
            for model, estimator, features in configurations:
                result = fit_outer_configuration(frame, country, model, estimator, features, CONTEXT_FEATURES)
                fit_id = f"synthetic|{country}|{model}|{estimator}"
                for (_, row), value in zip(held_out.iterrows(), result["prediction"]):
                    predictions.append({"fit_id": fit_id, "iso3": country, "window_start": row["window_start"], "window_end": row["window_end"], "model_id": model, "estimator": estimator, "observed": row["target"], "predicted": value, "synthetic_only": True})
                tuning.extend({"fit_id": fit_id, **candidate} for candidate in result["tuning"])
                preprocess.extend({"fit_id": fit_id, **row} for row in result["preprocessing"])
                fits.append({"fit_id": fit_id, "selected_alpha": result["selected_alpha"], "metadata": result["metadata"], "allocation": result["allocation"]})
    prediction_frame = pd.DataFrame(predictions)
    metric_rows = []
    for (model, estimator), group in prediction_frame.groupby(["model_id", "estimator"], sort=True):
        metric_rows.append({"model_id": model, "estimator": estimator, **score_metrics(group["observed"], group["predicted"], group["iso3"]), **{f"calibration_{key}": value for key, value in calibration_summary(group["observed"].to_numpy(), group["predicted"].to_numpy(), group["iso3"].to_numpy()).items()}})
    order_rows, tail_rows = [], []
    combined = prediction_frame[(prediction_frame["model_id"] == "M3") & (prediction_frame["estimator"] == "ridge")]
    for period, group in combined.groupby("window_start"):
        order_rows.append({"window_start": period, **period_order_summary(group)})
        for tail in ("low", "high"):
            tail_rows.append({"window_start": period, **tail_group_summary(group, tail)})
    outputs = {
        "predictions.csv": prediction_frame,
        "tuning.csv": pd.DataFrame(tuning),
        "preprocessing.csv": pd.DataFrame(preprocess),
        "metrics.csv": pd.DataFrame(metric_rows),
        "rank_order.csv": pd.DataFrame(order_rows),
        "tails.csv": pd.DataFrame(tail_rows),
    }
    for name, table in outputs.items():
        table.to_csv(output_directory / name, index=False)
    (output_directory / "fit_metadata.json").write_text(json.dumps(fits, indent=2, sort_keys=True), encoding="utf-8")
    (output_directory / "environment.json").write_text(json.dumps(environment_record(), indent=2, sort_keys=True), encoding="utf-8")
    manifest = {}
    for path in sorted(output_directory.iterdir()):
        if path.name != "manifest.json":
            manifest[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (output_directory / "manifest.json").write_text(json.dumps({"synthetic_only": True, "empirical_modelling_approved": False, "files": manifest}, indent=2, sort_keys=True), encoding="utf-8")
    return {"prediction_rows": len(prediction_frame), "fit_count": len(fits), "output_directory": str(output_directory)}

