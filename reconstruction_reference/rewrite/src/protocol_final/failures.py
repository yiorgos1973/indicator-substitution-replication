"""Paired-completeness rules for planned outer prediction cells."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable


def assess_paired_failures(records: Iterable[dict[str, object]]) -> dict[str, object]:
    records = list(records)
    planned = {str(row["cell_id"]) for row in records if row["model_id"] == "M0"}
    by_estimator_model: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in records:
        if bool(row["valid"]):
            by_estimator_model[(str(row["estimator"]), str(row["model_id"]))].add(str(row["cell_id"]))
    status = {}
    for estimator in ("ridge", "ols", "lasso"):
        complete = all(by_estimator_model[(estimator, model)] == planned for model in ("M0", "M1", "M2", "M3"))
        status[estimator] = {"complete": complete, "paired_primary_loss_allowed": estimator == "ridge" and complete}
    return {"planned_cell_count": len(planned), "estimators": status}

