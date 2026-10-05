"""Finite protocol execution manifest, including country-deletion refits."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

from .matrices import MatrixBundle


@dataclass(frozen=True)
class Job:
    job_id: str
    scenario_id: str
    direction: str
    construction_role: str
    model_id: str
    estimator: str
    deleted_country: str | None = None


def _job(scenario: str, direction: str, role: str, model: str, estimator: str, deleted: str | None = None) -> Job:
    parts = [scenario, direction, role, model, estimator, deleted or "none"]
    return Job("__".join(parts), scenario, direction, role, model, estimator, deleted)


def build_execution_plan(bundle: MatrixBundle) -> list[Job]:
    jobs: list[Job] = []
    for direction in ("HLO_to_P", "P_to_HLO"):
        jobs.append(_job("PRIMARY", direction, "primary_5y", "M0", "reference"))
        for estimator in ("ridge", "ols", "lasso"):
            for model in ("M1", "M2", "M3"):
                jobs.append(_job("PRIMARY", direction, "primary_5y", model, estimator))
    for identity in sorted(bundle.matrices):
        scenario, direction, role = identity
        if scenario == "PRIMARY":
            continue
        for model in ("M0", "M1", "M2", "M3"):
            jobs.append(_job(scenario, direction, role, model, "reference" if model == "M0" else "ridge"))
    primary = bundle.matrices[("PRIMARY", "HLO_to_P", "primary_5y")]
    for deleted in sorted(primary["iso3"].unique()):
        for model in ("M0", "M1", "M2", "M3"):
            jobs.append(_job("PRIMARY_DELETE", "HLO_to_P", "primary_5y", model, "reference" if model == "M0" else "ridge", deleted))
    identifiers = [job.job_id for job in jobs]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Duplicate execution job IDs")
    return jobs


def plan_hash(jobs: list[Job]) -> str:
    payload = [asdict(job) for job in jobs]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

