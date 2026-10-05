"""Canonical protocol, author approval, measurement, and frozen-input gates."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from .input_lock import PHASE_A_OUTPUTS


EXPECTED_PROTOCOL_HASH = "13fb577533c28acc88613fcf84549784e130bd658139ea7bbbfc7f77a3b8e773"
MEASUREMENT_CHECKS = tuple(f"MEAS_{number:02d}" for number in range(1, 10))


class GateError(RuntimeError):
    pass


@dataclass(frozen=True)
class GateResult:
    passed: bool
    protocol_hash: str
    checks: tuple[dict[str, object], ...]

    def require(self) -> None:
        if not self.passed:
            failed = [str(check["check_id"]) for check in self.checks if not check["passed"]]
            raise GateError("Phase B gate is closed: " + ", ".join(failed))


def canonical_protocol_hash(protocol: dict[str, object]) -> str:
    payload = dict(protocol)
    payload.pop("protocol_hash", None)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check(checks: list[dict[str, object]], check_id: str, passed: bool, detail: str) -> None:
    checks.append({"check_id": check_id, "passed": bool(passed), "detail": detail})


def _resolve(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if root.resolve() not in path.parents and path != root.resolve():
        raise GateError(f"Path escapes repository root: {relative}")
    return path


def evaluate_gate(
    repository_root: str | Path,
    protocol_path: str | Path,
    approval_path: str | Path,
    measurement_path: str | Path,
) -> GateResult:
    root = Path(repository_root).resolve()
    protocol_file = Path(protocol_path).resolve()
    approval_file = Path(approval_path).resolve()
    measurement_file = Path(measurement_path).resolve()
    protocol = json.loads(protocol_file.read_text(encoding="utf-8"))
    approval = json.loads(approval_file.read_text(encoding="utf-8"))
    checks: list[dict[str, object]] = []

    declared_hash = str(protocol.get("protocol_hash", ""))
    calculated_hash = canonical_protocol_hash(protocol)
    _check(checks, "PROTOCOL_EXPECTED_HASH", declared_hash == EXPECTED_PROTOCOL_HASH, declared_hash)
    _check(checks, "PROTOCOL_CANONICAL_HASH", calculated_hash == EXPECTED_PROTOCOL_HASH, calculated_hash)
    _check(checks, "APPROVAL_STATUS", approval.get("approval_status") == "approved" and approval.get("approval_record_status") == "author_approved", str(approval.get("approval_status")))
    _check(checks, "APPROVAL_SCOPE", approval.get("scope") == "empirical_prediction" and approval.get("empirical_modelling_approved") is True, str(approval.get("scope")))
    _check(checks, "APPROVAL_HASH", approval.get("protocol_hash") == EXPECTED_PROTOCOL_HASH, str(approval.get("protocol_hash")))
    _check(checks, "APPROVAL_PROTOCOL_BYTES", approval.get("protocol_file_sha256") == file_sha256(protocol_file), file_sha256(protocol_file))

    for item in protocol.get("data_reconciliation", {}).get("input_hashes", []):
        path = _resolve(root, str(item["repository_relative_path"]))
        actual = file_sha256(path) if path.is_file() else "missing"
        _check(checks, f"INPUT_HASH::{item['input_role']}", actual == item["sha256"], actual)
    for role, item in protocol.get("frozen_scientific_artifacts", {}).items():
        path = _resolve(root, str(item["repository_relative_path"]))
        actual = file_sha256(path) if path.is_file() else "missing"
        _check(checks, f"FROZEN_HASH::{role}", actual == item["sha256"], actual)
    for relative, expected in PHASE_A_OUTPUTS.items():
        path = _resolve(root, relative)
        actual = file_sha256(path) if path.is_file() else "missing"
        _check(checks, f"PHASE_A_OUTPUT_HASH::{relative}", actual == expected, actual)

    if measurement_file.is_file():
        measurement = json.loads(measurement_file.read_text(encoding="utf-8"))
        _check(checks, "MEAS_RECORD_SHA256_RECORDED", True, file_sha256(measurement_file))
        _check(checks, "MEAS_PROTOCOL_HASH", measurement.get("protocol_hash") == EXPECTED_PROTOCOL_HASH, str(measurement.get("protocol_hash")))
        overall_status = measurement.get("measurement_readiness_status", measurement.get("status"))
        readiness_flag = measurement.get("empirical_execution_ready", True)
        _check(checks, "MEAS_OVERALL_STATUS", overall_status == "passed" and readiness_flag is True, str(overall_status))
        _check(checks, "MEAS_PROTOCOL_BYTES", measurement.get("protocol_file_sha256") in (None, file_sha256(protocol_file)), str(measurement.get("protocol_file_sha256")))
        supplied = measurement.get("checks", {})
        if isinstance(supplied, list):
            supplied = {row.get("check_id"): row for row in supplied}
        for check_id in MEASUREMENT_CHECKS:
            row = supplied.get(check_id, {}) if isinstance(supplied, dict) else {}
            status = row.get("status") if isinstance(row, dict) else row
            _check(checks, check_id, status == "passed", str(status))
        for collection in ("input_manifest_hashes", "frozen_scientific_artifact_hashes", "measurement_source_manifest_hashes", "validation_evidence_artifacts"):
            for item in measurement.get(collection, []):
                relative = item.get("repository_relative_path")
                if not relative:
                    continue
                path = _resolve(root, str(relative))
                actual = file_sha256(path) if path.is_file() else "missing"
                _check(checks, f"MEAS_EVIDENCE_HASH::{relative}", actual == item.get("sha256"), actual)
    else:
        _check(checks, "MEAS_RECORD_PRESENT", False, str(measurement_file))
        for check_id in MEASUREMENT_CHECKS:
            _check(checks, check_id, False, "missing")
    return GateResult(all(bool(check["passed"]) for check in checks), declared_hash, tuple(checks))
