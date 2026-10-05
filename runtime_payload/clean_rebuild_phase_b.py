"""Rebuild Phase B in an allowlisted workspace with no Git metadata.

Preparation copies only hash-pinned sources. Full execution is deliberately
separate and requires an explicit flag plus an external production PASS record.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import pandas as pd


PROTOCOL_HASH = "13fb577533c28acc88613fcf84549784e130bd658139ea7bbbfc7f77a3b8e773"
THREAD_VARIABLES = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def resolve_inside(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"Path escapes root: {relative}")
    return path


def load_allowlist(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    required = {"source_path", "destination_path", "role", "bytes", "sha256"}
    if not rows or set(rows[0]) != required:
        raise ValueError(f"Allowlist must have exactly these columns: {sorted(required)}")
    if len({row["destination_path"] for row in rows}) != len(rows):
        raise ValueError("Duplicate allowlist destination")
    return rows


def prepare(repository: Path, workspace: Path, allowlist_path: Path) -> dict[str, object]:
    started = time.perf_counter()
    repository, workspace = repository.resolve(), workspace.resolve()
    if workspace.exists():
        raise FileExistsError(f"Workspace already exists: {workspace}")
    workspace.mkdir(parents=True)
    rows = load_allowlist(allowlist_path)
    copied = []
    for row in rows:
        source = resolve_inside(repository, row["source_path"])
        destination = resolve_inside(workspace, row["destination_path"])
        if not source.is_file():
            raise FileNotFoundError(source)
        if source.stat().st_size != int(row["bytes"]) or sha256(source) != row["sha256"]:
            raise ValueError(f"Frozen source mismatch: {row['source_path']}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        if destination.stat().st_size != int(row["bytes"]) or sha256(destination) != row["sha256"]:
            raise ValueError(f"Copied file mismatch: {row['destination_path']}")
        copied.append(row)
    forbidden = [path for path in workspace.rglob("*") if path.name == ".git" or path.suffix == ".pyc" or path.name == "__pycache__"]
    if forbidden:
        raise ValueError(f"Forbidden paths in extracted workspace: {forbidden}")
    metadata = workspace / ".clean_rebuild"
    metadata.mkdir()
    manifest = metadata / "PREPARED_SNAPSHOT_MANIFEST.csv"
    pd.DataFrame(copied).to_csv(manifest, index=False, lineterminator="\n")
    elapsed = time.perf_counter() - started
    environment = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "pandas": pd.__version__,
    }
    write_json(metadata / "ENVIRONMENT.json", environment)
    status = {
        "status": "PASS",
        "mode": "prepare_only",
        "workspace": str(workspace),
        "file_count": len(copied),
        "total_bytes": sum(int(row["bytes"]) for row in copied),
        "allowlist_sha256": sha256(allowlist_path),
        "snapshot_manifest_sha256": sha256(manifest),
        "git_metadata_present": False,
        "empirical_models_fitted": False,
        "elapsed_seconds": elapsed,
    }
    write_json(metadata / "PREPARE_STATUS.json", status)
    return status


def run_command(command: list[str], cwd: Path, log: Path) -> float:
    started = time.perf_counter()
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(cwd / "rewrite" / "src")
    for name in THREAD_VARIABLES:
        environment[name] = "1"
    completed = subprocess.run(command, cwd=cwd, env=environment, text=True, capture_output=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(
        "$ " + subprocess.list2cmdline(command) + "\n\nSTDOUT\n" + completed.stdout + "\nSTDERR\n" + completed.stderr,
        encoding="utf-8",
    )
    if completed.returncode:
        raise RuntimeError(f"Command failed ({completed.returncode}); see {log}")
    return time.perf_counter() - started


def close(left: object, right: object, absolute: float, relative: float) -> bool:
    if pd.isna(left) and pd.isna(right):
        return True
    if pd.isna(left) or pd.isna(right):
        return False
    return math.isclose(float(left), float(right), abs_tol=absolute, rel_tol=relative)


def compare_csv(production: Path, rebuilt: Path, keys: list[str], absolute: float, relative: float) -> dict[str, object]:
    left, right = pd.read_csv(production), pd.read_csv(rebuilt)
    if list(left.columns) != list(right.columns):
        raise ValueError(f"CSV schema differs: {production.name}")
    if len(left) != len(right):
        raise ValueError(f"CSV row count differs: {production.name}")
    if keys:
        for key in keys:
            if key not in left:
                raise ValueError(f"Missing comparison key {key}: {production.name}")
        left = left.sort_values(keys, kind="mergesort", na_position="first").reset_index(drop=True)
        right = right.sort_values(keys, kind="mergesort", na_position="first").reset_index(drop=True)
        if not left[keys].fillna("<NA>").astype(str).equals(right[keys].fillna("<NA>").astype(str)):
            raise ValueError(f"CSV identity differs: {production.name}")
    numeric_columns, exact_columns = [], []
    max_absolute = 0.0
    for column in left.columns:
        left_numeric = pd.to_numeric(left[column], errors="coerce")
        right_numeric = pd.to_numeric(right[column], errors="coerce")
        numeric = ((left[column].isna() | left_numeric.notna()).all() and (right[column].isna() | right_numeric.notna()).all())
        if numeric:
            numeric_columns.append(column)
            for a, b in zip(left_numeric, right_numeric):
                if not close(a, b, absolute, relative):
                    raise ValueError(f"Numeric mismatch: {production.name}:{column}")
                if not pd.isna(a) and not pd.isna(b):
                    max_absolute = max(max_absolute, abs(float(a) - float(b)))
        else:
            exact_columns.append(column)
            if not left[column].fillna("<NA>").astype(str).equals(right[column].fillna("<NA>").astype(str)):
                raise ValueError(f"Text mismatch: {production.name}:{column}")
    return {"numeric_columns": "|".join(numeric_columns), "exact_columns": "|".join(exact_columns), "max_absolute_difference": max_absolute}


def compare_json_values(left: object, right: object, absolute: float, relative: float, ignored: set[str], path: str = "") -> float:
    if path in ignored or path.rsplit(".", 1)[-1] in ignored:
        return 0.0
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            raise ValueError(f"JSON keys differ at {path or '<root>'}")
        return max((compare_json_values(left[key], right[key], absolute, relative, ignored, f"{path}.{key}".strip(".")) for key in left), default=0.0)
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            raise ValueError(f"JSON list length differs at {path}")
        return max((compare_json_values(a, b, absolute, relative, ignored, f"{path}[{index}]") for index, (a, b) in enumerate(zip(left, right))), default=0.0)
    if isinstance(left, (int, float)) and not isinstance(left, bool) and isinstance(right, (int, float)) and not isinstance(right, bool):
        if not close(left, right, absolute, relative):
            raise ValueError(f"JSON numeric mismatch at {path}")
        return abs(float(left) - float(right))
    if left != right:
        raise ValueError(f"JSON value differs at {path}")
    return 0.0


def compare_outputs(production: Path, rebuilt: Path, rules_path: Path, output: Path) -> list[dict[str, object]]:
    rules = pd.read_csv(rules_path).fillna("")
    results = []
    for rule in rules.to_dict("records"):
        relative = str(rule["path"])
        left, right = production / relative, rebuilt / relative
        if not left.is_file() or not right.is_file():
            raise FileNotFoundError(f"Comparison input missing: {relative}")
        mode = str(rule["mode"])
        result: dict[str, object] = {"path": relative, "mode": mode, "status": "PASS"}
        if mode == "byte_exact":
            result["production_sha256"] = sha256(left)
            result["rebuilt_sha256"] = sha256(right)
            if result["production_sha256"] != result["rebuilt_sha256"]:
                raise ValueError(f"Byte mismatch: {relative}")
        elif mode == "numeric_csv":
            result.update(compare_csv(left, right, str(rule["key_columns"]).split("|") if rule["key_columns"] else [], float(rule["absolute_tolerance"]), float(rule["relative_tolerance"])))
        elif mode == "numeric_json":
            left_value = json.loads(left.read_text(encoding="utf-8"))
            right_value = json.loads(right.read_text(encoding="utf-8"))
            ignored = set(str(rule["ignore_fields"]).split("|")) if rule["ignore_fields"] else set()
            result["max_absolute_difference"] = compare_json_values(left_value, right_value, float(rule["absolute_tolerance"]), float(rule["relative_tolerance"]), ignored)
            result["ignored_fields"] = "|".join(sorted(ignored))
        else:
            raise ValueError(f"Unknown comparison mode: {mode}")
        results.append(result)
    pd.DataFrame(results).to_csv(output, index=False, lineterminator="\n")
    return results


def validate_pass_record(record_path: Path, run: Path) -> dict[str, object]:
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if record.get("status") != "PASS" or record.get("material_failures") != 0:
        raise ValueError(f"Validation record is not PASS: {record_path}")
    if record.get("protocol_hash") != PROTOCOL_HASH:
        raise ValueError(f"Validation protocol mismatch: {record_path}")
    if record.get("validated_manifest_hash") != sha256(run / "MANIFEST.json"):
        raise ValueError(f"Validation record is not bound to supplied run: {record_path}")
    return record


def validate_reporting(output: Path, pass_path: Path) -> dict[str, object]:
    required = (
        "RUN_MANIFEST.json", "OUTPUT_MANIFEST.csv", "RESULTS_REGISTER.csv",
        "EXHIBIT_REGISTER.csv", "figures/F_PREDICTION_LOSS_render_check.csv",
    )
    missing = [name for name in required if not (output / name).is_file()]
    if missing:
        raise ValueError(f"Reporting outputs missing: {missing}")
    run_manifest = json.loads((output / "RUN_MANIFEST.json").read_text(encoding="utf-8"))
    if run_manifest.get("protocol_hash") != PROTOCOL_HASH or run_manifest.get("reporting_only") is not True or run_manifest.get("empirical_fitting_performed") is not False:
        raise ValueError("Reporting run manifest is not display-only")
    manifest = pd.read_csv(output / "OUTPUT_MANIFEST.csv")
    for row in manifest.itertuples(index=False):
        path = resolve_inside(output, str(row.path))
        if not path.is_file() or path.stat().st_size != int(row.bytes) or sha256(path) != row.sha256:
            raise ValueError(f"Reporting output manifest mismatch: {row.path}")
    render = pd.read_csv(output / "figures/F_PREDICTION_LOSS_render_check.csv")
    if not render["status"].astype(str).eq("PASS").all():
        raise ValueError("Reporting render check failed")
    record = {
        "status": "PASS",
        "protocol_hash": PROTOCOL_HASH,
        "reporting_only": True,
        "empirical_fitting_performed": False,
        "source_validation_status_sha256": sha256(pass_path),
        "output_manifest_sha256": sha256(output / "OUTPUT_MANIFEST.csv"),
        "run_manifest_sha256": sha256(output / "RUN_MANIFEST.json"),
    }
    write_json(output.parent / "DISPLAY_REPORTING_PASS.json", record)
    return record


def execute_full(args: argparse.Namespace, workspace: Path) -> dict[str, object]:
    started = time.perf_counter()
    production = Path(args.production_run).resolve()
    production_pass = Path(args.production_pass_record).resolve()
    validate_pass_record(production_pass, production)
    artifacts = workspace / "artifacts"
    run = artifacts / "empirical_run"
    validation = artifacts / "rebuild_validation"
    reporting = artifacts / "reporting"
    logs = workspace / ".clean_rebuild" / "logs"
    python = sys.executable
    timings = {}
    timings["empirical_seconds"] = run_command([
        python, "-m", "phase_b.cli", "--repository-root", str(workspace),
        "--output", str(run), "--execute",
    ], workspace, logs / "01_empirical.log")
    timings["no_fit_finalizer_seconds"] = run_command([
        python, "-m", "phase_b.finalize_run", "--run", str(run), "--run-id", args.run_id,
    ], workspace, logs / "02_no_fit_finalizer.log")
    timings["independent_validation_seconds"] = run_command([
        python, str(workspace / "rewrite/validation/phase_b_empirical/validate_empirical_run.py"),
        "--run-directory", str(run), "--output-directory", str(validation),
    ], workspace, logs / "03_independent_validation.log")
    rebuild_pass = validation / "VALIDATION_STATUS.json"
    validate_pass_record(rebuild_pass, run)
    comparison_path = artifacts / "PRODUCTION_REBUILD_COMPARISON.csv"
    comparison_started = time.perf_counter()
    compare_outputs(production, run, Path(args.comparison_rules).resolve(), comparison_path)
    timings["comparison_seconds"] = time.perf_counter() - comparison_started
    timings["display_reporting_seconds"] = run_command([
        python, "-m", "phase_b_reporting.cli", "--run", str(run), "--output", str(reporting),
        "--run-id", args.run_id, "--validation-status", str(rebuild_pass),
        "--phase-a-results", str(workspace / "rewrite/results/RESULTS_REGISTER.csv"),
        "--phase-a-exhibits", str(workspace / "rewrite/results/EXHIBIT_REGISTER.csv"),
    ], workspace, logs / "04_display_reporting.log")
    validate_reporting(reporting, rebuild_pass)
    timings["total_seconds"] = time.perf_counter() - started
    status = {
        "status": "PASS",
        "protocol_hash": PROTOCOL_HASH,
        "run_id": args.run_id,
        "production_manifest_sha256": sha256(production / "MANIFEST.json"),
        "rebuilt_manifest_sha256": sha256(run / "MANIFEST.json"),
        "production_pass_sha256": sha256(production_pass),
        "rebuild_pass_sha256": sha256(rebuild_pass),
        "comparison_sha256": sha256(comparison_path),
        "timings": timings,
    }
    write_json(workspace / ".clean_rebuild" / "FULL_REBUILD_STATUS.json", status)
    return status


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--repository-root", default=".")
    command.add_argument("--workspace", required=True)
    command.add_argument("--allowlist", default=str(Path(__file__).with_name("SOURCE_ALLOWLIST.csv")))
    command.add_argument("--comparison-rules", default=str(Path(__file__).with_name("COMPARISON_RULES.csv")))
    mode = command.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--execute-full", action="store_true")
    command.add_argument("--production-run")
    command.add_argument("--production-pass-record")
    command.add_argument("--run-id")
    return command


def main() -> int:
    args = parser().parse_args()
    if args.execute_full and not all((args.production_run, args.production_pass_record, args.run_id)):
        raise ValueError("Full execution requires --production-run, --production-pass-record, and --run-id")
    workspace = Path(args.workspace).resolve()
    status = prepare(Path(args.repository_root), workspace, Path(args.allowlist).resolve())
    if args.execute_full:
        status = execute_full(args, workspace)
    print(json.dumps(status, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
