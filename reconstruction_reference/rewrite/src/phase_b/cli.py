"""Command-line entry point. Preflight is read-only with respect to empirical fitting."""

from __future__ import annotations

import argparse
from pathlib import Path

from .orchestrator import build_preflight, execute_plan, write_preflight


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", default=".")
    parser.add_argument("--protocol", default="rewrite/protocol/author_approval/protocol_for_author_approval.json")
    parser.add_argument("--approval", default="rewrite/protocol/approvals/author_approval_2026-09-30.json")
    parser.add_argument("--measurement", default="rewrite/protocol/readiness/measurement_readiness_13fb5775.json")
    parser.add_argument("--sample-manifest", default="rewrite/analysis/protocol_final/scenario_sample_manifest.csv")
    parser.add_argument("--feature-manifest", default="rewrite/analysis/protocol_final/scenario_feature_manifest.csv")
    parser.add_argument("--output", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    root = Path(args.repository_root).resolve()
    resolve = lambda value: value if Path(value).is_absolute() else root / value
    gate, bundle, jobs = build_preflight(root, resolve(args.protocol), resolve(args.approval), resolve(args.measurement), resolve(args.sample_manifest), resolve(args.feature_manifest))
    write_preflight(args.output, gate, bundle, jobs)
    if args.execute:
        execute_plan(bundle, jobs, args.output, gate)


if __name__ == "__main__":
    main()
