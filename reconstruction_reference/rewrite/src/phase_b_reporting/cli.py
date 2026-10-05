from __future__ import annotations

import argparse

from .processor import build_reporting_package


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Phase B reports from validated saved ledgers only.")
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--validation-status", required=True, help="Independent PASS status JSON outside the immutable run directory.")
    parser.add_argument("--phase-a-results", default="rewrite/results/RESULTS_REGISTER.csv")
    parser.add_argument("--phase-a-exhibits", default="rewrite/results/EXHIBIT_REGISTER.csv")
    parser.add_argument("--allow-synthetic", action="store_true", help="Tests only; forbidden for manuscript reporting.")
    args = parser.parse_args()
    build_reporting_package(args.run, args.output, args.phase_a_results, args.phase_a_exhibits, args.run_id, args.validation_status, args.allow_synthetic)


if __name__ == "__main__":
    main()
