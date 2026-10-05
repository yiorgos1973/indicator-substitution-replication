"""Build outcome-blind historical WDI context features from the frozen panel."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_WDI = ROOT / "data" / "processed" / "wdi_bridge_1980_2017_long.csv"
DEFAULT_P_LEDGER = ROOT / "rewrite" / "analysis" / "temporal" / "p_record_contribution_ledger.csv"
DEFAULT_HLO_LEDGER = ROOT / "rewrite" / "analysis" / "temporal" / "hlo_record_contribution_ledger.csv"
DEFAULT_ELIGIBLE_CELLS = ROOT / "rewrite" / "analysis" / "temporal" / "temporal_eligible_cells.csv"
DEFAULT_OUTPUT = ROOT / "rewrite" / "analysis" / "context"

FEATURES = {
    "SE.PRM.ENRR": "Primary enrolment",
    "SE.SEC.ENRR": "Secondary enrolment",
    "SE.XPD.TOTL.GD.ZS": "Government education expenditure",
    "SH.DYN.MORT": "Under-five mortality",
    "SP.DYN.TFRT.IN": "Fertility",
    "SP.URB.TOTL.IN.ZS": "Urban population share",
    "IT.NET.USER.ZS": "Internet use",
}
HISTORY_LENGTHS = (5, 10, 12)
WINDOWS = ((1, 2000, 2004), (2, 2005, 2009), (3, 2010, 2014), (4, 2015, 2017))
DIRECTIONS = ("HLO_to_P", "P_to_HLO")
FEATURE_SETS = {
    "all_seven": list(FEATURES),
    "education_inputs": ["SE.PRM.ENRR", "SE.SEC.ENRR", "SE.XPD.TOTL.GD.ZS"],
    "demographic_context": ["SH.DYN.MORT", "SP.DYN.TFRT.IN", "SP.URB.TOTL.IN.ZS"],
    "demographic_plus_internet": ["SH.DYN.MORT", "SP.DYN.TFRT.IN", "SP.URB.TOTL.IN.ZS", "IT.NET.USER.ZS"],
    "six_without_education_expenditure": [c for c in FEATURES if c != "SE.XPD.TOTL.GD.ZS"],
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_wdi(path: Path) -> pd.DataFrame:
    wdi = pd.read_csv(path)
    required = {"iso3", "year", "indicator_code", "value"}
    assert required.issubset(wdi.columns)
    wdi = wdi[wdi["indicator_code"].isin(FEATURES)].copy()
    wdi["year"] = wdi["year"].astype(int)
    assert not wdi.duplicated(["iso3", "year", "indicator_code"]).any()
    return wdi


def coverage_reason(observed: int, requested: int, supported: int) -> str:
    if supported == 0:
        return "unsupported_source_period"
    if observed == 0:
        return "no_observed_history"
    proportion = observed / requested
    if proportion < 0.5:
        return "sparse_history_below_50pct"
    if proportion < 1:
        return "partial_history"
    return "complete_history"


def requested_years(window_start: int, window_end: int, construction: str, history_years: int | None) -> list[int]:
    if construction == "same_window_comparator":
        return list(range(window_start, window_end + 1))
    return list(range(window_start - int(history_years), window_start))


def build_block_ledger(wdi: pd.DataFrame) -> pd.DataFrame:
    countries = sorted(wdi["iso3"].unique())
    min_year, max_year = int(wdi["year"].min()), int(wdi["year"].max())
    rows = []
    constructions = [(f"block_preceding_{n}y", n) for n in HISTORY_LENGTHS]
    constructions.append(("same_window_comparator", None))

    for iso3 in countries:
        for window_number, start, end in WINDOWS:
            for direction in DIRECTIONS:
                for construction, history_years in constructions:
                    for indicator_code, indicator_name in FEATURES.items():
                        years = requested_years(start, end, construction, history_years)
                        for year in years:
                            rows.append({
                                "iso3": iso3,
                                "scheme": "five_year",
                                "window_number": window_number,
                                "window_start": start,
                                "window_end": end,
                                "terminal_window_partial": end - start + 1 < 5,
                                "target_direction": direction,
                                "schedule_target": "none_block_anchor",
                                "construction": construction,
                                "history_years": history_years,
                                "indicator_code": indicator_code,
                                "indicator_name": indicator_name,
                                "contribution_id": f"{iso3}_{start}_{construction}",
                                "assessment_date": np.nan,
                                "anchor_year": start,
                                "date_certainty": "window_anchor_exact",
                                "requested_year": year,
                                "source_year_supported": min_year <= year <= max_year,
                                "strict_pre_anchor": year < start if construction != "same_window_comparator" else np.nan,
                                "date_rule": "year_before_window_start" if construction != "same_window_comparator" else "same_window_nonhistorical_comparator",
                                "aggregation_weight": 1.0,
                            })

    ledger = pd.DataFrame(rows).merge(
        wdi[["iso3", "year", "indicator_code", "value"]],
        left_on=["iso3", "requested_year", "indicator_code"],
        right_on=["iso3", "year", "indicator_code"],
        how="left",
    ).drop(columns="year")
    ledger = ledger.rename(columns={"value": "observed_value"})
    group = ["iso3", "window_start", "target_direction", "construction", "indicator_code"]
    ledger["requested_year_count"] = ledger.groupby(group)["requested_year"].transform("size")
    ledger["observed_year_count"] = ledger["observed_value"].notna().groupby([ledger[c] for c in group]).transform("sum")
    ledger["nominal_annual_weight"] = 1 / ledger["requested_year_count"]
    ledger["annual_weight"] = np.where(
        ledger["observed_value"].notna(),
        1 / ledger["observed_year_count"].replace(0, np.nan),
        0.0,
    )
    ledger["effective_aggregation_weight"] = 1.0
    ledger["effective_total_weight"] = ledger["annual_weight"]
    ledger["missing_status"] = np.select(
        [ledger["observed_value"].notna(), ~ledger["source_year_supported"]],
        ["observed", "unsupported_source_period"],
        default="source_value_missing",
    )
    ledger["observed_year"] = ledger["requested_year"].where(ledger["observed_value"].notna())
    ledger["source_vintage_status"] = "retrospective_revised_snapshot; historical_release_availability_not_audited"
    return ledger


def summarize_ledger(ledger: pd.DataFrame, assessment_specific: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    keys = [
        "iso3", "scheme", "window_number", "window_start", "window_end",
        "terminal_window_partial", "target_direction", "schedule_target",
        "construction", "history_years", "indicator_code", "indicator_name",
    ]
    ledger = ledger.copy()
    ledger["weighted_value"] = ledger["observed_value"] * ledger["effective_total_weight"]

    rows = []
    for key, part in ledger.groupby(keys, dropna=False, sort=False):
        requested = len(part)
        observed = int(part["observed_value"].notna().sum())
        supported = int(part["source_year_supported"].sum())
        value = part["weighted_value"].sum(min_count=1)
        if assessment_specific:
            contribution_coverage = (
                part.groupby("contribution_id", sort=False)
                .agg(
                    aggregation_weight=("aggregation_weight", "first"),
                    requested=("requested_year", "size"),
                    observed=("observed_value", "count"),
                )
            )
            weighted_coverage = float(
                (contribution_coverage["aggregation_weight"] * contribution_coverage["observed"] / contribution_coverage["requested"]).sum()
            )
            contribution_count = len(contribution_coverage)
            observed_contribution_count = int((contribution_coverage["observed"] > 0).sum())
        else:
            weighted_coverage = observed / requested
            contribution_count = 1
            observed_contribution_count = int(observed > 0)
        row = dict(zip(keys, key))
        row.update({
            "context_mean": value,
            "requested_years": "|".join(str(int(x)) for x in sorted(part["requested_year"].unique())),
            "observed_years": "|".join(str(int(x)) for x in sorted(part.loc[part["observed_value"].notna(), "requested_year"].unique())),
            "requested_year_count": requested,
            "observed_year_count": observed,
            "coverage_proportion": observed / requested,
            "weighted_coverage_proportion": weighted_coverage,
            "contribution_count": contribution_count,
            "observed_contribution_count": observed_contribution_count,
            "missingness_reason": coverage_reason(observed, requested, supported),
            "all_reference_years_strictly_pre_anchor": bool(part["strict_pre_anchor"].dropna().all()),
        })
        rows.append(row)
    coverage = pd.DataFrame(rows)

    matrix_index = [
        "iso3", "scheme", "window_number", "window_start", "window_end",
        "terminal_window_partial", "target_direction", "schedule_target",
        "construction", "history_years",
    ]
    features = coverage.pivot(index=matrix_index, columns="indicator_code", values="context_mean").reset_index()
    features.columns.name = None
    coverage_matrix = coverage.pivot(index=matrix_index, columns="indicator_code", values="weighted_coverage_proportion").reset_index()
    coverage_matrix.columns = [*matrix_index, *[f"{c}__coverage" for c in coverage_matrix.columns[len(matrix_index):]]]
    return coverage, features, coverage_matrix


def coverage_summaries(coverage: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_summary = (
        coverage.groupby(["target_direction", "construction", "indicator_code", "indicator_name"], as_index=False)
        .agg(
            country_window_cells=("iso3", "size"),
            cells_with_any_year=("observed_year_count", lambda x: int((x > 0).sum())),
            cells_at_least_80pct=("weighted_coverage_proportion", lambda x: int((x >= 0.8).sum())),
            cells_complete=("weighted_coverage_proportion", lambda x: int((x == 1).sum())),
            median_weighted_coverage=("weighted_coverage_proportion", "median"),
        )
    )
    cell_minimum = (
        coverage.groupby(
            ["target_direction", "construction", "iso3", "window_start"], as_index=False
        )["weighted_coverage_proportion"].min()
    )
    all_feature_summary = (
        cell_minimum.groupby(["target_direction", "construction"], as_index=False)
        .agg(
            country_window_cells=("iso3", "size"),
            all_features_with_any_year=("weighted_coverage_proportion", lambda x: int((x > 0).sum())),
            all_features_at_least_80pct=("weighted_coverage_proportion", lambda x: int((x >= 0.8).sum())),
            all_features_complete=("weighted_coverage_proportion", lambda x: int((x == 1).sum())),
            median_minimum_feature_coverage=("weighted_coverage_proportion", "median"),
        )
    )
    return feature_summary, all_feature_summary


def feature_set_coverage(coverage: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature_set_id, indicators in FEATURE_SETS.items():
        selected = coverage[coverage["indicator_code"].isin(indicators)]
        minimum = (
            selected.groupby(
                ["target_direction", "construction", "iso3", "window_start"], as_index=False
            )["weighted_coverage_proportion"].min()
        )
        for (direction, construction), part in minimum.groupby(["target_direction", "construction"]):
            rows.append({
                "target_direction": direction,
                "construction": construction,
                "feature_set_id": feature_set_id,
                "indicator_codes": "|".join(indicators),
                "feature_count": len(indicators),
                "country_window_cells": len(part),
                "cells_all_features_any": int((part["weighted_coverage_proportion"] > 0).sum()),
                "cells_all_features_at_least_80pct": int((part["weighted_coverage_proportion"] >= 0.8).sum()),
                "cells_all_features_complete": int((part["weighted_coverage_proportion"] == 1).sum()),
                "median_minimum_feature_coverage": part["weighted_coverage_proportion"].median(),
            })
    return pd.DataFrame(rows)


def _five_year_rows(frame: pd.DataFrame) -> pd.DataFrame:
    values = frame["scheme"].astype(str).str.lower().str.replace("-", "_")
    return frame[values.isin(["five_year", "5_year", "5year"])].copy()


def load_assessment_schedule(p_path: Path, hlo_path: Path, eligible_path: Path) -> pd.DataFrame:
    p = _five_year_rows(pd.read_csv(p_path))
    eligible = _five_year_rows(pd.read_csv(eligible_path))
    eligible_keys = eligible[["iso3", "window_start", "window_end"]].drop_duplicates()
    p = p.merge(eligible_keys, on=["iso3", "window_start", "window_end"], how="inner")
    p["target_direction"] = "HLO_to_P"
    p["schedule_target"] = "P"
    p["assessment_date"] = p["measurement_year"]
    p["anchor_year"] = p["calendar_year_floor"].astype(int)
    p["aggregation_weight"] = p["normalized_composite_weight"]
    p["date_certainty"] = np.where(
        p["fractional_measurement_date"].astype(bool),
        p["date_precision_status"].astype(str) + "; anchor=floored_calendar_year",
        p["date_precision_status"].astype(str),
    )
    p["contribution_id"] = "P_" + p["record_id"].astype(str) + "_" + p.index.astype(str)

    hlo = _five_year_rows(pd.read_csv(hlo_path))
    hlo = hlo.merge(eligible_keys, on=["iso3", "window_start", "window_end"], how="inner")
    hlo["target_direction"] = "P_to_HLO"
    hlo["schedule_target"] = "HLO"
    hlo["assessment_date"] = hlo["year"]
    hlo["anchor_year"] = hlo["year"].astype(int)
    hlo["aggregation_weight"] = hlo["composite_weight"]
    hlo["date_certainty"] = "recorded_year"
    hlo["contribution_id"] = "HLO_" + hlo.index.astype(str)

    columns = [
        "iso3", "scheme", "window_number", "window_start", "window_end",
        "target_direction", "schedule_target", "assessment_date", "anchor_year",
        "date_certainty", "aggregation_weight", "contribution_id",
    ]
    schedule = pd.concat([p[columns], hlo[columns]], ignore_index=True)
    weight_sums = schedule.groupby(["iso3", "window_start", "target_direction"])["aggregation_weight"].sum()
    assert np.allclose(weight_sums, 1.0, atol=1e-9)
    return schedule


def build_assessment_ledger(wdi: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    min_year, max_year = int(wdi["year"].min()), int(wdi["year"].max())
    rows = []
    for record in schedule.itertuples(index=False):
        for history_years in HISTORY_LENGTHS:
            for indicator_code, indicator_name in FEATURES.items():
                for year in range(record.anchor_year - history_years, record.anchor_year):
                    rows.append({
                        "iso3": record.iso3,
                        "scheme": "five_year",
                        "window_number": record.window_number,
                        "window_start": record.window_start,
                        "window_end": record.window_end,
                        "terminal_window_partial": record.window_end - record.window_start + 1 < 5,
                        "target_direction": record.target_direction,
                        "schedule_target": record.schedule_target,
                        "construction": f"assessment_specific_{history_years}y",
                        "history_years": history_years,
                        "indicator_code": indicator_code,
                        "indicator_name": indicator_name,
                        "contribution_id": record.contribution_id,
                        "assessment_date": record.assessment_date,
                        "anchor_year": record.anchor_year,
                        "date_certainty": record.date_certainty,
                        "requested_year": year,
                        "source_year_supported": min_year <= year <= max_year,
                        "strict_pre_anchor": year < record.anchor_year,
                        "date_rule": "year_before_assessment_anchor",
                        "aggregation_weight": record.aggregation_weight,
                    })

    ledger = pd.DataFrame(rows).merge(
        wdi[["iso3", "year", "indicator_code", "value"]],
        left_on=["iso3", "requested_year", "indicator_code"],
        right_on=["iso3", "year", "indicator_code"],
        how="left",
    ).drop(columns="year").rename(columns={"value": "observed_value"})
    contribution_keys = ["contribution_id", "construction", "indicator_code"]
    ledger["requested_year_count"] = ledger.groupby(contribution_keys)["requested_year"].transform("size")
    ledger["observed_year_count"] = ledger["observed_value"].notna().groupby([ledger[c] for c in contribution_keys]).transform("sum")
    ledger["nominal_annual_weight"] = 1 / ledger["requested_year_count"]
    ledger["annual_weight"] = np.where(
        ledger["observed_value"].notna(),
        1 / ledger["observed_year_count"].replace(0, np.nan),
        0.0,
    )
    ledger["missing_status"] = np.select(
        [ledger["observed_value"].notna(), ~ledger["source_year_supported"]],
        ["observed", "unsupported_source_period"],
        default="source_value_missing",
    )
    ledger["observed_year"] = ledger["requested_year"].where(ledger["observed_value"].notna())
    ledger["source_vintage_status"] = "retrospective_revised_snapshot; historical_release_availability_not_audited"

    contribution = (
        ledger.assign(history_piece=ledger["observed_value"] * ledger["annual_weight"])
        .groupby(
            ["iso3", "window_start", "target_direction", "construction", "indicator_code", "contribution_id"],
            as_index=False,
        )
        .agg(
            contribution_history_value=("history_piece", lambda x: x.sum(min_count=1)),
            aggregation_weight=("aggregation_weight", "first"),
        )
    )
    contribution["available_weight_sum"] = contribution["aggregation_weight"].where(
        contribution["contribution_history_value"].notna(), 0
    ).groupby([
        contribution["iso3"], contribution["window_start"], contribution["target_direction"],
        contribution["construction"], contribution["indicator_code"],
    ]).transform("sum")
    contribution["effective_aggregation_weight"] = np.where(
        contribution["contribution_history_value"].notna(),
        contribution["aggregation_weight"] / contribution["available_weight_sum"].replace(0, np.nan),
        0.0,
    )
    ledger = ledger.merge(
        contribution[[
            "contribution_id", "construction", "indicator_code",
            "contribution_history_value", "effective_aggregation_weight",
        ]],
        on=["contribution_id", "construction", "indicator_code"],
        how="left",
    )
    ledger["effective_total_weight"] = ledger["annual_weight"] * ledger["effective_aggregation_weight"]
    return ledger


def write_protocol_files(output: Path, assessment_ready: bool) -> None:
    protocol_dir = ROOT / "rewrite" / "protocol" / "agent2"
    protocol_dir.mkdir(parents=True, exist_ok=True)
    protocol = {
        "status": "candidate_not_approved",
        "empirical_modelling_approved": False,
        "context_construction_candidates": ["block_preceding", "assessment_specific"],
        "history_duration_candidates_years": list(HISTORY_LENGTHS),
        "same_window_context": "comparator_only",
        "candidate_features": list(FEATURES),
        "excluded_primary_features": ["NY.GDP.PCAP.PP.KD", "SP.POP.TOTL"],
        "interpolation": "none",
        "annual_aggregation": "equal weight among observed requested years; coverage retained separately",
        "assessment_aggregation": "target-construction weights, renormalized only across contributions with an observed feature history",
        "assessment_specific_empirical_outputs_ready": assessment_ready,
        "selection_basis": "source meaning and coverage only; no target association or predictive performance inspected",
        "unapproved_fields": [
            "primary_context_construction", "primary_history_duration", "coverage_rule",
            "retained_feature_vector", "feature_transformations", "direction_hierarchy",
            "models_and_tuning", "validation_splits", "loss_weighting", "rank_aggregation",
            "uncertainty_procedure",
        ],
    }
    (protocol_dir / "protocol_candidate.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")


def write_input_manifest(paths: list[tuple[str, Path, bool]]) -> None:
    input_dir = ROOT / "rewrite" / "analysis" / "inputs" / "context"
    input_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for role, path, required in paths:
        exists = path.exists()
        rows.append({
            "input_role": role,
            "repository_relative_path": str(path.relative_to(ROOT)),
            "required_for_block_preceding": required and role == "frozen_annual_wdi_panel",
            "required_for_assessment_specific": required,
            "exists": exists,
            "sha256": sha256(path) if exists else "",
            "bytes": path.stat().st_size if exists else "",
            "status": "available" if exists else "pending_from_measurement_stage",
        })
    pd.DataFrame(rows).to_csv(input_dir / "INPUT_MANIFEST.csv", index=False)

    download_manifest = ROOT / "data" / "raw" / "wdi_bridge_1980_2017" / "wdi_download_manifest.csv"
    if download_manifest.exists():
        vintage = pd.read_csv(download_manifest)
        vintage = vintage[vintage["indicator_code"].isin(FEATURES)].copy()
        vintage["historical_reference_year_alignment"] = "audited_in_annual_provenance_ledgers"
        vintage["historical_release_availability"] = "not_established_from_retrospective_WDI_snapshot"
        vintage.to_csv(input_dir / "WDI_SOURCE_VINTAGE_AUDIT.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wdi", type=Path, default=DEFAULT_WDI)
    parser.add_argument("--p-ledger", type=Path, default=DEFAULT_P_LEDGER)
    parser.add_argument("--hlo-ledger", type=Path, default=DEFAULT_HLO_LEDGER)
    parser.add_argument("--eligible-cells", type=Path, default=DEFAULT_ELIGIBLE_CELLS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    wdi = load_wdi(args.wdi)
    block_ledger = build_block_ledger(wdi)
    assert block_ledger.loc[block_ledger["construction"] != "same_window_comparator", "strict_pre_anchor"].all()
    block_coverage, block_features, block_coverage_matrix = summarize_ledger(block_ledger, assessment_specific=False)
    block_feature_summary, block_all_feature_summary = coverage_summaries(block_coverage)
    block_ledger.to_csv(args.output / "block_preceding_annual_provenance_ledger.csv", index=False)
    block_coverage.to_csv(args.output / "block_preceding_coverage_long.csv", index=False)
    block_features.to_csv(args.output / "block_preceding_feature_matrix.csv", index=False)
    block_coverage_matrix.to_csv(args.output / "block_preceding_coverage_matrix.csv", index=False)
    block_feature_summary.to_csv(args.output / "block_preceding_coverage_by_feature_summary.csv", index=False)
    block_all_feature_summary.to_csv(args.output / "block_preceding_coverage_all_features_summary.csv", index=False)

    measurement_inputs_ready = args.p_ledger.exists() and args.hlo_ledger.exists() and args.eligible_cells.exists()
    eligible_count = 0
    if args.eligible_cells.exists():
        eligible = _five_year_rows(pd.read_csv(args.eligible_cells))
        eligible_keys = eligible[["iso3", "window_start", "window_end"]].drop_duplicates()
        eligible_count = len(eligible_keys)
        aligned_block_coverage = block_coverage.merge(
            eligible_keys, on=["iso3", "window_start", "window_end"], how="inner"
        )
        aligned_block_feature_summary, aligned_block_all_feature_summary = coverage_summaries(aligned_block_coverage)
        aligned_block_feature_sets = feature_set_coverage(aligned_block_coverage)
        aligned_block_coverage.to_csv(args.output / "aligned_cell_block_preceding_coverage_long.csv", index=False)
        aligned_block_feature_summary.to_csv(args.output / "aligned_cell_block_preceding_coverage_by_feature_summary.csv", index=False)
        aligned_block_all_feature_summary.to_csv(args.output / "aligned_cell_block_preceding_coverage_all_features_summary.csv", index=False)
        aligned_block_feature_sets.to_csv(args.output / "aligned_cell_block_preceding_candidate_vector_coverage.csv", index=False)

    assessment_ready = measurement_inputs_ready
    assessment_rows = 0
    if assessment_ready:
        schedule = load_assessment_schedule(args.p_ledger, args.hlo_ledger, args.eligible_cells)
        schedule.to_csv(args.output / "assessment_specific_schedule.csv", index=False)
        assessment_ledger = build_assessment_ledger(wdi, schedule)
        assert assessment_ledger["strict_pre_anchor"].all()
        assessment_coverage, assessment_features, assessment_coverage_matrix = summarize_ledger(
            assessment_ledger, assessment_specific=True
        )
        assessment_feature_summary, assessment_all_feature_summary = coverage_summaries(assessment_coverage)
        assessment_feature_sets = feature_set_coverage(assessment_coverage)
        assessment_ledger.to_csv(args.output / "assessment_specific_annual_provenance_ledger.csv", index=False)
        assessment_coverage.to_csv(args.output / "assessment_specific_coverage_long.csv", index=False)
        assessment_features.to_csv(args.output / "assessment_specific_feature_matrix.csv", index=False)
        assessment_coverage_matrix.to_csv(args.output / "assessment_specific_coverage_matrix.csv", index=False)
        assessment_feature_summary.to_csv(args.output / "assessment_specific_coverage_by_feature_summary.csv", index=False)
        assessment_all_feature_summary.to_csv(args.output / "assessment_specific_coverage_all_features_summary.csv", index=False)
        assessment_feature_sets.to_csv(args.output / "assessment_specific_candidate_vector_coverage.csv", index=False)
        assessment_rows = len(assessment_ledger)

    status = {
        "wdi_input": str(args.wdi.relative_to(ROOT)),
        "wdi_sha256": sha256(args.wdi),
        "wdi_rows": len(wdi),
        "countries": int(wdi["iso3"].nunique()),
        "features": list(FEATURES),
        "block_ledger_rows": len(block_ledger),
        "block_feature_rows": len(block_features),
        "assessment_specific_ready": assessment_ready,
        "eligible_five_year_cells": eligible_count,
        "assessment_ledger_rows": assessment_rows,
        "empirical_modelling_performed": False,
    }
    (args.output / "context_build_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    write_protocol_files(args.output, assessment_ready)
    write_input_manifest([
        ("frozen_annual_wdi_panel", args.wdi, True),
        ("P_measurement_schedule", args.p_ledger, True),
        ("HLO_measurement_schedule", args.hlo_ledger, True),
        ("eligible_temporal_cells", args.eligible_cells, True),
        ("WDI_download_manifest", ROOT / "data" / "raw" / "wdi_bridge_1980_2017" / "wdi_download_manifest.csv", False),
        ("WDI_indicator_metadata", ROOT / "data" / "raw" / "wdi_bridge_1980_2017" / "wdi_indicator_metadata.csv", False),
    ])
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
