"""Build outcome-blind context manifests for the proposed final protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
CONTEXT = ROOT / "rewrite" / "analysis" / "context"
TEMPORAL = ROOT / "rewrite" / "analysis" / "temporal"
OUTPUT = ROOT / "rewrite" / "analysis" / "protocol_final"

FEATURES = [
    "SE.PRM.ENRR",
    "SE.SEC.ENRR",
    "SE.XPD.TOTL.GD.ZS",
    "SH.DYN.MORT",
    "SP.DYN.TFRT.IN",
    "SP.URB.TOTL.IN.ZS",
    "IT.NET.USER.ZS",
]
DIRECTIONS = ["HLO_to_P", "P_to_HLO"]
COMPOSITION_EXCLUSIONS = {"BRA", "CAN", "KAZ", "MEX", "SRB", "USA", "ZAF"}
CELL_KEYS = ["iso3", "window_number", "window_start", "window_end"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact_usable(observed: pd.Series, requested: pd.Series) -> pd.Series:
    return 5 * observed >= 4 * requested


def text_join(values: pd.Series) -> str:
    return "|".join(sorted(values.dropna().astype(str).unique()))


def sample_hash(rows: pd.DataFrame) -> str:
    identities = sorted(f"{r.iso3}:{r.window_start}:{r.window_end}" for r in rows.itertuples())
    return hashlib.sha256("\n".join(identities).encode("utf-8")).hexdigest()


def load_cells() -> tuple[pd.DataFrame, pd.DataFrame]:
    cells = pd.read_csv(TEMPORAL / "temporal_eligible_cells.csv")
    cells = cells[cells["eligible_aligned_cell"]].copy()
    five = cells[cells["scheme"] == "5_year"].copy()
    four = cells[cells["scheme"] == "4_year"].copy()
    assert len(five) == 58
    assert len(four) == 58
    assert five[["p_score", "hlo_score"]].notna().all().all()
    assert four[["p_score", "hlo_score"]].notna().all().all()
    return five, four


def prepare_block_features() -> pd.DataFrame:
    coverage = pd.read_csv(CONTEXT / "aligned_cell_block_preceding_coverage_long.csv")
    coverage = coverage[
        (coverage["target_direction"] == "HLO_to_P")
        & coverage["construction"].isin(["block_preceding_5y", "block_preceding_10y", "block_preceding_12y"])
        & coverage["indicator_code"].isin(FEATURES)
    ].copy()
    coverage["history_years"] = coverage["history_years"].astype(int)
    coverage["minimum_observed_years"] = (4 * coverage["requested_year_count"] + 4) // 5
    coverage["feature_usable"] = exact_usable(
        coverage["observed_year_count"], coverage["requested_year_count"]
    )
    coverage["protocol_context_value"] = coverage["context_mean"].where(coverage["feature_usable"])
    coverage["feature_order"] = coverage["indicator_code"].map({v: i + 1 for i, v in enumerate(FEATURES)})
    coverage["coverage_rule"] = "5*n_obs>=4*T"
    return coverage


def build_four_year_features(four: pd.DataFrame) -> pd.DataFrame:
    wdi = pd.read_csv(ROOT / "data" / "processed" / "wdi_bridge_1980_2017_long.csv")
    wdi = wdi[wdi["indicator_code"].isin(FEATURES)].copy()
    assert not wdi.duplicated(["iso3", "year", "indicator_code"]).any()
    rows = []
    values = wdi.set_index(["iso3", "year", "indicator_code"])["value"]
    names = wdi.drop_duplicates("indicator_code").set_index("indicator_code")["indicator_name"]
    for cell in four.itertuples(index=False):
        requested_years = list(range(int(cell.window_start) - 5, int(cell.window_start)))
        for order, code in enumerate(FEATURES, start=1):
            series = pd.Series(
                [values.get((cell.iso3, year, code), pd.NA) for year in requested_years],
                index=requested_years,
                dtype="Float64",
            )
            observed = series.dropna()
            usable = 5 * len(observed) >= 4 * len(series)
            rows.append({
                "iso3": cell.iso3,
                "window_number": cell.window_number,
                "window_start": cell.window_start,
                "window_end": cell.window_end,
                "construction": "four_year_preceding_5y",
                "history_years": 5,
                "indicator_code": code,
                "indicator_name": names[code],
                "feature_order": order,
                "context_mean": observed.mean() if len(observed) else pd.NA,
                "protocol_context_value": observed.mean() if usable else pd.NA,
                "requested_years": "|".join(map(str, requested_years)),
                "observed_years": "|".join(map(str, observed.index)),
                "requested_year_count": len(series),
                "observed_year_count": len(observed),
                "minimum_observed_years": 4,
                "feature_usable": usable,
                "coverage_rule": "5*n_obs>=4*T",
            })
    return pd.DataFrame(rows)


def prepare_assessment_features() -> tuple[pd.DataFrame, pd.DataFrame]:
    ledger = pd.read_csv(CONTEXT / "assessment_specific_annual_provenance_ledger.csv")
    ledger = ledger[
        (ledger["construction"] == "assessment_specific_5y")
        & ledger["indicator_code"].isin(FEATURES)
    ].copy()
    contribution_keys = [
        *CELL_KEYS, "target_direction", "schedule_target", "construction", "history_years",
        "indicator_code", "indicator_name", "contribution_id",
    ]
    contributions = (
        ledger.groupby(contribution_keys, as_index=False)
        .agg(
            aggregation_weight=("aggregation_weight", "first"),
            requested_year_count=("requested_year_count", "first"),
            observed_year_count=("observed_year_count", "first"),
            contribution_history_value=("contribution_history_value", "first"),
        )
    )
    contributions["positive_weight"] = contributions["aggregation_weight"] > 0
    contributions["minimum_observed_years"] = (4 * contributions["requested_year_count"] + 4) // 5
    contributions["contribution_usable"] = (
        ~contributions["positive_weight"]
        | exact_usable(contributions["observed_year_count"], contributions["requested_year_count"])
    )
    contributions["coverage_rule"] = "5*n_obs>=4*T"

    feature_keys = contribution_keys[:-1]
    rows = []
    for key, part in contributions.groupby(feature_keys, sort=True):
        required = part[part["positive_weight"]]
        usable = bool(required["contribution_usable"].all())
        row = dict(zip(feature_keys, key))
        row.update({
            "feature_order": FEATURES.index(row["indicator_code"]) + 1,
            "positive_weight_contribution_count": len(required),
            "usable_positive_weight_contribution_count": int(required["contribution_usable"].sum()),
            "original_weight_sum": required["aggregation_weight"].sum(),
            "feature_usable": usable,
            "protocol_context_value": (
                (required["aggregation_weight"] * required["contribution_history_value"]).sum()
                if usable else pd.NA
            ),
            "contribution_ids": text_join(required["contribution_id"]),
            "unusable_contribution_ids": text_join(required.loc[~required["contribution_usable"], "contribution_id"]),
            "contextual_weights_renormalized": False,
            "coverage_rule": "every positive-weight contribution satisfies 5*n_obs>=4*T",
        })
        rows.append(row)
    return pd.DataFrame(rows), contributions


def eligibility(features: pd.DataFrame, extra_keys: list[str] | None = None) -> pd.DataFrame:
    keys = [*CELL_KEYS, *(extra_keys or [])]
    rows = []
    for key, part in features.groupby(keys, sort=True):
        row = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        usable = part.loc[part["feature_usable"], "indicator_code"]
        missing = part.loc[~part["feature_usable"], "indicator_code"]
        row.update({
            "usable_feature_count": int(part["feature_usable"].sum()),
            "missing_feature_count": int((~part["feature_usable"]).sum()),
            "usable_feature_codes": text_join(usable),
            "missing_feature_codes": text_join(missing),
            "row_eligible": int(part["feature_usable"].sum()) >= 5,
            "complete_seven_features": bool(part["feature_usable"].all()),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def coverage_corrections(
    assessment_features: pd.DataFrame,
    contributions: pd.DataFrame,
) -> pd.DataFrame:
    historical = pd.read_csv(CONTEXT / "assessment_specific_coverage_long.csv")
    historical = historical[
        (historical["construction"] == "assessment_specific_5y")
        & historical["indicator_code"].isin(FEATURES)
    ].copy()
    historical["old_float_classification"] = historical["weighted_coverage_proportion"] >= 0.8
    historical["corrected_integer_classification"] = exact_usable(
        historical["observed_year_count"], historical["requested_year_count"]
    )

    row_keys = [*CELL_KEYS, "target_direction"]
    row_effects = (
        historical.groupby(row_keys, as_index=False)
        .agg(
            old_usable_feature_count=("old_float_classification", "sum"),
            corrected_usable_feature_count=("corrected_integer_classification", "sum"),
        )
    )
    row_effects["old_row_eligible"] = row_effects["old_usable_feature_count"] >= 5
    row_effects["corrected_row_eligible"] = row_effects["corrected_usable_feature_count"] >= 5

    contribution_details = contributions.copy()
    contribution_details["contribution_count_detail"] = (
        contribution_details["contribution_id"].astype(str)
        + ":" + contribution_details["observed_year_count"].astype(str)
        + "/" + contribution_details["requested_year_count"].astype(str)
        + "@" + contribution_details["aggregation_weight"].map(lambda x: f"{x:.17g}")
    )
    detail_keys = [*row_keys, "indicator_code"]
    contribution_details = (
        contribution_details.groupby(detail_keys, as_index=False)
        .agg(
            contribution_ids=("contribution_id", text_join),
            contribution_count_details=("contribution_count_detail", lambda x: "|".join(sorted(x))),
        )
    )

    corrections = historical[
        historical["old_float_classification"] != historical["corrected_integer_classification"]
    ].copy()
    corrections = corrections.merge(row_effects, on=row_keys, how="left", validate="many_to_one")
    corrections = corrections.merge(contribution_details, on=detail_keys, how="left", validate="one_to_one")
    fixed_cols = [*detail_keys, "feature_usable", "positive_weight_contribution_count", "usable_positive_weight_contribution_count"]
    corrections = corrections.merge(
        assessment_features[fixed_cols].rename(columns={"feature_usable": "fixed_contribution_feature_usable"}),
        on=detail_keys,
        how="left",
        validate="one_to_one",
    )
    corrections["feature_classification_effect"] = (
        corrections["old_float_classification"].astype(str)
        + "_to_" + corrections["corrected_integer_classification"].astype(str)
    )
    corrections["row_eligibility_effect"] = (
        corrections["old_row_eligible"].astype(str)
        + "_to_" + corrections["corrected_row_eligible"].astype(str)
    )
    columns = [
        *row_keys, "schedule_target", "indicator_code", "indicator_name", "contribution_ids",
        "contribution_count_details", "requested_year_count", "observed_year_count",
        "weighted_coverage_proportion", "old_float_classification",
        "corrected_integer_classification", "feature_classification_effect",
        "old_usable_feature_count", "corrected_usable_feature_count", "old_row_eligible",
        "corrected_row_eligible", "row_eligibility_effect", "fixed_contribution_feature_usable",
        "positive_weight_contribution_count", "usable_positive_weight_contribution_count",
    ]
    return corrections[columns].sort_values(row_keys + ["indicator_code"]).reset_index(drop=True)


def add_cell_scores(manifest: pd.DataFrame, cells: pd.DataFrame) -> pd.DataFrame:
    columns = [
        *CELL_KEYS, "canonical_economy_name", "p_score", "hlo_score",
        "p_sample_id_count", "hlo_component_count", "composition_adjustment_exception",
    ]
    out = manifest.merge(cells[columns], on=CELL_KEYS, how="left", validate="many_to_one")
    out["target_score"] = out["p_score"].where(out["target_direction"] == "HLO_to_P", out["hlo_score"])
    out["source_indicator_score"] = out["hlo_score"].where(
        out["target_direction"] == "HLO_to_P", out["p_score"]
    )
    out["target_or_source_indicator_imputed"] = False
    out["context_imputation_performed"] = False
    return out


def make_scenario_rows(
    scenario_id: str,
    directions: list[str],
    variants: list[tuple[str, pd.DataFrame]],
    included: dict[str, set[tuple]],
    reason_maps: dict[str, dict[tuple, str]],
) -> pd.DataFrame:
    rows = []
    for direction in directions:
        keep = included[direction]
        reasons = reason_maps[direction]
        for role, eligible in variants:
            eligible_part = eligible
            if "target_direction" in eligible_part.columns:
                eligible_part = eligible_part[eligible_part["target_direction"] == direction]
            for cell in eligible_part.itertuples(index=False):
                identity = tuple(getattr(cell, key) for key in CELL_KEYS)
                row = {key: getattr(cell, key) for key in CELL_KEYS}
                row.update({
                    "scenario_id": scenario_id,
                    "target_direction": direction,
                    "construction_role": role,
                    "included": identity in keep,
                    "exclusion_reason": "included" if identity in keep else reasons[identity],
                    "usable_feature_count": cell.usable_feature_count,
                    "missing_feature_count": cell.missing_feature_count,
                    "missing_feature_codes": cell.missing_feature_codes,
                    "complete_seven_features": cell.complete_seven_features,
                    "sample_cell_id": f"{cell.iso3}_{cell.window_start}_{cell.window_end}",
                    "comparison_pair_id": f"{scenario_id}_{direction}",
                })
                rows.append(row)
    return pd.DataFrame(rows)


def key_set(frame: pd.DataFrame, flag: str = "row_eligible") -> set[tuple]:
    return set(map(tuple, frame.loc[frame[flag], CELL_KEYS].itertuples(index=False, name=None)))


def build_scenarios(
    five: pd.DataFrame,
    four: pd.DataFrame,
    block: pd.DataFrame,
    assessment: pd.DataFrame,
    s4_features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    b5 = block[block["construction"] == "block_preceding_5y"].copy()
    b10 = block[block["construction"] == "block_preceding_10y"].copy()
    b12 = block[block["construction"] == "block_preceding_12y"].copy()
    e5, e10, e12 = eligibility(b5), eligibility(b10), eligibility(b12)
    esa = eligibility(assessment, ["target_direction"])
    e4 = eligibility(s4_features)

    k5, k10, k12, k4 = map(key_set, [e5, e10, e12, e4])
    all5 = set(map(tuple, e5[CELL_KEYS].itertuples(index=False, name=None)))
    all4 = set(map(tuple, e4[CELL_KEYS].itertuples(index=False, name=None)))
    complete5 = key_set(e5, "complete_seven_features")
    noncomposition = {k for k in k5 if k[0] not in COMPOSITION_EXCLUSIONS}
    nonterminal = {k for k in k5 if k[2] != 2015}

    scenario_rows = []
    feature_rows = []

    def reasons(candidates: set[tuple], conditions: list[tuple[set[tuple], str]]) -> dict[tuple, str]:
        output = {}
        for identity in candidates:
            output[identity] = "included"
            for allowed, reason in conditions:
                if identity not in allowed:
                    output[identity] = reason
                    break
        return output

    def append(
        scenario_id: str,
        dirs: list[str],
        variants: list[tuple[str, pd.DataFrame, pd.DataFrame]],
        keep_by_direction: dict[str, set[tuple]],
        reason_by_direction: dict[str, dict[tuple, str]],
    ) -> None:
        scenario_rows.append(make_scenario_rows(
            scenario_id,
            dirs,
            [(role, eligible) for role, eligible, _ in variants],
            keep_by_direction,
            reason_by_direction,
        ))
        for direction in dirs:
            keep = keep_by_direction[direction]
            for role, _, features in variants:
                part = features.copy()
                if "target_direction" in part.columns:
                    part = part[part["target_direction"] == direction].copy()
                part["scenario_id"] = scenario_id
                part["target_direction"] = direction
                part["construction_role"] = role
                identities = pd.MultiIndex.from_frame(part[CELL_KEYS])
                part["included"] = identities.isin(pd.MultiIndex.from_tuples(keep, names=CELL_KEYS))
                part["requires_future_training_fold_imputation"] = part["included"] & ~part["feature_usable"]
                feature_rows.append(part)

    primary_keep = {d: k5 for d in DIRECTIONS}
    primary_reason = {d: reasons(all5, [(k5, "below_5_of_7_primary_features")]) for d in DIRECTIONS}
    append("PRIMARY", DIRECTIONS, [("primary_5y", e5, b5)], primary_keep, primary_reason)

    for scenario_id, alternative_set, alternative_e, alternative_features, role in [
        ("S10", k10, e10, b10, "alternative_10y"),
        ("S12", k12, e12, b12, "alternative_12y"),
    ]:
        intersection = k5 & alternative_set
        keep = {d: intersection for d in DIRECTIONS}
        reason = {d: reasons(all5, [
            (k5, "below_5_of_7_primary_features"),
            (alternative_set, f"below_5_of_7_{scenario_id.lower()}_features"),
        ]) for d in DIRECTIONS}
        append(scenario_id, DIRECTIONS, [
            ("primary_5y_matched", e5, b5),
            (role, alternative_e, alternative_features),
        ], keep, reason)

    sa_keep = {}
    sa_reason = {}
    for direction in DIRECTIONS:
        part = esa[esa["target_direction"] == direction]
        alt_set = key_set(part)
        sa_keep[direction] = k5 & alt_set
        sa_reason[direction] = reasons(all5, [
            (k5, "below_5_of_7_primary_features"),
            (alt_set, "below_5_of_7_assessment_fixed_features"),
        ])
    append("SA5", DIRECTIONS, [
        ("primary_5y_matched", e5, b5),
        ("assessment_specific_5y_fixed", esa, assessment),
    ], sa_keep, sa_reason)

    for scenario_id, scenario_set, reason_text in [
        ("SCC", complete5, "not_complete_seven_feature_primary_cell"),
        ("SCOMP", noncomposition, "composition_adjustment_country_excluded"),
        ("STERM", nonterminal, "terminal_2015_2017_window_excluded"),
    ]:
        keep = {d: scenario_set for d in DIRECTIONS}
        reason = {d: reasons(all5, [
            (k5, "below_5_of_7_primary_features"),
            (scenario_set, reason_text),
        ]) for d in DIRECTIONS}
        append(scenario_id, DIRECTIONS, [("restricted_primary_5y", e5, b5)], keep, reason)

    s4_keep = {d: k4 for d in DIRECTIONS}
    s4_reason = {d: reasons(all4, [(k4, "below_5_of_7_s4_features")]) for d in DIRECTIONS}
    append("S4", DIRECTIONS, [("four_year_preceding_5y", e4, s4_features)], s4_keep, s4_reason)

    cells = pd.concat(scenario_rows, ignore_index=True)
    features = pd.concat(feature_rows, ignore_index=True)
    five_and_four = pd.concat([five, four], ignore_index=True)
    return add_cell_scores(cells, five_and_four), features


def summarize_scenarios(cells: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    included = cells[cells["included"]].copy()
    summary = (
        included.groupby(["scenario_id", "target_direction", "construction_role"], as_index=False)
        .agg(
            retained_cells=("iso3", "size"),
            retained_countries=("iso3", "nunique"),
            windows=("window_start", "nunique"),
        )
    )
    summary["below_seven_country_floor"] = summary["retained_countries"] < 7
    summary["nested_design_feasible"] = ~summary["below_seven_country_floor"]

    samples = included.drop_duplicates(["scenario_id", "target_direction", *CELL_KEYS]).copy()
    hashes = []
    for (scenario, direction), part in samples.groupby(["scenario_id", "target_direction"]):
        hashes.append({
            "scenario_id": scenario,
            "target_direction": direction,
            "sample_id_sha256": sample_hash(part),
            "retained_cells": len(part),
            "retained_countries": part["iso3"].nunique(),
        })
    sample_ids = pd.DataFrame(hashes)
    samples = samples.merge(sample_ids, on=["scenario_id", "target_direction"], how="left")

    exclusions = (
        cells[~cells["included"]]
        .drop_duplicates(["scenario_id", "target_direction", *CELL_KEYS, "exclusion_reason"])
        [["scenario_id", "target_direction", *CELL_KEYS, "exclusion_reason"]]
        .sort_values(["scenario_id", "target_direction", "window_start", "iso3"])
    )
    return summary, samples, exclusions


def coverage_and_imputation_summary(features: pd.DataFrame) -> pd.DataFrame:
    included = features[features["included"]].copy()
    output = (
        included.groupby(
            ["scenario_id", "target_direction", "construction_role", "indicator_code", "indicator_name"],
            as_index=False,
        )
        .agg(
            retained_cells=("iso3", "size"),
            usable_observed_values=("feature_usable", "sum"),
            training_fold_imputations_required=("requires_future_training_fold_imputation", "sum"),
        )
    )
    output["imputation_performed"] = False
    output["notes"] = "future fold-specific context imputation requirement for M2/M3; no imputation performed"
    output.loc[output["scenario_id"] == "SCC", "notes"] = "complete seven-feature sensitivity; no context imputation required"
    output["feature_order"] = output["indicator_code"].map({v: i + 1 for i, v in enumerate(FEATURES)})
    return output.sort_values([
        "scenario_id", "target_direction", "construction_role", "feature_order"
    ]).reset_index(drop=True)


def scenario_window_summary(cells: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    five_windows = [(2000, 2004), (2005, 2009), (2010, 2014), (2015, 2017)]
    four_windows = [(2000, 2003), (2004, 2007), (2008, 2011), (2012, 2015), (2016, 2017)]
    rows = []
    for spec in summary.itertuples(index=False):
        windows = four_windows if spec.scenario_id == "S4" else five_windows
        part = cells[
            (cells["scenario_id"] == spec.scenario_id)
            & (cells["target_direction"] == spec.target_direction)
            & (cells["construction_role"] == spec.construction_role)
        ]
        for start, end in windows:
            window = part[(part["window_start"] == start) & (part["window_end"] == end)]
            rows.append({
                "scenario_id": spec.scenario_id,
                "target_direction": spec.target_direction,
                "construction_role": spec.construction_role,
                "window_start": start,
                "window_end": end,
                "candidate_cells": len(window),
                "retained_cells": int(window["included"].sum()),
                "window_status": "no_eligible_measurement_cells" if len(window) == 0 else "observed_candidate_cells",
            })
    return pd.DataFrame(rows)


def input_hashes() -> pd.DataFrame:
    paths = {
        "numerical_decisions_addendum": ROOT / "rewrite" / "protocol" / "Numerical_Decisions_Addendum.md",
        "frozen_annual_wdi_panel": ROOT / "data" / "processed" / "wdi_bridge_1980_2017_long.csv",
        "eligible_measurement_cells": TEMPORAL / "temporal_eligible_cells.csv",
        "block_coverage": CONTEXT / "aligned_cell_block_preceding_coverage_long.csv",
        "assessment_annual_ledger": CONTEXT / "assessment_specific_annual_provenance_ledger.csv",
        "assessment_historical_coverage": CONTEXT / "assessment_specific_coverage_long.csv",
    }
    return pd.DataFrame([{
        "input_role": role,
        "repository_relative_path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    } for role, path in paths.items()])


def validate(
    corrections: pd.DataFrame,
    cells: pd.DataFrame,
    features: pd.DataFrame,
    summary: pd.DataFrame,
    hashes: pd.DataFrame,
) -> pd.DataFrame:
    checks = []

    def add(check_id: str, ok: bool, evidence: str) -> None:
        checks.append({"check_id": check_id, "status": "PASS" if ok else "FAIL", "evidence": evidence})

    primary = cells[(cells["scenario_id"] == "PRIMARY") & (cells["target_direction"] == "HLO_to_P")]
    primary_kept = primary[primary["included"]]
    window_counts = primary_kept.groupby("window_start").size().to_dict()
    add("PF001_corrections_exact_18", len(corrections) == 18, f"observed={len(corrections)}")
    add("PF002_corrections_integer_true", corrections["corrected_integer_classification"].all(),
        "all corrected flags true")
    add("PF002B_corrections_old_float_false", (~corrections["old_float_classification"]).all(),
        "all 18 old direct-float flags false")
    add("PF003_primary_cells", len(primary_kept) == 53, f"observed={len(primary_kept)}")
    add("PF004_primary_countries", primary_kept["iso3"].nunique() == 38,
        f"observed={primary_kept['iso3'].nunique()}")
    add("PF005_primary_windows", window_counts == {2000: 7, 2005: 17, 2010: 20, 2015: 9}, str(window_counts))
    add("PF006_observed_targets_sources", cells.loc[cells["included"], ["target_score", "source_indicator_score"]].notna().all().all(),
        "all included target/source scores observed")
    add("PF007_no_target_source_imputation", not cells["target_or_source_indicator_imputed"].any(),
        "all rows false")
    add("PF008_no_context_imputation", not cells["context_imputation_performed"].any(), "all rows false")
    matched_ok = True
    for scenario in ["S10", "S12", "SA5"]:
        for direction in DIRECTIONS:
            part = cells[(cells["scenario_id"] == scenario) & (cells["target_direction"] == direction) & cells["included"]]
            sets = [set(map(tuple, x[CELL_KEYS].itertuples(index=False, name=None))) for _, x in part.groupby("construction_role")]
            matched_ok &= len(sets) == 2 and sets[0] == sets[1]
    add("PF009_matched_sensitivity_identities", matched_ok, "S10/S12/SA5 roles identical within direction")
    scc_missing = features[(features["scenario_id"] == "SCC") & features["included"]]["requires_future_training_fold_imputation"].sum()
    add("PF010_scc_no_imputation", scc_missing == 0, f"missing={int(scc_missing)}")
    s4_windows = set(cells.loc[cells["scenario_id"] == "S4", ["window_start", "window_end"]].itertuples(index=False, name=None))
    add("PF011_s4_eligible_windows", s4_windows == {(2000, 2003), (2004, 2007), (2008, 2011), (2012, 2015)},
        f"eligible={sorted(s4_windows)}; declared 2016-2017 has zero eligible cells")
    add("PF012_country_floor_flags", (
        summary["below_seven_country_floor"] == (summary["retained_countries"] < 7)
    ).all(), "flags follow declared floor")
    add("PF013_feature_order", set(features[["indicator_code", "feature_order"]].drop_duplicates().itertuples(index=False, name=None)) == set(zip(FEATURES, range(1, 8))),
        "fixed seven-feature order")
    count_rows = features[features["observed_year_count"].notna() & features["requested_year_count"].notna()]
    add("PF014_exact_integer_rule", (
        count_rows["feature_usable"]
        == exact_usable(count_rows["observed_year_count"], count_rows["requested_year_count"])
    ).all(), "block/S4 rows use 5*n_obs>=4*T; assessment uses fixed contribution rule")
    hash_ok = all(sha256(ROOT / row.repository_relative_path) == row.sha256 for row in hashes.itertuples(index=False))
    add("PF015_input_hashes", hash_ok, f"matched={len(hashes)}/{len(hashes)}")
    result = pd.DataFrame(checks)
    print(result.to_string(index=False))
    assert (result["status"] == "PASS").all()
    return result


def write_handoff(summary: pd.DataFrame, corrections: pd.DataFrame) -> None:
    pivot = summary.set_index(["scenario_id", "target_direction", "construction_role"])
    changed_rows = int((corrections["old_row_eligible"] != corrections["corrected_row_eligible"]).sum())
    correction_features = text_join(corrections["indicator_code"])
    lines = [
        "# Final protocol context-manifest handoff",
        "",
        "This build applies the numerical addendum to frozen, outcome-blind construction data only. No empirical model was fitted and no predictive performance was inspected.",
        "",
        "## Coverage correction",
        "",
        f"The ledger contains {len(corrections)} corrected assessment-specific cell-feature classifications. Each row records the cell, direction, feature, contributing measurement IDs, annual counts, old floating-point classification, corrected integer classification, and feature/row effects.",
        f"All corrections concern the education features ({correction_features}). {changed_rows} correction rows change their cell from below to at least five usable features; the other {len(corrections) - changed_rows} leave an already eligible cell eligible.",
        "",
        "## Scenario samples",
        "",
    ]
    for scenario in ["PRIMARY", "S10", "S12", "SA5", "SCC", "SCOMP", "STERM", "S4"]:
        part = summary[summary["scenario_id"] == scenario]
        counts = ", ".join(
            f"{r.target_direction}/{r.construction_role}: {int(r.retained_cells)} cells, {int(r.retained_countries)} countries"
            for r in part.itertuples(index=False)
        )
        lines.append(f"- {scenario}: {counts}.")
    lines.extend([
        "",
        "Matched S10, S12 and SA5 variants use identical country-period identities within each direction. SA5 remains direction-specific. SCC contains complete seven-feature primary cells and requires no context imputation. Scenarios with fewer than seven countries are flagged infeasible rather than modified.",
        "The declared S4 partial window 2016-2017 has zero eligible aligned measurement cells in the frozen temporal input. It remains explicit in `scenario_window_summary.csv`.",
        "",
        "## Reproduction",
        "",
        "```text",
        "py rewrite/src/protocol_final/context_manifest.py",
        "```",
        "",
        "Input hashes, exact identities, exclusions, feature requirements, sample hashes and validation results are stored beside this handoff.",
    ])
    (OUTPUT / "HANDOFF.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    five, four = load_cells()
    block = prepare_block_features()
    assessment, contributions = prepare_assessment_features()
    s4_features = build_four_year_features(four)
    corrections = coverage_corrections(assessment, contributions)
    scenario_cells, scenario_features = build_scenarios(five, four, block, assessment, s4_features)
    summary, identities, exclusions = summarize_scenarios(scenario_cells)
    windows = scenario_window_summary(scenario_cells, summary)
    scenario_cells = scenario_cells.merge(
        summary[[
            "scenario_id", "target_direction", "construction_role",
            "below_seven_country_floor", "nested_design_feasible",
        ]],
        on=["scenario_id", "target_direction", "construction_role"],
        how="left",
        validate="many_to_one",
    )
    scenario_cells["retained"] = scenario_cells["included"]
    scenario_cells["status"] = scenario_cells["included"].map({True: "retained", False: "excluded"})
    scenario_cells["feasibility"] = scenario_cells["nested_design_feasible"].map({
        True: "feasible", False: "infeasible_below_7_countries",
    })
    coverage_summary = coverage_and_imputation_summary(scenario_features)
    hashes = input_hashes()
    checks = validate(corrections, scenario_cells, scenario_features, summary, hashes)

    corrections.to_csv(OUTPUT / "coverage_threshold_corrections.csv", index=False)
    scenario_cells.to_csv(OUTPUT / "scenario_sample_manifest.csv", index=False)
    scenario_features.to_csv(OUTPUT / "scenario_feature_manifest.csv", index=False)
    summary.to_csv(OUTPUT / "scenario_summary.csv", index=False)
    windows.to_csv(OUTPUT / "scenario_window_summary.csv", index=False)
    coverage_summary.to_csv(OUTPUT / "coverage_and_imputation_summary.csv", index=False)
    identities.to_csv(OUTPUT / "scenario_included_cell_identities.csv", index=False)
    exclusions.to_csv(OUTPUT / "scenario_exclusions.csv", index=False)
    hashes.to_csv(OUTPUT / "input_hashes.csv", index=False)
    checks.to_csv(OUTPUT / "validation_results.csv", index=False)

    status = {
        "empirical_modelling_performed": False,
        "predictive_performance_inspected": False,
        "feature_order": FEATURES,
        "coverage_rule": "5*n_obs>=4*T",
        "minimum_features_per_cell": 5,
        "coverage_threshold_corrections": len(corrections),
        "scenario_count": scenario_cells["scenario_id"].nunique(),
        "input_hashes": hashes.to_dict(orient="records"),
    }
    (OUTPUT / "build_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    write_handoff(summary, corrections)


if __name__ == "__main__":
    main()
