from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, norm, pearsonr, spearmanr, t


REPO = Path(__file__).resolve().parents[3]
REWRITE = REPO / "rewrite"
INPUTS = REWRITE / "analysis" / "inputs" / "measurement"
CORE = REWRITE / "analysis" / "core"
TEMPORAL = REWRITE / "analysis" / "temporal"
VALIDATION = REWRITE / "validation" / "measurement"
DOCS = REWRITE / "docs" / "agent1"

NIQ_WORKBOOK = REPO / "data" / "raw" / "niq" / "v135" / "NIQ-DATA (V1.3.5).xlsx"
NIQ_MANUAL = REPO / "data" / "raw" / "niq" / "v135" / "MANUAL (V1.3.5).txt"
HLO_FILE = REPO / "data" / "raw" / "hlo_official_replication" / "official_filtered_hlo.dta"
RQ1_FILE = REPO / "outputs" / "headline_analysis_rq1_exact_sample.csv"
RQ2_FILE = REPO / "outputs" / "headline_analysis_rq2_exact_sample.csv"
ARCHIVED_RANKING = REPO / "outputs" / "headline_analysis_ranking_summary.csv"
ARCHIVED_GDP = REPO / "outputs" / "headline_analysis_correlation_difference.csv"

WINDOWS = {
    "annual": [(year, year) for year in range(2000, 2018)],
    "4_year": [(2000, 2003), (2004, 2007), (2008, 2011), (2012, 2015), (2016, 2017)],
    "5_year": [(2000, 2004), (2005, 2009), (2010, 2014), (2015, 2017)],
}
STANDARD_SUBJECTS = ("reading", "math", "science")
STANDARD_LEVELS = ("pri", "sec")
COMPOSITION_EXCEPTIONS = ("BRA", "CAN", "KAZ", "MEX", "SRB", "USA", "ZAF")
TOLERANCE = 1e-12


def snake(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = text.replace("&", " and ").replace("+", " plus ")
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower() or "unnamed"


def unique_names(names: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    output = []
    for name in names:
        number = seen.get(name, 0)
        output.append(name if number == 0 else f"{name}_{number + 1}")
        seen[name] = number + 1
    return output


def read_two_row_header(path: Path, sheet: str, usecols=None) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name=sheet, header=None, usecols=usecols)
    groups = raw.iloc[0].ffill()
    fields = raw.iloc[1]
    names = []
    for group, field in zip(groups, fields):
        group_name = snake(group)
        field_name = snake(field)
        names.append(field_name if group_name in {"unnamed", field_name} else f"{group_name}__{field_name}")
    frame = raw.iloc[2:].copy()
    frame.columns = unique_names(names)
    return frame.reset_index(drop=True)


def find_column(frame: pd.DataFrame, suffix: str) -> str:
    matches = [column for column in frame.columns if column.endswith(suffix)]
    assert len(matches) == 1, (suffix, matches)
    return matches[0]


def numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def recover_niq() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    records = read_two_row_header(NIQ_WORKBOOK, "REC")
    selection = read_two_row_header(NIQ_WORKBOOK, "SEL")
    rename = {
        find_column(records, "__id"): "sample_id",
        find_column(records, "__iso_3166_1_alpha_3"): "iso3",
        find_column(records, "__country_name"): "country",
        find_column(records, "__origin_type"): "origin_type",
        find_column(records, "__origin_specif"): "origin_detail",
        find_column(records, "__ses"): "ses_code",
        find_column(records, "__sample_comp"): "sample_composition",
        find_column(records, "__sample_char"): "sample_character_code",
        find_column(records, "__lowest_age"): "lowest_age",
        find_column(records, "__highest_age"): "highest_age",
        find_column(records, "__mean_age"): "mean_age",
        find_column(records, "__n_ind"): "sample_size",
        find_column(records, "__sample_rating"): "sample_rating",
        find_column(records, "__test_type"): "test_type",
        find_column(records, "__test_meas"): "test_measured",
        find_column(records, "__year_meas"): "measurement_year",
        find_column(records, "__year_std"): "norm_year",
        find_column(records, "__testing_rating"): "testing_rating",
        find_column(records, "__test_time_adjust"): "test_time_adjustment",
        find_column(records, "__country_cor"): "country_correction",
        find_column(records, "__method_rating"): "method_rating",
        find_column(records, "sample_iqs__iq_cor"): "corrected_iq",
        find_column(records, "__full_rating"): "full_rating",
        find_column(records, "__qn_factor"): "qn_factor",
        find_column(records, "references__short"): "reference_short",
        find_column(records, "references__full"): "reference_full",
    }
    records = records.rename(columns=rename)
    selection_id = find_column(selection, "__id")
    selection_flag = find_column(selection, "filter__y_n")
    selected = selection[[selection_id, selection_flag]].rename(
        columns={selection_id: "sample_id", selection_flag: "included_by_source"}
    )
    selected = selected[selected["sample_id"].notna()].groupby("sample_id", as_index=False).agg(
        included_by_source=("included_by_source", lambda values: "Y" if values.eq("Y").any() else "N")
    )
    records = records[records["sample_id"].notna()].copy()
    records["record_number_within_sample_id"] = records.groupby("sample_id").cumcount() + 1
    records["record_id"] = records["sample_id"].astype(str) + "_" + records["record_number_within_sample_id"].astype(str)
    records = records.merge(selected, on="sample_id", how="left", validate="many_to_one")
    numeric_columns = [
        "lowest_age", "highest_age", "mean_age", "sample_size", "sample_rating",
        "measurement_year", "norm_year", "testing_rating", "test_time_adjustment",
        "country_correction", "method_rating", "corrected_iq", "full_rating", "qn_factor",
    ]
    numeric(records, numeric_columns)
    keep = [
        "record_id", "sample_id", "iso3", "country", "included_by_source",
        "origin_type", "origin_detail", "ses_code", "sample_composition",
        "sample_character_code", "lowest_age", "highest_age", "mean_age",
        "sample_size", "sample_rating", "test_type", "test_measured",
        "measurement_year", "norm_year", "testing_rating", "test_time_adjustment",
        "country_correction", "method_rating", "corrected_iq", "full_rating",
        "qn_factor", "reference_short", "reference_full",
    ]
    records = records[keep]

    fav = read_two_row_header(NIQ_WORKBOOK, "FAV")
    fav = fav.rename(columns={
        fav.columns[0]: "iso3", fav.columns[1]: "country", fav.columns[2]: "niq_uw",
        fav.columns[3]: "niq_nw", fav.columns[4]: "niq_qnw", fav.columns[5]: "niq_sas",
        fav.columns[6]: "niq_qnw_sas", fav.columns[7]: "niq_qnw_sas_geo",
    })
    fav = fav[fav["iso3"].astype(str).str.fullmatch(r"[A-Z]{3}")].copy()
    numeric(fav, ["niq_uw", "niq_nw", "niq_qnw", "niq_sas", "niq_qnw_sas", "niq_qnw_sas_geo"])

    nat = read_two_row_header(NIQ_WORKBOOK, "NAT", usecols=range(18))
    nat = nat.rename(columns={
        "identification__iso_3166_1_alpha_3": "iso3",
        "identification__country_name": "country",
        "method__std_reest": "source_standard_reestimation",
        "niq__qnw": "source_qnw",
    })
    nat = nat[nat["iso3"].astype(str).str.fullmatch(r"[A-Z]{3}")].copy()
    numeric(nat, ["source_qnw"])
    nat = nat[["iso3", "country", "source_standard_reestimation", "source_qnw"]]

    write_csv(records, INPUTS / "niq_samples_v135_rebuilt.csv")
    write_csv(fav, INPUTS / "niq_country_scores_v135_rebuilt.csv")
    write_csv(nat, INPUTS / "niq_nation_method_v135_rebuilt.csv")
    return records, fav, nat


def recover_hlo() -> pd.DataFrame:
    hlo = pd.read_stata(HLO_FILE)
    standard = hlo[
        hlo["subject"].isin(STANDARD_SUBJECTS)
        & hlo["level"].isin(STANDARD_LEVELS)
        & hlo["year"].between(2000, 2017)
    ].copy()
    standard["year"] = standard["year"].astype(int)
    write_csv(standard, INPUTS / "hlo_official_filtered_standard_2000_2017.csv")
    return standard


def williams_and_zou(r_xy: float, r_zy: float, r_xz: float, n: int) -> dict[str, float]:
    determinant = 1 + 2 * r_xy * r_zy * r_xz - r_xy**2 - r_zy**2 - r_xz**2
    mean_correlation = (r_xy + r_zy) / 2
    statistic = (r_xy - r_zy) * np.sqrt(
        ((n - 1) * (1 + r_xz))
        / (2 * ((n - 1) / (n - 3)) * determinant + mean_correlation**2 * (1 - r_xz) ** 3)
    )
    p_value = 2 * t.sf(abs(statistic), n - 3)
    radius = norm.ppf(.975) / np.sqrt(n - 3)
    xy_interval = np.tanh(np.arctanh(r_xy) + np.array([-radius, radius]))
    zy_interval = np.tanh(np.arctanh(r_zy) + np.array([-radius, radius]))
    covariance = (
        (r_xz - .5 * r_xy * r_zy) * (1 - r_xy**2 - r_zy**2 - r_xz**2) + r_xz**3
    ) / ((1 - r_xy**2) * (1 - r_zy**2))
    difference = r_xy - r_zy
    lower = difference - np.sqrt(
        (r_xy - xy_interval[0]) ** 2 + (zy_interval[1] - r_zy) ** 2
        - 2 * covariance * (r_xy - xy_interval[0]) * (zy_interval[1] - r_zy)
    )
    upper = difference + np.sqrt(
        (xy_interval[1] - r_xy) ** 2 + (r_zy - zy_interval[0]) ** 2
        - 2 * covariance * (xy_interval[1] - r_xy) * (r_zy - zy_interval[0])
    )
    return {"williams_t": statistic, "williams_p": p_value, "zou_lower": lower, "zou_upper": upper}


def reproduce_core() -> tuple[pd.DataFrame, pd.DataFrame]:
    rq1 = pd.read_csv(RQ1_FILE)
    rq2 = pd.read_csv(RQ2_FILE)
    assert len(rq1) == 66 and rq1["iso3"].is_unique
    assert len(rq2) == 65 and rq2["iso3"].is_unique
    write_csv(rq1, CORE / "core_rq1_exact_sample.csv")
    write_csv(rq2, CORE / "core_rq2_exact_sample.csv")

    ranking = rq1[["iso3", "canonical_economy_name", "niq_qnw", "hlo"]].copy()
    ranking["rank_qnw"] = ranking["niq_qnw"].rank(ascending=False, method="average")
    ranking["rank_hlo"] = ranking["hlo"].rank(ascending=False, method="average")
    ranking["rank_displacement_hlo_minus_qnw"] = ranking["rank_hlo"] - ranking["rank_qnw"]
    ranking["absolute_rank_displacement"] = ranking["rank_displacement_hlo_minus_qnw"].abs()
    ranking["normalized_absolute_rank_displacement"] = ranking["absolute_rank_displacement"] / (len(ranking) - 1)
    qnw_top = ranking["niq_qnw"].quantile(.9, interpolation="linear")
    hlo_top = ranking["hlo"].quantile(.9, interpolation="linear")
    qnw_bottom = ranking["niq_qnw"].quantile(.1, interpolation="linear")
    hlo_bottom = ranking["hlo"].quantile(.1, interpolation="linear")
    ranking["top_qnw"] = ranking["niq_qnw"].ge(qnw_top)
    ranking["top_hlo"] = ranking["hlo"].ge(hlo_top)
    ranking["bottom_qnw"] = ranking["niq_qnw"].le(qnw_bottom)
    ranking["bottom_hlo"] = ranking["hlo"].le(hlo_bottom)
    qnw_difference = ranking["niq_qnw"].to_numpy()[:, None] - ranking["niq_qnw"].to_numpy()
    hlo_difference = ranking["hlo"].to_numpy()[:, None] - ranking["hlo"].to_numpy()
    strict = (qnw_difference != 0) & (hlo_difference != 0)
    reversal_count = int(np.triu((qnw_difference * hlo_difference < 0) & strict, 1).sum())
    strict_pairs = int(np.triu(strict, 1).sum())
    write_csv(ranking, CORE / "core_rank_ledger.csv")

    r_qnw_hlo = pearsonr(ranking["niq_qnw"], ranking["hlo"]).statistic
    rho = spearmanr(ranking["niq_qnw"], ranking["hlo"]).statistic
    tau = kendalltau(ranking["niq_qnw"], ranking["hlo"]).statistic
    r_qnw_gdp = pearsonr(rq2["niq_qnw"], rq2["log_gdp_pc_ppp_2017"]).statistic
    r_hlo_gdp = pearsonr(rq2["hlo"], rq2["log_gdp_pc_ppp_2017"]).statistic
    r_qnw_hlo_rq2 = pearsonr(rq2["niq_qnw"], rq2["hlo"]).statistic
    dependent = williams_and_zou(r_qnw_gdp, r_hlo_gdp, r_qnw_hlo_rq2, len(rq2))
    values = {
        "CORE_RQ1_N": len(rq1),
        "CORE_QNW_HLO_PEARSON": r_qnw_hlo,
        "CORE_QNW_HLO_SPEARMAN": rho,
        "CORE_QNW_HLO_KENDALL": tau,
        "CORE_RANK_MEAN_ABS": ranking["absolute_rank_displacement"].mean(),
        "CORE_RANK_MEDIAN_ABS": ranking["absolute_rank_displacement"].median(),
        "CORE_RANK_MAX_ABS": ranking["absolute_rank_displacement"].max(),
        "CORE_RANK_MEAN_NORMALIZED": ranking["normalized_absolute_rank_displacement"].mean(),
        "CORE_STRICT_PAIRS": strict_pairs,
        "CORE_ORDER_REVERSALS": reversal_count,
        "CORE_ORDER_REVERSAL_SHARE": reversal_count / strict_pairs,
        "CORE_TOP_OVERLAP": int((ranking["top_qnw"] & ranking["top_hlo"]).sum()),
        "CORE_BOTTOM_OVERLAP": int((ranking["bottom_qnw"] & ranking["bottom_hlo"]).sum()),
        "CORE_TAIL_DENOMINATOR": int(ranking["top_qnw"].sum()),
        "CORE_RQ2_N": len(rq2),
        "CORE_QNW_GDP_PEARSON": r_qnw_gdp,
        "CORE_HLO_GDP_PEARSON": r_hlo_gdp,
        "CORE_GDP_CORRELATION_DIFFERENCE_QNW_MINUS_HLO": r_qnw_gdp - r_hlo_gdp,
        "CORE_WILLIAMS_T": dependent["williams_t"],
        "CORE_WILLIAMS_P": dependent["williams_p"],
        "CORE_ZOU_LOWER": dependent["zou_lower"],
        "CORE_ZOU_UPPER": dependent["zou_upper"],
    }
    results = pd.DataFrame([{"result_id": key, "estimate": value} for key, value in values.items()])
    write_csv(results, CORE / "core_results.csv")

    archived_rank = pd.read_csv(ARCHIVED_RANKING).iloc[0]
    archived_gdp = pd.read_csv(ARCHIVED_GDP).iloc[0]
    comparisons = {
        "pearson_r": (r_qnw_hlo, archived_rank["pearson_r"]),
        "mean_absolute_rank_displacement": (ranking["absolute_rank_displacement"].mean(), archived_rank["mean_absolute_rank_displacement"]),
        "median_absolute_rank_displacement": (ranking["absolute_rank_displacement"].median(), archived_rank["median_absolute_rank_displacement"]),
        "maximum_absolute_rank_displacement": (ranking["absolute_rank_displacement"].max(), archived_rank["maximum_absolute_rank_displacement"]),
        "pairwise_order_reversal_count": (reversal_count, archived_rank["pairwise_order_reversal_count"]),
        "top_overlap_size": (values["CORE_TOP_OVERLAP"], archived_rank["top_overlap_size"]),
        "bottom_overlap_size": (values["CORE_BOTTOM_OVERLAP"], archived_rank["bottom_overlap_size"]),
        "r_qnw_gdp": (r_qnw_gdp, archived_gdp["r_qnw_gdp"]),
        "r_hlo_gdp": (r_hlo_gdp, archived_gdp["r_hlo_gdp"]),
        "difference_qnw_minus_hlo": (r_qnw_gdp - r_hlo_gdp, archived_gdp["difference_r_qnw_gdp_minus_r_hlo_gdp"]),
        "williams_p": (dependent["williams_p"], archived_gdp["williams_two_sided_p_value"]),
        "zou_lower": (dependent["zou_lower"], archived_gdp["zou_95_ci_lower"]),
        "zou_upper": (dependent["zou_upper"], archived_gdp["zou_95_ci_upper"]),
    }
    reconciliation = pd.DataFrame([
        {
            "quantity": key,
            "fresh_value": fresh,
            "archived_value": archived,
            "absolute_difference": abs(fresh - archived),
            "passes_tolerance_1e_12": abs(fresh - archived) <= TOLERANCE,
        }
        for key, (fresh, archived) in comparisons.items()
    ])
    write_csv(reconciliation, CORE / "core_reconciliation.csv")
    return results, reconciliation


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    return float(np.average(values.to_numpy(float), weights=weights.to_numpy(float)))


def weighted_sd(values: pd.Series, weights: pd.Series) -> float:
    mean = weighted_mean(values, weights)
    return float(np.sqrt(np.average((values.to_numpy(float) - mean) ** 2, weights=weights.to_numpy(float))))


def temporal_grid(sample: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scheme, windows in WINDOWS.items():
        for number, (start, end) in enumerate(windows, 1):
            for country in sample.itertuples(index=False):
                rows.append({
                    "scheme": scheme, "window_number": number, "window_start": start,
                    "window_end": end, "window_year_count": end - start + 1,
                    "window_label": str(start) if start == end else f"{start}-{end}",
                    "partial_terminal_window": scheme == "5_year" and start == 2015 and end == 2017,
                    "iso3": country.iso3, "canonical_economy_name": country.canonical_economy_name,
                })
    return pd.DataFrame(rows)


def p_measurement(samples: pd.DataFrame, sample_codes: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = samples[
        samples["included_by_source"].eq("Y")
        & samples["iso3"].isin(sample_codes)
        & samples["measurement_year"].notna()
        & samples["corrected_iq"].notna()
        & samples["qn_factor"].notna()
    ].copy()
    selected["calendar_year_floor"] = np.floor(selected["measurement_year"]).astype(int)
    selected["measurement_year_fraction"] = selected["measurement_year"] - selected["calendar_year_floor"]
    selected["fractional_measurement_date"] = selected["measurement_year_fraction"].abs().gt(TOLERANCE)
    selected["date_precision_status"] = np.where(
        selected["fractional_measurement_date"], "fractional_or_aggregated_year", "integer_year"
    )
    selected["repeated_sample_id_in_selected_data"] = selected.duplicated(["iso3", "sample_id"], keep=False)
    ledger_rows = []
    cell_rows = []
    for scheme, windows in WINDOWS.items():
        for number, (start, end) in enumerate(windows, 1):
            period = selected[selected["calendar_year_floor"].between(start, end)]
            for iso3, group in period.groupby("iso3", sort=False):
                total_weight = group["qn_factor"].sum()
                normalized = group["qn_factor"] / total_weight
                for (_, row), contribution_weight in zip(group.iterrows(), normalized):
                    ledger_rows.append({
                        "scheme": scheme, "window_number": number, "window_start": start,
                        "window_end": end, "iso3": iso3, "record_id": row["record_id"],
                        "sample_id": row["sample_id"], "measurement_year": row["measurement_year"],
                        "calendar_year_floor": row["calendar_year_floor"],
                        "fractional_measurement_date": row["fractional_measurement_date"],
                        "date_precision_status": row["date_precision_status"],
                        "measurement_year_fraction": row["measurement_year_fraction"],
                        "corrected_iq": row["corrected_iq"], "qn_factor": row["qn_factor"],
                        "normalized_composite_weight": contribution_weight,
                        "mean_age": row["mean_age"], "lowest_age": row["lowest_age"],
                        "highest_age": row["highest_age"], "sample_size": row["sample_size"],
                        "reference_short": row["reference_short"],
                        "repeated_sample_id_in_selected_data": row["repeated_sample_id_in_selected_data"],
                    })
                distinct_ids = group["sample_id"].nunique()
                sample_id_means = group.groupby("sample_id").apply(
                    lambda sample: pd.Series({
                        "sample_id_mean": weighted_mean(sample["corrected_iq"], sample["qn_factor"]),
                        "sample_id_weight": sample["qn_factor"].sum(),
                    }),
                    include_groups=False,
                )
                cell_rows.append({
                    "scheme": scheme, "window_number": number, "iso3": iso3,
                    "p_score": weighted_mean(group["corrected_iq"], group["qn_factor"]),
                    "p_weighted_record_sd": weighted_sd(group["corrected_iq"], group["qn_factor"]),
                    "p_between_sample_id_sd": weighted_sd(
                        sample_id_means["sample_id_mean"], sample_id_means["sample_id_weight"]
                    ) if distinct_ids >= 2 else np.nan,
                    "p_between_sample_id_sd_status": (
                        "descriptive_between_id_dispersion_not_certified_independent_studies"
                        if distinct_ids >= 2 else "unavailable_one_sample_id"
                    ),
                    "p_min": group["corrected_iq"].min(), "p_max": group["corrected_iq"].max(),
                    "p_record_count": len(group), "p_sample_id_count": distinct_ids,
                    "p_distinct_year_count": group["calendar_year_floor"].nunique(),
                    "p_measurement_year_min": group["measurement_year"].min(),
                    "p_measurement_year_max": group["measurement_year"].max(),
                    "p_fractional_date_count": group["fractional_measurement_date"].sum(),
                    "p_repeated_sample_id_record_count": group["repeated_sample_id_in_selected_data"].sum(),
                    "p_mean_age_nonmissing_count": group["mean_age"].notna().sum(),
                    "p_mean_age_min": group["mean_age"].min(), "p_mean_age_max": group["mean_age"].max(),
                    "p_total_qn_weight": total_weight,
                    "p_weight_effective_n": total_weight**2 / np.square(group["qn_factor"]).sum(),
                    "p_is_exact_published_qnw": False,
                })
    return pd.DataFrame(ledger_rows), pd.DataFrame(cell_rows)


def hlo_measurement(hlo: pd.DataFrame, sample_codes: set[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = hlo[hlo["code"].isin(sample_codes)].copy()
    record_rows = []
    component_rows = []
    cell_rows = []
    for scheme, windows in WINDOWS.items():
        for number, (start, end) in enumerate(windows, 1):
            period = data[data["year"].between(start, end)]
            for iso3, group in period.groupby("code", sort=False):
                cell_keys = ["year", "subject", "level"]
                year_cells = group.groupby(cell_keys, as_index=False).agg(
                    year_cell_hlo=("hlo", "mean"), tied_record_count=("hlo", "size"),
                )
                components = year_cells.groupby(["subject", "level"], as_index=False).agg(
                    component_hlo=("year_cell_hlo", "mean"), component_year_count=("year", "nunique"),
                    component_first_year=("year", "min"), component_last_year=("year", "max"),
                    component_temporal_sd=("year_cell_hlo", lambda x: x.std(ddof=0)),
                )
                components["level_subject_count"] = components.groupby("level")["subject"].transform("nunique")
                level_count = components["level"].nunique()
                components["component_to_composite_weight"] = (
                    1 / components["level_subject_count"] / level_count
                )
                levels = components.groupby("level", as_index=False).agg(
                    level_hlo=("component_hlo", "mean"), level_subject_count=("subject", "nunique")
                )
                eligible = level_count == 2 and components["subject"].nunique() >= 2
                score = levels["level_hlo"].mean()
                for row in components.itertuples(index=False):
                    component_rows.append({
                        "scheme": scheme, "window_number": number, "window_start": start,
                        "window_end": end, "iso3": iso3, "subject": row.subject, "level": row.level,
                        "component_hlo": row.component_hlo, "component_year_count": row.component_year_count,
                        "component_first_year": row.component_first_year,
                        "component_last_year": row.component_last_year,
                        "component_temporal_sd": row.component_temporal_sd,
                        "level_subject_count": row.level_subject_count,
                        "component_to_composite_weight": row.component_to_composite_weight,
                    })
                component_lookup = components.set_index(["subject", "level"])
                tie_lookup = year_cells.set_index(cell_keys)["tied_record_count"]
                for row in group.itertuples(index=False):
                    component = component_lookup.loc[(row.subject, row.level)]
                    tied_count = int(tie_lookup.loc[(row.year, row.subject, row.level)])
                    record_to_year_cell = 1 / tied_count
                    year_cell_to_component = 1 / component["component_year_count"]
                    component_to_level = 1 / component["level_subject_count"]
                    level_to_composite = 1 / level_count
                    record_rows.append({
                        "scheme": scheme, "window_number": number, "window_start": start,
                        "window_end": end, "iso3": iso3, "year": row.year,
                        "subject": row.subject, "level": row.level, "sourcetest": row.sourcetest,
                        "hlo": row.hlo, "hlo_se": row.hlo_se, "tied_record_count": tied_count,
                        "record_to_year_cell_weight": record_to_year_cell,
                        "year_cell_to_component_weight": year_cell_to_component,
                        "component_to_level_weight": component_to_level,
                        "level_to_composite_weight": level_to_composite,
                        "composite_weight": record_to_year_cell * year_cell_to_component * component_to_level * level_to_composite,
                    })
                cell_rows.append({
                    "scheme": scheme, "window_number": number, "iso3": iso3,
                    "hlo_score": score, "hlo_eligible": eligible,
                    "hlo_record_count": len(group), "hlo_year_cell_count": len(year_cells),
                    "hlo_distinct_year_count": group["year"].nunique(),
                    "hlo_component_count": len(components),
                    "hlo_subject_count": components["subject"].nunique(),
                    "hlo_level_count": level_count,
                    "hlo_raw_record_sd": group["hlo"].std(ddof=0),
                    "hlo_component_sd": components["component_hlo"].std(ddof=0),
                    "hlo_mean_within_component_temporal_sd": components["component_temporal_sd"].mean(),
                    "hlo_min": group["hlo"].min(), "hlo_max": group["hlo"].max(),
                    "hlo_se_available_count": group["hlo_se"].notna().sum(),
                    "hlo_se_min": group["hlo_se"].min(), "hlo_se_max": group["hlo_se"].max(),
                    "hlo_composite_se": np.nan,
                    "hlo_composite_se_status": "not_estimable_from_record_level_se_without_covariance",
                })
    return pd.DataFrame(record_rows), pd.DataFrame(component_rows), pd.DataFrame(cell_rows)


def reconstruct_qnw(samples: pd.DataFrame, nat: pd.DataFrame, rq2: pd.DataFrame) -> pd.DataFrame:
    selected = samples[samples["included_by_source"].eq("Y") & samples["iso3"].isin(rq2["iso3"])].copy()
    reconstructed = selected.groupby("iso3").apply(
        lambda group: pd.Series({
            "selected_record_count": len(group),
            "selected_sample_id_count": group["sample_id"].nunique(),
            "simple_full_history_qn_weighted_mean": weighted_mean(group["corrected_iq"], group["qn_factor"]),
        }), include_groups=False,
    ).reset_index()
    check = rq2[["iso3", "canonical_economy_name", "niq_qnw"]].merge(reconstructed, on="iso3", validate="one_to_one")
    check = check.merge(nat[["iso3", "source_standard_reestimation", "source_qnw"]], on="iso3", validate="one_to_one")
    check["published_minus_simple_weighted_mean"] = check["niq_qnw"] - check["simple_full_history_qn_weighted_mean"]
    check["composition_adjustment_exception"] = check["source_standard_reestimation"].eq("N")
    check["simple_mean_matches_published_qnw_1e_8"] = check["published_minus_simple_weighted_mean"].abs().lt(1e-8)
    check["temporal_p_construction_note"] = "Period P is QN-weighted corrected IQ; it is not exact published QNW."
    return check


def audit_measurement(samples: pd.DataFrame, hlo: pd.DataFrame, nat: pd.DataFrame, rq2: pd.DataFrame) -> None:
    selected = samples[samples["included_by_source"].eq("Y") & samples["iso3"].isin(rq2["iso3"])].copy()
    sample_ids = selected.groupby(["iso3", "sample_id"], as_index=False).agg(
        selected_record_count=("record_id", "size"),
        minimum_recorded_sample_size=("sample_size", "min"),
        maximum_recorded_sample_size=("sample_size", "max"),
        measurement_year_min=("measurement_year", "min"),
        measurement_year_max=("measurement_year", "max"),
        corrected_iq_min=("corrected_iq", "min"), corrected_iq_max=("corrected_iq", "max"),
        reference_count=("reference_short", "nunique"),
    )
    sample_ids["repeated_sample_id"] = sample_ids["selected_record_count"].gt(1)
    sample_ids["within_id_size_conflict"] = sample_ids["minimum_recorded_sample_size"].ne(sample_ids["maximum_recorded_sample_size"])
    write_csv(sample_ids, TEMPORAL / "p_sample_id_audit.csv")

    selected["calendar_year_floor"] = np.floor(selected["measurement_year"]).astype(int)
    potential_overlap = selected.groupby(
        ["iso3", "reference_short", "calendar_year_floor"], dropna=False, as_index=False
    ).agg(
        record_count=("record_id", "size"), sample_id_count=("sample_id", "nunique"),
        sample_ids=("sample_id", lambda x: " | ".join(sorted(x.astype(str).unique()))),
        origin_details=("origin_detail", lambda x: " | ".join(sorted(x.dropna().astype(str).unique()))),
        mean_age_min=("mean_age", "min"), mean_age_max=("mean_age", "max"),
    )
    potential_overlap = potential_overlap[potential_overlap["sample_id_count"].gt(1)].copy()
    potential_overlap["status"] = "candidate_overlap_same_country_reference_calendar_year_requires_source_review"
    write_csv(potential_overlap, TEMPORAL / "p_potential_overlap_audit.csv")

    fractional = selected[selected["measurement_year"].mod(1).abs().gt(TOLERANCE)].copy()
    fractional["calendar_year_floor"] = np.floor(fractional["measurement_year"]).astype(int)
    fractional["calendar_year_ceiling"] = np.ceil(fractional["measurement_year"]).astype(int)
    fractional["near_integer_boundary_within_0_1"] = np.minimum(
        fractional["measurement_year"].mod(1), 1 - fractional["measurement_year"].mod(1)
    ).le(.1)
    write_csv(fractional[[
        "record_id", "sample_id", "iso3", "measurement_year", "calendar_year_floor",
        "calendar_year_ceiling", "near_integer_boundary_within_0_1", "reference_short",
    ]], TEMPORAL / "p_fractional_date_audit.csv")

    boundary_rows = []
    for row in fractional.itertuples(index=False):
        for scheme, windows in WINDOWS.items():
            floor_window = next((i for i, (a, b) in enumerate(windows, 1) if a <= row.calendar_year_floor <= b), None)
            ceil_window = next((i for i, (a, b) in enumerate(windows, 1) if a <= row.calendar_year_ceiling <= b), None)
            boundary_rows.append({
                "record_id": row.record_id, "sample_id": row.sample_id, "iso3": row.iso3,
                "measurement_year": row.measurement_year, "scheme": scheme,
                "floor_window_number": floor_window, "ceiling_window_number": ceil_window,
                "window_assignment_changes_under_ceiling": floor_window != ceil_window,
            })
    write_csv(pd.DataFrame(boundary_rows), TEMPORAL / "p_date_boundary_audit.csv")

    age_audit = selected[[
        "record_id", "sample_id", "iso3", "lowest_age", "highest_age", "mean_age",
        "measurement_year", "sample_size", "sample_composition", "sample_character_code",
    ]].copy()
    age_audit["age_metadata_complete"] = age_audit[["lowest_age", "highest_age", "mean_age"]].notna().all(axis=1)
    age_audit["age_order_valid_when_complete"] = (
        age_audit["lowest_age"].le(age_audit["mean_age"]) & age_audit["mean_age"].le(age_audit["highest_age"])
    ).where(age_audit["age_metadata_complete"])
    age_audit["fractional_sample_size"] = age_audit["sample_size"].mod(1).abs().gt(TOLERANCE)
    write_csv(age_audit, TEMPORAL / "p_age_and_sample_size_audit.csv")

    hlo_duplicates = hlo[hlo.duplicated(["code", "year", "subject", "level"], keep=False)].copy()
    write_csv(hlo_duplicates, TEMPORAL / "hlo_retained_duplicate_cell_audit.csv")

    composition = nat[nat["iso3"].isin(rq2["iso3"])][["iso3", "country", "source_standard_reestimation", "source_qnw"]].copy()
    composition["composition_adjustment_exception"] = composition["source_standard_reestimation"].eq("N")
    composition["manual_rule_uses_2017_composition"] = composition["composition_adjustment_exception"]
    composition["period_adjustment_applied"] = False
    composition["reason"] = np.where(
        composition["composition_adjustment_exception"],
        "Source QNW uses a country-specific 2017 composition formula; it is not back-cast into period P.",
        "No source composition formula flagged, but period P remains a distinct construction from published QNW.",
    )
    write_csv(composition, TEMPORAL / "p_composition_exception_ledger.csv")


def date_assignment_sensitivity(samples: pd.DataFrame, grid: pd.DataFrame, sample_codes: set[str]) -> None:
    selected = samples[
        samples["included_by_source"].eq("Y")
        & samples["iso3"].isin(sample_codes)
        & samples["measurement_year"].notna()
        & samples["corrected_iq"].notna()
        & samples["qn_factor"].notna()
    ].copy()
    selected["calendar_year_ceiling"] = np.ceil(selected["measurement_year"]).astype(int)
    rows = []
    for scheme, windows in WINDOWS.items():
        for number, (start, end) in enumerate(windows, 1):
            period = selected[selected["calendar_year_ceiling"].between(start, end)]
            for iso3, group in period.groupby("iso3"):
                rows.append({
                    "scheme": scheme, "window_number": number, "iso3": iso3,
                    "p_score_ceiling_assignment": weighted_mean(group["corrected_iq"], group["qn_factor"]),
                    "p_record_count_ceiling_assignment": len(group),
                })
    ceiling = pd.DataFrame(rows)
    comparison = grid[[
        "scheme", "window_number", "window_start", "window_end", "iso3", "p_score",
        "p_record_count", "hlo_eligible", "eligible_aligned_cell",
    ]].merge(ceiling, on=["scheme", "window_number", "iso3"], how="left", validate="one_to_one")
    comparison["eligible_aligned_cell_ceiling_assignment"] = (
        comparison["p_score_ceiling_assignment"].notna() & comparison["hlo_eligible"]
    )
    comparison["p_presence_changes"] = comparison["p_score"].notna().ne(comparison["p_score_ceiling_assignment"].notna())
    comparison["aligned_eligibility_changes"] = comparison["eligible_aligned_cell"].ne(
        comparison["eligible_aligned_cell_ceiling_assignment"]
    )
    comparison["p_score_ceiling_minus_floor"] = comparison["p_score_ceiling_assignment"] - comparison["p_score"]
    write_csv(comparison, TEMPORAL / "p_date_assignment_sensitivity_cells.csv")
    summary = comparison.groupby("scheme", sort=False, as_index=False).agg(
        floor_aligned_eligible_cells=("eligible_aligned_cell", "sum"),
        ceiling_aligned_eligible_cells=("eligible_aligned_cell_ceiling_assignment", "sum"),
        cells_with_p_presence_change=("p_presence_changes", "sum"),
        cells_with_aligned_eligibility_change=("aligned_eligibility_changes", "sum"),
        max_absolute_p_score_change_common_cells=("p_score_ceiling_minus_floor", lambda x: x.abs().max()),
    )
    summary["ceiling_minus_floor_aligned_cells"] = (
        summary["ceiling_aligned_eligible_cells"] - summary["floor_aligned_eligible_cells"]
    )
    write_csv(summary, TEMPORAL / "p_date_assignment_sensitivity_summary.csv")


def build_temporal(samples: pd.DataFrame, hlo: pd.DataFrame, nat: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rq2 = pd.read_csv(RQ2_FILE)
    sample_codes = set(rq2["iso3"])
    grid = temporal_grid(rq2)
    p_ledger, p_cells = p_measurement(samples, sample_codes)
    hlo_ledger, hlo_components, hlo_cells = hlo_measurement(hlo, sample_codes)
    grid = grid.merge(p_cells, on=["scheme", "window_number", "iso3"], how="left", validate="one_to_one")
    grid = grid.merge(hlo_cells, on=["scheme", "window_number", "iso3"], how="left", validate="one_to_one")
    grid["has_p"] = grid["p_score"].notna()
    grid["has_hlo"] = grid["hlo_score"].notna()
    grid["hlo_eligible"] = grid["hlo_eligible"].eq(True)
    grid["eligible_aligned_cell"] = grid["has_p"] & grid["hlo_eligible"]
    grid["composition_adjustment_exception"] = grid["iso3"].isin(COMPOSITION_EXCEPTIONS)
    grid["exclusion_reason"] = np.select(
        [
            ~grid["has_p"] & ~grid["has_hlo"],
            ~grid["has_p"],
            ~grid["has_hlo"],
            grid["has_hlo"] & ~grid["hlo_eligible"],
        ],
        ["missing_p_and_hlo", "missing_p", "missing_hlo", "hlo_fails_both_levels_and_two_subjects_rule"],
        default="eligible",
    )
    reconstruction = reconstruct_qnw(samples, nat, rq2)
    audit_measurement(samples, hlo, nat, rq2)
    date_assignment_sensitivity(samples, grid, sample_codes)

    write_csv(p_ledger, TEMPORAL / "p_record_contribution_ledger.csv")
    write_csv(hlo_ledger, TEMPORAL / "hlo_record_contribution_ledger.csv")
    write_csv(hlo_components, TEMPORAL / "hlo_component_ledger.csv")
    write_csv(grid, TEMPORAL / "temporal_country_window_grid.csv")
    write_csv(grid[grid["eligible_aligned_cell"]].copy(), TEMPORAL / "temporal_eligible_cells.csv")
    write_csv(grid[~grid["eligible_aligned_cell"]].copy(), TEMPORAL / "temporal_exclusions.csv")
    write_csv(reconstruction, TEMPORAL / "p_full_history_qnw_reconstruction_check.csv")

    window_summary = grid.groupby(
        ["scheme", "window_number", "window_label", "window_start", "window_end"], sort=False, as_index=False
    ).agg(
        countries=("iso3", "size"), p_cells=("has_p", "sum"),
        hlo_cells=("has_hlo", "sum"), eligible_hlo_cells=("hlo_eligible", "sum"),
        aligned_eligible_cells=("eligible_aligned_cell", "sum"),
    )
    scheme_rows = []
    for scheme, group in grid.groupby("scheme", sort=False):
        eligible = group[group["eligible_aligned_cell"]]
        per_country = eligible.groupby("iso3").size()
        scheme_rows.append({
            "scheme": scheme, "window_count": group["window_number"].nunique(),
            "country_window_cells": len(group), "aligned_eligible_cells": len(eligible),
            "countries_with_aligned_cell": eligible["iso3"].nunique(),
            "countries_with_2plus_aligned_windows": per_country.ge(2).sum(),
            "one_sample_id_eligible_cells": eligible["p_sample_id_count"].eq(1).sum(),
            "median_p_records_per_eligible_cell": eligible["p_record_count"].median(),
            "median_hlo_components_per_eligible_cell": eligible["hlo_component_count"].median(),
        })
    scheme_summary = pd.DataFrame(scheme_rows)
    write_csv(window_summary, TEMPORAL / "temporal_window_summary.csv")
    write_csv(scheme_summary, TEMPORAL / "temporal_scheme_summary.csv")
    return grid, window_summary, scheme_summary


def write_manifest_and_dictionary() -> None:
    sources = [NIQ_WORKBOOK, NIQ_MANUAL, HLO_FILE, RQ1_FILE, RQ2_FILE, ARCHIVED_RANKING, ARCHIVED_GDP]
    manifest = pd.DataFrame([
        {
            "source_id": path.name,
            "repository_relative_path": path.relative_to(REPO).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "role": "frozen_input_read_only",
        }
        for path in sources
    ])
    write_csv(manifest, INPUTS / "measurement_input_manifest.csv")
    dictionary = pd.DataFrame([
        ["niq_samples_v135_rebuilt.csv", "Offline reconstruction of NIQ v1.3.5 REC records joined to source selection flags."],
        ["core_results.csv", "Freshly calculated frozen-core score, rank, tail, and GDP results."],
        ["core_reconciliation.csv", "Fresh-versus-archived numerical differences and 1e-12 checks."],
        ["temporal_country_window_grid.csv", "All 65 countries crossed with annual, four-year, and five-year windows; missing targets remain missing."],
        ["temporal_eligible_cells.csv", "Grid cells with P and HLO satisfying both levels plus at least two subjects."],
        ["p_record_contribution_ledger.csv", "Selected psychometric record dates, QN weights, normalized P weights, ages, and source IDs."],
        ["hlo_record_contribution_ledger.csv", "HLO record-to-year-cell-to-component-to-level-to-composite weights."],
        ["hlo_component_ledger.csv", "HLO subject-by-level component scores, dates, dispersion, and composite weights."],
        ["p_full_history_qnw_reconstruction_check.csv", "Simple QN-weighted history versus published QNW and seven composition exceptions."],
        ["p_fractional_date_audit.csv", "Fractional psychometric dates retained without inventing exact years."],
        ["p_date_boundary_audit.csv", "Floor-versus-ceiling window-assignment diagnostic for fractional dates."],
        ["p_sample_id_audit.csv", "Repeated source sample IDs and within-ID conflicts."],
        ["p_age_and_sample_size_audit.csv", "Age completeness/order and fractional sample-size audit."],
        ["hlo_retained_duplicate_cell_audit.csv", "Retained duplicate economy-year-subject-level HLO rows after the official source filter."],
    ], columns=["artifact", "description"])
    write_csv(dictionary, DOCS / "MEASUREMENT_DATA_DICTIONARY.csv")


def describe_column(column: str) -> str:
    exact = {
        "scheme": "Assessment-window scheme: annual, 4_year, or 5_year.",
        "window_number": "One-based window order within the scheme.",
        "window_start": "First included calendar year.",
        "window_end": "Last included calendar year.",
        "iso3": "ISO-style three-character economy identifier used by the frozen analysis.",
        "p_score": "Period P: corrected psychometric IQ weighted by source QN factor within the country-window.",
        "hlo_score": "Hierarchical HLO summary: years within component, subjects within level, then levels.",
        "eligible_aligned_cell": "True when P exists and HLO meets both-levels plus two-subjects eligibility.",
        "normalized_composite_weight": "Record QN factor divided by the country-window total QN factor.",
        "composite_weight": "Product of HLO record-to-cell, cell-to-component, component-to-level, and level-to-composite weights.",
        "hlo_composite_se": "Intentionally missing; a composite SE is not identified from record SEs without covariance information.",
        "p_is_exact_published_qnw": "Always false: period P is not the source's published full-history QNW.",
        "partial_terminal_window": "True only for the accepted 2015-2017 partial five-year terminal window.",
        "exclusion_reason": "Outcome-blind reason a country-window is not an aligned eligible measurement cell.",
    }
    if column in exact:
        return exact[column]
    if column.endswith("_count"):
        return "Count defined by the column prefix within the stated row scope."
    if column.endswith("_sd"):
        return "Descriptive standard deviation with population denominator (ddof=0), not a sampling standard error."
    if column.endswith("_weight"):
        return "Construction weight at the stage named by the column."
    if column.endswith("_year") or column.endswith("_year_min") or column.endswith("_year_max"):
        return "Recorded or derived year as specified by the column name."
    if column.startswith("has_") or column.startswith("eligible_") or column.endswith("_flag"):
        return "Boolean audit or eligibility indicator defined by the column name."
    return "Field retained or calculated as named; see MEASUREMENT_READINESS.md and the generating script for its construction."


def write_output_documentation() -> None:
    csvs = sorted([*INPUTS.glob("*.csv"), *CORE.glob("*.csv"), *TEMPORAL.glob("*.csv")])
    column_rows = []
    output_rows = []
    for path in csvs:
        frame = pd.read_csv(path)
        relative = path.relative_to(REWRITE).as_posix()
        output_rows.append({
            "artifact": relative,
            "rows": len(frame),
            "columns": len(frame.columns),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "generator": "rewrite/src/measurement/build_phase_a_measurement.py",
        })
        for column in frame.columns:
            column_rows.append({
                "artifact": relative,
                "column": column,
                "pandas_dtype": str(frame[column].dtype),
                "nonmissing_rows": int(frame[column].notna().sum()),
                "description": describe_column(column),
            })
    write_csv(pd.DataFrame(column_rows), DOCS / "MEASUREMENT_COLUMN_DICTIONARY.csv")
    write_csv(pd.DataFrame(output_rows), VALIDATION / "measurement_output_manifest.csv")

    source_manifest = pd.read_csv(INPUTS / "measurement_input_manifest.csv")
    preservation = []
    for row in source_manifest.itertuples(index=False):
        path = REPO / row.repository_relative_path
        current = sha256(path)
        preservation.append({
            "repository_relative_path": row.repository_relative_path,
            "before_sha256": row.sha256,
            "after_sha256": current,
            "unchanged": current == row.sha256,
        })
    write_csv(pd.DataFrame(preservation), VALIDATION / "protected_input_preservation.csv")


def write_readiness(core_results: pd.DataFrame, reconciliation: pd.DataFrame, window_summary: pd.DataFrame, scheme_summary: pd.DataFrame) -> None:
    values = core_results.set_index("result_id")["estimate"]
    five = window_summary[window_summary["scheme"].eq("5_year")]
    date_sensitivity = pd.read_csv(TEMPORAL / "p_date_assignment_sensitivity_summary.csv").set_index("scheme")
    overlap_groups = len(pd.read_csv(TEMPORAL / "p_potential_overlap_audit.csv"))
    age_audit = pd.read_csv(TEMPORAL / "p_age_and_sample_size_audit.csv")
    report = f"""# Agent 1 measurement-readiness report

Status: READY_FOR_CORE_REWRITE; temporal/context modelling remains WAITING_FOR_PROTOCOL_APPROVAL.

## Frozen core reproduction

- RQ1 contains {int(values['CORE_RQ1_N'])} economies; Pearson QNW-HLO r = {values['CORE_QNW_HLO_PEARSON']:.15f}.
- Median/mean/maximum absolute rank displacement = {values['CORE_RANK_MEDIAN_ABS']:.0f} / {values['CORE_RANK_MEAN_ABS']:.15f} / {values['CORE_RANK_MAX_ABS']:.0f}.
- Strict ordering reversals = {int(values['CORE_ORDER_REVERSALS'])} of {int(values['CORE_STRICT_PAIRS'])}.
- Upper/lower empirical-decile overlap = {int(values['CORE_TOP_OVERLAP'])}/{int(values['CORE_TAIL_DENOMINATOR'])} and {int(values['CORE_BOTTOM_OVERLAP'])}/{int(values['CORE_TAIL_DENOMINATOR'])}.
- RQ2 contains {int(values['CORE_RQ2_N'])} economies; QNW-GDP r = {values['CORE_QNW_GDP_PEARSON']:.15f}, HLO-GDP r = {values['CORE_HLO_GDP_PEARSON']:.15f}, difference = {values['CORE_GDP_CORRELATION_DIFFERENCE_QNW_MINUS_HLO']:.15f}.
- All {len(reconciliation)} fresh-versus-archived comparisons pass the absolute tolerance 1e-12: {bool(reconciliation['passes_tolerance_1e_12'].all())}.

## Temporal construction

- Annual, four-year and five-year aligned eligibility is calculated from the rebuilt source records; no target is imputed.
- Five-year eligible cells by window: {', '.join(str(int(value)) for value in five['aligned_eligible_cells'])}.
- The terminal 2015-2017 period is included and explicitly marked as partial.
- P is a within-window QN-weighted psychometric score and is never labelled exact published QNW.
- HLO is averaged across retained ties, years within subject-by-level, subjects within level, and then levels. Eligibility requires both levels and at least two subjects overall.

## Measurement warnings that remain visible

- Published QNW applies composition formulas for BRA, CAN, KAZ, MEX, SRB, USA and ZAF. These 2017 formulas are not back-cast into P.
- Fractional dates are preserved. Floor assignment follows the archived builder; the ceiling diagnostic records boundary-sensitive assignments but is not a replacement construction.
- Under ceiling rather than floor assignment, aligned cells are {int(date_sensitivity.loc['annual', 'ceiling_aligned_eligible_cells'])} annual, {int(date_sensitivity.loc['4_year', 'ceiling_aligned_eligible_cells'])} four-year and {int(date_sensitivity.loc['5_year', 'ceiling_aligned_eligible_cells'])} five-year, compared with 13/58/58 under the archived floor rule.
- Different sample IDs are not proof of independent study populations. The audit flags {overlap_groups} country-reference-year groups with multiple IDs as candidates for source review; it does not assert that they overlap.
- Complete lowest/mean/highest age metadata are available for {int(age_audit['age_metadata_complete'].sum())} of {len(age_audit)} selected records in the 65-economy frame.
- Observed-record and component dispersion are descriptive. A one-study cell has no between-study estimate. HLO record-level standard errors are not averaged into a composite standard error.
- Common calendar windows do not establish common birth cohorts, test content, or assessed populations.

## Gate

The core and measurement sections are ready for manuscript use. Empirical predictive modelling is not authorized until the unresolved protocol choices are explicitly approved. No predictive model was fitted here.
"""
    (DOCS / "MEASUREMENT_READINESS.md").write_text(report, encoding="utf-8")


def main() -> None:
    for path in (INPUTS, CORE, TEMPORAL, VALIDATION, DOCS):
        path.mkdir(parents=True, exist_ok=True)
    write_manifest_and_dictionary()
    samples, _, nat = recover_niq()
    hlo = recover_hlo()
    core_results, reconciliation = reproduce_core()
    _, window_summary, scheme_summary = build_temporal(samples, hlo, nat)
    write_readiness(core_results, reconciliation, window_summary, scheme_summary)
    run_metadata = {
        "python": sys.version,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "source_builder": Path(__file__).relative_to(REPO).as_posix(),
        "network_used": False,
        "predictive_models_fitted": False,
        "numerical_tolerance": TOLERANCE,
    }
    (VALIDATION / "measurement_run_metadata.json").write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")
    write_output_documentation()
    print(scheme_summary.to_string(index=False))


if __name__ == "__main__":
    main()
