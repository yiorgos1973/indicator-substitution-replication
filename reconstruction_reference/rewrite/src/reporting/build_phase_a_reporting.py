from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


REPO = Path(__file__).resolve().parents[3]
REWRITE = REPO / "rewrite"
CORE = REWRITE / "analysis" / "core"
TEMPORAL = REWRITE / "analysis" / "temporal"
CONTEXT = REWRITE / "analysis" / "context"
TABLES = REWRITE / "tables"
FIGURES = REWRITE / "figures"
RESULTS = REWRITE / "results"
DOCS = REWRITE / "docs" / "agent4"
VALIDATION = REWRITE / "validation" / "reporting"
RUN_ID = "phase-a-reporting-2026-09-29"
GENERATOR = "rewrite/src/reporting/build_phase_a_reporting.py"

SOURCES = {
    "core_results": CORE / "core_results.csv",
    "core_ranks": CORE / "core_rank_ledger.csv",
    "rq1": CORE / "core_rq1_exact_sample.csv",
    "rq2": CORE / "core_rq2_exact_sample.csv",
    "temporal_scheme": TEMPORAL / "temporal_scheme_summary.csv",
    "temporal_window": TEMPORAL / "temporal_window_summary.csv",
    "temporal_grid": TEMPORAL / "temporal_country_window_grid.csv",
    "date_sensitivity": TEMPORAL / "p_date_assignment_sensitivity_summary.csv",
    "block_all": CONTEXT / "aligned_cell_block_preceding_coverage_all_features_summary.csv",
    "block_feature": CONTEXT / "aligned_cell_block_preceding_coverage_by_feature_summary.csv",
    "assessment_all": CONTEXT / "assessment_specific_coverage_all_features_summary.csv",
    "assessment_feature": CONTEXT / "assessment_specific_coverage_by_feature_summary.csv",
    "bootstrap": REPO / "outputs" / "postreview_bootstrap_delta_r_summary.csv",
    "fixed_sensitivity": REPO / "outputs" / "v06_sensitivity_fixed_sample.csv",
    "coverage_sensitivity": REPO / "outputs" / "v06_sensitivity_coverage_changing.csv",
    "weight_sensitivity": REPO / "outputs" / "v06_sensitivity_weighting_estimand.csv",
    "membership_curve": REPO / "outputs" / "posthoc_rank_membership_curve.csv",
    "measurement_validation": REWRITE / "validation" / "measurement" / "measurement_validation_results.csv",
    "context_validation": REWRITE / "validation" / "context" / "context_validation_results.csv",
    "context_status": CONTEXT / "context_build_status.json",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def display_value(value: object) -> str:
    if pd.isna(value):
        return "—"
    if isinstance(value, (bool, np.bool_)):
        return "Yes" if value else "No"
    if isinstance(value, (float, np.floating)):
        if float(value).is_integer():
            return str(int(value))
        return f"{value:.3f}"
    return str(value)


def markdown_table(frame: pd.DataFrame) -> str:
    shown = frame.map(display_value)
    header = "| " + " | ".join(shown.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(shown.columns)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in shown.astype(str).to_numpy()]
    return "\n".join([header, separator, *rows])


def save_table(exhibit_id: str, title: str, source: pd.DataFrame, display_columns: list[str], notes: list[str]) -> None:
    write_csv(source, TABLES / f"{exhibit_id}_source.csv")
    display = source[display_columns].copy()
    body = f"# {title}\n\n{markdown_table(display)}\n\n" + "\n".join(f"- {note}" for note in notes) + "\n"
    (TABLES / f"{exhibit_id}_display.md").write_text(body, encoding="utf-8")


def source_hashes(names: list[str]) -> str:
    return " | ".join(f"{SOURCES[name].relative_to(REPO).as_posix()}={sha256(SOURCES[name])}" for name in names)


def verify_inputs() -> None:
    for path in SOURCES.values():
        assert path.exists(), path
    measurement = pd.read_csv(SOURCES["measurement_validation"])
    context = pd.read_csv(SOURCES["context_validation"])
    assert measurement["status"].eq("PASS").all()
    assert context["status"].eq("PASS").all()
    status = json.loads(SOURCES["context_status"].read_text(encoding="utf-8"))
    assert status["empirical_modelling_performed"] is False


def build_tables(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    core = data["core_results"].set_index("result_id")["estimate"]
    scheme = data["temporal_scheme"]
    five_windows = data["temporal_window"].query("scheme == '5_year'")

    indicators = pd.DataFrame([
        {
            "indicator": "Published QNW", "role": "Frozen core psychometric indicator",
            "assessed_evidence": "Selected psychometric samples in NIQ v1.3.5",
            "date_scope": "Source-specific dates; full-history country aggregate",
            "weighting": "Corrected IQ weighted by QN factor; seven countries also use source composition formulas",
            "analysis_coverage": f"{int(core['CORE_RQ1_N'])} common economies in RQ1",
            "important_boundary": "Not period P; composition formulas are not back-cast",
        },
        {
            "indicator": "HLO aggregate", "role": "Frozen core learning indicator",
            "assessed_evidence": "International and regional student assessments",
            "date_scope": "2000-2017",
            "weighting": "Years within subject×level; subjects within level; equal levels",
            "analysis_coverage": f"{int(core['CORE_RQ1_N'])} common economies in RQ1",
            "important_boundary": "Both levels and at least two subjects required; no composite SE inferred",
        },
        {
            "indicator": "Period P", "role": "Temporal psychometric target or source",
            "assessed_evidence": "Source-selected psychometric records in each calendar window",
            "date_scope": "Annual, 4-year, and 5-year windows; final 5-year window is 2015-2017",
            "weighting": "Corrected IQ weighted by within-window QN factor",
            "analysis_coverage": f"{int(scheme.loc[scheme['scheme'].eq('5_year'), 'aligned_eligible_cells'].iloc[0])} aligned cells / {int(scheme.loc[scheme['scheme'].eq('5_year'), 'countries_with_aligned_cell'].iloc[0])} countries",
            "important_boundary": "Distinct construction from published QNW for every country-window",
        },
        {
            "indicator": "Historical WDI context", "role": "Candidate contextual predictor block",
            "assessed_evidence": "Seven archived annual World Development Indicators",
            "date_scope": "Preceding 5, 10, or 12 years; same-window retained only as comparator",
            "weighting": "Equal observed years, then target-specific construction weights when assessment-specific",
            "analysis_coverage": f"{int(five_windows['aligned_eligible_cells'].sum())} five-year aligned cells screened",
            "important_boundary": "Historical country context, not exact cohort exposure or causal lag",
        },
    ])
    save_table(
        "T_INDICATORS", "Indicator definitions and measurement boundaries", indicators,
        list(indicators.columns),
        [
            "Coverage is calculated from the frozen core and eligible five-year temporal files.",
            "QNW is the published source aggregate; P is the study's period-specific construction.",
            "A common calendar window does not imply a common cohort, content domain, or assessed population.",
        ],
    )

    core_substitution = pd.DataFrame([
        ["Score association", "Pearson correlation", core["CORE_QNW_HLO_PEARSON"], "correlation", int(core["CORE_RQ1_N"]), "66 common economies"],
        ["Rank association", "Spearman correlation", core["CORE_QNW_HLO_SPEARMAN"], "correlation", int(core["CORE_RQ1_N"]), "66 common economies"],
        ["Rank association", "Kendall correlation", core["CORE_QNW_HLO_KENDALL"], "correlation", int(core["CORE_RQ1_N"]), "66 common economies"],
        ["Rank movement", "Median absolute displacement", core["CORE_RANK_MEDIAN_ABS"], "rank places", int(core["CORE_RQ1_N"]), "economies"],
        ["Rank movement", "Mean absolute displacement", core["CORE_RANK_MEAN_ABS"], "rank places", int(core["CORE_RQ1_N"]), "economies"],
        ["Rank movement", "Maximum absolute displacement", core["CORE_RANK_MAX_ABS"], "rank places", int(core["CORE_RQ1_N"]), "economies"],
        ["Strict pairs", "Ordering reversals", core["CORE_ORDER_REVERSALS"], "pairs", int(core["CORE_STRICT_PAIRS"]), "strict country pairs"],
        ["Upper group", "Common members", core["CORE_TOP_OVERLAP"], "economies", int(core["CORE_TAIL_DENOMINATOR"]), "members in each empirical-decile group"],
        ["Lower group", "Common members", core["CORE_BOTTOM_OVERLAP"], "economies", int(core["CORE_TAIL_DENOMINATOR"]), "members in each empirical-decile group"],
    ], columns=["output", "metric", "estimate", "unit", "n", "denominator"])
    save_table(
        "T_CORE_SUBSTITUTION", "Core indicator-substitution results", core_substitution,
        list(core_substitution.columns),
        [
            "The fixed RQ1 sample contains 66 economies; rank 1 is the highest score.",
            "Empirical deciles contain seven economies per indicator. Overlap is intersection divided by seven, not Jaccard similarity.",
            "Rank summaries describe published point scores, not uncertainty in latent or true ranks.",
        ],
    )

    bootstrap = data["bootstrap"].iloc[0]
    gdp_rows = [
        ["Primary", "QNW-GDP", "Pearson correlation", core["CORE_QNW_GDP_PEARSON"], np.nan, np.nan, "none", int(core["CORE_RQ2_N"])],
        ["Primary", "HLO-GDP", "Pearson correlation", core["CORE_HLO_GDP_PEARSON"], np.nan, np.nan, "none", int(core["CORE_RQ2_N"])],
        ["Primary contrast", "QNW minus HLO", "Difference in dependent correlations", core["CORE_GDP_CORRELATION_DIFFERENCE_QNW_MINUS_HLO"], core["CORE_ZOU_LOWER"], core["CORE_ZOU_UPPER"], "Zou 95% interval", int(core["CORE_RQ2_N"])],
        ["Primary contrast", "QNW minus HLO", "Difference in dependent correlations", bootstrap["observed_delta_r"], bootstrap["percentile_ci_95_lower"], bootstrap["percentile_ci_95_upper"], "Paired economy bootstrap percentile 95%", int(bootstrap["n_economies"])],
        ["Primary contrast", "QNW minus HLO", "Difference in dependent correlations", bootstrap["observed_delta_r"], bootstrap["bca_ci_95_lower"], bootstrap["bca_ci_95_upper"], "Paired economy bootstrap BCa 95%", int(bootstrap["n_economies"])],
        ["Primary test", "QNW minus HLO", "Williams two-sided p", core["CORE_WILLIAMS_P"], np.nan, np.nan, "Williams test", int(core["CORE_RQ2_N"])],
    ]
    fixed = data["fixed_sensitivity"]
    for row in fixed.itertuples(index=False):
        gdp_rows.append(["Fixed-sample sensitivity", row.scenario, "QNW-GDP minus HLO-GDP", row.qnw_minus_hlo_gdp, np.nan, np.nan, "none", int(row.gdp_n)])
    coverage = data["coverage_sensitivity"].query("sensitivity_class != 'reference'")
    for row in coverage.itertuples(index=False):
        gdp_rows.append(["Coverage-changing sensitivity", row.scenario, "QNW-GDP minus HLO-GDP", row.qnw_minus_hlo_gdp, np.nan, np.nan, "none", int(row.gdp_n)])
    weighting = data["weight_sensitivity"]
    for row in weighting.itertuples(index=False):
        gdp_rows.append(["Weighting sensitivity", row.estimand, "QNW-GDP minus HLO-GDP", row.qnw_minus_hlo_gdp, np.nan, np.nan, "none", int(row.n_economies)])
    gdp = pd.DataFrame(gdp_rows, columns=["section", "scenario", "metric", "estimate", "lower", "upper", "interval_or_test", "n_economies"])
    save_table(
        "T_GDP_CONTRAST", "GDP associations, paired contrast, and selected sensitivities", gdp,
        list(gdp.columns),
        [
            "The sign is always QNW-GDP correlation minus HLO-GDP correlation; negative values favour a larger observed HLO-GDP correlation.",
            "RQ2 uses natural-log 2017 PPP GDP per capita and equal economy weights unless the row says population-weighted.",
            "The paired bootstrap resamples complete economy triplets. Its interval is not a null distribution and supplies no bootstrap p-value here.",
            "QNW plus SAS includes school-assessment content and changes sample coverage; it is not direct QNW.",
            "Intervals containing zero do not establish equivalence.",
        ],
    )

    temporal_rows = []
    for row in scheme.itertuples(index=False):
        temporal_rows.append([
            "Temporal scheme", row.scheme, "all", row.country_window_cells,
            row.aligned_eligible_cells, row.countries_with_aligned_cell,
            row.countries_with_2plus_aligned_windows, row.one_sample_id_eligible_cells,
            np.nan, np.nan, np.nan, np.nan, np.nan,
        ])
    for row in five_windows.itertuples(index=False):
        temporal_rows.append([
            "Five-year window", "5_year", row.window_label, row.countries,
            row.aligned_eligible_cells, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
        ])
    block = data["block_all"].query("target_direction == 'HLO_to_P'")
    for row in block.itertuples(index=False):
        temporal_rows.append([
            "Block-preceding WDI", row.construction, "58 aligned five-year cells", row.country_window_cells,
            np.nan, np.nan, np.nan, np.nan,
            row.all_features_with_any_year, row.all_features_at_least_80pct, row.all_features_complete, np.nan, np.nan,
        ])
    assessment = data["assessment_all"]
    for row in assessment.itertuples(index=False):
        temporal_rows.append([
            f"Assessment-specific WDI ({row.target_direction})", row.construction,
            "58 aligned five-year cells", row.country_window_cells,
            np.nan, np.nan, np.nan, np.nan,
            row.all_features_with_any_year, row.all_features_at_least_80pct, row.all_features_complete, np.nan, np.nan,
        ])
    date_sensitivity = data["date_sensitivity"]
    cells_by_scheme = scheme.set_index("scheme")["country_window_cells"]
    for row in date_sensitivity.itertuples(index=False):
        temporal_rows.append([
            "Fractional-date assignment sensitivity", row.scheme, "floor versus ceiling", cells_by_scheme[row.scheme],
            row.floor_aligned_eligible_cells, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
            row.ceiling_aligned_eligible_cells, row.cells_with_aligned_eligibility_change,
        ])
    temporal = pd.DataFrame(temporal_rows, columns=[
        "section", "scheme_or_construction", "period", "candidate_cells",
        "aligned_eligible_cells", "countries", "countries_with_2plus_windows",
        "one_sample_id_cells", "all_7_any_year", "all_7_at_least_80pct", "all_7_complete",
        "ceiling_assignment_aligned_cells", "cells_changing_aligned_eligibility",
    ])
    save_table(
        "T_TEMPORAL_COVERAGE", "Temporal measurement and historical-context coverage", temporal,
        list(temporal.columns),
        [
            "Aligned measurement cells require P plus HLO with both levels and at least two subjects.",
            "The accepted five-year periods are 2000-2004, 2005-2009, 2010-2014, and the partial terminal period 2015-2017.",
            "All WDI thresholds are outcome-blind coverage diagnostics, not approved feature-retention rules.",
            "Assessment-specific coverage is direction-dependent because it uses the target's actual measurement schedule and construction weights.",
            "The date sensitivity changes only the assignment of fractional psychometric years from floor to ceiling; the archived floor rule remains primary.",
        ],
    )
    return {
        "T_INDICATORS": indicators,
        "T_CORE_SUBSTITUTION": core_substitution,
        "T_GDP_CONTRAST": gdp,
        "T_TEMPORAL_COVERAGE": temporal,
    }


def build_figures(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 11,
        "axes.labelsize": 9, "legend.fontsize": 8, "figure.dpi": 100,
        "svg.hashsalt": "niq-hlo-phase-a-reporting",
    })
    blue, orange, green, grey = "#0072B2", "#D55E00", "#009E73", "#666666"
    figure_data = {}

    ranks = data["core_ranks"].copy()
    ranks["label_display"] = np.where(ranks["absolute_rank_displacement"].ge(20), ranks["iso3"], "")
    write_csv(ranks, FIGURES / "F_CORE_RANKS_plot_data.csv")
    fig, ax = plt.subplots(figsize=(6.5, 5.6), constrained_layout=True)
    ax.scatter(ranks["rank_qnw"], ranks["rank_hlo"], s=30, color=blue, alpha=.78, edgecolor="white", linewidth=.35)
    limits = [67, 0]
    ax.plot([66, 1], [66, 1], linestyle="--", color=grey, linewidth=1, label="Same rank")
    for row in ranks[ranks["label_display"].ne("")].itertuples(index=False):
        ax.annotate(row.label_display, (row.rank_qnw, row.rank_hlo), xytext=(4, 3), textcoords="offset points", fontsize=8)
    ax.set(xlim=limits, ylim=limits, xlabel="Published QNW rank (1 = highest)", ylabel="HLO rank (1 = highest)", title="Country ranks differ despite strong score association")
    ax.set_xticks([66, 60, 50, 40, 30, 20, 10, 1])
    ax.set_yticks([66, 60, 50, 40, 30, 20, 10, 1])
    ax.grid(alpha=.2)
    ax.legend(frameon=False, loc="lower right")
    fig.savefig(FIGURES / "F_CORE_RANKS.svg", metadata={"Date": None})
    fig.savefig(FIGURES / "F_CORE_RANKS.png", dpi=300)
    plt.close(fig)
    figure_data["F_CORE_RANKS"] = ranks

    curve = data["membership_curve"].copy()
    write_csv(curve, FIGURES / "F_GROUP_CUTOFFS_plot_data.csv")
    fig, ax = plt.subplots(figsize=(6.5, 4.4), constrained_layout=True)
    for tail, color, marker in [("top", blue, "o"), ("bottom", orange, "s")]:
        part = curve[curve["tail"].eq(tail)].sort_values("cutoff_k")
        ax.plot(part["cutoff_k"], part["overlap_proportion_relative_to_k"], color=color, marker=marker, linewidth=1.8, label=f"{tail.title()} group")
    ax.set(xlabel="Number selected by each indicator (k)", ylabel="Shared members / k", title="Group agreement depends on the selection cutoff", xticks=sorted(curve["cutoff_k"].unique()), ylim=(0, 1))
    ax.grid(axis="y", alpha=.25)
    ax.legend(frameon=False)
    fig.savefig(FIGURES / "F_GROUP_CUTOFFS.svg", metadata={"Date": None})
    fig.savefig(FIGURES / "F_GROUP_CUTOFFS.png", dpi=300)
    plt.close(fig)
    figure_data["F_GROUP_CUTOFFS"] = curve

    scheme = data["temporal_scheme"].copy()
    windows = data["temporal_window"].query("scheme == '5_year'").copy()
    block = data["block_all"].query("target_direction == 'HLO_to_P'").copy()
    block_order = ["same_window_comparator", "block_preceding_5y", "block_preceding_10y", "block_preceding_12y"]
    block["plot_order"] = block["construction"].map({name: i for i, name in enumerate(block_order)})
    block = block.sort_values("plot_order")
    assessment = data["assessment_all"].copy()
    assessment["duration"] = assessment["construction"].str.extract(r"_(\d+)y")[0].astype(int)
    plot_rows = []
    for row in scheme.itertuples(index=False):
        plot_rows.append(["A", row.scheme, "aligned eligible cells", row.aligned_eligible_cells, "all"])
    for row in windows.itertuples(index=False):
        plot_rows.extend([
            ["B", row.window_label, "P cells", row.p_cells, "5_year"],
            ["B", row.window_label, "eligible HLO cells", row.eligible_hlo_cells, "5_year"],
            ["B", row.window_label, "aligned eligible cells", row.aligned_eligible_cells, "5_year"],
        ])
    for row in block.itertuples(index=False):
        label = {"same_window_comparator": "Same window", "block_preceding_5y": "Previous 5y", "block_preceding_10y": "Previous 10y", "block_preceding_12y": "Previous 12y"}[row.construction]
        plot_rows.extend([
            ["C", label, "≥1 year each", row.all_features_with_any_year, "all 7 WDI"],
            ["C", label, "≥80% each", row.all_features_at_least_80pct, "all 7 WDI"],
            ["C", label, "complete", row.all_features_complete, "all 7 WDI"],
        ])
    for row in assessment.itertuples(index=False):
        plot_rows.append(["D", f"{row.duration} years", row.target_direction, row.all_features_at_least_80pct, "all 7 WDI ≥80%"])
    coverage_plot = pd.DataFrame(plot_rows, columns=["panel", "category", "series", "value", "scope"])
    write_csv(coverage_plot, FIGURES / "F_TEMPORAL_COVERAGE_plot_data.csv")

    fig, axes = plt.subplots(2, 2, figsize=(8.3, 6.7), constrained_layout=True)
    ax = axes[0, 0]
    ax.bar(["Annual", "4-year", "5-year"], scheme["aligned_eligible_cells"], color=[grey, orange, blue])
    ax.set(title="A. Aligned measurement cells", ylabel="Country-window cells")
    ax.bar_label(ax.containers[0], padding=2)

    ax = axes[0, 1]
    x = np.arange(len(windows)); width = .24
    for offset, (column, label, color) in enumerate([
        ("p_cells", "P available", blue), ("eligible_hlo_cells", "HLO eligible", green),
        ("aligned_eligible_cells", "Aligned", orange),
    ]):
        ax.bar(x + (offset - 1) * width, windows[column], width, label=label, color=color)
    ax.set(title="B. Five-year measurement coverage", ylabel="Countries", xticks=x, xticklabels=windows["window_label"])
    ax.legend(frameon=False, ncol=3, fontsize=7)

    ax = axes[1, 0]
    x = np.arange(len(block)); width = .24
    labels = ["Same\nwindow", "Previous\n5y", "Previous\n10y", "Previous\n12y"]
    for offset, (column, label, color) in enumerate([
        ("all_features_with_any_year", "≥1 year", blue),
        ("all_features_at_least_80pct", "≥80%", orange),
        ("all_features_complete", "Complete", green),
    ]):
        ax.bar(x + (offset - 1) * width, block[column], width, label=label, color=color)
    ax.set(title="C. Block-based WDI coverage", ylabel="Cells with all 7 features", xticks=x, xticklabels=labels)
    ax.legend(frameon=False, fontsize=7)

    ax = axes[1, 1]
    for direction, color, marker in [("HLO_to_P", blue, "o"), ("P_to_HLO", orange, "s")]:
        part = assessment[assessment["target_direction"].eq(direction)].sort_values("duration")
        label = "P target schedule" if direction == "HLO_to_P" else "HLO target schedule"
        ax.plot(part["duration"], part["all_features_at_least_80pct"], marker=marker, color=color, linewidth=1.8, label=label)
    ax.set(title="D. Assessment-specific WDI coverage", xlabel="History duration (years)", ylabel="Cells with all 7 at ≥80%", xticks=[5, 10, 12])
    ax.legend(frameon=False, fontsize=7)
    for ax in axes.flat:
        ax.grid(axis="y", alpha=.2)
    fig.savefig(FIGURES / "F_TEMPORAL_COVERAGE.svg", metadata={"Date": None})
    fig.savefig(FIGURES / "F_TEMPORAL_COVERAGE.png", dpi=300)
    plt.close(fig)
    figure_data["F_TEMPORAL_COVERAGE"] = coverage_plot
    return figure_data


def result_row(result_id: str, family: str, metric: str, estimate: float, source: Path, **kwargs) -> dict[str, object]:
    row = {
        "result_id": result_id, "analysis_family": family, "status": "calculated_validated",
        "run_id": RUN_ID, "protocol_hash": "", "target": "", "model_or_comparison": "",
        "period": "", "metric": metric, "estimate": estimate, "lower": np.nan, "upper": np.nan,
        "interval_method": "", "units": "", "n_countries": np.nan, "n_cells": np.nan,
        "denominator_definition": "", "weighting_rule": "", "sample_id": "",
        "source_artifact": source.relative_to(REPO).as_posix(), "source_locator": "",
        "generator": GENERATOR, "input_hash": sha256(source), "interpretation": "", "limitation": "",
    }
    row.update(kwargs)
    return row


def build_results_register(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    core = data["core_results"]
    for record in core.itertuples(index=False):
        rq1_tokens = ("RQ1", "RANK", "ORDER", "OVERLAP", "TAIL", "QNW_HLO", "STRICT")
        is_rq1 = any(token in record.result_id for token in rq1_tokens)
        unit = "count"
        if any(token in record.result_id for token in ("PEARSON", "SPEARMAN", "KENDALL", "DIFFERENCE")):
            unit = "correlation or correlation difference"
        elif "P" == record.result_id[-1:] or "SHARE" in record.result_id or "NORMALIZED" in record.result_id:
            unit = "proportion"
        elif "RANK" in record.result_id and not record.result_id.endswith("N"):
            unit = "rank places"
        rows.append(result_row(
            record.result_id, "frozen_core", record.result_id, record.estimate, SOURCES["core_results"],
            units=unit, n_countries=66 if is_rq1 else 65,
            denominator_definition="Fixed 66-economy RQ1 sample" if is_rq1 else "Fixed 65-economy RQ2 sample",
            weighting_rule="Equal economies", sample_id="RQ1_66" if is_rq1 else "RQ2_65",
            source_locator=f"result_id={record.result_id}",
            limitation="Core point-score description; does not establish causal or construct equivalence.",
        ))
    for row in data["temporal_scheme"].itertuples(index=False):
        for metric in ["aligned_eligible_cells", "countries_with_aligned_cell", "countries_with_2plus_aligned_windows", "one_sample_id_eligible_cells"]:
            rows.append(result_row(
                f"TEMP_{row.scheme.upper()}_{metric.upper()}", "temporal_coverage", metric, getattr(row, metric), SOURCES["temporal_scheme"],
                period=row.scheme, units="count", n_countries=row.countries_with_aligned_cell,
                n_cells=row.aligned_eligible_cells, denominator_definition="Frozen 65-economy frame crossed with scheme windows",
                weighting_rule="P QN weights; hierarchical HLO", sample_id="RQ2_65_temporal",
                source_locator=f"scheme={row.scheme}",
                interpretation="Outcome-blind temporal measurement availability.",
                limitation="Cells repeat countries and are not independent countries.",
            ))
    for row in data["temporal_window"].query("scheme == '5_year'").itertuples(index=False):
        rows.append(result_row(
            f"TEMP_5Y_WINDOW_{row.window_number}_ALIGNED", "temporal_coverage", "aligned_eligible_cells", row.aligned_eligible_cells, SOURCES["temporal_window"],
            period=row.window_label, units="cells", n_cells=row.aligned_eligible_cells,
            denominator_definition="65 candidate country cells in the specified window",
            weighting_rule="P QN weights; hierarchical HLO", sample_id="RQ2_65_temporal",
            source_locator=f"scheme=5_year;window_number={row.window_number}",
            interpretation="Eligible aligned measurements in the specified period.",
            limitation="2015-2017 is a partial terminal window." if row.window_end == 2017 else "Calendar alignment does not imply cohort alignment.",
        ))
    block = data["block_all"].query("target_direction == 'HLO_to_P'")
    for row in block.itertuples(index=False):
        for metric in ["all_features_with_any_year", "all_features_at_least_80pct", "all_features_complete"]:
            rows.append(result_row(
                f"CTX_BLOCK_{row.construction.upper()}_{metric.upper()}", "historical_context_coverage", metric, getattr(row, metric), SOURCES["block_all"],
                target="both directions; identical block schedule", period=row.construction, units="cells", n_cells=58,
                denominator_definition="58 eligible aligned five-year cells", weighting_rule="Equal observed annual WDI values",
                sample_id="TEMP_5Y_58", source_locator=f"target_direction=HLO_to_P;construction={row.construction}",
                interpretation="Coverage of all seven candidate WDI features.",
                limitation="Coverage threshold is diagnostic and not an approved feature-retention rule.",
            ))
    for row in data["assessment_all"].itertuples(index=False):
        for metric in ["all_features_with_any_year", "all_features_at_least_80pct", "all_features_complete"]:
            rows.append(result_row(
                f"CTX_ASSESS_{row.target_direction}_{row.construction.upper()}_{metric.upper()}", "historical_context_coverage", metric, getattr(row, metric), SOURCES["assessment_all"],
                target=row.target_direction, period=row.construction, units="cells", n_cells=58,
                denominator_definition="58 eligible aligned five-year cells",
                weighting_rule="Annual means then target-specific assessment-contribution weights", sample_id="TEMP_5Y_58",
                source_locator=f"target_direction={row.target_direction};construction={row.construction}",
                interpretation="Coverage of all seven WDI features under the target-specific schedule.",
                limitation="Depends on target measurement metadata and remains a protocol candidate.",
            ))
    for row in data["date_sensitivity"].itertuples(index=False):
        rows.append(result_row(
            f"TEMP_DATE_CEILING_{row.scheme.upper()}_ALIGNED", "temporal_date_sensitivity",
            "ceiling_assigned aligned eligible cells", row.ceiling_aligned_eligible_cells, SOURCES["date_sensitivity"],
            period=row.scheme, units="cells", n_cells=row.ceiling_aligned_eligible_cells,
            denominator_definition="Frozen 65-economy temporal grid", weighting_rule="Ceiling fractional P dates; HLO unchanged",
            sample_id="RQ2_65_temporal", source_locator=f"scheme={row.scheme}",
            interpretation="Boundary sensitivity to an alternative fractional-date convention.",
            limitation="The floor convention remains primary; fractional dates do not identify exact dates.",
        ))
    for source_name, family in [("fixed_sensitivity", "fixed_sample_sensitivity"), ("coverage_sensitivity", "coverage_changing_sensitivity")]:
        for row in data[source_name].itertuples(index=False):
            rows.append(result_row(
                f"GDP_{family.upper()}_{str(row.scenario).upper().replace(' ', '_').replace('+', 'PLUS')}",
                family, "QNW-GDP r minus HLO-GDP r", row.qnw_minus_hlo_gdp, SOURCES[source_name],
                model_or_comparison=row.scenario, units="correlation difference", n_countries=row.gdp_n,
                denominator_definition="Scenario-specific common GDP sample", weighting_rule="Equal economies",
                sample_id=f"{family}_{row.gdp_n}", source_locator=f"scenario={row.scenario}",
                interpretation="Sensitivity of the observed GDP correlation contrast.",
                limitation="Scenario may change construction, sample coverage, or both; it is not a causal decomposition.",
            ))
    for row in data["weight_sensitivity"].itertuples(index=False):
        rows.append(result_row(
            f"GDP_WEIGHTING_{str(row.estimand).upper().replace(' ', '_').replace('-', '_')}",
            "weighting_sensitivity", "QNW-GDP r minus HLO-GDP r", row.qnw_minus_hlo_gdp, SOURCES["weight_sensitivity"],
            model_or_comparison=row.estimand, units="correlation difference", n_countries=row.n_economies,
            denominator_definition="Fixed 65-economy GDP sample", weighting_rule=row.estimand,
            sample_id="RQ2_65", source_locator=f"estimand={row.estimand}",
            interpretation="Sensitivity of the observed GDP correlation contrast to economy weights.",
            limitation="Population weighting changes the estimand and is not an individual-level correlation.",
        ))
    bootstrap = data["bootstrap"].iloc[0]
    for result_id, estimate, lower, upper, method in [
        ("GDP_DELTA_BOOTSTRAP_PERCENTILE", bootstrap.observed_delta_r, bootstrap.percentile_ci_95_lower, bootstrap.percentile_ci_95_upper, "paired economy percentile bootstrap"),
        ("GDP_DELTA_BOOTSTRAP_BCA", bootstrap.observed_delta_r, bootstrap.bca_ci_95_lower, bootstrap.bca_ci_95_upper, "paired economy BCa bootstrap"),
    ]:
        rows.append(result_row(
            result_id, "frozen_core_inference", "QNW-GDP r minus HLO-GDP r", estimate, SOURCES["bootstrap"],
            lower=lower, upper=upper, interval_method=method, units="correlation difference", n_countries=65,
            denominator_definition="65 complete economy triplets", weighting_rule="Equal economies; paired resampling",
            sample_id="RQ2_65", source_locator="single summary row",
            interpretation="Conditional uncertainty around the observed correlation contrast.",
            limitation="IID economy-triplet working model; interval inclusion of zero is not equivalence.",
        ))
    register = pd.DataFrame(rows)
    assert register["result_id"].is_unique
    write_csv(register, RESULTS / "RESULTS_REGISTER.csv")
    return register


def build_numerical_values(data: dict[str, pd.DataFrame], register: pd.DataFrame) -> None:
    core = data["core_results"].set_index("result_id")["estimate"]
    bootstrap = data["bootstrap"].iloc[0]
    scheme = data["temporal_scheme"].set_index("scheme")
    values = {
        "run_id": RUN_ID,
        "predictive_exhibits_produced": False,
        "core": {key: float(value) for key, value in core.items()},
        "gdp_bootstrap": {
            "percentile_95": [float(bootstrap.percentile_ci_95_lower), float(bootstrap.percentile_ci_95_upper)],
            "bca_95": [float(bootstrap.bca_ci_95_lower), float(bootstrap.bca_ci_95_upper)],
            "replications_valid": int(bootstrap.replications_valid),
        },
        "temporal": {
            scheme_name: {
                "aligned_cells": int(row.aligned_eligible_cells),
                "countries": int(row.countries_with_aligned_cell),
                "countries_2plus": int(row.countries_with_2plus_aligned_windows),
                "one_sample_id_cells": int(row.one_sample_id_eligible_cells),
            }
            for scheme_name, row in scheme.iterrows()
        },
        "historical_context": {
            "block_preceding": data["block_all"].query("target_direction == 'HLO_to_P'").to_dict(orient="records"),
            "assessment_specific": data["assessment_all"].to_dict(orient="records"),
        },
        "gdp_sensitivity_contrasts": {
            "fixed_sample": data["fixed_sensitivity"][["scenario", "gdp_n", "qnw_minus_hlo_gdp"]].to_dict(orient="records"),
            "coverage_changing": data["coverage_sensitivity"][["scenario", "gdp_n", "qnw_minus_hlo_gdp"]].to_dict(orient="records"),
            "weighting": data["weight_sensitivity"][["estimand", "n_economies", "qnw_minus_hlo_gdp"]].to_dict(orient="records"),
        },
        "input_hashes": {name: sha256(path) for name, path in SOURCES.items()},
        "result_register_sha256": sha256(RESULTS / "RESULTS_REGISTER.csv"),
    }
    (RESULTS / "numerical_values.json").write_text(json.dumps(values, indent=2, ensure_ascii=False), encoding="utf-8")


def build_captions(data: dict[str, pd.DataFrame]) -> None:
    core = data["core_results"].set_index("result_id")["estimate"]
    scheme = data["temporal_scheme"].set_index("scheme")
    block = data["block_all"].query("target_direction == 'HLO_to_P'").set_index("construction")
    captions = f"""# Phase A figure captions

## F_CORE_RANKS — Country-rank comparison

Published QNW rank is plotted against the hierarchical HLO rank for the fixed {int(core['CORE_RQ1_N'])}-economy sample. Rank 1 is the highest value on both reversed axes. The dashed line marks identical ranks. Labels identify economies with an absolute displacement of at least 20 places. The median absolute displacement is {core['CORE_RANK_MEDIAN_ABS']:.0f} places and the maximum is {core['CORE_RANK_MAX_ABS']:.0f}. These are ranks of published point estimates, not uncertainty intervals for true ranks.

Plain-language interpretation: the indicators broadly order countries similarly, but several countries move substantially.

## F_GROUP_CUTOFFS — Membership agreement across cutoffs

The figure reports the number of shared members divided by k when each indicator selects its top or bottom k economies, for k from {int(data['membership_curve']['cutoff_k'].min())} to {int(data['membership_curve']['cutoff_k'].max())}. The plotted measure is not Jaccard similarity. At the manuscript's seven-economy cutoff, upper overlap is {int(core['CORE_TOP_OVERLAP'])}/7 and lower overlap is {int(core['CORE_BOTTOM_OVERLAP'])}/7.

Plain-language interpretation: agreement about which countries enter an extreme group depends on both the tail and the chosen group size.

## F_TEMPORAL_COVERAGE — Temporal measurements and historical WDI coverage

Panel A compares aligned eligible country-window counts across the annual, four-year and five-year schemes ({int(scheme.loc['annual', 'aligned_eligible_cells'])}, {int(scheme.loc['4_year', 'aligned_eligible_cells'])}, and {int(scheme.loc['5_year', 'aligned_eligible_cells'])}). Panel B separates P availability, eligible HLO and their overlap within the four accepted five-year periods; 2015-2017 is a partial terminal window. Panel C shows how many of the 58 aligned five-year cells have all seven WDI features represented by at least one year, at least 80% of requested years, or every requested year under block timing. For a previous five-year history these counts are {int(block.loc['block_preceding_5y', 'all_features_with_any_year'])}, {int(block.loc['block_preceding_5y', 'all_features_at_least_80pct'])}, and {int(block.loc['block_preceding_5y', 'all_features_complete'])}; for twelve years they are {int(block.loc['block_preceding_12y', 'all_features_with_any_year'])}, {int(block.loc['block_preceding_12y', 'all_features_at_least_80pct'])}, and {int(block.loc['block_preceding_12y', 'all_features_complete'])}. Panel D shows the target-direction-specific assessment schedules at the 80% threshold.

Plain-language interpretation: pooling years yields more aligned measurements, while longer historical windows often have thinner annual coverage. Coverage thresholds are diagnostics, not approved model rules.
"""
    (FIGURES / "PHASE_A_FIGURE_CAPTIONS.md").write_text(captions, encoding="utf-8")


def build_exhibit_register() -> pd.DataFrame:
    entries = [
        ["T_INDICATORS", "Table 1", "produced_phase_a", "tables/T_INDICATORS_source.csv", "tables/T_INDICATORS_display.md", "Indicator definitions", "Indicators and prior comparisons", ["core_results", "temporal_scheme"]],
        ["T_CORE_SUBSTITUTION", "Table 2", "produced_phase_a", "tables/T_CORE_SUBSTITUTION_source.csv", "tables/T_CORE_SUBSTITUTION_display.md", "Core score/rank/group audit", "Results: core substitution", ["core_results"]],
        ["T_GDP_CONTRAST", "Table 3", "produced_phase_a", "tables/T_GDP_CONTRAST_source.csv", "tables/T_GDP_CONTRAST_display.md", "GDP contrast and sensitivities", "Results: GDP association", ["core_results", "bootstrap", "fixed_sensitivity", "coverage_sensitivity", "weight_sensitivity"]],
        ["T_TEMPORAL_COVERAGE", "Table 4", "produced_phase_a", "tables/T_TEMPORAL_COVERAGE_source.csv", "tables/T_TEMPORAL_COVERAGE_display.md", "Temporal and WDI coverage", "Methods/results: temporal extension", ["temporal_scheme", "temporal_window", "block_all", "assessment_all"]],
        ["T_PREDICTION", "Table 5", "not_produced_protocol_approval_required", "", "", "Empirical model comparison", "Not available in Phase A", []],
        ["F_CORE_RANKS", "Figure 1", "produced_phase_a", "figures/F_CORE_RANKS_plot_data.csv", "figures/F_CORE_RANKS.svg | figures/F_CORE_RANKS.png", "Core rank comparison", "Results: core substitution", ["core_ranks"]],
        ["F_GROUP_CUTOFFS", "Figure 2", "produced_phase_a", "figures/F_GROUP_CUTOFFS_plot_data.csv", "figures/F_GROUP_CUTOFFS.svg | figures/F_GROUP_CUTOFFS.png", "Tail membership across cutoffs", "Results: core substitution", ["membership_curve"]],
        ["F_TEMPORAL_COVERAGE", "Figure 3", "produced_phase_a", "figures/F_TEMPORAL_COVERAGE_plot_data.csv", "figures/F_TEMPORAL_COVERAGE.svg | figures/F_TEMPORAL_COVERAGE.png", "Temporal and historical coverage", "Methods/results: temporal extension", ["temporal_scheme", "temporal_window", "block_all", "assessment_all"]],
        ["F_PREDICTION_LOSS", "Figure 4", "not_produced_protocol_approval_required", "", "", "Out-of-sample loss comparison", "Not available in Phase A", []],
    ]
    rows = []
    for exhibit_id, display, status, source_file, display_file, title, location, names in entries:
        rows.append({
            "exhibit_id": exhibit_id, "display_number": display, "status": status,
            "title": title, "source_or_plot_data": source_file, "display_or_rendered_files": display_file,
            "manuscript_location": location, "generator": GENERATOR,
            "generating_command": "py rewrite/src/reporting/build_phase_a_reporting.py",
            "input_artifacts_and_sha256": source_hashes(names) if names else "",
            "run_id": RUN_ID,
        })
    register = pd.DataFrame(rows)
    write_csv(register, RESULTS / "EXHIBIT_REGISTER.csv")
    return register


def validate_rendering(exhibit_register: pd.DataFrame) -> None:
    rows = []
    for exhibit in ["F_CORE_RANKS", "F_GROUP_CUTOFFS", "F_TEMPORAL_COVERAGE"]:
        png = FIGURES / f"{exhibit}.png"
        svg = FIGURES / f"{exhibit}.svg"
        plot = FIGURES / f"{exhibit}_plot_data.csv"
        with Image.open(png) as image:
            dpi = image.info.get("dpi", (0, 0))
            width, height = image.size
        svg_text = svg.read_text(encoding="utf-8")
        rows.extend([
            [exhibit, "png_dimensions", "PASS" if width >= 1800 and height >= 1200 else "FAIL", f"{width}x{height}"],
            [exhibit, "png_dpi", "PASS" if min(dpi) >= 299 else "FAIL", str(dpi)],
            [exhibit, "svg_nonempty", "PASS" if len(svg_text) > 5000 and "<svg" in svg_text else "FAIL", f"bytes={svg.stat().st_size}"],
            [exhibit, "plot_data_nonempty", "PASS" if len(pd.read_csv(plot)) > 0 else "FAIL", f"rows={len(pd.read_csv(plot))}"],
        ])
    for exhibit in ["T_INDICATORS", "T_CORE_SUBSTITUTION", "T_GDP_CONTRAST", "T_TEMPORAL_COVERAGE"]:
        source = TABLES / f"{exhibit}_source.csv"
        display = TABLES / f"{exhibit}_display.md"
        rows.extend([
            [exhibit, "source_csv_nonempty", "PASS" if len(pd.read_csv(source)) > 0 else "FAIL", f"rows={len(pd.read_csv(source))}"],
            [exhibit, "editable_markdown", "PASS" if "|" in display.read_text(encoding="utf-8") else "FAIL", f"bytes={display.stat().st_size}"],
        ])
    predictive = exhibit_register[exhibit_register["exhibit_id"].isin(["T_PREDICTION", "F_PREDICTION_LOSS"])]
    rows.append(["PREDICTIVE_EXHIBITS", "approval_gate", "PASS" if predictive["status"].eq("not_produced_protocol_approval_required").all() else "FAIL", "No empirical predictive exhibit rendered"])
    checks = pd.DataFrame(rows, columns=["exhibit_id", "check", "status", "details"])
    write_csv(checks, VALIDATION / "reporting_render_checks.csv")
    assert checks["status"].eq("PASS").all()

    artifacts = sorted([*TABLES.glob("*"), *FIGURES.glob("*"), *RESULTS.glob("*")])
    manifest = pd.DataFrame([
        {
            "artifact": path.relative_to(REWRITE).as_posix(), "bytes": path.stat().st_size,
            "sha256": sha256(path), "generator": GENERATOR, "run_id": RUN_ID,
        }
        for path in artifacts if path.is_file()
    ])
    write_csv(manifest, VALIDATION / "reporting_output_manifest.csv")


def write_reporting_note(register: pd.DataFrame) -> None:
    produced = register[register["status"].eq("produced_phase_a")]
    note = f"""# Agent 4 Phase A reporting handoff

Status: Phase A core and coverage reporting complete; empirical prediction exhibits remain blocked by protocol approval.

## Produced

- {len(produced)} registered Phase A exhibits: four editable tables and three figures.
- Every table has a full-precision CSV and editable Markdown display.
- Every figure has exact plot data, SVG, 300-dpi PNG, and a manuscript-ready caption.
- `rewrite/results/RESULTS_REGISTER.csv` links numerical values to their source artifacts and hashes.
- `rewrite/results/numerical_values.json` supplies one machine-readable source for manuscript numbers and captions.
- `rewrite/results/EXHIBIT_REGISTER.csv` supplies stable IDs and display numbering.

## Not produced

`T_PREDICTION` and `F_PREDICTION_LOSS` are explicitly registered as `not_produced_protocol_approval_required`. No placeholder performance, empirical prediction, or invented interval appears in the reporting layer.

## Rebuild

```powershell
py rewrite/src/reporting/build_phase_a_reporting.py
py rewrite/validation/reporting/validate_phase_a_reporting.py
```

The reporting build reads validated calculation files only. It does not fit models, mutate analytical inputs, access the network, or write outside its assigned reporting directories.
"""
    (DOCS / "AGENT4_REPORTING_HANDOFF.md").write_text(note, encoding="utf-8")


def main() -> None:
    for path in (TABLES, FIGURES, RESULTS, DOCS, VALIDATION):
        path.mkdir(parents=True, exist_ok=True)
    verify_inputs()
    data = {name: pd.read_csv(path) for name, path in SOURCES.items() if path.suffix == ".csv"}
    build_tables(data)
    build_figures(data)
    results_register = build_results_register(data)
    build_numerical_values(data, results_register)
    build_captions(data)
    exhibit_register = build_exhibit_register()
    validate_rendering(exhibit_register)
    write_reporting_note(exhibit_register)
    print(exhibit_register[["exhibit_id", "display_number", "status"]].to_string(index=False))


if __name__ == "__main__":
    main()
