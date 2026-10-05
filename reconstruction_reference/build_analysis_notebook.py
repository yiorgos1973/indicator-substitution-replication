from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parent
NOTEBOOKS = ROOT / "notebooks"
NOTEBOOKS.mkdir(exist_ok=True)

nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.10"},
}

nb["cells"] = [
    nbf.v4.new_markdown_cell(
        """# 04 — Analysis-dataset construction

Constructs auditable country-level inputs for the measurement audit. It keeps latest-observation HLO subject × schooling-level measures separate, uses psychometric `QNW` as the primary NIQ series under audit, and forms exact common-economy samples before standardization or ranking. These six HLO measures are transparent sensitivities and inputs to Notebook 04b; none is selected here as the primary HLO composite.

This notebook performs **data transformation only**. It does not fit the paper's headline models, test hypotheses, interpret country scores as innate intelligence, or make causal claims.

GDP per capita PPP level in 2017 is retained for the planned confirmatory associational analysis. Subsequent annualized log growth from 2018 through 2024 is retained only as an exploratory, short-window, pandemic-affected outcome."""
    ),
    nbf.v4.new_code_cell(
        """from pathlib import Path
import time

import numpy as np
import pandas as pd

ROOT = Path.cwd().resolve()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent

PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)

GDP_CODE = "NY.GDP.PCAP.PP.KD"
POP_CODE = "SP.POP.TOTL"
GDP_LEVEL_YEAR = 2017
GROWTH_START_YEAR = 2018
GROWTH_END_YEAR = 2024

started = time.perf_counter()
print("Project root resolved.")"""
    ),
    nbf.v4.new_markdown_cell("## Load frozen processed sources"),
    nbf.v4.new_code_cell(
        """hlo = pd.read_csv(PROCESSED / "hlo_database.csv")
wdi = pd.read_csv(PROCESSED / "world_bank_indicators_long.csv")
niq = pd.read_csv(PROCESSED / "niq_country_scores_v135.csv")
niq_old = pd.read_csv(PROCESSED / "niq_country_scores_v133.csv")
provenance = pd.read_csv(PROCESSED / "niq_sample_summary_v135.csv")

print("HLO:", hlo.shape)
print("WDI:", wdi.shape)
print("NIQ v1.3.5:", niq.shape)
print("NIQ provenance summaries:", provenance.shape)"""
    ),
    nbf.v4.new_markdown_cell(
        """## Construct candidate HLO measures

For each economy, subject, and schooling level, select the latest observed year. When several source-test records occur in that same year, combine them by inverse-variance weighting if every record has a valid standard error; otherwise use their simple mean and leave the combined standard error missing. This avoids averaging across years or across substantively different subject/level constructs and avoids fabricating precision."""
    ),
    nbf.v4.new_code_cell(
        """keys = ["code", "subject", "level"]
hlo_subjects = hlo.loc[hlo["subject"].isin(["reading", "math", "science"])].copy()
latest_year = hlo_subjects.groupby(keys)["year"].transform("max")
hlo_latest_records = hlo_subjects.loc[hlo_subjects["year"].eq(latest_year)].copy()
hlo_latest_records["precision"] = 1 / hlo_latest_records["hlo_se"].pow(2)
hlo_latest_records["weighted_hlo"] = hlo_latest_records["hlo"] * hlo_latest_records["precision"]

hlo_latest = (
    hlo_latest_records.groupby(keys, as_index=False)
    .agg(
        country=("country", "first"),
        hlo_year=("year", "first"),
        weighted_hlo_sum=("weighted_hlo", "sum"),
        precision=("precision", "sum"),
        simple_hlo_mean=("hlo", "mean"),
        valid_se_count=("hlo_se", "count"),
        hlo_record_count=("hlo", "size"),
        hlo_total_n_res=("n_res", "sum"),
        hlo_source_tests=("sourcetest", lambda x: " | ".join(sorted(x.dropna().astype(str).unique()))),
        region=("region", "first"),
        income_group=("incomegroup", "first"),
    )
)
complete_precision = hlo_latest["valid_se_count"].eq(hlo_latest["hlo_record_count"])
hlo_latest["hlo"] = np.where(
    complete_precision,
    hlo_latest["weighted_hlo_sum"] / hlo_latest["precision"],
    hlo_latest["simple_hlo_mean"],
)
hlo_latest["hlo_se"] = np.where(complete_precision, np.sqrt(1 / hlo_latest["precision"]), np.nan)
hlo_latest["hlo_aggregation"] = np.where(complete_precision, "inverse_variance", "simple_mean_missing_se")
hlo_latest = hlo_latest.drop(columns=["weighted_hlo_sum", "precision", "simple_hlo_mean", "valid_se_count"]).rename(columns={"code": "iso3"})
hlo_latest["hlo_specification"] = hlo_latest["subject"] + "_" + hlo_latest["level"]

assert not hlo_latest.duplicated(["iso3", "hlo_specification"]).any()
hlo_latest.to_csv(PROCESSED / "hlo_latest_subject_level.csv", index=False)
hlo_latest.groupby(["subject", "level"]).agg(
    economies=("iso3", "nunique"),
    earliest_selected_year=("hlo_year", "min"),
    latest_selected_year=("hlo_year", "max"),
    median_hlo=("hlo", "median"),
).reset_index()"""
    ),
    nbf.v4.new_markdown_cell("## Assemble NIQ variants, version checks, and provenance"),
    nbf.v4.new_code_cell(
        """niq_columns = [
    "iso3", "niq_qnw", "niq_qnw_sas", "niq_qnw_sas_geo",
    "has_psychometric_data", "has_school_assessment_data",
    "geographically_imputed", "source_class",
]
old_columns = ["iso3", "niq_qnw", "niq_qnw_sas", "niq_qnw_sas_geo"]

niq_measurement = (
    niq[niq_columns]
    .merge(
        niq_old[old_columns].rename(columns={c: f"{c}_v133" for c in old_columns if c != "iso3"}),
        on="iso3",
        how="left",
        validate="one_to_one",
    )
    .merge(provenance, on="iso3", how="left", validate="one_to_one")
)

country_spec = hlo_latest.merge(niq_measurement, on="iso3", how="left", validate="many_to_one")
country_spec["eligible_hlo"] = country_spec["hlo"].notna()
country_spec["eligible_niq_qnw"] = country_spec["niq_qnw"].notna()
country_spec["eligible_ranking_primary"] = country_spec["eligible_hlo"] & country_spec["eligible_niq_qnw"]
country_spec.shape"""
    ),
    nbf.v4.new_markdown_cell("## Add fixed development outcomes and explicit eligibility flags"),
    nbf.v4.new_code_cell(
        """def indicator_at(code, year, name):
    return (
        wdi.loc[(wdi["indicator_code"] == code) & (wdi["year"] == year), ["iso3", "value"]]
        .rename(columns={"value": name})
    )

gdp_level = indicator_at(GDP_CODE, GDP_LEVEL_YEAR, f"gdp_pc_ppp_{GDP_LEVEL_YEAR}")
gdp_start = indicator_at(GDP_CODE, GROWTH_START_YEAR, f"gdp_pc_ppp_{GROWTH_START_YEAR}")
gdp_end = indicator_at(GDP_CODE, GROWTH_END_YEAR, f"gdp_pc_ppp_{GROWTH_END_YEAR}")
population = indicator_at(POP_CODE, GDP_LEVEL_YEAR, f"population_{GDP_LEVEL_YEAR}")

analysis_base = country_spec
for table in [gdp_level, gdp_start, gdp_end, population]:
    analysis_base = analysis_base.merge(table, on="iso3", how="left", validate="many_to_one")

analysis_base[f"log_gdp_pc_ppp_{GDP_LEVEL_YEAR}"] = np.log(analysis_base[f"gdp_pc_ppp_{GDP_LEVEL_YEAR}"])
analysis_base[f"annualized_log_gdp_growth_{GROWTH_START_YEAR}_{GROWTH_END_YEAR}"] = (
    np.log(analysis_base[f"gdp_pc_ppp_{GROWTH_END_YEAR}"])
    - np.log(analysis_base[f"gdp_pc_ppp_{GROWTH_START_YEAR}"])
) / (GROWTH_END_YEAR - GROWTH_START_YEAR)

analysis_base["eligible_gdp_level"] = analysis_base[f"gdp_pc_ppp_{GDP_LEVEL_YEAR}"].notna()
analysis_base["eligible_growth"] = analysis_base[[f"gdp_pc_ppp_{GROWTH_START_YEAR}", f"gdp_pc_ppp_{GROWTH_END_YEAR}"]].notna().all(axis=1)
analysis_base["eligible_population_weight"] = analysis_base[f"population_{GDP_LEVEL_YEAR}"].notna()
analysis_base.to_csv(PROCESSED / "analysis_country_specification_base.csv", index=False)
analysis_base.shape"""
    ),
    nbf.v4.new_markdown_cell("## Create exact common-sample ranking and outcome tables"),
    nbf.v4.new_code_cell(
        """ranking = analysis_base.loc[analysis_base["eligible_ranking_primary"]].copy()
ranking["rank_niq_qnw"] = ranking.groupby("hlo_specification")["niq_qnw"].rank(ascending=False, method="average")
ranking["rank_hlo"] = ranking.groupby("hlo_specification")["hlo"].rank(ascending=False, method="average")
ranking["rank_displacement_hlo_minus_niq"] = ranking["rank_hlo"] - ranking["rank_niq_qnw"]
ranking["absolute_rank_displacement"] = ranking["rank_displacement_hlo_minus_niq"].abs()

def outcome_table(eligible_column):
    table = ranking.loc[ranking[eligible_column]].copy()
    table["z_niq_qnw"] = table.groupby("hlo_specification")["niq_qnw"].transform(lambda x: (x - x.mean()) / x.std())
    table["z_hlo"] = table.groupby("hlo_specification")["hlo"].transform(lambda x: (x - x.mean()) / x.std())
    return table

gdp_level_analysis = outcome_table("eligible_gdp_level")
growth_analysis = outcome_table("eligible_growth")

ranking.to_csv(PROCESSED / "analysis_ranking_common_sample.csv", index=False)
gdp_level_analysis.to_csv(PROCESSED / "analysis_gdp_level_common_sample.csv", index=False)
growth_analysis.to_csv(PROCESSED / "analysis_growth_common_sample.csv", index=False)

ranking.groupby("hlo_specification").agg(
    common_countries=("iso3", "nunique"),
    hlo_year_min=("hlo_year", "min"),
    hlo_year_median=("hlo_year", "median"),
    hlo_year_max=("hlo_year", "max"),
).reset_index()"""
    ),
    nbf.v4.new_markdown_cell("## Sample-flow and exclusion audit"),
    nbf.v4.new_code_cell(
        """sample_flow = (
    analysis_base.groupby("hlo_specification", as_index=False)
    .agg(
        hlo_economies=("eligible_hlo", "sum"),
        qnw_hlo_common=("eligible_ranking_primary", "sum"),
        gdp_level_common=("eligible_gdp_level", lambda x: (x & analysis_base.loc[x.index, "eligible_ranking_primary"]).sum()),
        growth_common=("eligible_growth", lambda x: (x & analysis_base.loc[x.index, "eligible_ranking_primary"]).sum()),
        population_weight_common=("eligible_population_weight", lambda x: (x & analysis_base.loc[x.index, "eligible_ranking_primary"]).sum()),
    )
)
sample_flow["excluded_missing_qnw"] = sample_flow["hlo_economies"] - sample_flow["qnw_hlo_common"]
sample_flow["excluded_missing_gdp_level_after_common"] = sample_flow["qnw_hlo_common"] - sample_flow["gdp_level_common"]
sample_flow["excluded_missing_growth_after_common"] = sample_flow["qnw_hlo_common"] - sample_flow["growth_common"]

exclusions = analysis_base[[
    "iso3", "country", "hlo_specification", "hlo_year", "source_class",
    "eligible_hlo", "eligible_niq_qnw", "eligible_ranking_primary",
    "eligible_gdp_level", "eligible_growth", "eligible_population_weight",
]].copy()
exclusions["primary_exclusion_reason"] = np.select(
    [
        ~exclusions["eligible_niq_qnw"],
        ~exclusions["eligible_gdp_level"],
        ~exclusions["eligible_growth"],
        ~exclusions["eligible_population_weight"],
    ],
    ["missing_primary_qnw", "missing_gdp_level", "missing_growth_window", "missing_population_weight"],
    default="eligible_all_primary_tables",
)

sample_flow.to_csv(OUTPUTS / "analysis_sample_flow.csv", index=False)
exclusions.to_csv(OUTPUTS / "analysis_exclusion_audit.csv", index=False)
sample_flow"""
    ),
    nbf.v4.new_markdown_cell("## Data dictionary and construction report"),
    nbf.v4.new_code_cell(
        """dictionary_rows = [
    ("iso3", "ISO3-style economy code used for all joins"),
    ("hlo_specification", "HLO subject and schooling-level combination"),
    ("hlo_year", "Latest HLO year selected within economy and specification"),
    ("hlo", "Latest-year HLO; tied records combined by the documented aggregation rule"),
    ("hlo_se", "Combined standard error when all tied records provide valid uncertainty"),
    ("hlo_aggregation", "Inverse-variance combination, or simple mean when standard errors are incomplete"),
    ("niq_qnw", "Primary documented psychometric NIQ estimate, version 1.3.5"),
    ("niq_qnw_sas", "Sensitivity NIQ combining psychometric and school-assessment information"),
    ("niq_qnw_sas_geo", "Sensitivity NIQ additionally permitting geographic imputation"),
    ("log_gdp_pc_ppp_2017", "Natural log GDP per capita PPP in constant 2021 international dollars, 2017"),
    ("annualized_log_gdp_growth_2018_2024", "Annualized log GDP-per-capita PPP change, 2018–2024"),
    ("z_niq_qnw", "QNW standardized within the exact HLO-specification/outcome sample"),
    ("z_hlo", "HLO standardized within the exact HLO-specification/outcome sample"),
    ("rank_displacement_hlo_minus_niq", "HLO rank minus QNW rank within an exact common sample"),
    ("source_class", "NIQ source class supplied by the documented NIQ dataset"),
    ("mean_full_rating", "Mean author-defined sample quality rating across selected psychometric records"),
]
data_dictionary = pd.DataFrame(dictionary_rows, columns=["variable", "definition"])
data_dictionary.to_csv(OUTPUTS / "analysis_data_dictionary.csv", index=False)

flow_lines = "| " + " | ".join(sample_flow.columns) + " |\\n"
flow_lines += "| " + " | ".join(["---"] * len(sample_flow.columns)) + " |\\n"
flow_lines += "\\n".join(
    "| " + " | ".join(map(str, row)) + " |"
    for row in sample_flow.itertuples(index=False, name=None)
)
report = f'''# Analysis-dataset construction report

Generated by notebook 04. This stage transforms data and audits eligibility; it does not estimate headline associations.

## Construction decisions

- Primary NIQ: documented psychometric QNW, version 1.3.5.
- NIQ sensitivities: QNW+SAS, QNW+SAS+GEO, and version 1.3.3 fields.
- HLO: latest observation retained separately for every subject × schooling-level specification as transparent sensitivities; no primary composite is selected here.
- The isolated precomputed `average` HLO record is excluded; no cross-subject composite is imposed.
- Tied latest-year HLO records: inverse-variance combined when all standard errors are valid; otherwise simple mean with combined SE left missing.
- Confirmatory associational outcome retained for later analysis: GDP level in {GDP_LEVEL_YEAR}.
- Exploratory outcome retained for later analysis: annualized log growth {GROWTH_START_YEAR}–{GROWTH_END_YEAR}, a short pandemic-affected window.
- Rankings and z-scores: calculated only within exact common-country samples.
- Population: retained as an explicit potential weight, not silently applied.

## Sample flow

{flow_lines}

## Remaining design decisions

- Notebook 04b must compare published-style 2000–2017, latest-period, and balanced-component equal-weight HLO aggregates plus the six subject-by-level measures without NIQ or GDP comparisons.
- Freeze the primary HLO time estimand and minimum-component rule after that HLO-only audit.
- Create reporting-economy and UN member/observer sensitivity classifications from authoritative sources.
- Construct point estimates first and postpone probabilistic composite ranks until dependence-aware uncertainty is defensible.
- Freeze region definitions and the prospective analysis plan after data access before Notebook 05.
'''
(OUTPUTS / "analysis_dataset_construction_report.md").write_text(report, encoding="utf-8")
print(report)"""
    ),
    nbf.v4.new_markdown_cell("## Validation"),
    nbf.v4.new_code_cell(
        """assert not ranking.duplicated(["iso3", "hlo_specification"]).any()
assert not gdp_level_analysis.duplicated(["iso3", "hlo_specification"]).any()
assert not growth_analysis.duplicated(["iso3", "hlo_specification"]).any()

for table in [gdp_level_analysis, growth_analysis]:
    check = table.groupby("hlo_specification")[["z_niq_qnw", "z_hlo"]].agg(["mean", "std"])
    assert np.allclose(check.xs("mean", axis=1, level=1), 0, atol=1e-12)
    assert np.allclose(check.xs("std", axis=1, level=1), 1, atol=1e-12)

elapsed = time.perf_counter() - started
print(f"All key and standardization checks passed in {elapsed:.1f} seconds.")
print("Headline modelling remains intentionally deferred to notebook 05.")"""
    ),
]

path = NOTEBOOKS / "04_analysis_dataset_construction.ipynb"
nbf.write(nb, path)
print(path)
