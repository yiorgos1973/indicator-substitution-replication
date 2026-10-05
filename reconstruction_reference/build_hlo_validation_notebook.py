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

cells = []


def markdown(text):
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text):
    cells.append(nbf.v4.new_code_cell(text))


markdown(
    """# 04b — Transparent HLO aggregation and validation

This HLO-only notebook reconstructs and audits three transparent aggregate candidates before any NIQ or development-outcome analysis:

1. **Published-period:** 2000–2017 component means, equal subjects within level, then equal levels.
2. **Latest-period:** latest observation within each subject × level, then the same equal-weight hierarchy.
3. **Balanced:** the published-period construction restricted to economies observed in all six subject × level components.

It separately reproduces the official released-code aggregate and compares it with the documented equal-weight construction. The official source code uses a simple mean across retained records, so it is a replication benchmark rather than one of the three candidates.

The notebook does not load NIQ, GDP, population, or contextual data; calculate external correlations; fit hierarchical models; or construct probabilistic rank intervals."""
)

code(
    """from pathlib import Path
from datetime import datetime, timezone
import hashlib
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from scipy.stats import kendalltau, spearmanr

ROOT = Path.cwd().resolve()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent

RAW = ROOT / "data" / "raw" / "hlo_official_replication"
PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"
RAW.mkdir(parents=True, exist_ok=True)
PROCESSED.mkdir(parents=True, exist_ok=True)
OUTPUTS.mkdir(exist_ok=True)

COMMIT = "a61b6452555a080f113687ba6427927bba2d18f9"
BASE = f"https://raw.githubusercontent.com/measuringhumancapital/MHC/{COMMIT}/"
SOURCES = {
    "raw_hlo": BASE + "raw%20data/HLO_database.dta",
    "official_filtered_hlo": BASE + "analysis%20data/hlo_disag.dta",
    "official_country_aggregate": BASE + "analysis%20data/hlo_country.dta",
    "official_construction_code": BASE + "do%20files/MHC%20pre-analysis%20data.do",
}
STANDARD_SUBJECTS = ["math", "reading", "science"]
STANDARD_LEVELS = ["pri", "sec"]
COMPONENTS = [f"{subject}_{level}" for level in STANDARD_LEVELS for subject in STANDARD_SUBJECTS]

def markdown_table(frame):
    values = frame.fillna("").astype(str)
    text = "| " + " | ".join(values.columns) + " |\\n"
    text += "| " + " | ".join(["---"] * len(values.columns)) + " |\\n"
    text += "\\n".join("| " + " | ".join(row) + " |" for row in values.to_numpy())
    return text

started = time.perf_counter()
print("Project root resolved.")
print("Allowed analytical inputs: official HLO replication files only")"""
)

markdown("## Freeze official replication inputs")

code(
    """manifest_rows = []
for source_id, url in SOURCES.items():
    suffix = ".do" if source_id == "official_construction_code" else ".dta"
    path = RAW / f"{source_id}{suffix}"
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    path.write_bytes(response.content)
    manifest_rows.append({
        "source_id": source_id,
        "url": url,
        "github_commit": COMMIT,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "local_path": str(path.relative_to(ROOT)),
        "bytes": len(response.content),
        "sha256": hashlib.sha256(response.content).hexdigest(),
    })

source_manifest = pd.DataFrame(manifest_rows)
source_manifest.to_csv(OUTPUTS / "hlo_aggregation_source_manifest.csv", index=False)
source_manifest"""
)

markdown(
    """## Reproduce the official source-selection rule

The released Stata code sorts by economy, subject, level, and year; three times removes a non-nationally representative row when the immediately preceding retained row is from the same economy–subject–level series but a different source test; and then removes non-nationally representative Chinese rows. This sequential rule is reproduced literally and checked against the released `hlo_disag.dta` file.

After this official filter, economy–year–subject–level cells are unique. The candidate-construction rule nevertheless states that any tied retained records would receive a simple arithmetic mean; inverse-variance aggregation is not used because cross-record dependence is unavailable."""
)

code(
    """raw_hlo = pd.read_stata(RAW / "raw_hlo.dta")
official_filtered = pd.read_stata(RAW / "official_filtered_hlo.dta")
official_country = pd.read_stata(RAW / "official_country_aggregate.dta")

reproduced_filtered = raw_hlo.sort_values(
    ["code", "subject", "level", "year"], kind="mergesort"
).reset_index(drop=True)

for _ in range(3):
    previous = reproduced_filtered.shift()
    remove = (
        reproduced_filtered["n_res"].eq(0)
        & reproduced_filtered["code"].eq(previous["code"])
        & reproduced_filtered["subject"].eq(previous["subject"])
        & reproduced_filtered["level"].eq(previous["level"])
        & reproduced_filtered["sourcetest"].ne(previous["sourcetest"])
    )
    reproduced_filtered = reproduced_filtered.loc[~remove].reset_index(drop=True)

reproduced_filtered = reproduced_filtered.loc[
    ~(reproduced_filtered["code"].eq("CHN") & reproduced_filtered["n_res"].ne(1))
].reset_index(drop=True)

comparison_columns = ["code", "year", "subject", "level", "sourcetest", "n_res", "hlo"]
left = reproduced_filtered[comparison_columns].sort_values(comparison_columns).reset_index(drop=True)
right = official_filtered[comparison_columns].sort_values(comparison_columns).reset_index(drop=True)
pd.testing.assert_frame_equal(left, right, check_dtype=False)

cell_keys = ["code", "year", "subject", "level"]
tied_before = raw_hlo.duplicated(cell_keys, keep=False).sum()
tied_after = official_filtered.duplicated(cell_keys, keep=False).sum()

source_rule_audit = pd.DataFrame([
    {"stage": "raw_official_hlo", "records": len(raw_hlo), "economies": raw_hlo["code"].nunique(), "tied_cell_rows": tied_before, "nonrepresentative_records": raw_hlo["n_res"].eq(0).sum()},
    {"stage": "reproduced_official_filter", "records": len(reproduced_filtered), "economies": reproduced_filtered["code"].nunique(), "tied_cell_rows": tied_after, "nonrepresentative_records": reproduced_filtered["n_res"].eq(0).sum()},
    {"stage": "released_hlo_disag", "records": len(official_filtered), "economies": official_filtered["code"].nunique(), "tied_cell_rows": tied_after, "nonrepresentative_records": official_filtered["n_res"].eq(0).sum()},
])
source_rule_audit.to_csv(OUTPUTS / "hlo_source_test_audit.csv", index=False)
official_filtered.to_csv(PROCESSED / "hlo_official_filtered_records.csv", index=False)
source_rule_audit"""
)

markdown("## Reproduce the official released-code country aggregate")

code(
    """official_replication = (
    official_filtered.groupby(["code", "incomegroup"], as_index=False)
    .agg(
        official_raw_mean=("hlo", "mean"),
        official_raw_mean_male=("hlo_m", "mean"),
        official_raw_mean_female=("hlo_f", "mean"),
        official_record_count=("hlo", "size"),
    )
)
official_replication["official_reproduced_hlo"] = official_replication["official_raw_mean"].round(0)

official_check = official_country[["code", "hlo"]].merge(
    official_replication[["code", "official_reproduced_hlo"]],
    on="code",
    validate="one_to_one",
)
official_check["difference"] = official_check["official_reproduced_hlo"] - official_check["hlo"]
assert len(official_check) == 164
assert official_check["difference"].eq(0).all()
official_check.describe()"""
)

markdown(
    """## Construct the three candidates

All candidates exclude the isolated Chinese `subject = average` record because it is not one of the six standard subject × level components.

### Averaging order

- **Published-period:** simple mean within any tied economy–year–subject–level cell; mean across 2000–2017 years within each subject × level; equal mean across subjects within level; equal mean across available levels.
- **Latest-period:** select the latest year within each economy–subject–level; simple mean for any ties in that year; equal mean across subjects within level; equal mean across available levels.
- **Balanced:** published-period score, restricted to economies observed in all six standard components.

Effective measurement year follows the same weights as the score at every aggregation step."""
)

code(
    """standard = official_filtered.loc[
    official_filtered["subject"].isin(STANDARD_SUBJECTS)
    & official_filtered["level"].isin(STANDARD_LEVELS)
].copy()

year_cells = (
    standard.groupby(["code", "country", "region", "incomegroup", "year", "subject", "level"], as_index=False)
    .agg(hlo=("hlo", "mean"), tied_record_count=("hlo", "size"), source_tests=("sourcetest", lambda x: " | ".join(sorted(x.astype(str).unique()))))
)

period_components = (
    year_cells.groupby(["code", "country", "region", "incomegroup", "subject", "level"], as_index=False)
    .agg(
        component_hlo=("hlo", "mean"),
        component_effective_year=("year", "mean"),
        component_year_count=("year", "nunique"),
        component_first_year=("year", "min"),
        component_last_year=("year", "max"),
    )
)

period_levels = (
    period_components.groupby(["code", "country", "region", "incomegroup", "level"], as_index=False)
    .agg(
        level_hlo=("component_hlo", "mean"),
        level_effective_year=("component_effective_year", "mean"),
        level_subject_count=("subject", "nunique"),
    )
)

published = (
    period_levels.groupby(["code", "country", "region", "incomegroup"], as_index=False)
    .agg(
        published_period_hlo=("level_hlo", "mean"),
        published_period_effective_year=("level_effective_year", "mean"),
        published_level_count=("level", "nunique"),
    )
)

latest_year = year_cells.groupby(["code", "subject", "level"])["year"].transform("max")
latest_cells = year_cells.loc[year_cells["year"].eq(latest_year)].copy()
latest_levels = (
    latest_cells.groupby(["code", "country", "region", "incomegroup", "level"], as_index=False)
    .agg(
        level_hlo=("hlo", "mean"),
        level_effective_year=("year", "mean"),
        level_subject_count=("subject", "nunique"),
    )
)
latest = (
    latest_levels.groupby(["code", "country", "region", "incomegroup"], as_index=False)
    .agg(
        latest_period_hlo=("level_hlo", "mean"),
        latest_period_effective_year=("level_effective_year", "mean"),
        latest_level_count=("level", "nunique"),
    )
)

component_wide = period_components.pivot(
    index=["code", "country"], columns=["subject", "level"], values="component_hlo"
)
component_wide.columns = [f"{subject}_{level}" for subject, level in component_wide.columns]
component_wide = component_wide.reset_index()
for component in COMPONENTS:
    if component not in component_wide:
        component_wide[component] = np.nan
component_wide["balanced_all_six"] = component_wide[COMPONENTS].notna().all(axis=1)
component_wide["balanced_period_hlo"] = component_wide[COMPONENTS].mean(axis=1).where(component_wide["balanced_all_six"])

candidate_table = (
    published.merge(latest, on=["code", "country", "region", "incomegroup"], validate="one_to_one")
    .merge(component_wide[["code", "balanced_all_six", "balanced_period_hlo"] + COMPONENTS], on="code", validate="one_to_one")
    .merge(official_replication[["code", "official_raw_mean", "official_reproduced_hlo", "official_record_count"]], on="code", how="outer", validate="one_to_one")
)
candidate_table.to_csv(PROCESSED / "hlo_aggregate_candidates.csv", index=False)

assert candidate_table["published_period_hlo"].notna().sum() == 163
assert candidate_table["latest_period_hlo"].notna().sum() == 163
assert candidate_table["balanced_period_hlo"].notna().sum() == 49
candidate_table[["published_period_hlo", "latest_period_hlo", "balanced_period_hlo", "official_reproduced_hlo"]].describe()"""
)

markdown("## Audit component balance, missing years, sources, and temporal composition")

code(
    """economy_meta = standard.groupby("code", as_index=False).agg(
    country=("country", "first"),
    region=("region", "first"),
    income_group=("incomegroup", "first"),
    record_count=("hlo", "size"),
    observed_year_count=("year", "nunique"),
    first_year=("year", "min"),
    last_year=("year", "max"),
    source_test_count=("sourcetest", "nunique"),
    source_tests=("sourcetest", lambda x: " | ".join(sorted(x.astype(str).unique()))),
)

presence = period_components.assign(present=True).pivot(
    index="code", columns=["subject", "level"], values="present"
)
presence.columns = [f"has_{subject}_{level}" for subject, level in presence.columns]
presence = presence.reset_index()
for component in COMPONENTS:
    column = f"has_{component}"
    if column not in presence:
        presence[column] = False
    presence[column] = presence[column].fillna(False).astype(bool)

balance = economy_meta.merge(presence, on="code", validate="one_to_one")
has_columns = [f"has_{component}" for component in COMPONENTS]
balance["component_count"] = balance[has_columns].sum(axis=1)
balance["subject_count"] = standard.groupby("code")["subject"].nunique().reindex(balance["code"]).to_numpy()
balance["level_count"] = standard.groupby("code")["level"].nunique().reindex(balance["code"]).to_numpy()
balance["missing_components"] = balance.apply(
    lambda row: " | ".join(component for component in COMPONENTS if not row[f"has_{component}"]), axis=1
)
balance["missing_years_2000_2017"] = balance["code"].map(
    standard.groupby("code")["year"].apply(lambda x: " | ".join(map(str, sorted(set(range(2000, 2018)) - set(x.astype(int))))))
)
balance["eligible_primary_component_rule"] = balance["level_count"].eq(2) & balance["subject_count"].ge(2)
balance["eligible_balanced_all_six"] = balance["component_count"].eq(6)
balance.to_csv(PROCESSED / "hlo_component_balance_audit.csv", index=False)

temporal = candidate_table[[
    "code", "country", "published_period_effective_year", "latest_period_effective_year"
]].merge(
    balance[["code", "record_count", "observed_year_count", "first_year", "last_year", "component_count"]],
    on="code",
    how="outer",
    validate="one_to_one",
)
temporal["observed_year_span"] = temporal["last_year"] - temporal["first_year"]
temporal.to_csv(PROCESSED / "hlo_temporal_composition.csv", index=False)

balance["component_count"].value_counts().sort_index().rename_axis("component_count").to_frame("economies")"""
)

markdown("## Compare HLO-only coverage, distributions, and rank agreement")

code(
    """candidate_definitions = {
    "published_period": "published_period_hlo",
    "latest_period": "latest_period_hlo",
    "balanced_period": "balanced_period_hlo",
    "official_released_code": "official_reproduced_hlo",
}

summary_rows = []
for name, column in candidate_definitions.items():
    values = candidate_table[column].dropna()
    year_column = {
        "published_period": "published_period_effective_year",
        "latest_period": "latest_period_effective_year",
        "balanced_period": "published_period_effective_year",
    }.get(name)
    years = candidate_table.loc[candidate_table[column].notna(), year_column].dropna() if year_column else pd.Series(dtype=float)
    summary_rows.append({
        "construction": name,
        "economies": len(values),
        "mean_hlo": values.mean(),
        "sd_hlo": values.std(),
        "min_hlo": values.min(),
        "median_hlo": values.median(),
        "max_hlo": values.max(),
        "median_effective_year": years.median() if len(years) else np.nan,
        "effective_year_iqr": years.quantile(.75) - years.quantile(.25) if len(years) else np.nan,
    })
construction_summary = pd.DataFrame(summary_rows)
construction_summary.to_csv(OUTPUTS / "hlo_aggregation_validation_summary.csv", index=False)

agreement_rows = []
names = list(candidate_definitions)
for index, left_name in enumerate(names):
    for right_name in names[index + 1:]:
        left_column = candidate_definitions[left_name]
        right_column = candidate_definitions[right_name]
        common = candidate_table[["code", left_column, right_column]].dropna().copy()
        common["left_rank"] = common[left_column].rank(ascending=False, method="average")
        common["right_rank"] = common[right_column].rank(ascending=False, method="average")
        agreement_rows.append({
            "construction_1": left_name,
            "construction_2": right_name,
            "common_economies": len(common),
            "common_sample_iso3": " ".join(sorted(common["code"])),
            "pearson": common[left_column].corr(common[right_column]),
            "spearman": spearmanr(common[left_column], common[right_column]).statistic,
            "kendall_tau": kendalltau(common[left_column], common[right_column]).statistic,
            "median_absolute_score_difference": (common[left_column] - common[right_column]).abs().median(),
            "maximum_absolute_score_difference": (common[left_column] - common[right_column]).abs().max(),
            "median_absolute_rank_difference": (common["left_rank"] - common["right_rank"]).abs().median(),
            "maximum_absolute_rank_difference": (common["left_rank"] - common["right_rank"]).abs().max(),
        })
rank_agreement = pd.DataFrame(agreement_rows)
rank_agreement.to_csv(OUTPUTS / "hlo_rank_agreement.csv", index=False)

official_comparison = candidate_table[[
    "code", "country", "published_period_hlo", "official_raw_mean",
    "official_reproduced_hlo", "official_record_count",
]].dropna(subset=["published_period_hlo", "official_reproduced_hlo"]).copy()
official_comparison["published_minus_official"] = (
    official_comparison["published_period_hlo"] - official_comparison["official_reproduced_hlo"]
)
official_comparison["published_rank"] = official_comparison["published_period_hlo"].rank(ascending=False, method="average")
official_comparison["official_rank"] = official_comparison["official_reproduced_hlo"].rank(ascending=False, method="average")
official_comparison["rank_difference"] = official_comparison["published_rank"] - official_comparison["official_rank"]
official_comparison.to_csv(OUTPUTS / "hlo_official_aggregate_comparison.csv", index=False)

construction_summary, rank_agreement"""
)

code(
    """fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

for name, column in list(candidate_definitions.items())[:3]:
    axes[0].hist(candidate_table[column].dropna(), bins=18, alpha=.45, label=name)
axes[0].set(title="HLO candidate distributions", xlabel="HLO score", ylabel="Economies")
axes[0].legend(fontsize=8)

axes[1].hist(candidate_table["published_period_effective_year"].dropna(), bins=np.arange(2000, 2019) - .5, alpha=.6, label="published period")
axes[1].hist(candidate_table["latest_period_effective_year"].dropna(), bins=np.arange(2000, 2019) - .5, alpha=.6, label="latest period")
axes[1].set(title="Effective measurement year", xlabel="Weighted effective year", ylabel="Economies")
axes[1].legend(fontsize=8)

plot_data = candidate_table[["published_period_hlo", "latest_period_hlo"]].dropna()
axes[2].scatter(plot_data["published_period_hlo"], plot_data["latest_period_hlo"], alpha=.65, s=22)
limits = [plot_data.min().min(), plot_data.max().max()]
axes[2].plot(limits, limits, linestyle="--", color="black", linewidth=1)
axes[2].set(title="Published-period vs latest-period", xlabel="Published-period HLO", ylabel="Latest-period HLO")

fig.tight_layout()
fig.savefig(OUTPUTS / "hlo_aggregation_validation.png", dpi=180, bbox_inches="tight")
plt.show()"""
)

markdown("## Exclusion audit, data dictionary, and time-estimand decision")

code(
    """exclusions = balance[[
    "code", "country", "region", "income_group", "component_count", "subject_count", "level_count",
    "missing_components", "eligible_primary_component_rule", "eligible_balanced_all_six",
]].merge(
    candidate_table[["code", "published_period_hlo", "latest_period_hlo", "balanced_period_hlo"]],
    on="code",
    how="outer",
    validate="one_to_one",
)
exclusions["primary_exclusion_reason"] = np.select(
    [
        exclusions["published_period_hlo"].isna(),
        exclusions["level_count"].lt(2),
        exclusions["subject_count"].lt(2),
    ],
    ["no_standard_subject_level_component", "only_one_schooling_level", "only_one_subject"],
    default="eligible_primary_period_component_rule",
)

china = official_country.loc[official_country["code"].eq("CHN"), ["code"]].assign(
    country="China", region="East Asia & Pacific", income_group="Upper middle income",
    component_count=0, subject_count=0, level_count=0, missing_components=" | ".join(COMPONENTS),
    eligible_primary_component_rule=False, eligible_balanced_all_six=False,
    published_period_hlo=np.nan, latest_period_hlo=np.nan, balanced_period_hlo=np.nan,
    primary_exclusion_reason="only_nonstandard_precomputed_average_record",
)
exclusions = pd.concat([exclusions, china], ignore_index=True).drop_duplicates("code", keep="last")
exclusions.to_csv(OUTPUTS / "hlo_aggregate_exclusion_audit.csv", index=False)

dictionary_rows = [
    ("published_period_hlo", "2000–2017 mean within component; equal subjects within level; equal available levels"),
    ("published_period_effective_year", "Effective year using the published-period score's aggregation weights"),
    ("latest_period_hlo", "Latest year within component; equal subjects within level; equal available levels"),
    ("latest_period_effective_year", "Effective year using the latest-period score's aggregation weights"),
    ("balanced_period_hlo", "Published-period score restricted to all six standard components"),
    ("official_raw_mean", "Simple mean across records retained by the official source-selection code"),
    ("official_reproduced_hlo", "Official raw record mean rounded to the nearest whole HLO point"),
    ("component_count", "Number of observed standard subject × level components, from zero to six"),
    ("eligible_primary_component_rule", "Both schooling levels and at least two distinct subjects observed"),
    ("missing_years_2000_2017", "Calendar years without any retained standard HLO record"),
]
data_dictionary = pd.DataFrame(dictionary_rows, columns=["variable", "definition"])
data_dictionary.to_csv(OUTPUTS / "hlo_aggregate_data_dictionary.csv", index=False)

primary_eligible = int(balance["eligible_primary_component_rule"].sum())
decision = pd.DataFrame([
    {
        "candidate": "published_period_2000_2017",
        "time_estimand": "Mean learning performance over 2000–2017",
        "component_rule": "Equal subjects within level; equal levels",
        "candidate_coverage": int(candidate_table["published_period_hlo"].notna().sum()),
        "primary_rule_coverage": primary_eligible,
        "construct_meaning": "Period-average learning; matches the paper's stated 2000–2017 construct",
        "comparability": "Moderate overall; strengthened by requiring both levels and at least two subjects",
        "temporal_composition": "Uses all retained years; effective year reported for every economy",
        "decision": "PRIMARY",
        "reason": "Best alignment with the published period construct while the component rule limits one-domain or one-level proxies without collapsing coverage to the all-six sample",
    },
    {
        "candidate": "latest_period",
        "time_estimand": "Most recent observed performance within each component",
        "component_rule": "Equal subjects within level; equal levels",
        "candidate_coverage": int(candidate_table["latest_period_hlo"].notna().sum()),
        "primary_rule_coverage": primary_eligible,
        "construct_meaning": "Recent but component-asynchronous learning",
        "comparability": "Weaker temporal comparability because effective years differ across economies and components",
        "temporal_composition": "Latest component years; effective year reported for every economy",
        "decision": "SENSITIVITY",
        "reason": "Recency is useful, but unequal observation years and reliance on single latest waves make it less comparable as the headline measure",
    },
    {
        "candidate": "balanced_all_six_2000_2017",
        "time_estimand": "Mean learning performance over 2000–2017",
        "component_rule": "All six components required and equally weighted",
        "candidate_coverage": int(candidate_table["balanced_period_hlo"].notna().sum()),
        "primary_rule_coverage": int(candidate_table["balanced_period_hlo"].notna().sum()),
        "construct_meaning": "Fully balanced general-learning composite",
        "comparability": "Highest component comparability",
        "temporal_composition": "Uses all retained years within every component",
        "decision": "SENSITIVITY",
        "reason": "Maximum balance is scientifically valuable but the severe coverage reduction changes the target population",
    },
])
decision.to_csv(OUTPUTS / "hlo_time_estimand_decision.csv", index=False)
decision"""
)

markdown("## Construction report and completion gate")

code(
    """official_vs_documented = rank_agreement.loc[
    (rank_agreement["construction_1"] == "published_period")
    & (rank_agreement["construction_2"] == "official_released_code")
].iloc[0]
latest_comparison = rank_agreement.loc[
    (rank_agreement["construction_1"] == "published_period")
    & (rank_agreement["construction_2"] == "latest_period")
].iloc[0]
largest_official_difference = official_comparison.loc[
    official_comparison["published_minus_official"].abs().idxmax()
]

report = f'''# Transparent HLO aggregation and validation report

Generated by Notebook 04b using only HLO data and the official replication repository pinned at commit `{COMMIT}`.

## Official replication

- Reproduced the released source-selection code exactly: {len(reproduced_filtered):,} retained records and {reproduced_filtered["code"].nunique()} economies.
- The reproduced filtered records match the released `hlo_disag.dta` record-for-record.
- The released-code country aggregate is the simple record mean rounded to a whole HLO point; all {len(official_check)} official country records reproduce exactly.
- After official filtering there are {tied_after} tied economy–year–subject–level rows.
- The official aggregate includes one nonstandard Chinese `average_sec` record; the three candidates use only mathematics, reading, and science at primary and secondary levels.

## Candidate coverage and balance

{markdown_table(construction_summary.round(3))}

- Standard-component economies: {balance["code"].nunique()}.
- Economies meeting the primary component rule (both levels and at least two subjects): {primary_eligible}.
- Economies with all six components: {int(balance["eligible_balanced_all_six"].sum())}.
- Economies with only one component: {int(balance["component_count"].eq(1).sum())}.

## HLO-only agreement

- Published-period versus latest-period: Spearman {latest_comparison["spearman"]:.3f}, Kendall {latest_comparison["kendall_tau"]:.3f}, median absolute rank movement {latest_comparison["median_absolute_rank_difference"]:.1f}, maximum absolute rank movement {latest_comparison["maximum_absolute_rank_difference"]:.0f}, median absolute score difference {latest_comparison["median_absolute_score_difference"]:.2f}, and maximum absolute score difference {latest_comparison["maximum_absolute_score_difference"]:.2f}. Exact common sample: N = {int(latest_comparison["common_economies"])}; ISO3 = {latest_comparison["common_sample_iso3"]}.
- Documented equal-weight published-period construction versus official released-code aggregate: Spearman {official_vs_documented["spearman"]:.3f}, Kendall {official_vs_documented["kendall_tau"]:.3f}, median absolute rank movement {official_vs_documented["median_absolute_rank_difference"]:.1f}, maximum absolute rank movement {official_vs_documented["maximum_absolute_rank_difference"]:.0f}, median absolute score difference {official_vs_documented["median_absolute_score_difference"]:.2f}, and maximum absolute score difference {official_vs_documented["maximum_absolute_score_difference"]:.2f}. Exact common sample: N = {int(official_vs_documented["common_economies"])}; ISO3 = {official_vs_documented["common_sample_iso3"]}.
- The largest score difference is {largest_official_difference["country"]} ({largest_official_difference["code"]}): {largest_official_difference["published_minus_official"]:+.2f} HLO points under documented component weighting relative to the rounded released-code record mean.
- High agreement does not make the constructions identical. The released code differs from the documented aggregation rule: it weights economies according to their number of retained records, whereas the documented construction equalizes subjects and then levels.

## Completion-gate decision

**Selected primary time estimand:** mean HLO performance over 2000–2017 using the documented subject-then-level equal-weight order.

**Primary component eligibility:** both primary and secondary levels plus at least two distinct subjects. The score still uses equal subjects within each level and equal levels. This retains {primary_eligible} economies while preventing a one-subject or one-level observation from standing in for the general-learning composite.

**Sensitivities:** all available standard components for coverage, latest-period equal weighting for recency, and the all-six balanced sample for maximum component comparability. The official released-code aggregate is retained as a replication benchmark.

The choice is based only on construct meaning, component comparability, coverage, source rules, and temporal composition. No NIQ, GDP, population, hierarchical model, contextual covariate, or external association was loaded or examined.
'''
(OUTPUTS / "hlo_aggregation_construction_report.md").write_text(report, encoding="utf-8")
print(report)"""
)

markdown("## Validation")

code(
    """assert len(reproduced_filtered) == len(official_filtered) == 2005
assert official_check["difference"].eq(0).all()
assert tied_after == 0
assert not candidate_table.duplicated("code").any()
assert not balance.duplicated("code").any()
assert not exclusions.duplicated("code").any()
assert decision["decision"].eq("PRIMARY").sum() == 1
assert decision.loc[decision["decision"].eq("PRIMARY"), "candidate"].item() == "published_period_2000_2017"
assert primary_eligible == 82

elapsed = time.perf_counter() - started
print(f"All HLO-only reconstruction, key, coverage, and decision-gate checks passed in {elapsed:.1f} seconds.")
print("NIQ/GDP reproduction remains blocked until the decision log is updated from this completed gate.")"""
)

nb["cells"] = cells
path = NOTEBOOKS / "04b_transparent_HLO_aggregation_and_validation.ipynb"
nbf.write(nb, path)
print(path)
