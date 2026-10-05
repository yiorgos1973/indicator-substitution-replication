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
    """# 04d — Reproduction of the published NIQ–HLO comparison

This notebook reproduces the worldwide HLO correlations in Table 1 of Warne (2023) using the archived article supplement and its original specification:

- HLO IQ values constructed in the supplement from Angrist et al. (2021).
- Lynn and Becker score variants frozen in that supplement, which match NIQ v1.3.3.
- Pairwise-complete Pearson correlations.
- The exact economy sample reported for every coefficient.

This is a reproduction of prior work, not the project’s headline analysis. The 82-economy HLO measure selected in Notebook 04b is checked before and after the reproduction but is not modified or selected again.

Sources: [Warne (2023)](https://doi.org/10.1007/s40806-022-00351-y), [archived article text](https://gwern.net/doc/iq/2022-warne.pdf), and [Angrist et al. (2021)](https://doi.org/10.1038/s41586-021-03323-7)."""
)

code(
    """from pathlib import Path
import time

import pandas as pd

ROOT = Path.cwd().resolve()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent

PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"
DOCS = ROOT / "docs"

WARNE_DOI = "https://doi.org/10.1007/s40806-022-00351-y"
WARNE_ARTICLE = "https://gwern.net/doc/iq/2022-warne.pdf"
WARNE_SUPPLEMENT = "https://static-content.springer.com/esm/art%3A10.1007%2Fs40806-022-00351-y/MediaObjects/40806_2022_351_MOESM1_ESM.xlsx"

started = time.perf_counter()
print("Project root resolved.")"""
)

markdown("## Verify that the classification gate passed and the HLO measure is frozen")

code(
    """classification_validation = pd.read_csv(OUTPUTS / "economy_classification_validation.csv")
assert classification_validation["passed"].all()
assert "| A03 | Completed |" in (DOCS / "Research_Decision_and_Action_Log.md").read_text(encoding="utf-8")

hlo_audit = pd.read_csv(OUTPUTS / "hlo_aggregate_exclusion_audit.csv")
hlo_decision = pd.read_csv(OUTPUTS / "hlo_time_estimand_decision.csv")
frozen_hlo_count_before = int(hlo_audit["eligible_primary_component_rule"].sum())
assert frozen_hlo_count_before == 82
assert hlo_decision.loc[hlo_decision["decision"].eq("PRIMARY"), "candidate"].item() == "published_period_2000_2017"

print("Classification gate: passed")
print(f"Frozen HLO component-eligible measure before reproduction: {frozen_hlo_count_before} economies")"""
)

markdown(
    """## Original published specification

Warne’s Table 1 states that correlations use pairwise deletion and reports the number of countries in the upper triangle. Its HLO row reports correlations of .811 with QNW, .966 with SAS, .871 with QNW+SAS, .829 with QNW+SAS+GEO, and .140 with geographic-only IQ. The prose agrees except that it reports .896 for QNW+SAS and .830 for QNW+SAS+GEO.

The `.896` prose value is the adjacent Table 1 correlation between Patel and Sandefur (2020) and QNW+SAS. Accordingly, the table value `.871` is the auditable target for the HLO–QNW+SAS reproduction; the internal inconsistency is reported, not silently reconciled."""
)

code(
    """warne = pd.read_csv(PROCESSED / "warne_national_scores.csv")
niq_v133 = pd.read_csv(PROCESSED / "niq_country_scores_v133.csv")

iso3_column = "identification__unnamed"
name_column = "country_name__unnamed"
hlo_column = "agrist_et_al_2021_worldwide_data__hlo_iq"

warne[iso3_column] = warne[iso3_column].str.strip()

specifications = {
    "QNW": "niq_values_from_lynn_and_becker__qnw",
    "SAS": "niq_values_from_lynn_and_becker__sas",
    "QNW+SAS": "niq_values_from_lynn_and_becker__qnw_plus_sas",
    "QNW+SAS+GEO": "niq_values_from_lynn_and_becker__qnw_plus_sas_plus_geo",
    "GEO only": "niq_values_from_lynn_and_becker__geo_iq_only",
}

published_table = {
    "QNW": {"r": .811, "n": 112},
    "SAS": {"r": .966, "n": 98},
    "QNW+SAS": {"r": .871, "n": 130},
    "QNW+SAS+GEO": {"r": .829, "n": 158},
    "GEO only": {"r": .140, "n": 28},
}
published_prose = {
    "QNW": .811,
    "SAS": .966,
    "QNW+SAS": .896,
    "QNW+SAS+GEO": .830,
    "GEO only": .140,
}

results = []
sample_rows = []
for measure, niq_column in specifications.items():
    common = warne[[iso3_column, name_column, hlo_column, niq_column]].dropna().copy()
    common = common.sort_values(iso3_column)
    coefficient = common[hlo_column].corr(common[niq_column])
    for _, row in common.iterrows():
        sample_rows.append({
            "comparison": measure,
            "iso3": row[iso3_column],
            "economy_name": row[name_column],
            "warne_hlo_iq": row[hlo_column],
            "warne_niq_value": row[niq_column],
        })
    results.append({
        "comparison": measure,
        "published_table_r": published_table[measure]["r"],
        "published_prose_r": published_prose[measure],
        "reproduced_r": coefficient,
        "coefficient_difference_from_table": coefficient - published_table[measure]["r"],
        "matches_table_to_three_decimals": round(coefficient, 3) == published_table[measure]["r"],
        "published_table_n": published_table[measure]["n"],
        "reproduced_n": len(common),
        "sample_count_difference": len(common) - published_table[measure]["n"],
        "common_sample_iso3": " ".join(common[iso3_column]),
    })

reproduction = pd.DataFrame(results)
reproduction_samples = pd.DataFrame(sample_rows)
reproduction.to_csv(OUTPUTS / "prior_niq_hlo_reproduction_results.csv", index=False)
reproduction_samples.to_csv(OUTPUTS / "prior_niq_hlo_reproduction_exact_samples.csv", index=False)
reproduction"""
)

markdown("## Test the HKG/MAC/TWN sample-discrepancy hypothesis")

code(
    """excluded_reporting_codes = {"HKG", "MAC", "TWN"}
restriction_rows = []
restriction_sample_rows = []

for measure, niq_column in specifications.items():
    unrestricted = warne[[iso3_column, name_column, hlo_column, niq_column]].dropna().copy()
    restricted = unrestricted.loc[~unrestricted[iso3_column].isin(excluded_reporting_codes)].copy()
    restricted = restricted.sort_values(iso3_column)
    removed_codes = sorted(set(unrestricted[iso3_column]) - set(restricted[iso3_column]))
    coefficient = restricted[hlo_column].corr(restricted[niq_column])

    for _, row in restricted.iterrows():
        restriction_sample_rows.append({
            "comparison": measure,
            "iso3": row[iso3_column],
            "economy_name": row[name_column],
            "warne_hlo_iq": row[hlo_column],
            "warne_niq_value": row[niq_column],
        })

    restriction_rows.append({
        "comparison": measure,
        "published_table_r": published_table[measure]["r"],
        "restricted_r": coefficient,
        "matches_table_to_three_decimals": round(coefficient, 3) == published_table[measure]["r"],
        "published_table_n": published_table[measure]["n"],
        "restricted_n": len(restricted),
        "sample_size_recovered": len(restricted) == published_table[measure]["n"],
        "sample_count_discrepancy_eliminated": (
            len(unrestricted) != published_table[measure]["n"]
            and len(restricted) == published_table[measure]["n"]
        ),
        "removed_iso3": " ".join(removed_codes),
        "restricted_sample_iso3": " ".join(restricted[iso3_column]),
    })

restriction_test = pd.DataFrame(restriction_rows)
restriction_samples = pd.DataFrame(restriction_sample_rows)
restriction_test.to_csv(OUTPUTS / "prior_reproduction_hkg_mac_twn_test.csv", index=False)
restriction_samples.to_csv(OUTPUTS / "prior_reproduction_hkg_mac_twn_exact_samples.csv", index=False)

assert restriction_test["sample_size_recovered"].sum() == 3
assert restriction_test["sample_count_discrepancy_eliminated"].sum() == 2
assert restriction_test["matches_table_to_three_decimals"].sum() == 1
restriction_test"""
)

markdown("## Verify the NIQ edition represented in the supplement")

code(
    """edition_map = {
    "QNW": ("niq_values_from_lynn_and_becker__qnw", "niq_qnw"),
    "SAS": ("niq_values_from_lynn_and_becker__sas", "niq_sas"),
    "QNW+SAS": ("niq_values_from_lynn_and_becker__qnw_plus_sas", "niq_qnw_sas"),
    "QNW+SAS+GEO": ("niq_values_from_lynn_and_becker__qnw_plus_sas_plus_geo", "niq_qnw_sas_geo"),
}

edition_rows = []
for measure, (warne_column, v133_column) in edition_map.items():
    common = warne[[iso3_column, warne_column]].merge(
        niq_v133[["iso3", v133_column]], left_on=iso3_column, right_on="iso3", validate="one_to_one"
    ).dropna(subset=[warne_column, v133_column])
    edition_rows.append({
        "measure": measure,
        "common_economies": len(common),
        "maximum_absolute_difference": (common[warne_column] - common[v133_column]).abs().max(),
    })

edition_check = pd.DataFrame(edition_rows)
assert edition_check["maximum_absolute_difference"].eq(0).all()
edition_check.to_csv(OUTPUTS / "prior_reproduction_niq_edition_check.csv", index=False)
edition_check"""
)

markdown("## Reproduction assessment")

code(
    """def markdown_table(frame):
    values = frame.fillna("").astype(str)
    text = "| " + " | ".join(values.columns) + " |\\n"
    text += "| " + " | ".join(["---"] * len(values.columns)) + " |\\n"
    text += "\\n".join("| " + " | ".join(row) + " |" for row in values.to_numpy())
    return text

display_results = reproduction[[
    "comparison", "published_table_r", "published_prose_r", "reproduced_r",
    "published_table_n", "reproduced_n", "sample_count_difference",
    "matches_table_to_three_decimals",
]].copy()
display_results["reproduced_r"] = display_results["reproduced_r"].round(3)

restriction_display = restriction_test[[
    "comparison", "published_table_r", "restricted_r", "published_table_n",
    "restricted_n", "sample_size_recovered", "removed_iso3",
]].copy()
restriction_display["restricted_r"] = restriction_display["restricted_r"].round(3)

report = f'''# Prior NIQ–HLO reproduction report

## Specification

- Target: Warne (2023), Table 1 worldwide HLO comparisons.
- HLO: the article supplement's HLO-derived IQ variable.
- NIQ: Lynn and Becker score variables in the supplement; values match NIQ v1.3.3 exactly on every shared nonmissing record.
- Statistic: Pearson correlation with pairwise deletion.
- Source workbook: `{WARNE_SUPPLEMENT}`

## Results

{markdown_table(display_results)}

The archived supplement reproduces three Table 1 coefficients to the published three decimals: SAS (.966), QNW+SAS (.871), and geographic-only (.140). QNW reproduces as .809 rather than .811, and QNW+SAS+GEO as .830 rather than .829; both discrepancies are small.

The pairwise sample sizes in the current archived supplement are 2–3 economies larger than Table 1 for all comparisons except geographic-only, which reproduces N = 28 exactly. Because Table 1 does not identify the excluded economy codes, the historical 112/98/130/158 samples cannot be reconstructed uniquely from the current supplement. The exact current-supplement sample behind every reproduced coefficient is stored in `outputs/prior_niq_hlo_reproduction_exact_samples.csv` and as an ISO3 string in the results table.

## HKG/MAC/TWN exclusion test

{markdown_table(restriction_display)}

Removing Hong Kong, Macao, and Taiwan does **not** recover the published HLO row. It eliminates two sample-count discrepancies: QNW+SAS becomes 130 and QNW+SAS+GEO becomes 158. Geographic-only remains at its already matching N = 28 because none of the three economies occurs in that pairwise sample. QNW remains one economy too large (113 versus 112), whereas SAS becomes one economy too small (97 versus 98). Only the unchanged geographic-only coefficient reproduces to three decimals; the other restricted correlations move away from the tabulated values. This empirical result rejects HKG/MAC/TWN exclusion as a complete explanation of the publication discrepancy.

Exact restricted samples are stored in `outputs/prior_reproduction_hkg_mac_twn_exact_samples.csv`.

## Internal publication discrepancy

Table 1 reports HLO–QNW+SAS as .871, which the supplement reproduces. The article prose reports .896. In Table 1, .896 belongs to the adjacent Patel and Sandefur (2020)–QNW+SAS comparison. The reproduction therefore treats .871 as the tabulated target and records .896 as a prose inconsistency.

## Interpretation

This is an approximate, not exact, numerical reproduction of the published HLO row. The high correlations do not validate country-level interchangeability, do not resolve shared-source overlap for SAS-containing NIQ variants, and do not alter the frozen HLO construction.

Sources: [Warne (2023)]({WARNE_DOI}); [archived article]({WARNE_ARTICLE}); [Angrist et al. (2021)](https://doi.org/10.1038/s41586-021-03323-7).
'''

(OUTPUTS / "prior_niq_hlo_reproduction_report.md").write_text(report, encoding="utf-8")
print(report)"""
)

markdown("## Confirm that reproduction did not alter the frozen HLO measure")

code(
    """hlo_audit_after = pd.read_csv(OUTPUTS / "hlo_aggregate_exclusion_audit.csv")
frozen_hlo_count_after = int(hlo_audit_after["eligible_primary_component_rule"].sum())
assert frozen_hlo_count_after == frozen_hlo_count_before == 82

immutability = pd.DataFrame([{
    "check": "frozen_hlo_component_eligible_count_unchanged",
    "before_reproduction": frozen_hlo_count_before,
    "after_reproduction": frozen_hlo_count_after,
    "passed": frozen_hlo_count_before == frozen_hlo_count_after == 82,
}])
immutability.to_csv(OUTPUTS / "prior_reproduction_hlo_immutability_check.csv", index=False)

elapsed = time.perf_counter() - started
print(immutability.to_string(index=False))
print(f"Completed in {elapsed:.1f} seconds")"""
)

nb["cells"] = cells
nbf.write(nb, NOTEBOOKS / "04d_prior_niq_hlo_reproduction.ipynb")
print(NOTEBOOKS / "04d_prior_niq_hlo_reproduction.ipynb")
