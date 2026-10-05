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
        """# 03 — Documented national-IQ acquisition and provenance audit

Downloads and freezes NIQ Dataset versions 1.3.5 and 1.3.3 plus the supplementary workbook for Warne (2023). It builds separate country-, sample-, geographic-imputation-, and reference-level tables and audits their overlap with HLO.

This notebook does **not** treat national-IQ estimates as innate intelligence or as a causal exposure. Quality fields are author-defined metadata from the source workbook and are not independent quality validation. Raw publisher files remain private and excluded from Git pending clarification of redistribution terms."""
    ),
    nbf.v4.new_code_cell(
        """from pathlib import Path
from datetime import datetime, timezone
import hashlib
import io
import json
import re
import time
import zipfile

import numpy as np
import pandas as pd
import requests

ROOT = Path.cwd().resolve()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent

RAW = ROOT / "data" / "raw" / "niq"
PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"
RAW.mkdir(parents=True, exist_ok=True)
PROCESSED.mkdir(parents=True, exist_ok=True)
OUTPUTS.mkdir(exist_ok=True)

SOURCES = {
    "niq_v135": "https://viewoniq.org/wp-content/uploads/2023/10/NIQ-DATASET-V1.3.5.zip",
    "niq_v133": "https://viewoniq.org/wp-content/uploads/2019/07/NIQ-DATASET-V1.3.3.zip",
    "warne_2023": "https://static-content.springer.com/esm/art%3A10.1007%2Fs40806-022-00351-y/MediaObjects/40806_2022_351_MOESM1_ESM.xlsx",
}

session = requests.Session()
session.headers.update({"User-Agent": "IQ-HLO-research/1.0"})
started = time.perf_counter()
print("Project root resolved.")"""
    ),
    nbf.v4.new_markdown_cell("## Download, checksum, and freeze source files"),
    nbf.v4.new_code_cell(
        """download_paths = {}
download_records = []

for source_id, url in SOURCES.items():
    suffix = ".xlsx" if source_id == "warne_2023" else ".zip"
    path = RAW / f"{source_id}{suffix}"
    response = session.get(url, timeout=120)
    response.raise_for_status()
    path.write_bytes(response.content)
    download_paths[source_id] = path
    download_records.append({
        "source_id": source_id,
        "url": url,
        "local_path": str(path.relative_to(ROOT)),
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "bytes": len(response.content),
        "sha256": hashlib.sha256(response.content).hexdigest(),
        "content_type": response.headers.get("content-type", ""),
    })
    print(f"{source_id}: {len(response.content):,} bytes")

niq_manifest = pd.DataFrame(download_records)
niq_manifest.to_csv(PROCESSED / "niq_source_manifest.csv", index=False)
display(niq_manifest)"""
    ),
    nbf.v4.new_markdown_cell("## Extract versioned workbooks and documentation"),
    nbf.v4.new_code_cell(
        """workbooks = {}

for version, source_id in {"1.3.5": "niq_v135", "1.3.3": "niq_v133"}.items():
    version_dir = RAW / f"v{version.replace('.', '')}"
    version_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(download_paths[source_id]) as archive:
        members = archive.namelist()
        for member in members:
            if member.lower().endswith((".xlsx", ".txt")):
                target = version_dir / Path(member).name
                target.write_bytes(archive.read(member))
                if member.lower().endswith(".xlsx"):
                    workbooks[version] = target
        print(version, members)

for version, path in workbooks.items():
    print(version, pd.ExcelFile(path).sheet_names)"""
    ),
    nbf.v4.new_markdown_cell("## Parsing helpers"),
    nbf.v4.new_code_cell(
        """def snake(value):
    text = "" if pd.isna(value) else str(value)
    text = text.replace("&", " and ").replace("+", " plus ")
    text = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()
    return text or "unnamed"


def unique_names(names):
    seen = {}
    output = []
    for name in names:
        count = seen.get(name, 0)
        output.append(name if count == 0 else f"{name}_{count + 1}")
        seen[name] = count + 1
    return output


def read_two_row_header(path, sheet):
    raw = pd.read_excel(path, sheet_name=sheet, header=None)
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


def numeric(frame, columns):
    for column in columns:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def find_column(frame, suffix):
    matches = [column for column in frame.columns if column.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f"Expected one column ending in {suffix!r}; found {matches}")
    return matches[0]"""
    ),
    nbf.v4.new_markdown_cell("## Country-level score variants and provenance flags"),
    nbf.v4.new_code_cell(
        """def country_scores(path, version):
    frame = read_two_row_header(path, "FAV")
    stable = {
        frame.columns[0]: "iso3",
        frame.columns[1]: "country",
        frame.columns[2]: "niq_uw",
        frame.columns[3]: "niq_nw",
        frame.columns[4]: "niq_qnw",
        frame.columns[5]: "niq_sas",
        frame.columns[6]: "niq_qnw_sas",
        frame.columns[7]: "niq_qnw_sas_geo",
        frame.columns[8]: "niq_lv02",
        frame.columns[9]: "niq_lv02_geo",
        frame.columns[10]: "niq_lv12",
        frame.columns[11]: "niq_lv12_geo",
        frame.columns[12]: "lv12_minus_niq",
        frame.columns[13]: "niq_r",
    }
    frame = frame.rename(columns=stable)
    frame = frame[frame["iso3"].astype(str).str.fullmatch(r"[A-Z]{3}")].copy()
    score_columns = [column for column in frame.columns if column.startswith("niq_") or column == "lv12_minus_niq"]
    numeric(frame, score_columns)
    frame.insert(2, "version", version)
    frame["has_psychometric_data"] = frame["niq_qnw"].notna()
    frame["has_school_assessment_data"] = frame["niq_sas"].notna()
    frame["geographically_imputed"] = frame["niq_qnw_sas"].isna() & frame["niq_qnw_sas_geo"].notna()
    frame["source_class"] = np.select(
        [
            frame["has_psychometric_data"] & frame["has_school_assessment_data"],
            frame["has_psychometric_data"],
            frame["has_school_assessment_data"],
            frame["geographically_imputed"],
        ],
        ["psychometric_and_assessment", "psychometric_only", "assessment_only", "geographic_only"],
        default="no_score",
    )
    return frame


country_v135 = country_scores(workbooks["1.3.5"], "1.3.5")
country_v133 = country_scores(workbooks["1.3.3"], "1.3.3")
country_v135.to_csv(PROCESSED / "niq_country_scores_v135.csv", index=False)
country_v133.to_csv(PROCESSED / "niq_country_scores_v133.csv", index=False)

print(country_v135["source_class"].value_counts(dropna=False))
display(country_v135[["iso3", "country", "niq_qnw", "niq_sas", "niq_qnw_sas", "niq_qnw_sas_geo", "source_class"]].head(10))"""
    ),
    nbf.v4.new_markdown_cell("## Sample-level provenance and author-defined quality fields"),
    nbf.v4.new_code_cell(
        """samples = read_two_row_header(workbooks["1.3.5"], "REC")
selection = read_two_row_header(workbooks["1.3.5"], "SEL")

sample_rename = {
    find_column(samples, "__id"): "sample_id",
    find_column(samples, "__iso_3166_1_alpha_3"): "iso3",
    find_column(samples, "__country_name"): "country",
    find_column(samples, "__origin_type"): "origin_type",
    find_column(samples, "__origin_specif"): "origin_detail",
    find_column(samples, "__ses"): "ses_code",
    find_column(samples, "__sample_comp"): "sample_composition",
    find_column(samples, "__sample_char"): "sample_character_code",
    find_column(samples, "__mean_age"): "mean_age",
    find_column(samples, "__n_ind"): "sample_size",
    find_column(samples, "__sample_rating"): "sample_rating",
    find_column(samples, "__test_type"): "test_type",
    find_column(samples, "__test_meas"): "test_measured",
    find_column(samples, "__year_meas"): "measurement_year",
    find_column(samples, "__year_std"): "norm_year",
    find_column(samples, "__testing_rating"): "testing_rating",
    find_column(samples, "__test_time_adjust"): "test_time_adjustment",
    find_column(samples, "__country_cor"): "country_correction",
    find_column(samples, "__method_rating"): "method_rating",
    find_column(samples, "sample_iqs__iq_cor"): "corrected_iq",
    find_column(samples, "__full_rating"): "full_rating",
    find_column(samples, "__qn_factor"): "qn_factor",
    find_column(samples, "references__short"): "reference_short",
    find_column(samples, "references__full"): "reference_full",
}
samples = samples.rename(columns=sample_rename)

selection_id = find_column(selection, "__id")
selection_flag = find_column(selection, "filter__y_n")
included = selection[[selection_id, selection_flag]].rename(columns={selection_id: "sample_id", selection_flag: "included_by_source"})
included = included[included["sample_id"].notna()].groupby("sample_id", as_index=False).agg(
    included_by_source=("included_by_source", lambda values: "Y" if values.eq("Y").any() else "N")
)
samples = samples[samples["sample_id"].notna()].copy()
samples["record_number_within_sample_id"] = samples.groupby("sample_id").cumcount() + 1
samples["record_id"] = samples["sample_id"].astype(str) + "_" + samples["record_number_within_sample_id"].astype(str)
samples = samples.merge(included, on="sample_id", how="left", validate="many_to_one")

numeric(samples, [
    "mean_age", "sample_size", "sample_rating", "measurement_year", "norm_year",
    "testing_rating", "test_time_adjustment", "country_correction", "method_rating",
    "corrected_iq", "full_rating", "qn_factor",
])
samples.to_csv(PROCESSED / "niq_samples_v135.csv", index=False)

selected_samples = samples[samples["included_by_source"].eq("Y")].copy()
provenance_fields = [
    "origin_type", "origin_detail", "ses_code", "sample_composition", "sample_character_code",
    "mean_age", "sample_size", "sample_rating", "test_type", "test_measured",
    "measurement_year", "norm_year", "testing_rating", "test_time_adjustment",
    "country_correction", "method_rating", "corrected_iq", "full_rating", "qn_factor",
    "reference_short", "reference_full",
]
provenance_coverage = pd.concat([
    pd.DataFrame({
        "sample_scope": scope,
        "field": provenance_fields,
        "records": len(frame),
        "nonmissing_n": [frame[field].notna().sum() for field in provenance_fields],
        "nonmissing_pct": [100 * frame[field].notna().mean() for field in provenance_fields],
        "unique_n": [frame[field].nunique(dropna=True) for field in provenance_fields],
    })
    for scope, frame in {"all_records": samples, "source_included_records": selected_samples}.items()
], ignore_index=True)
provenance_coverage.to_csv(OUTPUTS / "niq_sample_provenance_field_coverage.csv", index=False)

sample_summary = selected_samples.groupby("iso3", as_index=False).agg(
    selected_sample_count=("sample_id", "nunique"),
    selected_record_count=("record_id", "nunique"),
    summed_record_sample_size=("sample_size", "sum"),
    earliest_measurement_year=("measurement_year", "min"),
    latest_measurement_year=("measurement_year", "max"),
    median_measurement_year=("measurement_year", "median"),
    mean_sample_rating=("sample_rating", "mean"),
    mean_testing_rating=("testing_rating", "mean"),
    mean_method_rating=("method_rating", "mean"),
    mean_full_rating=("full_rating", "mean"),
    mean_qn_factor=("qn_factor", "mean"),
    test_types=("test_type", lambda x: "|".join(sorted(set(x.dropna().astype(str))))),
    reference_count=("reference_short", "nunique"),
)
sample_summary.to_csv(PROCESSED / "niq_sample_summary_v135.csv", index=False)

print("Sample records:", len(samples))
print("Unique sample IDs:", samples["sample_id"].nunique())
print("Source-included records:", len(selected_samples))
print("Source-included unique sample IDs:", selected_samples["sample_id"].nunique())
display(sample_summary.head())"""
    ),
    nbf.v4.new_markdown_cell("## Geographic imputations and reference ledger"),
    nbf.v4.new_code_cell(
        """geo = read_two_row_header(workbooks["1.3.5"], "GEO")
geo = geo.rename(columns={geo.columns[0]: "iso3", geo.columns[1]: "country"})
geo = geo[geo["iso3"].astype(str).str.fullmatch(r"[A-Z]{3}")].copy()
geo.to_csv(PROCESSED / "niq_geographic_imputations_v135.csv", index=False)

references = pd.read_excel(workbooks["1.3.5"], sheet_name="REF", header=1)
references.columns = unique_names([snake(column) for column in references.columns])
references = references.dropna(how="all")
references.to_csv(PROCESSED / "niq_references_v135.csv", index=False)

print("Geographic table rows:", len(geo))
print("Reference rows:", len(references))"""
    ),
    nbf.v4.new_markdown_cell("## Version instability audit"),
    nbf.v4.new_code_cell(
        """version_comparison = country_v133[["iso3", "country", "niq_qnw", "niq_qnw_sas", "niq_qnw_sas_geo", "source_class"]].merge(
    country_v135[["iso3", "country", "niq_qnw", "niq_qnw_sas", "niq_qnw_sas_geo", "source_class"]],
    on="iso3", how="outer", suffixes=("_v133", "_v135"), validate="one_to_one"
)
for score in ["niq_qnw", "niq_qnw_sas", "niq_qnw_sas_geo"]:
    version_comparison[f"{score}_change"] = version_comparison[f"{score}_v135"] - version_comparison[f"{score}_v133"]
version_comparison.to_csv(PROCESSED / "niq_version_comparison.csv", index=False)

version_stats = pd.DataFrame([
    {
        "score": score,
        "common_countries": version_comparison[[f"{score}_v133", f"{score}_v135"]].dropna().shape[0],
        "mean_change": version_comparison[f"{score}_change"].mean(),
        "median_absolute_change": version_comparison[f"{score}_change"].abs().median(),
        "maximum_absolute_change": version_comparison[f"{score}_change"].abs().max(),
        "correlation": version_comparison[[f"{score}_v133", f"{score}_v135"]].corr().iloc[0, 1],
    }
    for score in ["niq_qnw", "niq_qnw_sas", "niq_qnw_sas_geo"]
])
version_stats.to_csv(OUTPUTS / "niq_version_instability_summary.csv", index=False)
display(version_stats)"""
    ),
    nbf.v4.new_markdown_cell("## Warne supplement and HLO overlap audit"),
    nbf.v4.new_code_cell(
        """warne_book = pd.ExcelFile(download_paths["warne_2023"])
warne_sheets = pd.DataFrame({"sheet": warne_book.sheet_names})
warne_sheets.to_csv(PROCESSED / "warne_supplement_sheets.csv", index=False)

warne_national = read_two_row_header(download_paths["warne_2023"], "National scores")
warne_national.to_csv(PROCESSED / "warne_national_scores.csv", index=False)

hlo = pd.read_csv(PROCESSED / "hlo_database.csv")
hlo_country = hlo.groupby(["code", "country"], as_index=False).agg(
    hlo_records=("hlo", "size"),
    hlo_mean=("hlo", "mean"),
    hlo_median_se=("hlo_se", "median"),
    hlo_first_year=("year", "min"),
    hlo_last_year=("year", "max"),
    hlo_years=("year", "nunique"),
    hlo_subjects=("subject", "nunique"),
    hlo_levels=("level", "nunique"),
)

overlap = country_v135[[
    "iso3", "country", "niq_qnw", "niq_sas", "niq_qnw_sas", "niq_qnw_sas_geo",
    "has_psychometric_data", "has_school_assessment_data", "geographically_imputed", "source_class",
]].merge(sample_summary, on="iso3", how="left", validate="one_to_one").merge(
    hlo_country, left_on="iso3", right_on="code", how="left", validate="one_to_one"
)
overlap["has_hlo"] = overlap["hlo_records"].notna()
overlap.to_csv(PROCESSED / "niq_hlo_provenance_overlap.csv", index=False)

overlap_summary = overlap.groupby("source_class", dropna=False).agg(
    niq_countries=("iso3", "count"),
    countries_with_hlo=("has_hlo", "sum"),
    mean_selected_samples=("selected_sample_count", "mean"),
    median_measurement_year=("median_measurement_year", "median"),
    mean_full_rating=("mean_full_rating", "mean"),
).reset_index()
overlap_summary["hlo_coverage_pct"] = 100 * overlap_summary["countries_with_hlo"] / overlap_summary["niq_countries"]
overlap_summary.to_csv(OUTPUTS / "niq_hlo_provenance_coverage.csv", index=False)
display(overlap_summary)"""
    ),
    nbf.v4.new_markdown_cell("## Integrity checks and research-readiness decision"),
    nbf.v4.new_code_cell(
        """assert country_v135["iso3"].is_unique
assert country_v133["iso3"].is_unique
assert samples["record_id"].is_unique
assert overlap["iso3"].is_unique
assert country_v135.loc[country_v135["geographically_imputed"], "niq_qnw_sas"].isna().all()
assert country_v135.loc[country_v135["geographically_imputed"], "niq_qnw_sas_geo"].notna().all()

psychometric_hlo = int((overlap["has_psychometric_data"] & overlap["has_hlo"]).sum())
direct_hlo = int((overlap["source_class"].ne("geographic_only") & overlap["has_hlo"]).sum())
geo_hlo = int((overlap["geographically_imputed"] & overlap["has_hlo"]).sum())

report = f'''# Documented NIQ acquisition and provenance audit

## Decision: PROVENANCE-AWARE ANALYSIS IS FEASIBLE

- NIQ v1.3.5 contains {len(country_v135)} country rows.
- {int(country_v135["has_psychometric_data"].sum())} countries have a QNW psychometric estimate.
- {int(country_v135["has_school_assessment_data"].sum())} countries have an SAS estimate.
- {int(country_v135["geographically_imputed"].sum())} countries are geographic-only under the operational rule `QNW+SAS` missing and `QNW+SAS+GEO` present.
- The sample table contains {len(samples)} records representing {samples["sample_id"].nunique()} unique source sample IDs; {len(selected_samples)} records representing {selected_samples["sample_id"].nunique()} unique IDs are marked included by the source workbook.
- {psychometric_hlo} countries have both QNW psychometric data and HLO.
- {direct_hlo} non-geographic-only NIQ countries have HLO.
- {geo_hlo} geographic-only NIQ countries have HLO.

## Primary measurement comparison

Use NIQ `QNW` versus prespecified subject- and level-specific HLO constructions. This minimizes criterion contamination from school assessments included in `QNW+SAS`.

## Required sensitivity comparisons

1. NIQ v1.3.3 versus v1.3.5.
2. QNW psychometric-only versus QNW+SAS.
3. Direct-source countries versus geographic-only estimates.
4. Constant-country sample versus expanded coverage.
5. Equal-country versus population weighting.
6. HLO uncertainty propagated into rank simulations.

## Interpretation constraints

- Workbook quality ratings are author-defined metadata, not independent validation.
- Geographic imputation is identified structurally and must not be treated as observed psychometric evidence.
- HLO is a learning measure, not a gold-standard intelligence test.
- The current WorldData-like `iq_nation.csv` remains a secondary undocumented-snapshot audit.
- No causal or innate-intelligence claims are supported.
'''

(OUTPUTS / "documented_niq_provenance_audit.md").write_text(report, encoding="utf-8")
print(report)
print(f"Notebook completed in {time.perf_counter() - started:.1f} seconds.")"""
    ),
]

output = NOTEBOOKS / "03_documented_niq_acquisition_and_audit.ipynb"
nbf.write(nb, output)
print(output)
