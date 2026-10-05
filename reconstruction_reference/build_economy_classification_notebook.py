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
    """# 04c — Frozen economy-classification gate

This notebook freezes the economy population before any NIQ score is loaded. It classifies every economy in the HLO universe using only authoritative World Bank and United Nations sources.

The population rules are:

1. **Primary population:** current World Bank reporting economies, including territories and other separately reported economies. The World Bank explicitly uses *country* interchangeably with *economy* and states that this does not imply political independence.
2. **Sensitivity population:** primary-population economies that are current UN members or non-member permanent-observer states.
3. **Measurement eligibility:** the already frozen HLO component rule is joined only after the classification is complete. It does not determine any classification flag.

Manual decisions are limited to Hong Kong, Macao, Taiwan, Kosovo, Palestine, and the historical Serbia and Montenegro record. No NIQ, GDP, population, or contextual data are loaded."""
)

code(
    """from pathlib import Path
from datetime import datetime, timezone
from io import StringIO
import hashlib
import json
import re
import time

import pandas as pd
import requests

ROOT = Path.cwd().resolve()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent

RAW = ROOT / "data" / "raw" / "economy_classification"
PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"
RAW.mkdir(parents=True, exist_ok=True)
OUTPUTS.mkdir(exist_ok=True)

ACCESS_DATE = "2026-07-21"
WORLD_BANK_API_URL = "https://api.worldbank.org/v2/country?format=json&per_page=400"
WORLD_BANK_GROUPS_URL = "https://datahelpdesk.worldbank.org/knowledgebase/articles/906519-world-bank-country-and-lending-groups"
UN_MEMBER_URL = "https://digitallibrary.un.org/record/4082085/files/member_states_auths_2026-04-30.csv"
UN_MEMBER_PAGE_URL = "https://www.un.org/about-us/member-states"
UN_OBSERVER_URL = "https://www.un.org/en/about-us/non-member-states"
UN_M49_URL = "https://unstats.un.org/unsd/methodology/m49/overview/"
UN_M49_NOTES_URL = "https://unstats.un.org/unsd/methodology/m49/"

SOURCE_VERSIONS = {
    "world_bank": "World Bank Country and Lending Groups, fiscal year 2027",
    "un_members": "UN Member States authority file, 2026-04-30, version 1",
    "un_observers": "UN Non-Member States page, accessed 2026-07-21",
    "un_m49": "UN M49 online classification, accessed 2026-07-21",
}

started = time.perf_counter()
print("Project root resolved.")
print("Classification inputs only: HLO economy identifiers, World Bank classifications, and UN classifications")"""
)

markdown("## Freeze authoritative source snapshots")

code(
    """sources = {
    "world_bank_country_api": WORLD_BANK_API_URL,
    "world_bank_country_groups_fy2027": WORLD_BANK_GROUPS_URL,
    "un_member_states_2026_04_30_v1": UN_MEMBER_URL,
    "un_m49": UN_M49_URL,
}

suffixes = {
    "world_bank_country_api": ".json",
    "world_bank_country_groups_fy2027": ".html",
    "un_member_states_2026_04_30_v1": ".csv",
    "un_m49": ".html",
}

manifest_rows = []
source_bytes = {}
for source_id, url in sources.items():
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    content = response.content
    source_bytes[source_id] = content
    path = RAW / f"{source_id}_{ACCESS_DATE}{suffixes[source_id]}"
    path.write_bytes(content)
    manifest_rows.append({
        "source_id": source_id,
        "url": url,
        "source_version": SOURCE_VERSIONS[
            "world_bank" if source_id.startswith("world_bank") else
            "un_members" if source_id.startswith("un_member") else
            "un_m49"
        ],
        "access_date": ACCESS_DATE,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "local_path": str(path.relative_to(ROOT)),
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    })

manifest_rows.append({
    "source_id": "un_non_member_permanent_observers",
    "url": UN_OBSERVER_URL,
    "source_version": SOURCE_VERSIONS["un_observers"],
    "access_date": ACCESS_DATE,
    "downloaded_at_utc": "",
    "local_path": "",
    "bytes": "",
    "sha256": "",
})

source_manifest = pd.DataFrame(manifest_rows)
source_manifest.to_csv(OUTPUTS / "economy_classification_source_manifest.csv", index=False)
source_manifest"""
)

markdown(
    """## Build the authoritative reference sets

The World Bank API labels aggregate regions with `region.id == "NA"`; those records are excluded before matching. The FY2027 country-groups page supplies Taiwan, China, which appears in the current regional table but not in the country API response.

The UN authority file contains historical names and interrupted name periods. A record is current when its most recent start date is later than its most recent end date. This rule returns exactly 193 current member states. The two current non-member permanent-observer states are the Holy See and the State of Palestine; only Palestine appears in the HLO universe."""
)

code(
    """world_bank_payload = json.loads(source_bytes["world_bank_country_api"])
world_bank_all = pd.json_normalize(world_bank_payload[1])
world_bank_reporting = world_bank_all.loc[world_bank_all["region.id"].ne("NA")].copy()
world_bank_aggregates = world_bank_all.loc[world_bank_all["region.id"].eq("NA")].copy()

world_bank_groups_html = source_bytes["world_bank_country_groups_fy2027"].decode("utf-8")
world_bank_region_tables = pd.read_html(StringIO(world_bank_groups_html))[1:8]
world_bank_region_names = [
    "East Asia & Pacific",
    "Europe & Central Asia",
    "Latin America & Caribbean",
    "Middle East, North Africa, Afghanistan & Pakistan",
    "North America",
    "South Asia",
    "Sub-Saharan Africa",
]
world_bank_page_regions = {}
for region_name, table in zip(world_bank_region_names, world_bank_region_tables):
    for economy_name in table.stack().dropna().astype(str):
        world_bank_page_regions[economy_name.strip()] = region_name

assert "Taiwan, China" in world_bank_page_regions
assert "current 2027 fiscal year" in world_bank_groups_html

un_members = pd.read_csv(
    StringIO(source_bytes["un_member_states_2026_04_30_v1"].decode("utf-8")),
    dtype=str,
)

def most_recent_date(value):
    dates = re.findall(r"\\d{4}-\\d{2}-\\d{2}", "" if pd.isna(value) else value)
    return max(dates) if dates else ""

un_members["last_start"] = un_members["Start date"].map(most_recent_date)
un_members["last_end"] = un_members["End date"].map(most_recent_date)
un_current = un_members.loc[
    un_members["last_start"].gt(un_members["last_end"])
    & un_members["ISO Code"].notna()
].copy()
assert len(un_current) == 193
assert un_current["ISO Code"].nunique() == 193

un_member_codes = set(un_current["ISO Code"])
un_observer_codes = {"PSE", "VAT"}

un_m49 = pd.read_html(
    StringIO(source_bytes["un_m49"].decode("utf-8"))
)[0]
un_m49 = un_m49.rename(columns={
    "ISO-alpha3 Code": "iso3",
    "Country or Area": "un_m49_name",
    "M49 Code": "m49_code",
})
un_m49["m49_code"] = un_m49["m49_code"].astype(str).str.zfill(3)
un_m49_lookup = un_m49.set_index("iso3")

print(f"World Bank API reporting economies: {len(world_bank_reporting)}")
print(f"World Bank FY2027 reporting economies after Taiwan supplement: {len(world_bank_reporting) + 1}")
print(f"Current UN members: {len(un_member_codes)}")
print(f"UN non-member permanent observers: {len(un_observer_codes)}")"""
)

markdown("## Classify every HLO economy")

code(
    """hlo = pd.read_csv(PROCESSED / "hlo_database.csv")
hlo_universe = hlo[["code", "country"]].drop_duplicates("code").rename(
    columns={"code": "iso3", "country": "hlo_name"}
)
assert len(hlo_universe) == 164

wb_lookup = world_bank_reporting.set_index("id")
crosswalk = hlo_universe.copy()
crosswalk["world_bank_name"] = crosswalk["iso3"].map(wb_lookup["name"])
crosswalk["frozen_world_bank_region"] = crosswalk["iso3"].map(wb_lookup["region.value"])

# Taiwan is on the World Bank FY2027 economy list but absent from the API response.
crosswalk.loc[crosswalk["iso3"].eq("TWN"), "world_bank_name"] = "Taiwan, China"
crosswalk.loc[crosswalk["iso3"].eq("TWN"), "frozen_world_bank_region"] = "East Asia & Pacific"

crosswalk["world_bank_reporting_economy"] = crosswalk["world_bank_name"].notna()
crosswalk["un_member"] = crosswalk["iso3"].isin(un_member_codes)
crosswalk["un_permanent_observer"] = crosswalk["iso3"].isin(un_observer_codes)
crosswalk["un_member_or_permanent_observer"] = (
    crosswalk["un_member"] | crosswalk["un_permanent_observer"]
)
crosswalk["un_m49_name"] = crosswalk["iso3"].map(un_m49_lookup["un_m49_name"])
crosswalk["m49_code"] = crosswalk["iso3"].map(un_m49_lookup["m49_code"])

crosswalk["dependent_territory"] = False
crosswalk["special_administrative_region"] = crosswalk["iso3"].isin(["HKG", "MAC"])
crosswalk["other_separately_reporting_economy"] = crosswalk["iso3"].isin(["TWN", "XKX"])

canonical_overrides = {
    "HKG": "Hong Kong SAR, China",
    "MAC": "Macao SAR, China",
    "PSE": "State of Palestine",
    "TWN": "Taiwan, China",
    "XKX": "Kosovo",
    "SCG": "Serbia and Montenegro (historical entity)",
}
crosswalk["canonical_economy_name"] = crosswalk["world_bank_name"].fillna(
    crosswalk["un_m49_name"]
).fillna(crosswalk["hlo_name"])
for iso3, name in canonical_overrides.items():
    crosswalk.loc[crosswalk["iso3"].eq(iso3), "canonical_economy_name"] = name

crosswalk["iso3_status"] = "current ISO-alpha3"
crosswalk.loc[crosswalk["iso3"].eq("XKX"), "iso3_status"] = "World Bank user-assigned code; not an ISO 3166 alpha-3 code"
crosswalk.loc[crosswalk["iso3"].eq("SCG"), "iso3_status"] = "retired ISO-alpha3 code for a dissolved historical entity"
crosswalk.loc[crosswalk["iso3"].eq("TWN"), "iso3_status"] = "current ISO-alpha3; not separately listed in UN M49"

crosswalk["manual_decision"] = crosswalk["iso3"].isin(["HKG", "MAC", "TWN", "XKX", "PSE", "SCG"])
manual_justifications = {
    "HKG": "World Bank FY2027 reporting economy; UN M49 identifies Hong Kong as a Special Administrative Region of China; not a UN member or permanent-observer state.",
    "MAC": "World Bank FY2027 reporting economy; UN M49 identifies Macao as a Special Administrative Region of China; not a UN member or permanent-observer state.",
    "TWN": "World Bank FY2027 lists Taiwan, China separately in East Asia and Pacific although it is absent from the country API and UN M49; retain as an other separately reporting economy; not a UN member or permanent-observer state.",
    "XKX": "World Bank FY2027 lists Kosovo separately and assigns XKX; UN M49 treats Kosovo within Serbia and XKX is not an ISO code; retain as an other separately reporting economy; not a UN member or permanent-observer state.",
    "PSE": "World Bank reports West Bank and Gaza separately; UN M49 uses State of Palestine (PSE); the UN classifies the State of Palestine as a non-member permanent-observer state.",
    "SCG": "Historical HLO record using the retired SCG code. UN M49 records the 2006 dissolution of Serbia and Montenegro; it is not a current World Bank reporting economy or current UN member/observer and is excluded from both populations.",
}
crosswalk["manual_justification"] = crosswalk["iso3"].map(manual_justifications).fillna("")

crosswalk["include_primary_reporting_economy"] = crosswalk["world_bank_reporting_economy"]
crosswalk["include_sensitivity_un_member_observer"] = (
    crosswalk["include_primary_reporting_economy"]
    & crosswalk["un_member_or_permanent_observer"]
)

common_sources = "World Bank country API; World Bank FY2027 Country and Lending Groups; UN Member States authority file; UN M49"
common_versions = "; ".join([
    SOURCE_VERSIONS["world_bank"],
    SOURCE_VERSIONS["un_members"],
    SOURCE_VERSIONS["un_m49"],
])
crosswalk["classification_sources"] = common_sources
crosswalk["classification_versions"] = common_versions
crosswalk["classification_access_date"] = ACCESS_DATE
crosswalk.loc[crosswalk["iso3"].eq("PSE"), "classification_sources"] += "; UN Non-Member States page"
crosswalk.loc[crosswalk["iso3"].eq("PSE"), "classification_versions"] += "; " + SOURCE_VERSIONS["un_observers"]

column_order = [
    "iso3", "canonical_economy_name", "hlo_name", "world_bank_name", "un_m49_name", "m49_code", "iso3_status",
    "world_bank_reporting_economy", "un_member", "un_permanent_observer", "un_member_or_permanent_observer",
    "dependent_territory", "special_administrative_region", "other_separately_reporting_economy",
    "frozen_world_bank_region", "classification_sources", "classification_versions", "classification_access_date",
    "manual_decision", "manual_justification", "include_primary_reporting_economy",
    "include_sensitivity_un_member_observer",
]
crosswalk = crosswalk[column_order].sort_values("iso3").reset_index(drop=True)
crosswalk.to_csv(OUTPUTS / "economy_classification_crosswalk.csv", index=False)
crosswalk.head()"""
)

markdown(
    """## Validate the gate

The classification is frozen before the HLO component-eligibility field is joined. The final two validation rows merely report intersections with the already frozen 82-economy HLO measure; they do not alter the population flags."""
)

code(
    """aggregate_codes = set(world_bank_aggregates["id"])
ambiguous_codes = {"HKG", "MAC", "TWN", "XKX", "PSE", "SCG"}

expected_primary = crosswalk["world_bank_reporting_economy"]
expected_sensitivity = expected_primary & crosswalk["un_member_or_permanent_observer"]

checks = [
    ("all_hlo_economies_classified", len(crosswalk) == 164 and crosswalk["iso3"].nunique() == 164, "164 unique HLO codes"),
    ("all_eligibility_decisions_nonmissing", crosswalk[["include_primary_reporting_economy", "include_sensitivity_un_member_observer"]].notna().all().all(), "No missing primary or sensitivity decision"),
    ("no_world_bank_aggregate_in_primary", not crosswalk.loc[crosswalk["include_primary_reporting_economy"], "iso3"].isin(aggregate_codes).any(), "World Bank aggregate codes excluded"),
    ("all_ambiguous_cases_documented", set(crosswalk.loc[crosswalk["manual_decision"], "iso3"]) == ambiguous_codes and crosswalk.loc[crosswalk["manual_decision"], "manual_justification"].ne("").all(), "HKG, MAC, TWN, XKX, PSE, SCG"),
    ("primary_rule_reproduced", crosswalk["include_primary_reporting_economy"].equals(expected_primary), "Current World Bank reporting economy"),
    ("sensitivity_rule_reproduced", crosswalk["include_sensitivity_un_member_observer"].equals(expected_sensitivity), "Primary and UN member/observer"),
    ("un_member_reference_count", len(un_member_codes) == 193, "193 current UN members"),
]

hlo_eligibility = pd.read_csv(OUTPUTS / "hlo_aggregate_exclusion_audit.csv")[[
    "code", "eligible_primary_component_rule"
]].rename(columns={"code": "iso3", "eligible_primary_component_rule": "hlo_primary_component_eligible"})
sample_membership = crosswalk.merge(hlo_eligibility, on="iso3", how="left", validate="one_to_one")
sample_membership["in_frozen_primary_hlo_reporting_sample"] = (
    sample_membership["hlo_primary_component_eligible"]
    & sample_membership["include_primary_reporting_economy"]
)
sample_membership["in_frozen_primary_hlo_un_sensitivity_sample"] = (
    sample_membership["hlo_primary_component_eligible"]
    & sample_membership["include_sensitivity_un_member_observer"]
)
sample_membership.to_csv(OUTPUTS / "economy_classification_sample_membership.csv", index=False)

checks.extend([
    ("frozen_primary_hlo_measure_unchanged", sample_membership["hlo_primary_component_eligible"].sum() == 82, "82 economies"),
    ("reporting_population_hlo_intersection", sample_membership["in_frozen_primary_hlo_reporting_sample"].sum() == 82, "82 economies"),
    ("un_sensitivity_hlo_intersection", sample_membership["in_frozen_primary_hlo_un_sensitivity_sample"].sum() == 79, "79 economies"),
])

validation = pd.DataFrame(checks, columns=["check", "passed", "evidence"])
assert validation["passed"].all()
validation.to_csv(OUTPUTS / "economy_classification_validation.csv", index=False)

ambiguous = crosswalk.loc[crosswalk["manual_decision"]].copy()
ambiguous.to_csv(OUTPUTS / "economy_classification_ambiguous_cases.csv", index=False)

sample_flow = pd.DataFrame([
    {"stage": "HLO economy universe", "economies": len(crosswalk)},
    {"stage": "Primary World Bank reporting-economy population", "economies": crosswalk["include_primary_reporting_economy"].sum()},
    {"stage": "UN member/permanent-observer sensitivity population", "economies": crosswalk["include_sensitivity_un_member_observer"].sum()},
    {"stage": "Frozen HLO component-eligible measure", "economies": sample_membership["hlo_primary_component_eligible"].sum()},
    {"stage": "Frozen HLO × primary reporting population", "economies": sample_membership["in_frozen_primary_hlo_reporting_sample"].sum()},
    {"stage": "Frozen HLO × UN sensitivity population", "economies": sample_membership["in_frozen_primary_hlo_un_sensitivity_sample"].sum()},
])
sample_flow.to_csv(OUTPUTS / "economy_classification_sample_flow.csv", index=False)

validation"""
)

markdown("## Data dictionary")

code(
    """definitions = {
    "iso3": "HLO three-character economy identifier; exceptions and retired codes are identified in iso3_status.",
    "canonical_economy_name": "Frozen display name selected from authoritative World Bank or UN sources, with documented overrides for ambiguous cases.",
    "hlo_name": "Economy name in the HLO source file.",
    "world_bank_name": "Current World Bank FY2027 reporting-economy name.",
    "un_m49_name": "Current English country-or-area name in UN M49.",
    "m49_code": "Current three-digit UN M49 statistical code.",
    "iso3_status": "Status of the three-character identifier, including World Bank-assigned and retired exceptions.",
    "world_bank_reporting_economy": "True when listed as a current World Bank FY2027 reporting economy.",
    "un_member": "True for a current member in the UN Member States authority file.",
    "un_permanent_observer": "True for a current non-member permanent-observer state.",
    "un_member_or_permanent_observer": "Union of current UN members and non-member permanent-observer states.",
    "dependent_territory": "True for a dependent territory in the HLO universe; no HLO economy met this definition.",
    "special_administrative_region": "True for Hong Kong SAR, China or Macao SAR, China.",
    "other_separately_reporting_economy": "True for a separately reported economy that is neither a UN member/observer, dependent territory, nor SAR: Taiwan or Kosovo here.",
    "frozen_world_bank_region": "World Bank FY2027 region frozen on the access date.",
    "classification_sources": "Authoritative sources supporting the row classification.",
    "classification_versions": "Frozen source editions or dated online snapshots.",
    "classification_access_date": "Date on which online classifications were accessed.",
    "manual_decision": "True when the row required an explicit documented classification decision.",
    "manual_justification": "Reason and source interpretation for a manual decision.",
    "include_primary_reporting_economy": "Primary-population inclusion: current World Bank reporting economy.",
    "include_sensitivity_un_member_observer": "Sensitivity-population inclusion: primary population and UN member or permanent observer.",
}
data_dictionary = pd.DataFrame([
    {"column": column, "definition": definitions[column]} for column in column_order
])
data_dictionary.to_csv(OUTPUTS / "economy_classification_data_dictionary.csv", index=False)
data_dictionary"""
)

markdown("## Write the construction report")

code(
    """def markdown_table(frame):
    values = frame.fillna("").astype(str)
    text = "| " + " | ".join(values.columns) + " |\\n"
    text += "| " + " | ".join(["---"] * len(values.columns)) + " |\\n"
    text += "\\n".join("| " + " | ".join(row) + " |" for row in values.to_numpy())
    return text

excluded_primary = crosswalk.loc[~crosswalk["include_primary_reporting_economy"], ["iso3", "canonical_economy_name"]]
excluded_sensitivity = sample_membership.loc[
    sample_membership["in_frozen_primary_hlo_reporting_sample"]
    & ~sample_membership["in_frozen_primary_hlo_un_sensitivity_sample"],
    ["iso3", "canonical_economy_name"]
]

report = f'''# Economy classification gate report

**Frozen:** {ACCESS_DATE}
**Scope:** Classification only; no NIQ, GDP, population, or contextual data were loaded.

## Decision

The primary target population is current World Bank FY2027 reporting economies. The sensitivity population is the subset that consists of current UN members or non-member permanent-observer states. These flags were determined before joining the already frozen HLO component-eligibility rule.

The World Bank states that *country* is used interchangeably with *economy* and does not imply political independence. UN M49 classifies countries or areas for statistical use and likewise warns that its groupings do not imply a position on political affiliation.

## Population flow

{markdown_table(sample_flow)}

The HLO universe has 164 identifiers. The primary reporting-economy population has 163; only the dissolved historical `SCG` record is excluded. The frozen 82-economy HLO component-eligible measure is unchanged. Its UN-member/permanent-observer sensitivity intersection has exactly 79 economies; the three excluded codes are:

{markdown_table(excluded_sensitivity)}

## Ambiguous cases

{markdown_table(ambiguous[["iso3", "canonical_economy_name", "world_bank_reporting_economy", "un_member", "un_permanent_observer", "dependent_territory", "special_administrative_region", "other_separately_reporting_economy", "include_primary_reporting_economy", "include_sensitivity_un_member_observer", "manual_justification"]])}

No dependent territory occurs in the 164-economy HLO universe. This zero is a feature of this source universe, not a claim that no dependent territories exist. Hong Kong and Macao are coded specifically as special administrative regions, not as dependent territories. Taiwan and Kosovo are coded as other separately reporting economies. Palestine is a UN non-member permanent-observer state. Serbia and Montenegro is a dissolved historical entity and is excluded.

## Primary exclusions

{markdown_table(excluded_primary)}

## Validation

{markdown_table(validation)}

## Authoritative sources

- World Bank, *Country and Lending Groups*, FY2027: {WORLD_BANK_GROUPS_URL}
- World Bank Country API snapshot: {WORLD_BANK_API_URL}
- United Nations Dag Hammarskjöld Library, *UN Member States authority file*, 2026-04-30 version 1: {UN_MEMBER_URL}
- United Nations, *Member States*: {UN_MEMBER_PAGE_URL}
- United Nations, *Non-Member States*: {UN_OBSERVER_URL}
- United Nations Statistics Division, *M49*: {UN_M49_NOTES_URL}

The machine-readable crosswalk is `outputs/economy_classification_crosswalk.csv`. Every row contains the source version, access date, population flags, and any manual justification.
'''

(OUTPUTS / "economy_classification_report.md").write_text(report, encoding="utf-8")

elapsed = time.perf_counter() - started
print(report)
print(f"Completed in {elapsed:.1f} seconds")"""
)

nb["cells"] = cells
nbf.write(nb, NOTEBOOKS / "04c_economy_classification_gate.ipynb")
print(NOTEBOOKS / "04c_economy_classification_gate.ipynb")
