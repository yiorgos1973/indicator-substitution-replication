"""Acquire and audit the outcome-free 65-economy WDI bridge panel."""

from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import json
import urllib.request
import zipfile

import pandas as pd


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw" / "wdi_bridge_1980_2017"
PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"
START_YEAR = 1980
END_YEAR = 2017
YEARS = list(range(START_YEAR, END_YEAR + 1))
WDI_URL = "https://api.worldbank.org/v2/en/indicator/{code}?downloadformat=csv"

ECONOMIES = [
    "ARE", "ARG", "AUS", "AUT", "BGR", "BHR", "BRA", "BWA", "CAN", "CHL",
    "COL", "CRI", "CYP", "DEU", "DNK", "DOM", "EGY", "ESP", "FIN", "GHA",
    "GRC", "HKG", "HRV", "HUN", "IDN", "IRL", "IRN", "ISL", "ISR", "ITA",
    "JOR", "JPN", "KAZ", "KGZ", "KOR", "KWT", "LTU", "LVA", "MAR", "MEX",
    "MLT", "MNG", "MUS", "NLD", "NOR", "NZL", "OMN", "PER", "PHL", "POL",
    "PRT", "QAT", "ROU", "RUS", "SAU", "SGP", "SRB", "SVK", "SVN", "SWE",
    "THA", "TUR", "UKR", "USA", "ZAF",
]

INDICATORS = {
    "SE.PRM.ENRR": "School enrollment, primary (% gross)",
    "SE.SEC.ENRR": "School enrollment, secondary (% gross)",
    "SE.XPD.TOTL.GD.ZS": "Government expenditure on education, total (% of GDP)",
    "SH.DYN.MORT": "Mortality rate, under-5 (per 1,000 live births)",
    "SP.DYN.TFRT.IN": "Fertility rate, total (births per woman)",
    "SP.URB.TOTL.IN.ZS": "Urban population (% of total population)",
    "IT.NET.USER.ZS": "Individuals using the Internet (% of population)",
    "SP.DYN.LE00.IN": "Life expectancy at birth, total (years)",
    "NY.GDP.PCAP.PP.KD": "GDP per capita, PPP (constant 2021 international $)",
    "SP.POP.TOTL": "Population, total",
}


def checksum(content):
    return sha256(content).hexdigest()


def read_zip_csv(content, prefix, **kwargs):
    with zipfile.ZipFile(BytesIO(content)) as archive:
        name = next(name for name in archive.namelist() if Path(name).name.startswith(prefix))
        return pd.read_csv(archive.open(name), **kwargs), name


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    download_rows = []
    indicator_metadata = []
    frames = []
    country_names = {}

    for position, (code, label) in enumerate(INDICATORS.items(), start=1):
        url = WDI_URL.format(code=code)
        with urllib.request.urlopen(url, timeout=120) as response:
            content = response.read()
        archive_path = RAW / f"wdi_{code}.zip"
        archive_path.write_bytes(content)

        wide, data_name = read_zip_csv(content, f"API_{code}_", skiprows=4)
        indicator_info, indicator_name = read_zip_csv(content, "Metadata_Indicator", skiprows=0)
        row = indicator_info.loc[indicator_info["INDICATOR_CODE"].eq(code)].iloc[0]
        indicator_metadata.append({
            "indicator_code": code,
            "requested_label": label,
            "source_indicator_name": row.get("INDICATOR_NAME", label),
            "unit_of_measure": row.get("UNIT_OF_MEASURE", ""),
            "periodicity": row.get("PERIODICITY", ""),
            "source_metadata_member": indicator_name,
        })

        for _, source_row in wide[["Country Code", "Country Name"]].drop_duplicates().iterrows():
            if pd.notna(source_row["Country Code"]):
                country_names[str(source_row["Country Code"])] = source_row["Country Name"]

        year_columns = [str(year) for year in YEARS]
        panel = wide[["Country Code", "Country Name", *year_columns]].rename(
            columns={"Country Code": "iso3", "Country Name": "economy_name"}
        )
        panel = panel[panel["iso3"].isin(ECONOMIES)]
        panel = panel.melt(
            id_vars=["iso3", "economy_name"],
            value_vars=year_columns,
            var_name="year",
            value_name="value",
        )
        panel["year"] = panel["year"].astype(int)
        panel["value"] = pd.to_numeric(panel["value"], errors="coerce")
        panel["indicator_code"] = code
        panel["indicator_name"] = label
        frames.append(panel)
        download_rows.append({
            "source_id": f"wdi_bridge_{code}",
            "source_type": "official World Bank Indicators API v2 download archive",
            "indicator_code": code,
            "indicator_label": label,
            "url": url,
            "local_path": str(archive_path.relative_to(ROOT)),
            "archive_member": data_name,
            "bytes": len(content),
            "sha256": checksum(content),
            "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        })
        print(f"[{position:02d}/{len(INDICATORS)}] {code}: {len(content):,} bytes")

    expected = pd.MultiIndex.from_product(
        [ECONOMIES, YEARS, list(INDICATORS)],
        names=["iso3", "year", "indicator_code"],
    ).to_frame(index=False)
    panel = pd.concat(frames, ignore_index=True)
    panel = expected.merge(panel, on=["iso3", "year", "indicator_code"], how="left")
    panel["economy_name"] = panel["economy_name"].fillna(panel["iso3"].map(country_names))
    panel["indicator_name"] = panel["indicator_name"].fillna(panel["indicator_code"].map(INDICATORS))
    panel = panel[["iso3", "economy_name", "year", "value", "indicator_code", "indicator_name"]]
    panel = panel.sort_values(["indicator_code", "iso3", "year"]).reset_index(drop=True)

    if set(panel["iso3"]) != set(ECONOMIES):
        raise ValueError("Panel economy set does not match the requested 65 ISO3 codes")
    if set(panel["year"]) != set(YEARS):
        raise ValueError("Panel years do not match 1980-2017")
    if set(panel["indicator_code"]) != set(INDICATORS):
        raise ValueError("Panel indicator set does not match the requested 10 series")
    if panel.duplicated(["iso3", "year", "indicator_code"]).any():
        raise ValueError("Panel contains duplicate economy-year-indicator keys")
    if len(panel) != len(ECONOMIES) * len(YEARS) * len(INDICATORS):
        raise ValueError("Panel does not contain the complete requested key grid")

    panel_path = PROCESSED / "wdi_bridge_1980_2017_long.csv"
    panel.to_csv(panel_path, index=False)
    pd.DataFrame(download_rows).to_csv(RAW / "wdi_download_manifest.csv", index=False)
    pd.DataFrame(indicator_metadata).to_csv(RAW / "wdi_indicator_metadata.csv", index=False)
    pd.DataFrame(
        [{"iso3": iso3, "economy_name": country_names.get(iso3, iso3)} for iso3 in ECONOMIES]
    ).to_csv(RAW / "wdi_bridge_country_metadata.csv", index=False)

    panel["window"] = panel["year"].map(
        lambda year: "1980-1999" if year <= 1999 else "2000-2017"
    )
    window_years = {"1980-1999": 20, "2000-2017": 18}
    economy_coverage = panel.groupby(
        ["indicator_code", "indicator_name", "window", "iso3", "economy_name"],
        dropna=False,
    ).agg(
        expected_observations=("value", "size"),
        observed_nonmissing=("value", "count"),
    ).reset_index()
    economy_coverage["coverage_pct"] = (
        100 * economy_coverage["observed_nonmissing"] / economy_coverage["expected_observations"]
    )
    economy_coverage["meets_80pct_year_rule"] = economy_coverage["coverage_pct"] >= 80
    economy_coverage.to_csv(OUTPUTS / "wdi_bridge_coverage_by_economy.csv", index=False)

    indicator_coverage = economy_coverage.groupby(
        ["indicator_code", "indicator_name", "window"], dropna=False
    ).agg(
        expected_observations=("expected_observations", "sum"),
        observed_nonmissing=("observed_nonmissing", "sum"),
        economies_with_any_observation=("observed_nonmissing", lambda x: int((x > 0).sum())),
        economies_meeting_80pct_year_rule=("meets_80pct_year_rule", "sum"),
    ).reset_index()
    indicator_coverage["observation_coverage_pct"] = (
        100 * indicator_coverage["observed_nonmissing"] / indicator_coverage["expected_observations"]
    )
    indicator_coverage["economy_any_observation_pct"] = (
        100 * indicator_coverage["economies_with_any_observation"] / len(ECONOMIES)
    )
    indicator_coverage["economies_meeting_80pct_year_pct"] = (
        100 * indicator_coverage["economies_meeting_80pct_year_rule"] / len(ECONOMIES)
    )
    indicator_coverage["provisional_80pct_economies_rule"] = (
        indicator_coverage["economies_meeting_80pct_year_pct"] >= 80
    )
    indicator_coverage["window_years"] = indicator_coverage["window"].map(window_years)
    indicator_coverage.to_csv(OUTPUTS / "wdi_bridge_coverage_by_indicator_window.csv", index=False)

    validation = {
        "retrieved_run_started_utc": started.isoformat(),
        "retrieved_run_finished_utc": datetime.now(timezone.utc).isoformat(),
        "panel_relative_path": str(panel_path.relative_to(ROOT)),
        "panel_sha256": checksum(panel_path.read_bytes()),
        "economies_requested": ECONOMIES,
        "economy_count": len(ECONOMIES),
        "years": YEARS,
        "year_count": len(YEARS),
        "indicators": list(INDICATORS),
        "indicator_count": len(INDICATORS),
        "panel_rows": len(panel),
        "missing_value_rows": int(panel["value"].isna().sum()),
        "expected_panel_rows": len(ECONOMIES) * len(YEARS) * len(INDICATORS),
        "taiwan_in_panel": "TWN" in set(panel["iso3"]),
        "source_endpoints": {code: WDI_URL.format(code=code) for code in INDICATORS},
    }
    (RAW / "wdi_bridge_validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    print(f"Panel: {len(panel):,} rows; missing values: {validation['missing_value_rows']:,}")
    print(f"Elapsed: {(datetime.now(timezone.utc) - started).total_seconds():.2f} seconds")


if __name__ == "__main__":
    main()
