# v0.6 source acquisition and reconstruction boundary

`SOURCE_MANIFEST.csv` is the authoritative 21-source ledger. It records the source version or access date, acquisition URL, expected filename, byte count where recorded, SHA-256, bundled form, and redistribution decision.

## Two distinct reproduction levels

The one-command offline run is a **derived-output reproduction**. Its NIQ/QNW, HLO, WDI, Warne/GHW, classification, and prior-study values are frozen processed inputs. Those values are not silently reacquired, and the package does not claim to regenerate them from third-party raw sources.

The package also includes the existing upstream notebook builders and Notebooks 01--06. They document the source-to-processed transformations and can be used for an optional source reconstruction after the omitted raw files have been reacquired and hash-verified. That optional workflow is network-dependent where a notebook calls a provider and is not part of the offline acceptance gates.

Natural Earth Admin 0 Countries v5.1.1 is the exception: `data/external/natural_earth_10m_admin_0_countries_2026-07-23.zip` is public-domain material included in full, and Notebook 08 regenerates the coordinate-derived outputs from it during the offline run.

## Transformation entry points

| Source family | Expected reacquired file | Transformation entry point | Offline package status |
|---|---|---|---|
| NIQ v1.3.5 and v1.3.3 | filenames and hashes in `SOURCE_MANIFEST.csv` | `build_niq_notebook.py`; Notebooks 03, 04, 05, and 06 | raw archives omitted; reduced QNW and audit inputs frozen |
| Official HLO catalog and replication files | filenames and commit-locked hashes in `SOURCE_MANIFEST.csv` | `build_notebooks.py`, `build_hlo_validation_notebook.py`; Notebooks 01, 04, and 04b | raw files omitted; selected HLO inputs and validation reports frozen |
| World Development Indicators | WDI GDP and population ZIPs in `SOURCE_MANIFEST.csv` | `build_notebooks.py`; Notebooks 01, 04, and 06 | raw API archives omitted; required 2017 values frozen |
| World Bank/UN classifications | API/page/file snapshots in `SOURCE_MANIFEST.csv` | `build_economy_classification_notebook.py`; Notebook 04c | raw snapshots omitted; frozen 217-row World Bank reporting-economy table and classification ledgers included |
| Warne publisher supplement | `40806_2022_351_MOESM1_ESM.xlsx` | `build_prior_reproduction_notebook.py`; Notebook 04d; `build_v06_editorial_tables.py` | minimum five-column processed slice included |
| GHW Table A4 | article PDF named in `SOURCE_MANIFEST.csv` | `build_editorial_redesign_analysis.py` using the frozen per-economy Table A4 ledger | PDF omitted; exact values and layer audit frozen |
| Henss preprint | `6IBSKW.pdf` | `build_v06_editorial_tables.py` using the frozen Warne-derived slice and the documented exclusions | PDF omitted; displayed coefficients reconstructed |
| Parra--Kirkegaard | article PDF plus `iqresults.csv` and `iqs.csv` | `build_parra_kirkegaard_audit.py` | raw files omitted; hash-locked audit outputs frozen |
| Natural Earth v5.1.1 | included ZIP above | `build_methodological_sensitivity_notebook.py`; Notebook 08 | raw ZIP bundled and regenerated offline |

The GDP series is `NY.GDP.PCAP.PP.KD` (GDP per capita, purchasing-power-parity, constant 2021 international dollars). The [indicator page](https://data.worldbank.org/indicator/NY.GDP.PCAP.PP.KD) and [WDI metadata page](https://databank.worldbank.org/metadataglossary/world-development-indicators/series/NY.GDP.PCAP.PP.KD) document the series. The frozen API route is `https://api.worldbank.org/v2/en/indicator/NY.GDP.PCAP.PP.KD?downloadformat=csv`; it was retrieved July 20, 2026 at `2026-07-20T14:54:05.733113+00:00`. `SOURCE_MANIFEST.csv` preserves its SHA-256 `7ef5882fb5a4417977d2de1236fb99e47e3d6982c1ef2c6eb57550d56e665c87`.

## Henss correction

The v0.6 audit does reproduce the Henss displayed correlations to the reported two-decimal precision after applying the documented KHM, CUB, and PAK exclusions. Pairwise sample sizes are reconstructed from the frozen comparison slice because the displayed Henss table does not report them. The package therefore claims numerical reconstruction of the displayed coefficients, not independent recovery of every historical row-processing decision.

## Optional source reconstruction sequence

After reacquiring the omitted files at their notebook-expected locations and verifying the manifest hashes, rebuild and execute the relevant notebook in order. For example:

```text
python build_niq_notebook.py
python execute_notebook.py notebooks/03_documented_niq_acquisition_and_audit.ipynb --timeout=300
python build_hlo_validation_notebook.py
python execute_notebook.py notebooks/04b_transparent_HLO_aggregation_and_validation.ipynb --timeout=300
python build_economy_classification_notebook.py
python execute_notebook.py notebooks/04c_economy_classification_gate.ipynb --timeout=300
python build_analysis_notebook.py
python execute_notebook.py notebooks/04_analysis_dataset_construction.ipynb --timeout=300
```

The complete notebook sequence is 01, 02, 03, 04, 04b, 04c, 04d, 05, and 06. Builders create notebook definitions; notebooks perform the transformations. Hash mismatches mean source drift and must not be accepted silently.
