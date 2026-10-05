# NIQ and HLO reproducibility supplement

This supplement lets a reader inspect the analysis samples and reproduce the main comparisons in the accompanying manuscript. It contains frozen country-level analysis inputs, the archived numerical code, expected results and a portable reproduction command.

**Distribution status:** submission candidate for author review. The computational checks have passed. Permission to distribute the derived NIQ, learning and prior-study values with a journal submission still needs to be documented in `DATA_PERMISSIONS.csv`. This candidate has not been sent to a journal or made public.

## Start here

The manuscript compares three things: selected published correlations, the country-level outputs produced by two indicators, and prediction between historical measurements. The fixed-sample comparison has 66 economies; the GDP comparison has 65. The primary historical sample has 53 observations in 38 countries.

| Manuscript material | Files | Reproduction coverage |
|---|---|---|
| Table 2 and source audit | `original/source_audit/` | Published/reconstructed coefficients, sample identities and available frozen comparison values. The main command does not independently reacquire or reconstruct the published source archives. |
| Table 3 and fixed-sample comparison | `original/data/core/`, `original/tables/T_GDP_CONTRAST_source.csv` | Recalculates score/rank associations, rank displacement, tail overlap, pair reversals, GDP correlations, Williams test, Zou interval and the 99,999-draw paired BCa interval. |
| Table 4 and historical prediction | `original/data/prediction/model_matrices.csv` | Refits both directions with the archived nested country-held-out routines. Default mode covers 20 primary and benchmark jobs. |
| Historical sensitivities and country deletions | `original/results/model_runs/phase_b_20260930_13fb5775_v2/` | Full mode also refits 80 sensitivity jobs and 152 country-deletion jobs, for 252 jobs in total. |
| Indicator construction and coverage | `original/protocol/`, `original/tables/` | Documents the construction rules and retained coverage summaries. The reproduction starts from the resulting frozen analysis inputs. |

Each data directory includes the available column dictionaries. `MANIFEST.csv` lists every packaged file with its byte count and SHA-256 hash. `SOURCE_PROVENANCE.csv` links unchanged archived files to their commit and repository-relative source path.

## Run the checks

Use Python 3.12 and create a separate environment. From the extracted supplement directory:

```text
python -m venv .venv
```

Activate it with `.venv\Scripts\activate` on Windows or `source .venv/bin/activate` on macOS/Linux, then run:

```text
python -m pip install -r requirements.txt
python reproduce.py --output reproduced_primary
python reproduce.py --full --output reproduced_full
```

Use a fresh output directory for each run. Dependency installation needs Internet access; the calculations use the included files and make no network requests. The script checks file hashes and the archived protocol hash before calculating. A successful run writes `VALIDATION.json` with `status: PASS`, comparison CSVs and the reconstructed primary held-out predictions.

The tested full run took about two minutes in the preparation environment. Runtime depends on the computer. It matched 24 core/BCa quantities and four loss summaries for each of 252 prediction jobs: 1,008 prediction-metric comparisons. Absolute tolerance is 1e-10 and relative tolerance is 1e-8. The tested environment is recorded in `verification/VALIDATION.json`; `requirements.txt` specifies its numerical packages. The original production environment remains recorded in `original/code/requirements-phase-b.txt` and the original run settings. The latter archival file includes a Python-version line and is not the pip-install entry point.

## What the checks establish

The command reproduces calculations from frozen derived inputs. It uses the original numerical routines and compares their outputs with the archived results. This checks execution, sample handling and agreement with the archive in the recorded environment. The full run covers the primary models, OLS/Lasso benchmarks, all retained Ridge sensitivities and the country deletions.

Upstream acquisition, psychometric corrections, HLO harmonization, annual WDI downloads and record-to-window construction are outside this portable run. The raw provider files and the full record-level ledgers remain outside the supplement. The historical source-acquisition guide and manifest in `original/docs/` identify the recorded sources and acquisition routes; that guide describes the older full repository layout. Its notebook commands are not commands for this compact supplement. No missing source archive should be replaced silently with a newer download.

The source-specific audit, GDP coverage/weighting sensitivities, and temporal coverage tables are retained evidence. They are not all recalculated by this command. Calibration, predictive ranks and tails are not separate validation targets in the portable loss-comparison check. The original validation summaries are included as provenance and remain distinct from the new reproduction report.

## Protocol and review status

The numerical protocol was approved on 30 September 2026. `PROTOCOL_STATUS.json` summarizes the separately inspected approval record without the author's identity. Some unchanged protocol documents retain their earlier “pending approval” headings because they are the exact documents subsequently approved. Their date and wording record that sequence. The manuscript's simpler design account remains authoritative for readers: predictive rules were fixed before predictive fitting, after the earlier comparisons were known.

The included files are an explicit selection from the validated author archive. Author identity and approval correspondence, private reviews, local paths, notebooks, raw source archives and record-level measurement ledgers are excluded. `EXCLUSIONS.csv` records those choices. The historical review-only notice is retained for context; it does not itself establish permission for the newly assembled historical inputs. Complete the permission decisions in `DATA_PERMISSIONS.csv` before uploading the candidate to a journal. A public release requires its own documented data and code licensing decisions.
