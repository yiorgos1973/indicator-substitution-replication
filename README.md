# Indicator substitution replication

This package contains the data, existing numerical code and display inputs needed to verify the paper's results, regenerate its empirical graphs and run the documented frozen-input refits.

Use **Python 3.12** and run commands from this directory. Choose a new output directory for each verification or refit.

```bash
python -m pip install -r requirements.txt
python reproduce_all.py --mode verify --output verification_run
python -m pip install -r requirements_figures.txt
python figures/plot_figures.py
```

Verification checks archived numerical comparisons, sample identities, country folds and file hashes. Plotting recreates the two empirical graphs under `figures/`. `figures/METHODOLOGY_Workflow.svg` is the editable methodological schematic.

## Tables

`tables/Table_1.csv` through `Table_4.csv` contain the exact archived main-table cells. The supplementary displays use these existing result ledgers and rules:

| Table | Source and selection |
| --- | --- |
| S1 | `runtime_payload/expected_empirical/METRICS.csv`: PRIMARY; MSE/RMSE to three decimals |
| S2 | Same ledger: scenarios other than PRIMARY and PRIMARY_DELETE; three decimals |
| S3 | Same ledger: PRIMARY_DELETE; pivot MSE by country/model and form M1 minus M3 and M2 minus M3; three decimals |
| S4 | `tables/archived/PHASE_B_RANK_SUMMARY_SOURCE.csv`: PRIMARY; rank loss to six decimals |
| S5 | `runtime_payload/expected_empirical/CALIBRATION.csv`: PRIMARY, ridge/reference; three decimals and explicit undefined values |
| S6 | `tables/archived/TAIL_SUMMARIES.csv`: primary Ridge jobs; counts, overlap and flags |
| S7 | `tables/archived/T_GDP_CONTRAST_source.csv`: estimates/limits to six decimals, with interval/test labels |
| S8 | `tables/archived/T_TEMPORAL_COVERAGE_source.csv`: first six columns |

These are archived table values and display rules. The existing public entry point verifies results; it does not export formatted tables.

## Numerical refits

Refits require the **six exact additional files** listed in `ACQUIRED_INPUTS_REQUIRED.csv`, preserving their relative paths below `ACQUIRED_FILES`. Their public distribution route remains unresolved. See [input instructions](ACQUISITION_AND_RECONSTRUCTION.md).

```bash
python reproduce_all.py --mode prepare --authorized-input-root ACQUIRED_FILES --output prepared_run
python reproduce_all.py --mode frozen --authorized-input-root ACQUIRED_FILES --output refitted_run
```

`prepare` checks hashes without fitting. `frozen` reruns the unchanged 252-job workflow and compares numerical ledgers. The complete author-held run passed in approximately 21.45 minutes; the engine reports progress and estimated time.

Attribution and access terms are in [PUBLIC_ACCESS_NOTICE.txt](PUBLIC_ACCESS_NOTICE.txt).
