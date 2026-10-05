# Indicator substitution replication

Data, code and supporting documentation for an output-specific audit of national-IQ and Harmonized Learning Outcomes. This repository supports archived-result verification, empirical-figure regeneration and documented full-refit routes. Manuscripts are maintained privately.

## Install and verify

Use **Python 3.12**. Run commands from the repository root and choose a new output directory for each run.

```bash
python -m pip install -r requirements.txt
python reproduce_all.py --mode verify --output verification_run
```

Verification checks file hashes, 24 archived core comparisons, 1,008 archived loss comparisons, 252 job identities, 22 matrices, 233,688 split rows and 6,666 fold groups. It also checks the primary sample of 53 observations in 38 countries. A successful run writes `verification_run/VALIDATION.json` with `status: PASS`. This route checks archived evidence and deterministic identities without fitting models.

## Regenerate empirical figures

```bash
python -m pip install -r requirements_figures.txt
python figures/plot_figures.py
```

This regenerates rank/displacement and normalized prediction-loss figures from the distributed CSVs, without fitting models. Outputs appear under `figures/`; the existing `INTELLIGENCE_Figure_*` filenames preserve the tested interface. For the optional `Reviewer_Walkthrough.ipynb`, install `requirements_notebook.txt`.

## Full frozen-input refits

Full refits require **ten exact, separately authorized project-derived files** listed in [ACQUIRED_INPUTS_REQUIRED.csv](ACQUIRED_INPUTS_REQUIRED.csv). Preserve their relative paths beneath `ACQUIRED_FILES`. These are derived dependencies, not files downloadable under those names from providers. Their public distribution route remains unresolved.

```bash
python reproduce_all.py --mode prepare --authorized-input-root ACQUIRED_FILES --output prepared_run
python reproduce_all.py --mode frozen --authorized-input-root ACQUIRED_FILES --output refitted_run
```

`prepare` verifies the required hashes and assembles the runtime without fitting. Missing or changed dependencies are recorded in `MISSING_INPUTS.csv`, and execution stops before fitting. `frozen` runs the unchanged numerical workflow once those exact files are supplied.

The complete author-held frozen-input workflow passed 252 jobs, 24 core comparisons, 1,008 loss comparisons and 11 independent empirical checks on 4 October 2026. The full run took approximately 21.45 minutes; actual duration depends on the machine.

## Source reconstruction and attribution

Full upstream reconstruction additionally requires the documented original **93-file source workspace**, construction code and dependency records. See [ACQUISITION_AND_RECONSTRUCTION.md](ACQUISITION_AND_RECONSTRUCTION.md), [OFFICIAL_SOURCE_REGISTRY.csv](OFFICIAL_SOURCE_REGISTRY.csv) and `runtime_payload/SOURCE_ALLOWLIST.csv`. The files under `reconstruction_reference/` preserve the existing transformation code for that separate workspace. Current provider downloads can differ from the hash-pinned historical versions.

Frozen HLO and WDI data retain their recorded provider terms, attribution and notices. Original summaries, code and supporting records retain the validated scientific payload. [PUBLIC_ACCESS_NOTICE.txt](PUBLIC_ACCESS_NOTICE.txt) records the current publication scope; [PERMISSIONS_CURRENT.csv](PERMISSIONS_CURRENT.csv) records included materials and ten omissions. Historical notices retain their original dates and scope. Public access does not assign a blanket reuse license.
