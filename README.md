# Indicator substitution replication

Open **METHODOLOGY_Replication.ipynb** in Jupyter with Python 3.12 and choose **Run All**. The notebook downloads its pinned analysis code and public source data, reconstructs the samples, fits the country-holdout models, and exports four main tables, eight supplementary tables and three figures. The full model run previously took approximately 22 minutes; progress and estimated remaining time are printed.

The paper reports the results recorded for the submission package dated **5 October 2026**. Re-running the analysis with subsequently updated source data may yield different estimates.

`DATA_VERSION = "current"` downloads the current World Bank extracts. Set it to `"paper"` to use the attributed July/September 2026 World Bank snapshots. NIQ editions, the HLO replication commit and the earlier-study supplement are specified in the notebook support code. Each run records the retrieval time, source hashes, sample sizes and numerical differences from the archived results.

Outputs are saved under `replication_output/`: `tables/`, `figures/`, `model_run/`, `SOURCE_DOWNLOADS.csv`, `DATA_VERSION.json`, `comparisons/` and `VALIDATION.json`. Tables 1 and the methodological workflow are fixed definitions; the empirical tables and figures are regenerated from the reconstructed data and fitted predictions.

For an optional check of the archived results without refitting:

```bash
python -m pip install -r requirements.txt
python reproduce_all.py --output verification_run
```

The repository contains replication materials and archived scientific results. The manuscript and author records are held separately. Source attribution is in `PUBLIC_ACCESS_NOTICE.txt` and `data/wdi_frozen/ATTRIBUTION.csv`.
