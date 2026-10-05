# Source acquisition and reconstruction

Run `METHODOLOGY_Replication.ipynb` using Python 3.12. Its setup cell downloads the pinned public analysis code and installs the numerical requirements. The source cell downloads NIQ v1.3.5 and v1.3.3, the Measuring Human Capital HLO replication files, the published Warne supplementary workbook, and nine World Bank indicators.

The notebook reconstructs NIQ and HLO aggregations, country samples, historical cells, contextual predictors and model matrices. It uses the original estimation, weighting, sample rules, country folds and tuning grids. The default uses current World Bank downloads; `DATA_VERSION = "paper"` selects the attributed archived World Bank extracts.

The paper's dated results remain the reference. Provider revisions may change subsequent estimates. Download hashes, retrieval dates, sample sizes and table/ledger differences are written to the run directory. No separate author-held input files are required by this workflow.
