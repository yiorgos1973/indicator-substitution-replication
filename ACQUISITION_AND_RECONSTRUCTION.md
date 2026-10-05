# Acquisition and reconstruction: exact scope

1. **Distributed evidence now:** verify mode, the executed walkthrough, all original result summaries, sample identities and folds. Frozen HLO/WDI sources are included with attribution. This route requires no NIQ or Warne acquisition, no private author checkout and no network access after dependency installation.
2. **Exact downstream refitting:** acquire the ten files named in ACQUIRED_INPUTS_REQUIRED.csv on an authorized basis and preserve their relative paths. They are frozen project-derived dependencies, not files downloadable by those names from a provider. The full authorized capsule in the author archive contains them. An author/provider or institutional rights decision is required before the authors can supply that capsule or its omitted files to reviewers. Do not claim that the ten derived-file hashes are hashes of the source ZIPs. `--mode prepare` verifies their exact bytes and both original manifests; `--mode frozen` invokes the unchanged validated scientific workflow. The preparation route was tested locally with the existing authorized author inputs, but those inputs do not accompany the distributed archive.
3. **Provider reconstruction:** OFFICIAL_SOURCE_REGISTRY.csv lists the 21 historical sources, exact releases/commit, official URLs, filenames, byte counts and hashes. It is a source registry, not a permission grant. WDI_FROZEN_ATTRIBUTION.csv additionally records all eleven included July/September archives, members, dates and hashes. HLO_FILE_LINEAGE.csv maps the catalogue workbook and the two exact Stata inputs. Current WDI downloads are mutable: use the included frozen files, not a new API response. NIQ editions v1.3.5 and v1.3.3 and the Warne workbook must match the registry; an absent or revised source stays missing.

Official unresolved source routes:
- NIQ v1.3.5: https://viewoniq.org/wp-content/uploads/2023/10/NIQ-DATASET-V1.3.5.zip ; SHA-256 e786460da3ff8a03065251160168b88bce8884321c1427901492e8842731666a ; 3,826,519 bytes.
- NIQ v1.3.3: https://viewoniq.org/wp-content/uploads/2019/07/NIQ-DATASET-V1.3.3.zip ; SHA-256 bfd444e1c7ad4c17a50c7b236f50d379464dd27fd58ba9be49a0850b27a14888 ; 3,994,559 bytes.
- Warne (2023), DOI 10.1007/s40806-022-00351-y, publisher workbook: https://static-content.springer.com/esm/art%3A10.1007%2Fs40806-022-00351-y/MediaObjects/40806_2022_351_MOESM1_ESM.xlsx ; SHA-256 9eb05090b71a4fd728e13872e788eed9bd382594871265a941acedc297c4989d ; 424,047 bytes.

The included reconstruction_reference files preserve the actual existing transformation code. They are not a closed raw-source reproduction environment. The historical sequence, in an authorized source workspace with its dependency records and exact frozen sources, is:
```
python build_niq_notebook.py
python execute_notebook.py notebooks/03_documented_niq_acquisition_and_audit.ipynb --timeout=300
python build_hlo_validation_notebook.py
python execute_notebook.py notebooks/04b_transparent_HLO_aggregation_and_validation.ipynb --timeout=300
python build_economy_classification_notebook.py
python execute_notebook.py notebooks/04c_economy_classification_gate.ipynb --timeout=300
python build_analysis_notebook.py
python execute_notebook.py notebooks/04_analysis_dataset_construction.ipynb --timeout=300
python build_prior_reproduction_notebook.py
python execute_notebook.py notebooks/04d_prior_niq_hlo_reproduction.ipynb --timeout=300
```
These legacy acquisition notebooks can contact mutable providers and depend on earlier frozen classification, audit and construction files. Merely executing their builder writes a notebook; it does not reconstruct the data. Inspect the included code and exact registry first; do not run acquisition cells to replace already frozen inputs. The historical notebook filenames and source-workspace command sequence are supplied for traceability, not asserted to run from this partial distribution.

The approved Phase B source-workspace rebuild uses the 93 hash-pinned sources in runtime_payload/SOURCE_ALLOWLIST.csv, including measurement/context outputs and private approval provenance. Its exact command, in the original authorized workspace, is:
```
python rewrite/validation/phase_b_clean_rebuild/clean_rebuild_phase_b.py --repository-root . --workspace NEW_EMPTY_WORKSPACE --execute-full --production-run rewrite/analysis/prediction/phase_b_20260930_13fb5775_v2 --production-pass-record rewrite/validation/phase_b_empirical/production_phase_b_20260930_13fb5775_v2/VALIDATION_STATUS.json --run-id phase_b_20260930_13fb5775_v2
```
That route is historically attested, not newly executed here and not advertised as self-contained in this reviewer ZIP. The former full capsule's original/candidate dependency manifests remain unchanged. They name omitted files deliberately; only the top-level manifest enumerates the actually distributed payload.

New validation checks all local archived source hashes; the original HLO sequential filtering rule against the exact filtered input; frozen WDI selection/long-panel values; and authorized full-capsule preparation. This does not independently reproduce every NIQ psychometric correction, source study, coverage sensitivity or earlier published sample identity. Source-audit and upstream construction results remain archived. No permission or public access link is manufactured.
