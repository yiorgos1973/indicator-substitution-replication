# Third-party notices

`SOURCE_MANIFEST.csv` is authoritative for source version, route, expected filename, hash, bundled form, and redistribution decision.

- Natural Earth Admin 0 Countries v5.1.1 is public-domain material. The original ZIP is included so the cited coordinate ledger and post hoc spatial outputs can be regenerated.
- World Bank WDI, HLO, and country-classification sources are represented by frozen processed values and source records. The package does not assert a broader redistribution license for each raw catalog item.
- The NIQ v1.3.3/v1.3.5 archives, the Warne publisher supplement, the GHW article PDF/Table A4, the Henss article, and the Parra--Kirkegaard article/OSF files are not redistributed as raw source archives. Exact acquisition routes and SHA-256 values are supplied.
- `data/processed/warne_national_scores.csv` in the package is a minimum five-column processed slice used only for confidential numerical verification. It is not the publisher workbook and carries no independent redistribution grant.
- NIQ, HLO, and WDI processed inputs are reduced to the rows and columns read by the included offline analyses; they are frozen inputs, not outputs regenerated from the omitted upstream datasets.
- The Parra acquisition builder checks the two recorded SHA-256 values and fails on source drift. It is optional and excluded from the offline core.

The optional source-reconstruction notebooks and their builders are included for transformation provenance. Their network-dependent acquisition stages are outside the one-command offline acceptance run; see `docs/Source_Acquisition_v0.6.md`.

Nothing in this archive changes the ownership or licensing of cited source material.
