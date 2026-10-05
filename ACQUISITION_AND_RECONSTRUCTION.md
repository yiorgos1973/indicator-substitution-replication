# Exact inputs for numerical refits

`ACQUIRED_INPUTS_REQUIRED.csv` identifies six additional frozen, project-derived files by relative path, byte count and SHA-256 hash. They are dependencies of the existing numerical workflow, rather than provider-download filenames. Their public sharing route remains unresolved.

Place separately authorized copies beneath `ACQUIRED_FILES`, preserving every recorded relative path. Run the `prepare` command in README.md before the `frozen` command. Preparation checks all six hashes and reports absent or changed inputs in `MISSING_INPUTS.csv`, stopping before fitting.

Archived table inspection, archived-result verification and empirical plotting use the included files. Raw-source acquisition and reconstruction are outside this minimal frozen-input package.
