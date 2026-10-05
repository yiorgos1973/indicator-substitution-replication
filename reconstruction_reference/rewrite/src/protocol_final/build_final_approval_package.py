"""Build the numerically complete author-approval package without empirical fitting."""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn


ROOT = Path(__file__).resolve().parents[3]
ARCHIVE = ROOT / "rewrite" / "protocol" / "archive" / "author_approval_pre_numerical_addendum_2026-09-30"
CONTEXT = ROOT / "rewrite" / "analysis" / "protocol_final"
OUTPUT = ROOT / "rewrite" / "protocol" / "author_approval"
SYNTHETIC = ROOT / "rewrite" / "validation" / "protocol_final_synthetic"

PREDICTORS = [
    "SE.PRM.ENRR",
    "SE.SEC.ENRR",
    "SE.XPD.TOTL.GD.ZS",
    "SH.DYN.MORT",
    "SP.DYN.TFRT.IN",
    "SP.URB.TOTL.IN.ZS",
    "IT.NET.USER.ZS",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(payload: dict) -> str:
    value = dict(payload)
    value.pop("protocol_hash", None)
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def records(frame: pd.DataFrame) -> list[dict]:
    return json.loads(frame.to_json(orient="records"))


def file_record(path: Path) -> dict[str, object]:
    return {
        "repository_relative_path": path.relative_to(ROOT).as_posix(),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }


def build_protocol() -> dict:
    previous_path = ARCHIVE / "protocol_for_author_approval.json"
    protocol = json.loads(previous_path.read_text(encoding="utf-8"))
    scenario_summary = pd.read_csv(CONTEXT / "scenario_summary.csv")
    input_hashes = pd.read_csv(CONTEXT / "input_hashes.csv")

    protocol["specification_id"] = "niq_hlo_integrated_protocol_numerically_complete_v2_2026-09-30"
    protocol["specification_status"] = "fully_specified_pending_author_approval"
    protocol["protocol_status"] = "fully_specified_pending_author_approval"
    protocol["execution_scope"] = "protocol_preparation_and_synthetic_validation_only"
    protocol["empirical_modelling_approved"] = False
    protocol["authority"] = {
        "accepted_scientific_plan": "docs/rewrite/docs/NIQ_HLO_Revision_Plan.md",
        "implementation_context": "docs/rewrite/docs/Implementation_Brief.md",
        "master_runbook": "rewrite/docs/CODEX_MASTER_RUNBOOK.md",
        "integrated_review": "rewrite/protocol/agent2/Protocol_Review.md Section 4",
        "numerical_amendment": "rewrite/protocol/Numerical_Decisions_Addendum.md version 1.0",
        "proposal_is_not_approval": True,
        "selection_basis": "Prespecified scientific meaning and outcome-blind coverage only; no empirical predictive result was inspected.",
    }
    protocol["predictor_vector"] = {
        "context_feature_order": PREDICTORS,
        "combined_model_order": ["required_other_indicator", *PREDICTORS],
        "retain_all_seven": True,
        "excluded": [
            "GDP",
            "population",
            "country identifiers",
            "region",
            "calendar year or window indicators",
            "ranks",
            "coverage or missingness flags",
            "uncertainty or audit quantities",
        ],
        "nonlinear_terms": False,
        "interactions": False,
        "feature_selection": False,
        "positivity_constraints": False,
        "prediction_clipping": False,
    }
    protocol["historical_feature_construction"] = {
        "primary": "five_year_block_preceding",
        "annual_observation_unit": "one unique calendar year per country-feature",
        "duplicate_rule": "duplicate country-feature-year rows require explicit source reconciliation and are never double-counted",
        "coverage_rule": "5 * observed_year_count >= 4 * requested_year_count",
        "minimum_observed_years_formula": "(4 * requested_year_count + 4) // 5",
        "thresholds": {"5_year": 4, "10_year": 8, "12_year": 10},
        "usable_mean": "arithmetic mean of available requested annual values after threshold passes",
        "missing_values_are_zero": False,
        "retain_requested_and_observed_year_ledger": True,
        "assessment_specific_fixed_contribution": {
            "positive_weight_requirement": "every positive-weight target contribution must have a usable history for the feature",
            "failure_action": "mark combined context feature missing",
            "renormalize_context_weights": False,
            "target_contributions_changed": False,
            "direction_specific_schedule": True,
        },
        "historical_renormalized_candidate": "preserved_and_labelled_not_executed",
    }
    protocol["row_and_missingness_rules"] = {
        "minimum_usable_context_features": 5,
        "total_context_features": 7,
        "maximum_missing_context_features": 2,
        "same_planned_rows_for_M0_to_M3": True,
        "required_target": "observed_only_never_imputed",
        "required_other_indicator": "observed_only_never_imputed",
        "wdi_missingness_changes_target_contributions": False,
        "context_imputation": {
            "method": "ordinary_unweighted_training_subset_median",
            "even_count_rule": "mean_of_two_middle_order_statistics",
            "fit_on_training_only": True,
            "apply_training_median_unchanged_to_training_and_holdout": True,
            "missingness_indicators": False,
            "all_missing_training_feature": "structurally_infeasible_fit",
        },
    }
    protocol["validation"] = {
        "outer_design": "leave_one_country_out",
        "outer_order": "ISO3_ascending_ASCII",
        "all_country_periods_held_out_together": True,
        "repeated_outer_splits": False,
        "inner_folds": 5,
        "inner_group": "country",
        "inner_hash_string": "NIQ-HLO-inner|20260930|<ISO3>",
        "inner_hash": "SHA-256 over UTF-8",
        "inner_order": "hex_digest_ascending_then_ISO3_ascending_ASCII",
        "inner_assignment": "zero_based_sorted_position_modulo_5",
        "balanced_unit": "country_count_not_row_count",
        "same_allocation_for_models_on_same_country_set": True,
        "rebuild_after_scenario_or_deletion_change": True,
        "save_every_realized_allocation": True,
        "minimum_scenario_countries": 7,
        "below_minimum_action": "mark_infeasible_do_not_reduce_folds",
    }
    protocol["weighting"] = {
        "training_weight": "N / (C * m_i)",
        "training_weight_sum": "N",
        "training_country_total": "N / C",
        "recalculate_within_each_training_subset": True,
        "evaluation_weight": "1 / (C * m_i)",
        "evaluation_weight_sum": 1,
        "evaluation_country_total": "1 / C",
        "use_for_fitting_and_predictor_target_scaling": True,
        "prohibited_additional_weights": [
            "population",
            "psychometric sample size",
            "QN factor",
            "number of tests",
            "measurement precision",
        ],
    }
    protocol["preprocessing"] = {
        "order": ["training_median_imputation", "weighted_standardization"],
        "weighted_mean": "sum(weight * x) / sum(weight)",
        "weighted_variance": "sum(weight * (x - mean)**2) / sum(weight)",
        "variance_divisor": "sum_of_weights_population_form",
        "target_standardized_with_same_training_rule": True,
        "predictions_inverse_transformed_before_scoring": True,
        "constant_threshold": "weighted_sd <= 1e-12 * max(1, abs(weighted_mean))",
        "constant_predictor_action": "preserve_schema_position_set_standardized_train_and_holdout_column_to_zero_and_flag",
        "constant_target_action": "predict_weighted_training_mean_and_leave_selected_alpha_null",
        "inner_constant_target_action": "same_constant_predictions_score_every_alpha",
    }
    protocol["numerical_execution"] = {
        "array_storage": "dense",
        "dtype": "float64",
        "row_order": ["ISO3_ascending_ASCII", "window_start_ascending", "window_end_ascending"],
        "unique_row_key": ["ISO3", "window_start", "window_end"],
        "intercept": "unpenalized",
        "thread_pools": 1,
        "environment": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
        },
    }
    protocol["model_comparison"] = {
        "M0": {
            "predictors": [],
            "fit": "equal_country_weighted_outer_training_target_mean_in_original_units",
            "role": "score_error_reference_only",
            "rank_baseline": False,
        },
        "M1": {"predictors": ["required_other_indicator"]},
        "M2": {"predictors": PREDICTORS},
        "M3": {"predictors": ["required_other_indicator", *PREDICTORS]},
        "ridge": {
            "role": "primary",
            "objective": "sum(weight * residual**2) + alpha * sum(beta**2)",
            "solver": "svd",
            "fit_intercept": True,
            "positive": False,
            "grid": [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000],
        },
        "weighted_ols": {
            "role": "primary_sample_benchmark",
            "design": "sqrt(weight) multiplied augmented [1, Z] and standardized target",
            "solver": "SVD_minimum_norm_least_squares",
            "rcond": 1e-12,
            "record": ["numerical_rank", "singular_values", "rank_deficiency_warning"],
        },
        "lasso": {
            "role": "primary_sample_benchmark",
            "objective": "sum(weight * residual**2) / (2 * N) + alpha * sum(abs(beta))",
            "fit_intercept": True,
            "selection": "cyclic",
            "positive": False,
            "warm_start": False,
            "precompute": False,
            "tol": 1e-8,
            "max_iter": 100000,
            "grid": [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1],
            "retry": {
                "allowed_once_on_convergence_warning": True,
                "fresh_estimator": True,
                "max_iter": 1000000,
                "persistent_nonconvergence": "failed_fit",
            },
            "record": ["attempts", "n_iter", "dual_gap"],
        },
        "outer_performance_algorithm_selection": False,
    }
    protocol["tuning"] = {
        "scope": "within_each_outer_training_set",
        "candidate_prediction_units": "original_target_units_after_inverse_transform",
        "candidate_score": "mean_of_validation_country_MSE_after_concatenating_all_five_inner_holdouts",
        "fold_MSE_unweighted_average_prohibited": True,
        "tie_rule": "candidate <= L_min + 1e-10 + 1e-8 * max(1, abs(L_min))",
        "tie_choice": "largest_alpha",
        "all_inner_folds_required": True,
        "partial_candidate_scoring": False,
        "no_valid_candidate_action": "model_run_incomplete",
        "outer_refit": "rebuild_preprocessing_and_estimator_from_scratch_on_full_outer_training_set",
        "grid_boundary_action": "report_without_expanding_grid",
    }
    protocol["evaluation"] = {
        "score_metrics": {
            "MAE": "mean_over_countries(mean_over_country_cells(abs(pred-y)))",
            "MSE": "mean_over_countries(mean_over_country_cells((pred-y)**2))",
            "RMSE": "sqrt(MSE)",
            "bias": "mean_over_countries(mean_over_country_cells(pred-y))",
            "unitless_MSE_improvement_vs_M0": "1 - MSE_model / MSE_M0; null when MSE_M0 is zero",
        },
        "direction_units_kept_separate": True,
        "global_test_target_standardization": False,
        "calibration": {
            "equation": "observed_y = intercept + slope * outer_held_out_prediction",
            "fit": "equal_country_weighted_least_squares_in_original_target_units",
            "pooled": "one_per_direction_labelled_as_mixing_periods",
            "period_specific_minimum_countries": 3,
            "period_specific_prediction_variation": "must_exceed_constant_threshold",
            "undefined_slope": "null_with_reason",
            "adjust_scored_predictions": False,
            "p_values_or_standard_errors": False,
        },
    }
    protocol["rank_order_and_tail_evaluation"] = {
        "scope": "within_scenario_period_direction_on_identical_evaluated_country_set",
        "rank_one": "highest",
        "ties": "average_ranks_for_exact_full_precision_ties",
        "pre_rank_rounding": False,
        "normalized_absolute_displacement": "abs(rank_prediction-rank_target)/(n-1)",
        "aggregation": "within_country_across_periods_then_across_countries",
        "one_country_period": "rank_undefined_but_retained_for_score_metrics",
        "raw_source_and_fitted_M1_separate": True,
        "improvement": "loss_baseline - loss_M3",
        "percentage_improvement": "100 * difference / loss_baseline only when loss_baseline > 0",
        "correlations": ["Spearman_per_period", "Kendall_tau_b_per_period"],
        "constant_correlation": "null",
        "strict_reversals": "exclude_pairs_tied_on_either_variable_and_report_numerator_denominator_fraction",
        "correlation_p_values": False,
        "tail_percentiles": [10, 90],
        "tail_interpolation": "linear",
        "tail_inclusion": "at_or_beyond_own_variable_threshold",
        "tail_outputs": ["target_count", "prediction_count", "intersection", "target_recall", "Jaccard"],
        "tail_small_group_flag": "either_group_count_below_3",
        "all_score_tie_flag": True,
        "M0_rank_or_tail": False,
        "flag_repeated_intercept_only_models": True,
    }
    protocol["uncertainty"] = {
        "prediction_intervals": False,
        "cross_validation_significance_tests": False,
        "new_coefficient_inference": False,
        "original_GDP_uncertainty": "unchanged",
        "country_deletion": {
            "scope": "primary_five_year_HLO_to_P_Ridge_M0_to_M3_only",
            "delete": "all_cells_of_each_primary_sample_country_in_turn",
            "refit": "full_nested_LOCO_including_preprocessing_tuning_and_refitting",
            "recalculate": ["evaluation_weights", "within_period_ranks", "M3_minus_baseline_comparisons"],
            "summary": ["minimum", "median", "maximum", "number_of_sign_changes"],
            "interpretation": "sensitivity_not_confidence_interval",
            "other_directions_estimators_or_sensitivities": False,
        },
    }
    protocol["finite_analysis_manifest"] = {
        "primary": {
            "directions": ["HLO_to_P", "P_to_HLO"],
            "models": ["M0", "M1", "M2", "M3"],
            "estimators_M1_to_M3": ["Ridge", "weighted_OLS", "Lasso"],
        },
        "ridge_only_sensitivities_both_directions_M0_to_M3": {
            "S10": "ten_year_block_preceding",
            "S12": "twelve_year_block_preceding",
            "SA5": "five_year_assessment_specific_fixed_contribution",
            "SCC": "complete_seven_feature_primary_cells_no_imputation",
            "SCOMP": "remove_BRA_CAN_KAZ_MEX_SRB_USA_ZAF",
            "STERM": "remove_2015_2017_cells",
            "S4": "four_year_targets_2000_2003_2004_2007_2008_2011_2012_2015_2016_2017_with_preceding_five_year_context",
        },
        "matched_intersection_rules": {
            "S10_S12_SA5": "freeze exact primary-and-alternative eligible intersection before fitting; refit both constructions with identical splits and weights",
            "SA5": "direction_specific_intersections",
            "SCC_SCOMP_STERM": "conditional_refits_on_restricted_samples",
            "S4": "within_scheme_comparisons_only",
        },
        "diagnostics_without_new_models": [
            "single_vs_multiple_study",
            "date_uncertain_vs_exact_date",
            "HLO_component_depth",
            "number_of_imputed_features",
            "target_metadata_availability",
            "coefficients",
            "selected_alphas",
            "grid_edge_selections",
            "convergence_flags",
            "calibration",
        ],
        "disabled": [
            "life_expectancy_replacement",
            "same_window_predictive_models",
            "new_forward_period_tests",
            "bootstrap_intervals",
            "synthetic_missing_target_experiments",
            "other_algorithms",
            "new_variables",
        ],
    }
    protocol["failure_handling"] = {
        "planned_test_cells_identical_for_M0_to_M3": True,
        "valid_fit": "finite_complete_training_only_pipeline_with_declared_convergence_status",
        "failed_rows_dropped_for_one_model_only": False,
        "required_Ridge_failure": "mark_scenario_incomplete_keep_partial_diagnostics_no_primary_paired_loss",
        "OLS_or_Lasso_failure": "mark_benchmark_incomplete_do_not_suppress_Ridge_or_substitute_estimator",
        "sensitivity_failure": "report_without_invalidating_unrelated_scenarios",
        "below_seven_countries": "infeasible_from_coverage",
        "numerical_retries": "only_predeclared_Lasso_retry",
        "implementation_bug": "versioned_change_log_full_affected_rerun_independent_verification",
        "post_result_scientific_change": "dated_amendment_labelled_post_result",
    }
    protocol["measurement_gate"] = {
        "required_checks": {
            "MEAS_01": "source_and_exact_row_identity",
            "MEAS_02": "P_and_HLO_construction_weights",
            "MEAS_03": "date_and_period_boundaries",
            "MEAS_04": "duplicate_and_overlap_audit",
            "MEAS_05": "composition_adjustment_exceptions",
            "MEAS_06": "HLO_component_coverage_and_eligibility",
            "MEAS_07": "historical_context_endpoints_and_coverage",
            "MEAS_08": "target_and_predictor_missingness",
            "MEAS_09": "absence_of_GDP_identifier_rank_outcome_derived_audit_and_coverage_leakage",
        },
        "status": "matching_protocol_hash_and_all_passed_required_before_empirical_execution",
    }
    protocol["approval_gate"] = {
        "current_state": "closed",
        "empirical_modelling_approved": False,
        "self_approval_prohibited": True,
        "approval_record": "separate_author_controlled_record_referencing_exact_protocol_hash",
        "matching_measurement_record": "must_reference_same_hash_and_pass_MEAS_01_through_MEAS_09",
        "hash_design": "canonical_UTF8_JSON_recursively_sorted_keys_compact_separators_SHA256_excluding_only_protocol_hash",
        "gate_must_refuse_when": [
            "approval_status_is_not_approved",
            "approval_scope_is_not_empirical_prediction",
            "approval_hash_differs",
            "measurement_record_hash_differs",
            "any_required_measurement_check_is_absent_or_not_passed",
        ],
    }
    protocol.pop("unresolved_conventions", None)
    protocol.pop("current_data_reconciliation", None)
    protocol.pop("prespecified_sensitivities", None)
    protocol.pop("sensitivity_limits", None)
    protocol.pop("failure_and_exclusion_rules_resolved_by_review", None)
    protocol["remaining_scientific_or_numerical_choices"] = []
    protocol["remaining_author_action"] = "explicitly approve this exact new protocol hash in a separate record"
    protocol["data_reconciliation"] = {
        "selection_was_outcome_blind": True,
        "scenario_summary": records(scenario_summary),
        "scenario_window_summary": records(
            pd.read_csv(CONTEXT / "scenario_window_summary.csv")
        ),
        "coverage_threshold_correction_count": int(
            len(pd.read_csv(CONTEXT / "coverage_threshold_corrections.csv"))
        ),
        "input_hashes": records(input_hashes),
        "target_or_source_indicator_imputation_performed": False,
        "context_imputation_performed": False,
        "empirical_modelling_performed": False,
    }
    protocol["frozen_scientific_artifacts"] = {
        "sample_manifest": file_record(OUTPUT / "sample_manifest.csv"),
        "coverage_and_imputation_summary": file_record(
            OUTPUT / "coverage_and_imputation_summary.csv"
        ),
        "coverage_threshold_corrections": file_record(
            CONTEXT / "coverage_threshold_corrections.csv"
        ),
        "scenario_summary": file_record(CONTEXT / "scenario_summary.csv"),
        "scenario_window_summary": file_record(
            CONTEXT / "scenario_window_summary.csv"
        ),
    }
    protocol["governing_document_hashes"] = [
        file_record(ROOT / "docs" / "rewrite" / "docs" / "NIQ_HLO_Revision_Plan.md"),
        file_record(ROOT / "docs" / "rewrite" / "docs" / "Implementation_Brief.md"),
        file_record(ROOT / "rewrite" / "docs" / "CODEX_MASTER_RUNBOOK.md"),
        file_record(ROOT / "rewrite" / "protocol" / "agent2" / "Protocol_Review.md"),
        file_record(ROOT / "rewrite" / "protocol" / "Numerical_Decisions_Addendum.md"),
    ]
    protocol["provenance"] = {
        "predecessor_protocol_hash": "113537313ea06ebb91dc718a5e7498b7d37aa3c52588c6f7b279ed34567e28f1",
        "predecessor_protocol_file": file_record(previous_path),
        "predecessor_package_archive": ARCHIVE.relative_to(ROOT).as_posix(),
        "addendum_sha256": sha256(
            ROOT / "rewrite" / "protocol" / "Numerical_Decisions_Addendum.md"
        ),
        "git_branch_at_build": "wdi-and-code-only-2026-09-21",
        "git_commit_at_build": "e01a8fb5239766bf8b97077b76863490c8b61642",
    }
    implementation_paths = sorted(
        path
        for path in (ROOT / "rewrite" / "src" / "protocol_final").glob("*.py")
        if path.name != Path(__file__).name
    )
    test_paths = sorted(SYNTHETIC.glob("*.py"))
    protocol["implementation_identifiers"] = {
        "package_builder": file_record(Path(__file__)),
        "numerical_convention_modules": [file_record(path) for path in implementation_paths],
        "synthetic_test_modules": [file_record(path) for path in test_paths],
        "empirical_runner_executed": False,
    }
    protocol["implementation_readiness"] = {
        "data_construction_rules_implemented_and_checked": True,
        "numerical_conventions_encoded_and_synthetically_tested": True,
        "empirical_execution_allowed": False,
        "reason": "Author approval and a matching passed measurement-gate record are still absent.",
    }
    protocol["protocol_hash_algorithm"] = "sha256_canonical_json_v1_excluding_protocol_hash"
    protocol["protocol_hash"] = canonical_hash(protocol)
    return protocol


def readable_document(protocol: dict) -> str:
    summary = pd.DataFrame(protocol["data_reconciliation"]["scenario_summary"])
    summary_table = summary.to_markdown(index=False)
    environment = protocol["numerical_execution"]["environment"]
    return f"""# NIQ–HLO integrated protocol for author approval

Status: **fully specified proposal pending author approval**  
Empirical modelling approved: **no**  
Scientific specification hash: `{protocol['protocol_hash']}`

## Author action

The numerical addendum has been integrated. The scientific and numerical protocol is complete; no empirical predictive model was fitted. The only remaining author action is to approve or reject the exact hash above in a separate record. Validation success does not constitute approval.

## Scope and fixed sample construction

The primary analysis uses period-specific P and HLO for 2000–2004, 2005–2009, 2010–2014 and partial 2015–2017, with five calendar years of block-preceding WDI context. `HLO_to_P` is primary and `P_to_HLO` is secondary. P is the period-specific QN-weighted psychometric score, not exact published QNW.

The ordered context vector is: `{'`, `'.join(PREDICTORS)}`. In M3 the required other indicator comes first. GDP, population, identifiers, region, calendar/window indicators, ranks, coverage flags and uncertainty/audit fields are excluded.

Coverage uses integer arithmetic: `5 × n_observed >= 4 × T`, with minimum `((4 × T) + 4) // 5`. Thus the thresholds are 4/5, 8/10 and 10/12. A passing feature is the arithmetic mean of its observed requested years. A cell needs at least five of seven usable features. Targets and the required other indicator must be observed and are never imputed. WDI missingness never changes target contributions.

The assessment-specific sensitivity requires every positive-weight contribution to pass coverage. It does not renormalize contextual weights. The prior renormalized construction remains preserved as a historical candidate.

## Reconciled outcome-blind samples

{summary_table}

The correction ledger contains **{protocol['data_reconciliation']['coverage_threshold_correction_count']}** float-versus-integer threshold corrections. The manifest and correction ledger use the exact integer rule. Scenario feasibility requires at least seven countries.

## Models and validation

M0 is the equal-country weighted outer-training mean. M1 uses the other indicator, M2 the seven context features, and M3 both. Ridge is primary; weighted OLS and Lasso are bounded primary-sample benchmarks. No outer-test performance selects an estimator.

Outer validation is leave-one-country-out in ascending ASCII ISO3 order. Inner tuning uses exactly five country folds. Countries are ordered by SHA-256 of `NIQ-HLO-inner|20260930|<ISO3>` and assigned by zero-based position modulo five. All rows of a country stay together, and the allocation is rebuilt after any scenario or deletion changes the country set.

For a fitting subset with N rows, C countries and m_i rows for country i, training weight is `N / (C × m_i)`; evaluation weight is `1 / (C × m_i)`. Each country therefore has equal total weight. No population, sample-size, QN, test-count or precision weight is added.

Context missingness is imputed with the ordinary unweighted median of observed training rows. Weighted population-form moments standardize predictors and target after imputation. A standard deviation no larger than `1e-12 × max(1, |mean|)` is constant. Constant predictor columns stay in position and become zero after standardization. A constant training target produces its weighted mean as the prediction and has no selected alpha.

Ridge uses an unpenalized intercept, SVD solver and grid `[0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]`. Weighted OLS uses the square-root-weighted augmented design and SVD minimum-norm least squares with `rcond=1e-12`. Lasso uses cyclic coordinate descent, `tol=1e-8`, `max_iter=100000`, grid `[0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1]`, and one fresh retry at `max_iter=1000000` after a convergence warning.

Inner tuning concatenates original-unit predictions across all five held-out country folds and averages country-specific MSE. A score is tied when it is no greater than `L_min + 1e-10 + 1e-8 × max(1, |L_min|)`; the largest tied alpha is selected. Every alpha must complete every inner fold.

## Evaluation and bounded sensitivities

MAE, MSE, RMSE and bias give each country equal weight; RMSE is the square root of equal-country MSE. HLO and P errors remain in their own units. Calibration is equal-country weighted `observed = a + b × held-out prediction`, diagnostic only, with no p-values or standard errors.

Ranks are period-specific on identical evaluated countries, highest score rank 1, with average exact-tie ranks. Normalized displacement is `|predicted rank − target rank| / (n − 1)`, averaged within country and then across countries. Spearman and Kendall tau-b are period-specific. Strict reversals exclude pairs tied on either variable. Tail groups use each variable's own linearly interpolated 10th and 90th percentiles and report actual counts, recall and Jaccard overlap.

The finite Ridge-only sensitivity set is S10, S12, SA5, SCC, SCOMP, STERM and S4. S10, S12 and direction-specific SA5 compare primary and alternative constructions on frozen intersections with identical splits and weights. S4 is interpreted within its own measurement scheme. No combinatorial search is authorized.

In the frozen S4 measurement input, the declared partial 2016–2017 window has zero eligible aligned cells. S4 therefore contains 49 cells in 33 countries across the four nonempty windows; the empty declared window remains explicit in the machine-readable window summary.

The country-deletion refit sensitivity is limited to primary five-year `HLO_to_P` Ridge M0–M3. Every deletion rebuilds the entire nested evaluation. Its minimum, median, maximum and sign-change count are sensitivity summaries, not confidence intervals.

## Failure and approval controls

M0–M3 share planned test cells. A required Ridge failure makes that scenario incomplete; no favourable subset loss is published. OLS or Lasso failure marks only that benchmark incomplete. The only numerical retry is the declared Lasso retry.

The empirical gate remains closed. It requires a separate author approval record and a matching measurement record for this exact hash. The supplied approval template is explicitly unapproved.

## Tested environment

Python {environment['python']}; NumPy {environment['numpy']}; SciPy {environment['scipy']}; scikit-learn {environment['scikit_learn']}; pandas {environment['pandas']}; one numerical thread. Dense float64 arrays and deterministic row ordering are required.
"""


def changes_document(protocol: dict) -> str:
    return f"""# Changes from the predecessor integrated protocol

The predecessor seven-file package is preserved unchanged under `rewrite/protocol/archive/author_approval_pre_numerical_addendum_2026-09-30/`. Its canonical scientific hash is `113537313ea06ebb91dc718a5e7498b7d37aa3c52588c6f7b279ed34567e28f1`.

The new canonical scientific hash is `{protocol['protocol_hash']}`.

`Numerical_Decisions_Addendum.md` resolved every numerical field that the predecessor left open. The integrated specification now fixes the exact integer coverage rule, deterministic five-fold allocation, training and evaluation weight normalization, imputation and weighted scaling moments, constant-value tolerance, Ridge/weighted-OLS/Lasso objectives and controls, tuning tie rule, score/calibration/rank/tail definitions, country-deletion procedure, finite sensitivity matrix, and paired failure behavior.

The scientific scope and primary sample-selection rules are unchanged. The primary five-year construction still retains 53 cells in 38 countries, with 36 contextual entries reserved for training-fold imputation. The exact manifests now also identify every finite sensitivity sample and matched comparison intersection. The 18 threshold corrections are preserved in a row-level ledger rather than back-written into earlier outputs.

No target or required source indicator was imputed. No empirical predictive model was fitted. The approval template remains false and separately references the new hash.
"""


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    sample = pd.read_csv(CONTEXT / "scenario_sample_manifest.csv").drop(
        columns=["p_score", "hlo_score", "target_score", "source_indicator_score"]
    )
    sample.to_csv(OUTPUT / "sample_manifest.csv", index=False, lineterminator="\n")
    shutil.copyfile(
        CONTEXT / "coverage_and_imputation_summary.csv",
        OUTPUT / "coverage_and_imputation_summary.csv",
    )
    protocol = build_protocol()
    protocol_path = OUTPUT / "protocol_for_author_approval.json"
    protocol_path.write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (OUTPUT / "PROTOCOL_FOR_AUTHOR_APPROVAL.md").write_text(
        readable_document(protocol), encoding="utf-8"
    )
    (OUTPUT / "changes_from_previous_candidate.md").write_text(
        changes_document(protocol), encoding="utf-8"
    )
    approval = {
        "approval_record_status": "template_not_approved",
        "approval_status": "not_approved",
        "empirical_modelling_approved": False,
        "scope": "empirical_prediction",
        "protocol_file": "protocol_for_author_approval.json",
        "protocol_hash_algorithm": protocol["protocol_hash_algorithm"],
        "protocol_hash": protocol["protocol_hash"],
        "protocol_file_sha256": sha256(protocol_path),
        "approved_by": None,
        "approved_at": None,
        "approval_statement": None,
        "remaining_choices_resolved": True,
        "instruction": "Do not modify this template into an approval. Create a separate dated author-controlled approval record for the exact protocol hash.",
    }
    (OUTPUT / "approval_template.json").write_text(
        json.dumps(approval, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    package_files = [
        "protocol_for_author_approval.json",
        "PROTOCOL_FOR_AUTHOR_APPROVAL.md",
        "sample_manifest.csv",
        "coverage_and_imputation_summary.csv",
        "changes_from_previous_candidate.md",
        "approval_template.json",
    ]
    checksums = "".join(
        f"{sha256(OUTPUT / name)}  {name}\n" for name in package_files
    )
    (OUTPUT / "SHA256SUMS.txt").write_text(checksums, encoding="utf-8")
    print(protocol["protocol_hash"])


if __name__ == "__main__":
    main()
