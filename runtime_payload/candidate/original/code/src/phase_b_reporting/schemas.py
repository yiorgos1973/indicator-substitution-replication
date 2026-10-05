RESULT_COLUMNS = [
    "result_id", "analysis_family", "status", "run_id", "protocol_hash", "target",
    "model_or_comparison", "period", "metric", "estimate", "lower", "upper",
    "interval_method", "units", "n_countries", "n_cells", "denominator_definition",
    "weighting_rule", "sample_id", "source_artifact", "source_locator", "generator",
    "input_hash", "interpretation", "limitation",
]

PREDICTION_COLUMNS = [
    "run_id", "protocol_hash", "direction", "scheme", "history_definition", "model_id",
    "outer_split_id", "country", "window_start", "window_end", "observed_target",
    "predicted_target", "raw_source_indicator", "absolute_error", "squared_error",
    "selected_hyperparameters", "training_country_set_id", "preprocessing_record_id",
    "status", "failure_reason",
]

PROTOCOL_HASH = "13fb577533c28acc88613fcf84549784e130bd658139ea7bbbfc7f77a3b8e773"
GENERATOR = "rewrite/src/phase_b_reporting/processor.py"

