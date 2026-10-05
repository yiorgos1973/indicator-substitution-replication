"""Synthetic reference implementation of the proposed final numerical protocol.

This package does not authorize or load empirical predictive data.
"""

from .core import (
    StructuralInfeasibility,
    classify_history,
    deterministic_inner_folds,
    equal_country_evaluation_weights,
    equal_country_fit_weights,
    fit_preprocessor,
)
from .scenario import run_synthetic_scenario

__all__ = [
    "StructuralInfeasibility",
    "classify_history",
    "deterministic_inner_folds",
    "equal_country_evaluation_weights",
    "equal_country_fit_weights",
    "fit_preprocessor",
    "run_synthetic_scenario",
]
