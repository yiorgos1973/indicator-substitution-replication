"""Phase B empirical orchestration, protected by immutable pre-fit gates."""

import os

for _thread_variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_thread_variable] = "1"

from .gate import EXPECTED_PROTOCOL_HASH, GateError, evaluate_gate
from .matrices import build_modelling_matrices
from .plan import build_execution_plan

__all__ = ["EXPECTED_PROTOCOL_HASH", "GateError", "evaluate_gate", "build_modelling_matrices", "build_execution_plan"]
