"""Single-thread numerical-runtime policy and environment recording."""

from __future__ import annotations

import os
import platform


THREAD_ENVIRONMENT_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)


def enforce_one_thread_environment() -> None:
    for name in THREAD_ENVIRONMENT_VARIABLES:
        os.environ[name] = "1"


enforce_one_thread_environment()


def environment_record() -> dict[str, object]:
    import numpy
    import pandas
    import scipy
    import sklearn
    from threadpoolctl import threadpool_info

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "thread_environment": {
            name: os.environ.get(name) for name in THREAD_ENVIRONMENT_VARIABLES
        },
        "threadpools_observed": threadpool_info(),
        "policy": "Reference rebuilds restrict numerical thread pools to one thread.",
    }
