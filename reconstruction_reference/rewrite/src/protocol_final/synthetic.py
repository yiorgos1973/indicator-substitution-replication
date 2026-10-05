"""Deterministic synthetic-only fixture; it contains no project outcome data."""

from __future__ import annotations

import numpy as np
import pandas as pd


CONTEXT_FEATURES = (
    "SE.PRM.ENRR", "SE.SEC.ENRR", "SE.XPD.TOTL.GD.ZS", "SH.DYN.MORT",
    "SP.DYN.TFRT.IN", "SP.URB.TOTL.IN.ZS", "IT.NET.USER.ZS",
)


def make_synthetic_fixture() -> pd.DataFrame:
    rng = np.random.default_rng(20260930)
    rows = []
    for country_number in range(8):
        iso3 = f"X{country_number:02d}"
        country_signal = (country_number - 3.5) / 2
        for period_number, start in enumerate((2000, 2005, 2010)):
            source = country_signal + 0.25 * period_number + rng.normal(0, 0.08)
            contexts = country_signal + np.arange(7) * 0.2 + 0.1 * period_number + rng.normal(0, 0.05, 7)
            contexts[-1] = 7.0
            target = 3.0 + 1.4 * source + 0.3 * contexts[0] - 0.2 * contexts[3] + rng.normal(0, 0.05)
            row = {"iso3": iso3, "window_start": start, "window_end": min(start + 4, 2017), "source_score": source, "target": target}
            row.update(dict(zip(CONTEXT_FEATURES, contexts)))
            rows.append(row)
    frame = pd.DataFrame(rows)
    frame.loc[[1, 8, 19], "SE.XPD.TOTL.GD.ZS"] = np.nan
    frame.loc[[4, 14], "SP.DYN.TFRT.IN"] = np.nan
    return frame.sort_values(["iso3", "window_start", "window_end"]).reset_index(drop=True)

