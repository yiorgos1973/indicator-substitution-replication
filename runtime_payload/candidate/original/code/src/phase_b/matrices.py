"""Construct frozen scenario/direction modelling matrices without fitting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


FEATURES = (
    "SE.PRM.ENRR", "SE.SEC.ENRR", "SE.XPD.TOTL.GD.ZS", "SH.DYN.MORT",
    "SP.DYN.TFRT.IN", "SP.URB.TOTL.IN.ZS", "IT.NET.USER.ZS",
)
KEY = ["iso3", "window_start", "window_end", "scenario_id", "target_direction", "construction_role"]


@dataclass(frozen=True)
class MatrixBundle:
    matrices: dict[tuple[str, str, str], pd.DataFrame]
    manifest: pd.DataFrame


def build_modelling_matrices(sample_manifest: str | Path, feature_manifest: str | Path) -> MatrixBundle:
    samples = pd.read_csv(sample_manifest)
    features = pd.read_csv(feature_manifest)
    retained = samples[samples["included"].astype(str).str.lower().eq("true")].copy()
    retained = retained[retained["retained"].astype(str).str.lower().eq("true")]
    selected_features = features[features["included"].astype(str).str.lower().eq("true")].copy()
    selected_features = selected_features[selected_features["indicator_code"].isin(FEATURES)]
    # Phase A stores direction-invariant block histories once under HLO_to_P.
    # Reuse them only when the same scenario/role has no direction-specific rows.
    required_identities = retained[["scenario_id", "target_direction", "construction_role"]].drop_duplicates()
    available = set(map(tuple, selected_features[["scenario_id", "target_direction", "construction_role"]].drop_duplicates().to_numpy()))
    copies = []
    for scenario, direction, role in required_identities.itertuples(index=False, name=None):
        if (scenario, direction, role) not in available and direction == "P_to_HLO":
            source = selected_features[(selected_features["scenario_id"] == scenario) & (selected_features["target_direction"] == "HLO_to_P") & (selected_features["construction_role"] == role)].copy()
            if source.empty:
                raise ValueError(f"No context features for {(scenario, direction, role)}")
            source["target_direction"] = direction
            copies.append(source)
    if copies:
        selected_features = pd.concat([selected_features, *copies], ignore_index=True)
    if selected_features.duplicated(KEY + ["indicator_code"]).any():
        raise ValueError("Duplicate scenario-cell-feature rows")
    wide = selected_features.pivot(index=KEY, columns="indicator_code", values="protocol_context_value").reset_index()
    frame = retained.merge(wide, on=KEY, how="left", validate="one_to_one")
    matrices: dict[tuple[str, str, str], pd.DataFrame] = {}
    manifest_rows = []
    for identity, group in frame.groupby(["scenario_id", "target_direction", "construction_role"], sort=True):
        matrix = group[["iso3", "window_start", "window_end", "sample_cell_id", "p_score", "hlo_score", "target_score", "source_indicator_score", "usable_feature_count", "missing_feature_count", *FEATURES]].copy()
        matrix = matrix.sort_values(["iso3", "window_start", "window_end"], kind="mergesort").reset_index(drop=True)
        if matrix.duplicated(["iso3", "window_start", "window_end"]).any():
            raise ValueError(f"Duplicate country-period rows in {identity}")
        if matrix[["target_score", "source_indicator_score"]].isna().any().any():
            raise ValueError(f"Missing required score in {identity}")
        if (matrix[list(FEATURES)].notna().sum(axis=1) < 5).any():
            raise ValueError(f"Fewer than five context features in {identity}")
        if identity[0] == "SCC" and matrix[list(FEATURES)].isna().any().any():
            raise ValueError("SCC must contain seven complete context features")
        matrices[identity] = matrix
        manifest_rows.append({"scenario_id": identity[0], "target_direction": identity[1], "construction_role": identity[2], "rows": len(matrix), "countries": matrix["iso3"].nunique(), "cell_identity_sha256": _cell_hash(matrix)})
    _validate_matched_pairs(matrices)
    return MatrixBundle(matrices, pd.DataFrame(manifest_rows))


def _cell_hash(frame: pd.DataFrame) -> str:
    import hashlib
    values = "\n".join(frame["sample_cell_id"].astype(str).sort_values())
    return hashlib.sha256(values.encode("utf-8")).hexdigest()


def _validate_matched_pairs(matrices: dict[tuple[str, str, str], pd.DataFrame]) -> None:
    alternative_roles = {"S10": "alternative_10y", "S12": "alternative_12y", "SA5": "assessment_specific_5y_fixed"}
    for scenario, alternative_role in alternative_roles.items():
        for direction in ("HLO_to_P", "P_to_HLO"):
            alternative = matrices[(scenario, direction, alternative_role)]
            primary = matrices[(scenario, direction, "primary_5y_matched")]
            if set(alternative["sample_cell_id"]) != set(primary["sample_cell_id"]):
                raise ValueError(f"Unmatched comparison cells: {scenario} {direction}")
