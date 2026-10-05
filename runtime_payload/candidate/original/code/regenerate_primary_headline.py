from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, norm, pearsonr, spearmanr, t


ROOT = Path(__file__).resolve().parent
INPUTS = ROOT / "outputs"
BOOTSTRAP_REPLICATIONS = 9_999
BOOTSTRAP_SEED = 20_260_721
ALPHA = .05


def standardized(values):
    return (values - values.mean()) / values.std(ddof=1)


def hc3_simple_regression(x, y):
    design = np.column_stack([np.ones(len(x)), x])
    inverse_xtx = np.linalg.inv(design.T @ design)
    coefficients = inverse_xtx @ design.T @ y
    fitted = design @ coefficients
    residuals = y - fitted
    leverage = np.sum((design @ inverse_xtx) * design, axis=1)
    meat = design.T @ ((residuals / (1 - leverage))[:, None] ** 2 * design)
    covariance = inverse_xtx @ meat @ inverse_xtx
    return coefficients, np.sqrt(np.diag(covariance)), residuals, leverage


def wild_bootstrap_p_value(x, y, observed_t):
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    restricted_fitted = np.repeat(y.mean(), len(y))
    restricted_residuals = y - restricted_fitted
    bootstrap_t = []
    for _ in range(BOOTSTRAP_REPLICATIONS):
        weights = rng.choice([-1, 1], size=len(y))
        y_star = restricted_fitted + restricted_residuals * weights
        coefficients, standard_errors, _, _ = hc3_simple_regression(x, y_star)
        bootstrap_t.append(coefficients[1] / standard_errors[1])
    bootstrap_t = np.asarray(bootstrap_t)
    p_value = (np.sum(np.abs(bootstrap_t) >= abs(observed_t)) + 1) / (BOOTSTRAP_REPLICATIONS + 1)
    return p_value, bootstrap_t


def fisher_interval(correlation, n, confidence=.95):
    radius = norm.ppf((1 + confidence) / 2) / np.sqrt(n - 3)
    return np.tanh(np.arctanh(correlation) + np.array([-radius, radius]))


def williams_and_zou(r_niq_gdp, r_hlo_gdp, r_niq_hlo, n):
    determinant = 1 + 2 * r_niq_gdp * r_hlo_gdp * r_niq_hlo - r_niq_gdp ** 2 - r_hlo_gdp ** 2 - r_niq_hlo ** 2
    mean_correlation = (r_niq_gdp + r_hlo_gdp) / 2
    williams_t = (r_niq_gdp - r_hlo_gdp) * np.sqrt(
        ((n - 1) * (1 + r_niq_hlo))
        / (2 * ((n - 1) / (n - 3)) * determinant + mean_correlation ** 2 * (1 - r_niq_hlo) ** 3)
    )
    williams_p = 2 * t.sf(abs(williams_t), n - 3)
    niq_interval = fisher_interval(r_niq_gdp, n)
    hlo_interval = fisher_interval(r_hlo_gdp, n)
    covariance_correlation = (
        (r_niq_hlo - .5 * r_niq_gdp * r_hlo_gdp)
        * (1 - r_niq_gdp ** 2 - r_hlo_gdp ** 2 - r_niq_hlo ** 2)
        + r_niq_hlo ** 3
    ) / ((1 - r_niq_gdp ** 2) * (1 - r_hlo_gdp ** 2))
    difference = r_niq_gdp - r_hlo_gdp
    zou_lower = difference - np.sqrt(
        (r_niq_gdp - niq_interval[0]) ** 2
        + (hlo_interval[1] - r_hlo_gdp) ** 2
        - 2 * covariance_correlation * (r_niq_gdp - niq_interval[0]) * (hlo_interval[1] - r_hlo_gdp)
    )
    zou_upper = difference + np.sqrt(
        (niq_interval[1] - r_niq_gdp) ** 2
        + (r_hlo_gdp - hlo_interval[0]) ** 2
        - 2 * covariance_correlation * (niq_interval[1] - r_niq_gdp) * (r_hlo_gdp - hlo_interval[0])
    )
    return determinant, williams_t, williams_p, difference, zou_lower, zou_upper


def regenerate(output_dir):
    rq1 = pd.read_csv(INPUTS / "headline_analysis_rq1_exact_sample.csv").sort_values("iso3")
    rq2 = pd.read_csv(INPUTS / "headline_analysis_rq2_exact_sample.csv").sort_values("iso3")
    assert len(rq1) == 66 and rq1["iso3"].is_unique
    assert len(rq2) == 65 and rq2["iso3"].is_unique
    assert set(rq2["iso3"]).issubset(rq1["iso3"])

    ranking = rq1[["iso3", "canonical_economy_name", "niq_qnw", "hlo"]].copy()
    ranking["rank_niq_qnw"] = ranking["niq_qnw"].rank(ascending=False, method="average")
    ranking["rank_hlo"] = ranking["hlo"].rank(ascending=False, method="average")
    ranking["rank_displacement_hlo_minus_niq"] = ranking["rank_hlo"] - ranking["rank_niq_qnw"]
    ranking["absolute_rank_displacement"] = ranking["rank_displacement_hlo_minus_niq"].abs()
    top_niq_cutoff = ranking["niq_qnw"].quantile(.90, interpolation="linear")
    top_hlo_cutoff = ranking["hlo"].quantile(.90, interpolation="linear")
    bottom_niq_cutoff = ranking["niq_qnw"].quantile(.10, interpolation="linear")
    bottom_hlo_cutoff = ranking["hlo"].quantile(.10, interpolation="linear")
    ranking["top_decile_niq"] = ranking["niq_qnw"].ge(top_niq_cutoff)
    ranking["top_decile_hlo"] = ranking["hlo"].ge(top_hlo_cutoff)
    ranking["bottom_decile_niq"] = ranking["niq_qnw"].le(bottom_niq_cutoff)
    ranking["bottom_decile_hlo"] = ranking["hlo"].le(bottom_hlo_cutoff)
    assert ranking[["niq_qnw", "hlo"]].nunique().eq(len(ranking)).all()
    assert ranking[["top_decile_niq", "top_decile_hlo", "bottom_decile_niq", "bottom_decile_hlo"]].sum().eq(7).all()

    niq_difference = ranking["niq_qnw"].to_numpy()[:, None] - ranking["niq_qnw"].to_numpy()
    hlo_difference = ranking["hlo"].to_numpy()[:, None] - ranking["hlo"].to_numpy()
    strict_pair = (niq_difference != 0) & (hlo_difference != 0)
    pairwise_reversals = int(np.triu((niq_difference * hlo_difference < 0) & strict_pair, k=1).sum())
    strict_pairs = int(np.triu(strict_pair, k=1).sum())
    pearson = pearsonr(ranking["niq_qnw"], ranking["hlo"])
    spearman = spearmanr(ranking["niq_qnw"], ranking["hlo"])
    kendall = kendalltau(ranking["niq_qnw"], ranking["hlo"])
    ranking_summary = pd.DataFrame([{
        "n_economies": len(ranking),
        "common_sample_iso3": " ".join(ranking["iso3"]),
        "pearson_r": pearson.statistic,
        "pearson_p_value": pearson.pvalue,
        "spearman_rho": spearman.statistic,
        "spearman_p_value": spearman.pvalue,
        "kendall_tau": kendall.statistic,
        "kendall_p_value": kendall.pvalue,
        "mean_absolute_rank_displacement": ranking["absolute_rank_displacement"].mean(),
        "median_absolute_rank_displacement": ranking["absolute_rank_displacement"].median(),
        "maximum_absolute_rank_displacement": ranking["absolute_rank_displacement"].max(),
        "strict_pair_count": strict_pairs,
        "pairwise_order_reversal_count": pairwise_reversals,
        "pairwise_order_reversal_share": pairwise_reversals / strict_pairs,
        "top_niq_cutoff": top_niq_cutoff,
        "top_hlo_cutoff": top_hlo_cutoff,
        "top_niq_size": int(ranking["top_decile_niq"].sum()),
        "top_hlo_size": int(ranking["top_decile_hlo"].sum()),
        "top_overlap_size": int((ranking["top_decile_niq"] & ranking["top_decile_hlo"]).sum()),
        "bottom_niq_cutoff": bottom_niq_cutoff,
        "bottom_hlo_cutoff": bottom_hlo_cutoff,
        "bottom_niq_size": int(ranking["bottom_decile_niq"].sum()),
        "bottom_hlo_size": int(ranking["bottom_decile_hlo"].sum()),
        "bottom_overlap_size": int((ranking["bottom_decile_niq"] & ranking["bottom_decile_hlo"]).sum()),
    }])

    association_rows = []
    bootstrap_rows = []
    influence_frames = []
    y = standardized(rq2["log_gdp_pc_ppp_2017"]).to_numpy()
    for measure, column in {"QNW": "niq_qnw", "HLO": "hlo"}.items():
        x = standardized(rq2[column]).to_numpy()
        coefficients, standard_errors, residuals, leverage = hc3_simple_regression(x, y)
        beta = coefficients[1]
        beta_se = standard_errors[1]
        degrees_of_freedom = len(rq2) - 2
        critical_value = t.ppf(1 - ALPHA / 2, degrees_of_freedom)
        t_value = beta / beta_se
        bootstrap_p_value, bootstrap_t = wild_bootstrap_p_value(x, y, t_value)
        mse = np.sum(residuals ** 2) / degrees_of_freedom
        influence_frames.append(pd.DataFrame({
            "iso3": rq2["iso3"].to_numpy(),
            "economy_name": rq2["canonical_economy_name"].to_numpy(),
            "model": measure,
            "leverage": leverage,
            "internally_studentized_residual": residuals / np.sqrt(mse * (1 - leverage)),
            "cooks_distance": (residuals ** 2 / (2 * mse)) * (leverage / (1 - leverage) ** 2),
        }))
        association_rows.append({
            "model": measure,
            "predictor": column,
            "outcome": "log_gdp_pc_ppp_2017",
            "n_economies": len(rq2),
            "common_sample_iso3": " ".join(rq2["iso3"]),
            "standardized_coefficient": beta,
            "hc3_standard_error": beta_se,
            "hc3_ci_lower": beta - critical_value * beta_se,
            "hc3_ci_upper": beta + critical_value * beta_se,
            "hc3_t_value": t_value,
            "hc3_degrees_of_freedom": degrees_of_freedom,
            "hc3_p_value": 2 * t.sf(abs(t_value), degrees_of_freedom),
            "r_squared": np.corrcoef(x, y)[0, 1] ** 2,
            "wild_bootstrap_p_value": bootstrap_p_value,
        })
        bootstrap_rows.append({
            "model": measure,
            "null_hypothesis": "standardized_coefficient = 0",
            "weights": "Rademacher (-1, +1)",
            "restricted_model": "intercept_only",
            "studentization": "HC3",
            "replications": BOOTSTRAP_REPLICATIONS,
            "seed": BOOTSTRAP_SEED,
            "two_sided_p_value": bootstrap_p_value,
            "bootstrap_t_mean": bootstrap_t.mean(),
            "bootstrap_t_standard_deviation": bootstrap_t.std(ddof=1),
        })

    r_niq_gdp = pearsonr(rq2["niq_qnw"], rq2["log_gdp_pc_ppp_2017"]).statistic
    r_hlo_gdp = pearsonr(rq2["hlo"], rq2["log_gdp_pc_ppp_2017"]).statistic
    r_niq_hlo = pearsonr(rq2["niq_qnw"], rq2["hlo"]).statistic
    determinant, williams_t, williams_p, difference, zou_lower, zou_upper = williams_and_zou(r_niq_gdp, r_hlo_gdp, r_niq_hlo, len(rq2))
    assert determinant >= 0 and zou_lower <= difference <= zou_upper
    correlation_difference = pd.DataFrame([{
        "comparison": "QNW-GDP minus HLO-GDP",
        "n_economies": len(rq2),
        "common_sample_iso3": " ".join(rq2["iso3"]),
        "r_qnw_gdp": r_niq_gdp,
        "r_hlo_gdp": r_hlo_gdp,
        "r_qnw_hlo": r_niq_hlo,
        "difference_r_qnw_gdp_minus_r_hlo_gdp": difference,
        "correlation_matrix_determinant": determinant,
        "williams_t": williams_t,
        "williams_degrees_of_freedom": len(rq2) - 3,
        "williams_two_sided_p_value": williams_p,
        "zou_95_ci_lower": zou_lower,
        "zou_95_ci_upper": zou_upper,
        "formula_implementation": "standard Williams t and Zou 2007 overlapping-correlation interval; cross-checked against cocor source",
    }])

    def correlation_pair(frame):
        return {
            "r_qnw_gdp": frame["niq_qnw"].corr(frame["log_gdp_pc_ppp_2017"]),
            "r_hlo_gdp": frame["hlo"].corr(frame["log_gdp_pc_ppp_2017"]),
        }

    leave_one_out = pd.DataFrame([
        {"excluded_iso3": iso3, "n_economies": len(rq2) - 1, **correlation_pair(rq2.loc[~rq2["iso3"].eq(iso3)])}
        for iso3 in rq2["iso3"]
    ])
    leave_one_region_out = pd.DataFrame([
        {
            "excluded_region": region,
            "n_economies": int(rq2["frozen_world_bank_region"].ne(region).sum()),
            **correlation_pair(rq2.loc[rq2["frozen_world_bank_region"].ne(region)]),
        }
        for region in sorted(rq2["frozen_world_bank_region"].dropna().unique())
    ])

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "headline_analysis_rank_displacement.csv": ranking,
        "headline_analysis_ranking_summary.csv": ranking_summary,
        "headline_analysis_association_results.csv": pd.DataFrame(association_rows),
        "headline_analysis_wild_bootstrap_results.csv": pd.DataFrame(bootstrap_rows),
        "headline_analysis_influence.csv": pd.concat(influence_frames, ignore_index=True),
        "headline_analysis_correlation_difference.csv": correlation_difference,
        "headline_analysis_leave_one_out.csv": leave_one_out,
        "headline_analysis_leave_one_region_out.csv": leave_one_region_out,
    }
    for name, frame in outputs.items():
        frame.to_csv(output_dir / name, index=False, lineterminator="\n")
    return outputs


def main():
    parser = ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "regenerated_headline")
    arguments = parser.parse_args()
    outputs = regenerate(arguments.output_dir.resolve())
    print(f"Regenerated {len(outputs)} primary headline outputs in {arguments.output_dir.resolve()}")


if __name__ == "__main__":
    main()
