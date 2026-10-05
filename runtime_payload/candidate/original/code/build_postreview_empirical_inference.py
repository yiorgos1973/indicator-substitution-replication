from hashlib import sha256
import json
from pathlib import Path
import platform
import sys
from time import perf_counter

import numpy as np
import pandas as pd
import scipy
from scipy.stats import norm


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
REPLICATIONS = 99_999
SEED = 20260917
BATCH_SIZE = 5_000
VARIABLES = ["niq_qnw", "hlo", "log_gdp_pc_ppp_2017"]
INPUTS = [
    "headline_analysis_rq2_exact_sample.csv",
    "headline_analysis_correlation_difference.csv",
    "headline_analysis_association_results.csv",
    "headline_analysis_wild_bootstrap_results.csv",
    "spatial_correlation_if_audit.csv",
]


def input_hashes():
    return {name: sha256((OUTPUTS / name).read_bytes()).hexdigest() for name in INPUTS}


class PairedCorrelationBootstrap:
    def __init__(self, sample, archived):
        self.sample = sample
        self.values = sample[VARIABLES].to_numpy(dtype=float)
        assert len(sample) == 65
        assert sample["iso3"].is_unique
        assert not sample[["iso3", *VARIABLES]].isna().any().any()
        assert np.isfinite(self.values).all()
        assert (self.values.var(axis=0) > 0).all()
        assert " ".join(sample["iso3"]) == archived["common_sample_iso3"]
        self.observed = self.correlations(self.values)
        expected = archived[[
            "r_qnw_gdp", "r_hlo_gdp", "difference_r_qnw_gdp_minus_r_hlo_gdp"
        ]].to_numpy(dtype=float)
        np.testing.assert_allclose(self.observed, expected, atol=1e-12, rtol=0)
        np.testing.assert_allclose(
            self.observed, [0.6159047248355083, 0.6895192024555438, -0.07361447762003548],
            atol=1e-12, rtol=0,
        )

    @staticmethod
    def correlations(values):
        centered = values - values.mean(axis=-2, keepdims=True)
        sums_of_squares = np.sum(centered**2, axis=-2)
        cross_products = np.sum(centered[..., :2] * centered[..., 2:3], axis=-2)
        denominators = np.sqrt(sums_of_squares[..., :2] * sums_of_squares[..., 2:3])
        correlations = np.full_like(cross_products, np.nan)
        np.divide(cross_products, denominators, out=correlations, where=denominators > 0)
        valid = (sums_of_squares > 0).all(axis=-1) & np.isfinite(correlations).all(axis=-1)
        delta = correlations[..., 0] - correlations[..., 1]
        result = np.concatenate([correlations, delta[..., None]], axis=-1)
        return np.where(valid[..., None], result, np.nan)

    def distribution(self):
        rng = np.random.Generator(np.random.PCG64(SEED))
        statistics = np.empty((REPLICATIONS, 3))
        started = perf_counter()
        for start in range(0, REPLICATIONS, BATCH_SIZE):
            stop = min(start + BATCH_SIZE, REPLICATIONS)
            indices = rng.integers(0, 65, size=(stop - start, 65), dtype=np.int64)
            statistics[start:stop] = self.correlations(self.values[indices])
            elapsed = perf_counter() - started
            eta = elapsed / stop * (REPLICATIONS - stop)
            print(f"Bootstrap {stop:,}/{REPLICATIONS:,}: elapsed {elapsed:.2f}s; estimated remaining {eta:.2f}s.", flush=True)
        result = pd.DataFrame(statistics, columns=["r_qnw_gdp", "r_hlo_gdp", "delta_r"])
        result.insert(0, "replication", np.arange(1, REPLICATIONS + 1))
        result["valid"] = np.isfinite(statistics).all(axis=1)
        invalid = int((~result["valid"]).sum())
        print(f"Invalid replications: {invalid}/{REPLICATIONS:,}.", flush=True)
        return result

    def jackknife(self):
        statistics = np.array([
            self.correlations(np.delete(self.values, index, axis=0)) for index in range(65)
        ])
        assert np.isfinite(statistics).all(), "Undefined jackknife statistic; stop and investigate."
        result = self.sample[["iso3", "canonical_economy_name"]].rename(columns={
            "iso3": "omitted_iso3", "canonical_economy_name": "omitted_economy_name"
        }).copy()
        result[["r_qnw_gdp", "r_hlo_gdp", "delta_r"]] = statistics
        return result

    def summary(self, distribution, jackknife):
        valid = distribution.loc[distribution["valid"], "delta_r"].to_numpy()
        invalid = REPLICATIONS - len(valid)
        assert invalid <= 0.001 * REPLICATIONS, "Invalid replications exceed 0.1%; stop and investigate."
        fraction_below = (np.sum(valid < self.observed[2]) + 0.5 * np.sum(valid == self.observed[2])) / len(valid)
        z0 = norm.ppf(fraction_below)
        deviations = jackknife["delta_r"].mean() - jackknife["delta_r"].to_numpy()
        acceleration = np.sum(deviations**3) / (6 * np.sum(deviations**2)**1.5)
        bias_shift = z0 + norm.ppf([0.025, 0.975])
        adjusted = norm.cdf(z0 + bias_shift / (1 - acceleration * bias_shift))
        assert np.isfinite([z0, acceleration, *adjusted]).all(), "Undefined BCa calculation."
        assert 0 < adjusted[0] < adjusted[1] < 1, "Undefined BCa quantile probabilities."
        percentile = np.quantile(valid, [0.025, 0.975], method="linear")
        bca = np.quantile(valid, adjusted, method="linear")
        assert percentile[0] <= percentile[1] and bca[0] <= bca[1]
        result = {
            "n_economies": 65,
            "replications_requested": REPLICATIONS,
            "replications_valid": len(valid),
            "invalid_replications": invalid,
            "seed": SEED,
            "observed_r_qnw_gdp": self.observed[0],
            "observed_r_hlo_gdp": self.observed[1],
            "observed_delta_r": self.observed[2],
            "bootstrap_mean_delta_r": valid.mean(),
            "bootstrap_median_delta_r": np.median(valid),
            "bootstrap_se_delta_r": valid.std(ddof=1),
            "percentile_ci_95_lower": percentile[0],
            "percentile_ci_95_upper": percentile[1],
            "bca_ci_95_lower": bca[0],
            "bca_ci_95_upper": bca[1],
        }
        probabilities = [0.01, 0.025, 0.05, 0.50, 0.95, 0.975, 0.99]
        names = ["q01", "q025", "q05", "q50", "q95", "q975", "q99"]
        result.update(zip([f"bootstrap_{name}" for name in names], np.quantile(valid, probabilities, method="linear")))
        result.update({
            "bca_bias_correction_z0": z0,
            "bca_acceleration": acceleration,
            "bca_adjusted_probability_lower": adjusted[0],
            "bca_adjusted_probability_upper": adjusted[1],
            "generator": "numpy.random.Generator(PCG64)",
            "batch_size": BATCH_SIZE,
            "quantile_method": "linear",
            "bootstrap_se_ddof": 1,
            "summary_population": "valid replications only",
            "input_row_order": "stored CSV order",
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
            "pandas_version": pd.__version__,
        })
        return result


def inference_crosswalk(summary, archived):
    delta_target = "rho(QNW, log GDP) minus rho(HLO, log GDP)"
    delta = summary["observed_delta_r"]
    rows = []

    def add(method, target, point, reference, assumption, interpretation, lower=np.nan, upper=np.nan, p=np.nan):
        rows.append({
            "method": method, "target": target, "point_estimate": point,
            "interval_lower": lower, "interval_upper": upper, "p_value_if_applicable": p,
            "resampling_or_reference_distribution": reference,
            "key_assumption": assumption, "interpretation": interpretation,
        })

    for name, key in [("QNW", "observed_r_qnw_gdp"), ("HLO", "observed_r_hlo_gdp")]:
        add(f"{name}-GDP Pearson r", f"Observed {name}-GDP correlation", summary[key],
            "None; fixed-sample description", "Published scores and equal-economy weighting are taken as given.",
            "Describes association across these 65 economies; no causal interpretation.")
    add("Observed delta_r", "Observed QNW-GDP minus HLO-GDP correlation", delta,
        "None; fixed-sample description", "Both correlations use the same 65 economies.",
        "Negative values mean the HLO-GDP correlation is larger in this sample.")
    add("Williams test", delta_target, delta, "Student-t, 62 degrees of freedom",
        "Independent comparable economy triplets and the classical multivariate-normal correlation model.",
        "Tests equality of overlapping correlations; non-rejection does not establish equivalence.",
        p=archived["williams_two_sided_p_value"])
    add("Zou 95% interval", delta_target, delta, "Fisher-z intervals with overlapping-correlation covariance",
        "IID comparable triplets; Fisher-z and normal-theory correlation covariance approximations.",
        "Analytic uncertainty for the difference; coverage depends on the working model.",
        archived["zou_95_ci_lower"], archived["zou_95_ci_upper"])
    for method, prefix in [("percentile", "percentile"), ("BCa", "bca")]:
        add(f"Paired case bootstrap {method} 95% interval", delta_target, delta,
            "99,999 whole-economy samples with replacement; ordinary sampling distribution",
            "IID comparable economy triplets, nonzero variance, finite moments and bootstrap regularity.",
            "Post-review sensitivity; does not address spatial dependence, selective coverage or score error.",
            summary[f"{prefix}_ci_95_lower"], summary[f"{prefix}_ci_95_upper"])
    association = pd.read_csv(OUTPUTS / "headline_analysis_association_results.csv")
    wild = pd.read_csv(OUTPUTS / "headline_analysis_wild_bootstrap_results.csv")
    for row in association.itertuples():
        target = f"Separate standardized {row.model} regression coefficient"
        add(f"HC3 {row.model} regression", target, row.standardized_coefficient,
            "HC3 covariance; approximate Student-t, 63 degrees of freedom",
            "Independent observations, finite moments and regularity for robust regression inference.",
            "Targets one coefficient; does not test the difference between the two correlations.",
            row.hc3_ci_lower, row.hc3_ci_upper, row.hc3_p_value)
        bootstrap = wild.loc[wild["model"].eq(row.model)].iloc[0]
        add(f"Existing wild bootstrap {row.model}", target, row.standardized_coefficient,
            "9,999 null-imposed Rademacher samples; HC3 studentization; seed 20260721",
            "Independent regression disturbances and validity of the restricted residual bootstrap.",
            "Tests a separate zero coefficient; does not test the correlation difference.",
            p=bootstrap["two_sided_p_value"])
    spatial = pd.read_csv(OUTPUTS / "spatial_correlation_if_audit.csv")
    spatial = spatial.loc[spatial["first_indicator"].eq("QNW") & spatial["second_indicator"].eq("HLO")]
    assert spatial["bandwidth_km"].tolist() == [1500, 3000]
    for row in spatial.itertuples():
        add(f"Existing spatial correlation-IF sensitivity ({row.bandwidth_km:,} km)", delta_target,
            row.contrast, "Bartlett spatial HAC of paired correlation influence; approximate Student-t, 63 df",
            "Weak spatial dependence compatible with the chosen distance kernel and asymptotic approximation.",
            "Approximate contrast sensitivity at a fixed bandwidth; does not remedy selective coverage.",
            row.if_ci_lower, row.if_ci_upper, row.if_p_value)
    return pd.DataFrame(rows)


def technical_note(summary, archived, hashes):
    s = summary
    return f"""# Post-review empirical inference results

## Purpose and scope

The paper, **Substituting National-IQ and Harmonized-Learning Indicators: An Output-Specific Replication and Sensitivity Audit**, was accepted for peer review and then rejected after peer review. This analysis adds an empirical sampling distribution to Part II's indicator-substitution audit. It addresses the request to make sampling uncertainty and its assumptions explicit. It does not modify Part I's source-specific replication findings or rewrite the manuscript.

The specification was committed before this computation in `docs/Post_Review_Inference_Protocol_2026-09-17.md`. This is a post-review sensitivity, not a preregistered analysis. The archived Williams test and Zou interval retain their primary analytic role. All historical inputs and original results remain unchanged.

## Sample, statistic and interpretation

The input is `outputs/headline_analysis_rq2_exact_sample.csv`, in its stored row order: exactly 65 unique economies with complete QNW, HLO and natural-log 2017 PPP GDP per capita. Equal-economy weighting is unchanged. The complete membership is recorded in the historical CSV and correlation-difference output. Input SHA-256: `{hashes['headline_analysis_rq2_exact_sample.csv']}`.

QNW denotes the national-IQ indicator; HLO denotes harmonized learning outcomes. The statistic is `delta_r = r(QNW, log GDP) - r(HLO, log GDP)`. Its two correlations are {s['observed_r_qnw_gdp']:.15f} and {s['observed_r_hlo_gdp']:.15f}; the difference is **{s['observed_delta_r']:.15f}**. The negative sign means HLO has the larger observed GDP correlation on this fixed sample.

These observed quantities describe the published values. Repeated-sampling inference concerns a hypothetical population of comparable economy triplets under an IID working model, nonzero marginal variances and sufficient finite moments. The economies were not randomly selected. This bootstrap does not propagate country-score measurement error or harmonization uncertainty. It does not account for cross-economy dependence or selective coverage. More resamples reduce simulation error; they do not establish these assumptions.

## Empirical uncertainty

Each bootstrap sample resamples whole economies, keeping QNW, HLO, and GDP together. There are {s['replications_requested']:,} requested draws of 65 rows with replacement, seed {s['seed']}, with **{s['invalid_replications']} invalid replications**. Summaries use the {s['replications_valid']:,} valid replications; no replacement draws are introduced.

| Quantity | Value |
|---|---:|
| Observed difference | {s['observed_delta_r']:.9f} |
| Bootstrap mean | {s['bootstrap_mean_delta_r']:.9f} |
| Bootstrap median | {s['bootstrap_median_delta_r']:.9f} |
| Bootstrap standard error (ddof=1) | {s['bootstrap_se_delta_r']:.9f} |
| Percentile 95% interval | [{s['percentile_ci_95_lower']:.9f}, {s['percentile_ci_95_upper']:.9f}] |
| BCa 95% interval | [{s['bca_ci_95_lower']:.9f}, {s['bca_ci_95_upper']:.9f}] |
| Archived Williams two-sided p | {archived['williams_two_sided_p_value']:.9f} |
| Archived Zou 95% interval | [{archived['zou_95_ci_lower']:.9f}, {archived['zou_95_ci_upper']:.9f}] |

The empirical intervals include zero and values in both directions. They do not establish equivalence of the indicators or demonstrate actual coverage for this country sample. The method estimates uncertainty around the observed contrast. **No p-value was calculated or inferred from the ordinary paired case-bootstrap distribution.** It is not a null-imposed distribution.

The inference crosswalk separates the contrast from the two individual slopes. Existing HC3 and null-imposed wild-bootstrap results concern the individual regression coefficients. Existing spatial contrast sensitivities use the correlation influence-function output at both 1,500 and 3,000 km. Those intervals are approximate and assumption-dependent. Individual regression spatial results are not substituted for contrast inference.

## BCa and simulation audit

The generator is `numpy.random.Generator(numpy.random.PCG64(20260917))`. Sequential calls to `integers(0, 65, size=(batch_size, 65), dtype=np.int64)` use batches of 5,000, with a final batch of 4,999. No other random draws use this generator. Pearson correlations use centered cross-products and sums of squared deviations. Quantiles use NumPy's `method="linear"` convention.

The same stored bootstrap distribution supplies the percentile and BCa intervals. For the 65 leave-one-economy contrasts `theta_i`, define `u_i = mean(theta_i) - theta_i`. Acceleration is `sum(u_i**3) / (6 * sum(u_i**2)**1.5)`. Bias correction is `z0 = norm.ppf((count(theta_boot < theta_observed) + 0.5 * count(theta_boot == theta_observed)) / B_valid)`. At nominal probability `alpha`, the adjusted probability is `norm.cdf(z0 + (z0 + norm.ppf(alpha)) / (1 - acceleration * (z0 + norm.ppf(alpha))))`.

- Bias correction z0: {s['bca_bias_correction_z0']:.15f}.
- Jackknife acceleration: {s['bca_acceleration']:.15f}.
- Adjusted lower probability: {s['bca_adjusted_probability_lower']:.15f}.
- Adjusted upper probability: {s['bca_adjusted_probability_upper']:.15f}.

All draws retain their replication number and validity flag. A degenerate draw has zero marginal variance or a nonfinite correlation; all its numerical statistics are missing. The process stops above 0.1% invalid draws or for undefined jackknife/BCa quantities. It neither replaces invalid draws nor clips undefined BCa probabilities.

## Files and reproduction

- `build_postreview_empirical_inference.py`: deterministic offline builder.
- `outputs/postreview_bootstrap_delta_r_distribution.csv`: all 99,999 draws.
- `outputs/postreview_bootstrap_delta_r_summary.csv`: estimates, quantiles, BCa audit and environment.
- `outputs/postreview_delta_r_jackknife.csv`: all 65 omissions, with names.
- `outputs/postreview_inference_crosswalk.csv`: method targets, results and assumptions.
- `tests/check_postreview_empirical_inference.py`: source integrity, numerical checks and network-blocked deterministic rerun.

Run `python build_postreview_empirical_inference.py` and `python tests/check_postreview_empirical_inference.py`. The builder prints measured elapsed time and estimated time remaining after each batch. Runtime is operational metadata printed to the console, not inserted into deterministic scientific files. Environment: Python {s['python_version']}; NumPy {s['numpy_version']}; SciPy {s['scipy_version']}; pandas {s['pandas_version']}.

Input bytes must match either the committed Windows baseline hash or the exact baseline Git-blob hash. This permits Git's recorded line-ending conversion across operating systems without normalizing or changing input content. Each input's bytes must remain identical before and after computation. Observed correlations and the contrast must match the archived values within absolute tolerance 1e-12, with zero relative tolerance. Independent validation is reported separately in `docs/Post_Review_Independent_Validation.md`.

## Methodological sources

- Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall/CRC.
- SciPy 1.15.3, paired BCa bootstrap documentation: https://docs.scipy.org/doc/scipy-1.15.3/reference/generated/scipy.stats.bootstrap.html
- Diedenhofen, B., & Musch, J. (2015), overlapping-correlation comparisons: https://doi.org/10.1371/journal.pone.0121945
- Zou, G. Y. (2007), intervals for correlation differences: https://doi.org/10.1037/1082-989X.12.4.399
- Abadie et al. (2020), sampling-based and design-based uncertainty: https://doi.org/10.3982/ECTA12675
- Historical spatial specification: `docs/Spatial_Correlation_IF_Specification_v0.6.md`.
"""


def main():
    assert sys.flags.optimize == 0, "Run without python -O; scientific assertions must remain active."
    started = perf_counter()
    print("Post-review paired bootstrap: first batch provides the runtime estimate.", flush=True)
    hashes_before = input_hashes()
    baseline = json.loads((ROOT / "docs/Post_Review_Historical_Hashes_2026-09-17.json").read_text())
    git_blobs = json.loads((ROOT / "docs/Post_Review_Git_Blob_Hashes_2026-09-17.json").read_text())
    for name, digest in hashes_before.items():
        assert digest in {baseline[f"outputs/{name}"], git_blobs[f"outputs/{name}"]}, f"Historical input changed: {name}"
    sample = pd.read_csv(OUTPUTS / INPUTS[0])
    archived = pd.read_csv(OUTPUTS / INPUTS[1]).iloc[0]
    analysis = PairedCorrelationBootstrap(sample, archived)
    distribution = analysis.distribution()
    distribution.to_csv(OUTPUTS / "postreview_bootstrap_delta_r_distribution.csv", index=False, float_format="%.17g")
    jackknife = analysis.jackknife()
    summary = analysis.summary(distribution, jackknife)
    jackknife.to_csv(OUTPUTS / "postreview_delta_r_jackknife.csv", index=False, float_format="%.17g")
    pd.DataFrame([summary]).to_csv(OUTPUTS / "postreview_bootstrap_delta_r_summary.csv", index=False, float_format="%.17g")
    inference_crosswalk(summary, archived).to_csv(OUTPUTS / "postreview_inference_crosswalk.csv", index=False, float_format="%.17g")
    (ROOT / "docs/Post_Review_Inference_Results.md").write_text(technical_note(summary, archived, hashes_before), encoding="utf-8")
    assert input_hashes() == hashes_before, "Historical input was rewritten."
    print(f"Observed delta: {summary['observed_delta_r']:.15f}; bootstrap SE: {summary['bootstrap_se_delta_r']:.15f}.")
    print(f"Percentile CI: [{summary['percentile_ci_95_lower']:.15f}, {summary['percentile_ci_95_upper']:.15f}].")
    print(f"BCa CI: [{summary['bca_ci_95_lower']:.15f}, {summary['bca_ci_95_upper']:.15f}].")
    print(f"Post-review inference finished in {perf_counter() - started:.2f}s.")


if __name__ == "__main__":
    main()
