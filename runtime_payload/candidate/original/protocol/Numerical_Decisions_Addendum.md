# NIQ–HLO: numerical decisions for the executable protocol

Version: 1.0 — 30 September 2026  
Status: **proposed final numerical specification; author approval is still required**.

## 0. Purpose, evidence and authority

This addendum supplies explicit recommendations for the numerical choices left open in the integrated approval-package completion report. They are proposed choices, not settings recovered from the new JSON and not results inferred from predictive performance.

It supplements the accepted `NIQ_HLO_Revision_Plan.md`, `Implementation_Brief.md`, master runbook and `Protocol_Review.md`. Preserve those originals. Keep the original RQ1/RQ2 calculations, data, inference and results unchanged.

The latest report states that there are 53 primary cells across 38 countries, distributed 7 / 17 / 20 / 9 across the four assessment periods. It reports 36 missing contextual feature entries to be imputed within training folds, 49/49 validation checks and 18 corrected floating-point coverage classifications. This addendum does **not** independently verify that report or the seven files referenced by local Windows paths. “36” means missing entries in the analysis matrix, not the total number of imputation operations across nested fits.

The reported predecessor scientific hash is:

`113537313ea06ebb91dc718a5e7498b7d37aa3c52588c6f7b279ed34567e28f1`

Integrating these numerical choices will create a **new specification and hash**. Do not approve or execute the predecessor merely because its construction checks passed. No empirical model was fitted to prepare this addendum.

## 1. Scientific choices to retain

Keep five-year block-preceding WDI context as the proposed primary construction, with assessment windows 2000–2004, 2005–2009, 2010–2014 and partial 2015–2017. Keep all seven context variables, without a feature-level screening step that removes education variables.

Use this fixed predictor order:

1. `SE.PRM.ENRR`
2. `SE.SEC.ENRR`
3. `SE.XPD.TOTL.GD.ZS`
4. `SH.DYN.MORT`
5. `SP.DYN.TFRT.IN`
6. `SP.URB.TOTL.IN.ZS`
7. `IT.NET.USER.ZS`

The other indicator precedes that block in a combined-model design matrix. Exclude GDP, population, country identifiers, region, calendar year/window indicators, ranks, coverage flags and uncertainty/audit quantities from predictors. Period labels remain available for splitting, construction and evaluation. No nonlinear transformations, interactions, feature selection, positivity constraints or prediction clipping are added.

Retain `HLO_to_P` as primary and `P_to_HLO` as the prespecified secondary direction. `P` is the period-specific QN-weighted psychometric score, **not exact published QNW**. Target scores and source-indicator scores must both be observed. Neither is imputed or changed because WDI is missing.

For each direction, retain M0 (training-mean reference), M1 (other indicator), M2 (context), M3 (indicator + context). Ridge is primary; ordinary weighted linear regression and Lasso are bounded benchmarks on the primary sample. “Ordinary linear regression” here means no coefficient penalty, but with the agreed equal-country observation weights; it does not mean changing to unweighted rows.

## 2. Exact coverage rule and correction audit

For an integer count of requested years `T > 0` and observed usable years `n_obs`, use:

```python
usable = 5 * n_obs >= 4 * T
minimum_observed_years = (4 * T + 4) // 5
```

These represent the exact 80% rule. They give 4/5, 8/10 and 10/12 years. Do not calculate the boundary with rounded percentages or binary floating-point ratios.

One calendar year is one annual observation per country-feature. Missing values do not become zero. Duplicate country-feature-year rows require an explicit source check; do not count them twice. A usable historical mean is the arithmetic mean of the available requested annual values. Otherwise, the feature is missing. Retain the complete requested/observed-year ledger.

Retain cells with at least five usable context features; at most two may require fold-specific feature imputation. This row rule applies equally to M0–M3 within a scenario: do not give the indicator-only model a different evaluation sample.

For assessment-specific histories, each positive-weight target contribution must have a usable history for a feature. If one does not, the combined context feature is missing. Do not renormalize across missing contributions. Existing target scores and their contribution weights remain unchanged. The relevant assessment schedule is direction-specific.

Save `coverage_threshold_corrections.csv` with the country-period-feature-contribution identifiers, requested/observed counts, old and corrected classifications, and effects on feature and row eligibility. Reconcile the 18 corrections reported by Codex. Do not force that count if the actual correction ledger differs. The new sample manifest must use the corrected integer rule.

## 3. Split allocation: countries are the grouping unit

### Outer validation

Use leave-one-country-out validation. Hold out all rows of each country together. Iterate over the ISO3 country codes in ascending ASCII order. Train using all other countries in that scenario's fixed sample.

No repeated outer split design is added. Every retained cell receives one outer-held-out prediction per model configuration in its scenario.

### Inner validation

For each outer training country set, construct exactly five grouped inner folds as follows:

1. For each ISO3 code, calculate SHA-256 of the UTF-8 string `NIQ-HLO-inner|20260930|<ISO3>`.
2. Sort countries by the resulting hexadecimal digest, using ISO3 as the secondary key.
3. Assign the country at zero-based position `j` to inner validation fold `j % 5`.
4. Keep all its period rows in that fold.

This is an explicit deterministic, seed-labelled allocation, balanced in country count rather than row count. Do not replace it with a library default, stratify using targets, or search over seeds. Apply the same allocation to all estimators and predictor sets evaluated on the same country set. Recreate it by the same rule after a scenario or deletion changes the country set; save every realized allocation.

Training/preprocessing must use only the applicable training fold. A scenario needs at least seven countries to support this fixed nested design: after one outer holdout, at least six training countries remain and every inner training fold has at least four countries. This is a computational feasibility floor, **not** evidence of adequate statistical power. Mark smaller scenarios infeasible rather than reducing the number of folds.

## 4. Equal-country weights

In any fitting subset with `N` rows, `C` countries and `m_i` rows belonging to country `i`, set:

`sample_weight[i,w] = N / (C * m_i)`.

The weights sum to `N`, with total weight `N/C` for each country. Recalculate from the particular inner or outer **training** subset, never from the original full-sample country counts.

Use these weights for coefficient fitting and the feature/target centering and scaling below. For evaluation, equivalent weights are `1 / (C * m_i)`, which sum to one. Do not additionally weight by population, psychometric sample size, QN factor, number of tests or measurement precision. Such source weights belong to indicator construction, not country-level predictive fitting.

This fixes the normalization that determines the numerical meaning of Ridge's penalty. Retain the stated alpha grids under this normalization; do not rescale them to obtain a preferred result. Ridge and Lasso alpha values do not have identical objective-function meanings [S1–S2].

## 5. Imputation, standardization and constant values

### Imputation

Within every applicable training subset, compute the **ordinary unweighted median of observed country-period values** for each context feature. For an even number of observations, use the mean of the two middle values. This makes the earlier training-fold-median proposal exact; it is not a weighted-median imputer.

Apply each training median unchanged to missing values in training and validation/test rows. Do not impute separately for the held-out country, borrow its other periods, use its outcome, or add missingness indicators. If any required context feature has no observed training values, record a structurally infeasible fit; do not drop that feature or replace it with zero.

### Standardization

After imputation, compute each predictor's weighted training mean and population-form weighted variance:

`mu = sum(weight * x) / sum(weight)`

`variance = sum(weight * (x - mu)**2) / sum(weight)`.

Use `z = (x - mu) / sqrt(variance)`. Use the same rule for the training target, separately within each target direction. Predict in standardized target units, then invert the target transformation before computing validation losses and final score metrics. No scaling is estimated over the full sample or over held-out rows. Weighted standardization and population-form variance are supported by the documented scaling API [S3]; the exact numerical rule here is the specification.

A column is numerically constant when its weighted standard deviation is no larger than `1e-12 * max(1, abs(mu))`. Preserve its schema position but set its standardized train and test column to zero; record the flag. Do not infer variation for a feature the training set did not identify.

If the **training target** is numerically constant by the same rule, use its weighted training mean as the constant prediction for that fit, record the degeneracy and leave selected alpha null. This is a deterministic special case, not a reason to inspect validation targets or alter the split. If it occurs in an inner fit, its constant predictions enter the prescribed validation score for every alpha.

## 6. Models, objectives and solver conventions

Use dense float64 arrays, the predictor order in Section 1, and row order `(ISO3 ascending ASCII, window_start ascending, window_end ascending)`, with one unique row per country-period in a scenario. Use an unpenalized intercept. Pin the actually tested Python, NumPy, SciPy and scikit-learn versions. No forced upgrade to a newly released version is required. Restrict numerical thread pools to one thread for the reference rebuild, and record the numerical libraries and platform.

### M0

Use the equal-country weighted mean of the applicable outer-training target in original units. Save it as a score-error reference. Do not interpret its cross-fold-varying constants as a substantive country-rank baseline.

### Ridge

For standardized predictors `Z` and target `z_y`, fit the objective:

`sum(sample_weight * (z_y - intercept - Z @ beta)**2) + alpha * sum(beta**2)`.

Use `Ridge(solver="svd", fit_intercept=True, positive=False)` or independently validated equivalent code. SVD is a noniterative solver; a Ridge tolerance parameter does not set its solution accuracy [S1]. No coefficient sign restriction is imposed.

Grid:

`[0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]`.

### Ordinary weighted linear regression

Use the unpenalized weighted least-squares objective. Solve the augmented matrix `[1, Z]` and standardized target after multiplying each row by `sqrt(sample_weight)`, using a minimum-norm SVD least-squares solve with relative singular-value cutoff `rcond=1e-12`. Record numerical rank and singular values. Rank deficiency produces the declared minimum-norm solution and a warning, not a switch to another estimator. Do not interpret nonidentified individual coefficients.

### Lasso

Use the objective:

`sum(sample_weight * residual**2) / (2*N) + alpha * sum(abs(beta))`.

Use `fit_intercept=True`, `selection="cyclic"`, `positive=False`, `warm_start=False`, `precompute=False`, `tol=1e-8`, `max_iter=100000`. Initialize every fit afresh. Lasso internally normalizes sample weights to sum to the sample count; the weights specified above already have that sum [S2].

Grid:

`[0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1]`.

One predetermined numerical retry is allowed for a convergence warning: repeat the same fit from a fresh estimator with `max_iter=1000000`, leaving all data, tolerance and scientific choices unchanged. Record both attempts, iterations and dual gap. Persistent nonconvergence is handled as a failed fit, not hidden by loosening tolerance.

## 7. Hyperparameter selection and numerical ties

For each alpha and predictor set, concatenate its five inner-validation prediction sets, each returned to original target units. Calculate MSE within each validation country, then average the country MSEs. Each outer-training country appears exactly once as an inner-validation country. Do not average fold MSEs without accounting for different country counts.

Let the smallest valid candidate score be `L_min`. Treat a candidate as numerically tied when:

`L_candidate <= L_min + 1e-10 + 1e-8 * max(1, abs(L_min))`.

Among tied candidates, choose the **largest alpha**. This is a numerical tie convention, not a statistical equivalence margin or a one-standard-error rule.

A candidate alpha must complete all inner folds. A persistently nonconvergent alpha is ineligible and logged; do not score it on only the folds that succeeded. A structural all-missing-feature failure invalidates the corresponding fit rather than being solved through alpha selection. If no candidate is valid, the model run is incomplete under Section 12.

After selection, refit preprocessing and the estimator from scratch on the full outer training set, then predict its held-out country. Never refit calibration on the test country. A best alpha at a grid boundary is reported; it does not authorize expanding the grid after viewing results.

## 8. Score loss, calibration and scale

For each model, direction and evaluated scenario define:

`MAE = mean_over_countries(mean_over_that_country's_cells(abs(pred - y)))`.

`MSE = mean_over_countries(mean_over_that_country's_cells((pred - y)**2))`.

`RMSE = sqrt(MSE)`.

Do not average country RMSEs and label that value the above RMSE. Report mean prediction bias using the same weights, with bias defined as `pred - y`. Keep HLO-unit and P-unit errors separate. A unitless MSE improvement against M0 may additionally be reported as `1 - MSE_model/MSE_M0`, with null when the denominator is zero. No global test-target standardization is used to make the two units appear comparable.

For descriptive calibration, fit observed `y = a + b * outer_held_out_prediction` by equal-country weighted least squares in original target units. Report intercept `a`, slope `b` and mean prediction bias. Compute one pooled summary per direction, explicitly labelled as mixing periods, and period-specific summaries when there are at least three countries and prediction variation exceeds the constant-value threshold. For undefined slope, report null and the reason. Calibration is diagnostic only: do not use it to adjust scored predictions. No coefficient p-values or standard errors are generated for these diagnostic fits.

## 9. Rank, ordering and group rules

### Ranks

Within each scenario, assessment period and direction, rank the observed targets, raw source indicators and corresponding held-out predictions on **the identical evaluated country set**. Rank 1 is highest. Use average ranks for exact ties in full-precision values. Do not round before ranking or break ties alphabetically; use ISO3 only for deterministic display ordering.

For a period with `n_w >= 2`:

`d_iw = abs(rank_prediction_iw - rank_target_iw) / (n_w - 1)`.

Average `d_iw` within each country over its evaluable periods and then across countries. Report period-specific losses and country/period denominators too. A one-country period is undefined for rank comparisons, remains in score evaluation and is explicitly excluded from the rank denominator.

Keep raw-source rank loss and fitted M1 rank loss separate. Improvement against baseline B is `loss_B - loss_M3`, so positive is better for M3. Percentage improvement is `100 * difference / loss_B` only if `loss_B > 0`.

These pooled cross-fitted rankings summarize outer-held-out **score predictions**. They are not a jointly held-out ranking of an entirely new set of countries. Report that limitation, do not treat pair comparisons as independent, and do not attach an unqualified new-country ranking guarantee.

### Supporting ordering summaries

Compute Spearman and Kendall tau-b separately within periods. Report null for undefined constant-score cases. Report strict pair reversals excluding pairs tied on either variable, with numerator, denominator and fraction. Do not obtain a p-value by treating all pairs or periods as independent. Do not average correlations across periods into a new headline coefficient.

### Tail groups

Use each variable's own sample 10th and 90th percentiles within the period, with linear interpolation. Include all scores at or beyond their respective thresholds. Report actual target/predicted group counts, intersection count, target-group recall and Jaccard overlap. Do not assume equal group sizes or force exactly one member. These are descriptive secondary outputs. Flag periods where either selected group contains fewer than three countries; keep the values but label them small-group results. All-score ties can produce nonselective groups and must be flagged rather than interpreted as successful discrimination.

Do not produce a substantive rank/tail baseline for M0. Also flag indicator/context models that repeatedly reduce to intercept-only fits.

## 10. Uncertainty and country-deletion refits

Do not add prediction intervals, significance tests from cross-validation variability, or new coefficient-inference claims. Original GDP uncertainty calculations remain unchanged.

For the primary five-year block specification and primary `HLO_to_P` direction, perform a full country-deletion sensitivity of the Ridge family:

- Delete all cells of country k.
- Rebuild the entire nested country-held-out evaluation on the remaining sample for M0–M3, including imputation, scaling, tuning and refitting.
- Recalculate evaluation weights, within-period ranks and M3-minus-baseline performance comparisons according to the declared sign convention.
- Repeat for every country of the primary sample, irrespective of whether the original result favours M3.

This is not accomplished by deleting that country's saved errors without refitting the remaining models. Report the minimum, median and maximum loss differences and the number of deletions changing the sign of the comparison. These are sensitivity ranges, **not confidence intervals**. Deletion changes training data and the comparison population; it does not isolate one causal source of disagreement.

No deletion-refit grid is added to reverse-direction, alternative-estimator or other sensitivity runs. This keeps the analysis bounded. Preserve every deletion result and flag incomplete refits.

## 11. Finite analysis manifest: no combinatorial search

### Primary runs

On the primary sample, run M0 and M1–M3 for Ridge, unpenalized weighted linear regression and Lasso, in both directions. M0 need not be computed repeatedly for identical splits. Estimators are reported separately; do not select a final algorithm using outer-test performance.

### Ridge-only sensitivity families, both directions

Run the following separately, changing only the named aspect:

| ID | Change |
| --- | --- |
| S10 | Ten-year block-preceding context |
| S12 | Twelve-year block-preceding context |
| SA5 | Five-year assessment-specific context with the fixed-contribution rule |
| SCC | Primary construction restricted to complete seven-feature cells; no imputation |
| SCOMP | Remove BRA, CAN, KAZ, MEX, SRB, USA and ZAF |
| STERM | Remove 2015–2017 assessment cells |
| S4 | Four-year windows: 2000–2003, 2004–2007, 2008–2011, 2012–2015 and partial 2016–2017; preceding five-year block context |

For each family fit M0–M3, not merely the combined model. Do not cross histories with alternative algorithms, exclusions and other sensitivity choices.

For S10, S12 and SA5, determine the exact intersection of primary-eligible and sensitivity-eligible country-period cells **before** fitting. Run both the primary historical construction and the alternative construction on that same intersection, with identical country splits and weights. Comparison sample sizes can therefore be below the reported availability totals (48, 42 or 51). Freeze each actual intersection manifest. Equal counts in the two directions do not establish identical cell identities.

For SCC, SCOMP and STERM, report conditional M3-versus-M1/M2 comparisons on the restricted sample. A difference from the full primary result reflects changed training/evaluation coverage as well as any named restriction; do not call it an isolated effect of imputation or composition. If original primary predictions are displayed on the restricted evaluation set, label them as such and recalculate ranks and weights there; do not confuse that display with a restricted-sample refit.

S4 constructs different targets and different observation windows. Report within-scheme predictive differences; do not match four- and five-year cells as though they were identical targets or infer a causal advantage from their raw sample-wide loss difference.

### Diagnostics without additional prediction models

Report error summaries from saved primary predictions for single- versus multiple-study cells, date-uncertain versus exact-date cells, HLO component depth, number of imputed features, and target-metadata availability. Use source-defined flags fixed before modelling. Show coefficients, selected alphas, grid-edge selections, convergence flags and calibration only as diagnostics. Do not use them to select a new specification.

Preserve outcome-blind date-boundary construction checks already completed in Phase A, clearly labelled. Do not add new empirical rounding/exclusion models without a dated amendment. Life-expectancy replacement, same-window predictive models, new forward-period tests, bootstrap intervals, synthetic missing-target experiments, other algorithms and new variables are **not enabled** in this version.

## 12. Paired failures and readiness

All M0–M3 comparisons within a scenario use identical planned test cells. A fitted result is valid only if it is finite, follows the complete training-only pipeline and has the declared convergence status.

Do not drop a failed country's rows in only one model. If a required Ridge model cannot produce valid predictions for every planned outer test cell, mark the corresponding primary scenario incomplete, retain the valid partial outputs as diagnostic and do not publish a primary paired loss on a favourable subset.

A failed OLS/Lasso benchmark does not suppress valid Ridge results. Mark that benchmark incomplete and do not use a different estimator to fill its failures. A sensitivity failure does not invalidate unrelated completed scenarios, but must be reported. A scenario below the declared country-count floor is marked infeasible from coverage, without trying different sample rules until it fits.

Numerical retries are limited to the predeclared Lasso retry. Fixing an implementation bug is allowed with a versioned change log, full rerun of affected results and independent verification. Any change to the scientific specification after predictive results are visible requires a dated amendment identifying the change as post-result; do not relabel it prospective.

## 13. Integration, validation and final approval

Codex must now:

1. Read the actual integrated JSON and author-facing document locally. Apply this addendum to their open fields, preserving all originals. If a proposed choice conflicts with an already approved decision, identify the exact conflict; do not guess approval.
2. Record the training-weight formulas, split algorithm, tolerances, objectives, solver controls, all evaluation rules and the finite execution manifest in the machine-readable specification. No decision field above may remain `TBD` or an unresolved candidate list.
3. Preserve and verify the corrected source/sample hashes, including exact sensitivity intersections and the coverage-correction ledger. No new empirical model is fitted during integration.
4. Add synthetic unit tests for weights, country splits, training-only transformations, numerical tie selection, threshold integers, constant columns, calibration, rank/tail ties and paired failures. Expand independent validation beyond the reported earlier 49 checks; the old count is not a target to preserve.
5. Include a hand-checkable exact coverage fixture for 4/5 passing, 3/5 failing, 8/10 passing, 9/12 failing and 10/12 passing. Check equal-country loss by hand, and verify that perturbing a held-out target cannot alter that fold's fit, preprocessing, split or alpha selection.
6. Rebuild the readable approval document and recompute the scientific-content hash using the project's documented canonicalization. Include every normative numerical setting and the frozen input/sample-manifest hashes in the scientific payload. Keep the actual code/environment identifiers in the final approval package as well. Preserve the predecessor hash as provenance. Do not exclude these conventions from hashing as “only implementation”.
7. Keep approval in a separate record that references the new immutable scientific hash. Keep `empirical_modelling_approved=false` until the author explicitly approves that final hash. Do not treat this addendum, a prior plan approval or validation success as permission to fit.
8. Return the updated seven-file approval package and independent validation report as **one ZIP attachment**, not only local file links. Keep core reporting, citation work and unrelated Phase A work running.

The next return should contain a fully specified protocol and a short author-readable decision summary. It should not ask again for choices that this addendum specifies. A genuine contradiction, missing source or infeasible measurement is an exception to resolve, not permission to invent a favourable answer.

After reconciliation and validation, the remaining action is the author's approval of the **new exact protocol hash**. No journal submission, payment, public data release or destructive repository operation is authorized here.

## 14. Source notes

The study choices and previously proposed model grids come from the accepted project plan and `Protocol_Review.md`. The latest sample/check counts and predecessor hash above come from the user's pasted Codex status report and remain reported rather than independently verified in this addendum.

The following official documentation was checked for API and objective-function details, not to claim that it scientifically validates our chosen numerical cutoffs or study design:

- [S1] scikit-learn, Ridge: objective, SVD solver, sample weights and solver-specific tolerance. https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html
- [S2] scikit-learn, Lasso: objective, sample-weight normalization, cyclic updates and convergence controls. https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html
- [S3] scikit-learn, StandardScaler: weighted fitting and population-form variance convention. https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html

Use the locally pinned, tested package versions and record any API compatibility issue. The equations and rules above, rather than unrecorded library defaults, define the intended computations.
