# NIQ–HLO integrated protocol for author approval

Status: **fully specified proposal pending author approval**  
Empirical modelling approved: **no**  
Scientific specification hash: `13fb577533c28acc88613fcf84549784e130bd658139ea7bbbfc7f77a3b8e773`

## Author action

The numerical addendum has been integrated. The scientific and numerical protocol is complete; no empirical predictive model was fitted. The only remaining author action is to approve or reject the exact hash above in a separate record. Validation success does not constitute approval.

## Scope and fixed sample construction

The primary analysis uses period-specific P and HLO for 2000–2004, 2005–2009, 2010–2014 and partial 2015–2017, with five calendar years of block-preceding WDI context. `HLO_to_P` is primary and `P_to_HLO` is secondary. P is the period-specific QN-weighted psychometric score, not exact published QNW.

The ordered context vector is: `SE.PRM.ENRR`, `SE.SEC.ENRR`, `SE.XPD.TOTL.GD.ZS`, `SH.DYN.MORT`, `SP.DYN.TFRT.IN`, `SP.URB.TOTL.IN.ZS`, `IT.NET.USER.ZS`. In M3 the required other indicator comes first. GDP, population, identifiers, region, calendar/window indicators, ranks, coverage flags and uncertainty/audit fields are excluded.

Coverage uses integer arithmetic: `5 × n_observed >= 4 × T`, with minimum `((4 × T) + 4) // 5`. Thus the thresholds are 4/5, 8/10 and 10/12. A passing feature is the arithmetic mean of its observed requested years. A cell needs at least five of seven usable features. Targets and the required other indicator must be observed and are never imputed. WDI missingness never changes target contributions.

The assessment-specific sensitivity requires every positive-weight contribution to pass coverage. It does not renormalize contextual weights. The prior renormalized construction remains preserved as a historical candidate.

## Reconciled outcome-blind samples

| scenario_id   | target_direction   | construction_role            |   retained_cells |   retained_countries |   windows | below_seven_country_floor   | nested_design_feasible   |
|:--------------|:-------------------|:-----------------------------|-----------------:|---------------------:|----------:|:----------------------------|:-------------------------|
| PRIMARY       | HLO_to_P           | primary_5y                   |               53 |                   38 |         4 | False                       | True                     |
| PRIMARY       | P_to_HLO           | primary_5y                   |               53 |                   38 |         4 | False                       | True                     |
| S10           | HLO_to_P           | alternative_10y              |               48 |                   36 |         4 | False                       | True                     |
| S10           | HLO_to_P           | primary_5y_matched           |               48 |                   36 |         4 | False                       | True                     |
| S10           | P_to_HLO           | alternative_10y              |               48 |                   36 |         4 | False                       | True                     |
| S10           | P_to_HLO           | primary_5y_matched           |               48 |                   36 |         4 | False                       | True                     |
| S12           | HLO_to_P           | alternative_12y              |               42 |                   33 |         4 | False                       | True                     |
| S12           | HLO_to_P           | primary_5y_matched           |               42 |                   33 |         4 | False                       | True                     |
| S12           | P_to_HLO           | alternative_12y              |               42 |                   33 |         4 | False                       | True                     |
| S12           | P_to_HLO           | primary_5y_matched           |               42 |                   33 |         4 | False                       | True                     |
| S4            | HLO_to_P           | four_year_preceding_5y       |               49 |                   33 |         4 | False                       | True                     |
| S4            | P_to_HLO           | four_year_preceding_5y       |               49 |                   33 |         4 | False                       | True                     |
| SA5           | HLO_to_P           | assessment_specific_5y_fixed |               51 |                   36 |         4 | False                       | True                     |
| SA5           | HLO_to_P           | primary_5y_matched           |               51 |                   36 |         4 | False                       | True                     |
| SA5           | P_to_HLO           | assessment_specific_5y_fixed |               51 |                   36 |         4 | False                       | True                     |
| SA5           | P_to_HLO           | primary_5y_matched           |               51 |                   36 |         4 | False                       | True                     |
| SCC           | HLO_to_P           | restricted_primary_5y        |               29 |                   23 |         4 | False                       | True                     |
| SCC           | P_to_HLO           | restricted_primary_5y        |               29 |                   23 |         4 | False                       | True                     |
| SCOMP         | HLO_to_P           | restricted_primary_5y        |               45 |                   32 |         4 | False                       | True                     |
| SCOMP         | P_to_HLO           | restricted_primary_5y        |               45 |                   32 |         4 | False                       | True                     |
| STERM         | HLO_to_P           | restricted_primary_5y        |               44 |                   34 |         3 | False                       | True                     |
| STERM         | P_to_HLO           | restricted_primary_5y        |               44 |                   34 |         3 | False                       | True                     |

The correction ledger contains **18** float-versus-integer threshold corrections. The manifest and correction ledger use the exact integer rule. Scenario feasibility requires at least seven countries.

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

Python 3.12.8; NumPy 1.26.4; SciPy 1.13.1; scikit-learn 1.5.1; pandas 2.2.2; one numerical thread. Dense float64 arrays and deterministic row ordering are required.
