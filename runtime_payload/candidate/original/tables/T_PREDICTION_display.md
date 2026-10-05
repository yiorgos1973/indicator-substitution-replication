# Out-of-sample prediction comparison

## Primary models and benchmarks

| direction | scenario | specification | comparison | mse | rmse | mse_improvement_vs_m0 | n_countries | n_cells | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HLO → P | PRIMARY | M0 Reference | Primary five-year construction | 110.086 | 10.492 | 0.000 | 38 | 53.000 | calculated_validated |
| HLO → P | PRIMARY | M1 Ridge | Primary five-year construction | 51.974 | 7.209 | 0.528 | 38 | 53.000 | calculated_validated |
| HLO → P | PRIMARY | M2 Ridge | Primary five-year construction | 64.223 | 8.014 | 0.417 | 38 | 53.000 | calculated_validated |
| HLO → P | PRIMARY | M3 Ridge | Primary five-year construction | 61.795 | 7.861 | 0.439 | 38 | 53.000 | calculated_validated |
| HLO → P | PRIMARY | M1 OLS (benchmark) | Primary five-year construction | 48.456 | 6.961 | 0.560 | 38 | 53.000 | calculated_validated |
| HLO → P | PRIMARY | M2 OLS (benchmark) | Primary five-year construction | 65.087 | 8.068 | 0.409 | 38 | 53.000 | calculated_validated |
| HLO → P | PRIMARY | M3 OLS (benchmark) | Primary five-year construction | 58.829 | 7.670 | 0.466 | 38 | 53.000 | calculated_validated |
| HLO → P | PRIMARY | M1 Lasso (benchmark) | Primary five-year construction | 50.377 | 7.098 | 0.542 | 38 | 53.000 | calculated_validated |
| HLO → P | PRIMARY | M2 Lasso (benchmark) | Primary five-year construction | 60.478 | 7.777 | 0.451 | 38 | 53.000 | calculated_validated |
| HLO → P | PRIMARY | M3 Lasso (benchmark) | Primary five-year construction | 48.837 | 6.988 | 0.556 | 38 | 53.000 | calculated_validated |
| P → HLO | PRIMARY | M0 Reference | Primary five-year construction | 5682.209 | 75.380 | 0.000 | 38 | 53.000 | calculated_validated |
| P → HLO | PRIMARY | M1 Ridge | Primary five-year construction | 2408.481 | 49.076 | 0.576 | 38 | 53.000 | calculated_validated |
| P → HLO | PRIMARY | M2 Ridge | Primary five-year construction | 1299.342 | 36.046 | 0.771 | 38 | 53.000 | calculated_validated |
| P → HLO | PRIMARY | M3 Ridge | Primary five-year construction | 1130.524 | 33.623 | 0.801 | 38 | 53.000 | calculated_validated |
| P → HLO | PRIMARY | M1 OLS (benchmark) | Primary five-year construction | 2399.600 | 48.986 | 0.578 | 38 | 53.000 | calculated_validated |
| P → HLO | PRIMARY | M2 OLS (benchmark) | Primary five-year construction | 1421.069 | 37.697 | 0.750 | 38 | 53.000 | calculated_validated |
| P → HLO | PRIMARY | M3 OLS (benchmark) | Primary five-year construction | 1256.098 | 35.441 | 0.779 | 38 | 53.000 | calculated_validated |
| P → HLO | PRIMARY | M1 Lasso (benchmark) | Primary five-year construction | 2405.347 | 49.044 | 0.577 | 38 | 53.000 | calculated_validated |
| P → HLO | PRIMARY | M2 Lasso (benchmark) | Primary five-year construction | 1305.575 | 36.133 | 0.770 | 38 | 53.000 | calculated_validated |
| P → HLO | PRIMARY | M3 Lasso (benchmark) | Primary five-year construction | 1143.685 | 33.818 | 0.799 | 38 | 53.000 | calculated_validated |

## Matched history sensitivities

| direction | scenario | specification | comparison | summary_metric | summary_estimate | n_countries | status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| HLO → P | S10 | M3 Ridge | alternative_10y versus primary_5y_matched; M3 ridge | matched_primary_mse_minus_alternative_mse | 7.738 | 36 | calculated_validated |
| P → HLO | S10 | M3 Ridge | alternative_10y versus primary_5y_matched; M3 ridge | matched_primary_mse_minus_alternative_mse | 120.340 | 36 | calculated_validated |
| HLO → P | S12 | M3 Ridge | alternative_12y versus primary_5y_matched; M3 ridge | matched_primary_mse_minus_alternative_mse | 16.538 | 33 | calculated_validated |
| P → HLO | S12 | M3 Ridge | alternative_12y versus primary_5y_matched; M3 ridge | matched_primary_mse_minus_alternative_mse | -85.195 | 33 | calculated_validated |
| HLO → P | SA5 | M3 Ridge | assessment_specific_5y_fixed versus primary_5y_matched; M3 ridge | matched_primary_mse_minus_alternative_mse | -0.434 | 36 | calculated_validated |
| P → HLO | SA5 | M3 Ridge | assessment_specific_5y_fixed versus primary_5y_matched; M3 ridge | matched_primary_mse_minus_alternative_mse | -34.319 | 36 | calculated_validated |

## Other named sensitivities

| direction | scenario | specification | comparison | mse | rmse | mse_improvement_vs_m0 | n_countries | n_cells | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HLO → P | S4 | M3 Ridge | four_year_preceding_5y | 72.157 | 8.495 | 0.365 | 33 | 49.000 | calculated_validated |
| P → HLO | S4 | M3 Ridge | four_year_preceding_5y | 1056.781 | 32.508 | 0.779 | 33 | 49.000 | calculated_validated |
| HLO → P | SCC | M3 Ridge | restricted_primary_5y | 58.211 | 7.630 | 0.436 | 23 | 29.000 | calculated_validated |
| P → HLO | SCC | M3 Ridge | restricted_primary_5y | 1116.995 | 33.421 | 0.802 | 23 | 29.000 | calculated_validated |
| HLO → P | SCOMP | M3 Ridge | restricted_primary_5y | 67.705 | 8.228 | 0.406 | 32 | 45.000 | calculated_validated |
| P → HLO | SCOMP | M3 Ridge | restricted_primary_5y | 1061.829 | 32.586 | 0.826 | 32 | 45.000 | calculated_validated |
| HLO → P | STERM | M3 Ridge | restricted_primary_5y | 67.892 | 8.240 | 0.417 | 34 | 44.000 | calculated_validated |
| P → HLO | STERM | M3 Ridge | restricted_primary_5y | 1085.811 | 32.952 | 0.805 | 34 | 44.000 | calculated_validated |

## Country-deletion sensitivity

| direction | scenario | specification | comparison | summary_metric | summary_estimate | n_countries | status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| HLO → P | PRIMARY_DELETE | M3 Ridge | M3 Ridge improvement over M1 Ridge under full country-deletion refits | country_deletion_max_mse_improvement | 2.606 | 38 | calculated_validated |
| HLO → P | PRIMARY_DELETE | M3 Ridge | M3 Ridge improvement over M1 Ridge under full country-deletion refits | country_deletion_median_mse_improvement | -9.785 | 38 | calculated_validated |
| HLO → P | PRIMARY_DELETE | M3 Ridge | M3 Ridge improvement over M1 Ridge under full country-deletion refits | country_deletion_min_mse_improvement | -15.456 | 38 | calculated_validated |
| HLO → P | PRIMARY_DELETE | M3 Ridge | M3 Ridge improvement over M1 Ridge | country_deletion_sign_changes | 1.000 | 38 | calculated_validated |
| HLO → P | PRIMARY_DELETE | M3 Ridge | M3 Ridge improvement over M2 Ridge under full country-deletion refits | country_deletion_max_mse_improvement | 12.566 | 38 | calculated_validated |
| HLO → P | PRIMARY_DELETE | M3 Ridge | M3 Ridge improvement over M2 Ridge under full country-deletion refits | country_deletion_median_mse_improvement | 3.981 | 38 | calculated_validated |
| HLO → P | PRIMARY_DELETE | M3 Ridge | M3 Ridge improvement over M2 Ridge under full country-deletion refits | country_deletion_min_mse_improvement | -2.662 | 38 | calculated_validated |
| HLO → P | PRIMARY_DELETE | M3 Ridge | M3 Ridge improvement over M2 Ridge | country_deletion_sign_changes | 8.000 | 38 | calculated_validated |
