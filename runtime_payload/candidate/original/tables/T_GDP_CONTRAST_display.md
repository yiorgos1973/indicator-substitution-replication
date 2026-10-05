# GDP associations, paired contrast, and selected sensitivities

| section | scenario | metric | estimate | lower | upper | interval_or_test | n_economies |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Primary | QNW-GDP | Pearson correlation | 0.616 | — | — | none | 65 |
| Primary | HLO-GDP | Pearson correlation | 0.690 | — | — | none | 65 |
| Primary contrast | QNW minus HLO | Difference in dependent correlations | -0.074 | -0.207 | 0.045 | Zou 95% interval | 65 |
| Primary contrast | QNW minus HLO | Difference in dependent correlations | -0.074 | -0.237 | 0.073 | Paired economy bootstrap percentile 95% | 65 |
| Primary contrast | QNW minus HLO | Difference in dependent correlations | -0.074 | -0.247 | 0.066 | Paired economy bootstrap BCa 95% | 65 |
| Primary test | QNW minus HLO | Williams two-sided p | 0.218 | — | — | Williams test | 65 |
| Fixed-sample sensitivity | Primary reference | QNW-GDP minus HLO-GDP | -0.074 | — | — | none | 65 |
| Fixed-sample sensitivity | QNW v1.3.3 | QNW-GDP minus HLO-GDP | -0.082 | — | — | none | 65 |
| Fixed-sample sensitivity | Latest-period HLO | QNW-GDP minus HLO-GDP | -0.094 | — | — | none | 65 |
| Coverage-changing sensitivity | UN member-observer population | QNW-GDP minus HLO-GDP | -0.076 | — | — | none | 64 |
| Coverage-changing sensitivity | QNW plus SAS | QNW-GDP minus HLO-GDP | 0.009 | — | — | none | 80 |
| Coverage-changing sensitivity | QNW plus SAS plus geographic imputation | QNW-GDP minus HLO-GDP | 0.009 | — | — | none | 80 |
| Coverage-changing sensitivity | Balanced all-six HLO | QNW-GDP minus HLO-GDP | -0.034 | — | — | none | 41 |
| Coverage-changing sensitivity | All-available published HLO | QNW-GDP minus HLO-GDP | -0.005 | — | — | none | 107 |
| Weighting sensitivity | Equal-economy association | QNW-GDP minus HLO-GDP | -0.074 | — | — | none | 65 |
| Weighting sensitivity | Population-weighted association | QNW-GDP minus HLO-GDP | -0.133 | — | — | none | 65 |

- The sign is always QNW-GDP correlation minus HLO-GDP correlation; negative values favour a larger observed HLO-GDP correlation.
- RQ2 uses natural-log 2017 PPP GDP per capita and equal economy weights unless the row says population-weighted.
- The paired bootstrap resamples complete economy triplets. Its interval is not a null distribution and supplies no bootstrap p-value here.
- QNW plus SAS includes school-assessment content and changes sample coverage; it is not direct QNW.
- Intervals containing zero do not establish equivalence.
