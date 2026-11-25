# Submission log

Fill in one row per Kaggle submission, and give each one a matching Git tag.

| Tag | Date | Who | File | Change | Local CV (cost/20k, old depot) | 5%-depot sim. | Public LB | Notes |
|---|---|---|---|---|---|---|---|---|
| sub-01-2026-11-__ | | | sub_v1.csv | V1 baseline: logistic regression, p > 0.5, median budget | 52,998 ± 182 | n/a | | |
| sub-02-2026-11-__ | | | sub_v2.csv | V2: LightGBM + Platt + decision rule + depot alignment | 19,906 ± 680 | 121,109 | | |
| sub-03-2026-11-__ | | | sub_v3.csv | V3: + cost model, censoring weights, stable-sensor LGB/XGB/Cat ensemble, Platt, prior π′ = 0.0477 | 14,256 ± 725 | 64,818 | | **final** |

Earlier team submission (before this pipeline): `submission_em.csv`, public LB 20,277.33. On the same folds and scoring it costs 16,662 (old depot) and 77,534 (5% depot).

**Final selection rule:** choose the best *local CV* model whose public LB is consistent with it (V3). Do not pick by public LB alone, because the public board is only 30% of test (about 6,000 vehicles).
