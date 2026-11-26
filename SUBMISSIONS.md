# Submission log

Fill in one row per Kaggle submission, and give each one a matching Git tag.

| Tag | Date | Who | File | Change | Local CV (cost/20k, old depot) | 5%-depot sim. | Public LB | Notes |
|---|---|---|---|---|---|---|---|---|
| sub-01-2026-11-__ | | | sub_v1.csv | V1 baseline: logistic regression, p > 0.5, median budget | 52,998 ± 182 | n/a | | |
| sub-02-2026-11-__ | | | sub_v2.csv | V2: LightGBM + Platt + decision rule + depot alignment | 19,906 ± 680 | 121,109 | | |
| sub-03-2026-11-25 | | | sub_v3.csv (first) | V3a: simple cost model, stable-sensor LGB/XGB/Cat ensemble, Platt, prior π′ = 0.0477 | 14,256 ± 725 | 64,818 | **19,515** | |
| sub-04-2026-11-25 | | | sub_v3.csv (V3b) | V3b: + engineered cost model (log-ratios, histogram shares, ridge blend), prior π′ = 0.035 | 14,190 ± 657 | 65,774 | **17,832** | **final (selected)** |
| sub-05-2026-11-25 | | | sub_v3c.csv | V3c: + cost-weighted training, histogram shape features, row subsampling in the cost model | 14,132 ± 307 | 63,342 | slightly worse than 17,832 (fill in) | not selected |

Earlier team submission (before this pipeline): `submission_em.csv`, public LB 20,277.33. On the same folds and scoring it costs 16,662 (old depot) and 77,534 (5% depot).

**Final selection rule:** choose the best *local CV* model whose public LB is consistent with it (V3). Do not pick by public LB alone, because the public board is only 30% of test (about 6,000 vehicles).
