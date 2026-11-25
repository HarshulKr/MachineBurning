# Fleet Maintenance Planning: ML Project 2 (CSE 343 / ECE 363, Monsoon 2026)

Kaggle competition: <https://www.kaggle.com/competitions/mlprojectfleet2>

| File | What it is |
|---|---|
| `fleet_final_report.ipynb` | **Final report notebook.** Runs top to bottom in about 12 minutes and writes the three CSVs. Upload it to Kaggle as-is, because it finds `/kaggle/input/**/train.csv` on its own. |
| `submissions/sub_v1.csv` | V1 baseline: logistic regression, flag if p > 0.5, budget = median cost |
| `submissions/sub_v2.csv` | V2: LightGBM calibrated with Platt scaling, Bayes-optimal flag and budget, depot harmonisation |
| `submissions/sub_v3.csv` | **V3 (final):** V2, plus a per-vehicle repair-cost model, censoring weights, a LightGBM/XGBoost/CatBoost ensemble on depot-stable sensors, and a correction for the depot's failure prior |
| `SUBMISSIONS.md` | Submission log: tag, date, who, change, CV, public LB |
| `VIVA_NOTES.md` | What every member must be able to explain (factor 5) |
| `experiments/` | Exploratory scripts behind the decisions; their results are summarised in notebook section 11 |

## Results (out-of-fold cost per 20,000 vehicles; lower is better)

| Version | Key change | Old-depot CV | Simulated 5%-failure depot |
|---|---|---|---|
| Flag nobody | reference | 72,167 | 543,189 |
| V1 | logistic regression, p > 0.5, median budget | 52,998 ± 182 | n/a |
| V2 | LightGBM + Platt calibration + expected-cost decision rule | 19,906 ± 680 | 121,109 |
| V3 | + cost model, censoring weights, stable-sensor GBM ensemble, prior correction π′ = 0.0477 | **14,256 ± 725** | **64,818** |
| Earlier team pipeline (`submission_em.csv`, LB 20,277) | same folds and scoring | 16,662 | 77,534 |

## Rules compliance
Every method is on the competition's explicit list: logistic regression (V1, and Platt calibration), linear regression (re-scaling the cost predictions), gradient boosting (LightGBM, XGBoost, CatBoost), simple model combination (averaging the three), mixture models and EM (the failure-prior estimate). The libraries are numpy, pandas, scikit-learn, lightgbm, xgboost, catboost and matplotlib, and the only data used is the competition's `train.csv` and `test.csv`.

## The four findings that drive the solution
1. **Decision theory.** The optimal budget is the 0.20/0.22 = **90.9% quantile** of the predictive cost distribution, which puts a point mass 1−p at zero cost. Flag a vehicle when the expected cost with inspection is below p·E[c | fail]. The flag threshold falls out of the 10 / 0.20 / 0.02 costs, so nothing is hand-tuned.
2. **Repair cost is predictable** from the sensors (log-space R² ≈ 0.70), which makes per-vehicle budgets possible. This was the single biggest gain from V2 to V3.
3. **Depot shift in the sensors.** Histogram s054–s063 is offset by one bin in test (re-aligned, and used in the cost model), s047 is all-missing in test, s120 and s122 are re-scaled (dropped), and s055/s062 are rounded to tens and s057 to hundreds in test (train rounded to match).
4. **Depot shift in the failure rate.** About 66 independent sensors and EM over the latent label all say the new depot fails at about **4.8%**, against 0.66% in train. Calibrated probabilities are moved to that prior with Bayes' rule. On a simulated 5% depot this cuts cost from 82.6k to 64.8k.

## How to reproduce
```bash
pip install numpy pandas scipy scikit-learn lightgbm xgboost catboost matplotlib jupyter
mkdir data && cp /path/to/{train,test}.csv data/
jupyter nbconvert --to notebook --execute fleet_final_report.ipynb
```

## Git workflow (required by the project rules)
Each member commits their own work from their own GitHub account, because the commits are the evidence of contribution. After every Kaggle submission:
```bash
git add -A && git commit -m "V3: cost model + stable ensemble + prior correction; CV 14256±725"
git tag -a sub-NN-2026-11-DD -m "sub_v3.csv; public LB <score>"
git push && git push --tags
```
Then add a row to `SUBMISSIONS.md`. Remember to add the TA as a collaborator on the repo.
