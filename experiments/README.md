# Exploratory experiments

These scripts are the analysis behind the notebook's decisions; notebook section 11 summarises their results. Run them from a working folder that has `src/` next to the scripts and the competition CSVs in `../data/`. Some scripts reuse cached arrays (`oof_cache.npz` etc.) written by `exp_main.py` and `exp_eval.py`.

They use only methods from the competition's explicit list:
- logistic regression, including Platt calibration
- linear and ridge regression
- gradient boosting
- random forest
- SVM
- k-NN
- Gaussian process
- PCA
- mixture models and EM

| Script | Question it answered |
|---|---|
| eda1 / eda2 / eda3 | Censoring table, adversarial validation, histogram bin offset, whether repair cost is predictable |
| exp_tune / exp_tune2 | LightGBM and logistic-regression hyperparameters (log-loss, AP) |
| exp_main / exp_eval | Censoring variants, dropping s120/s122, policies scored on the exact metric |
| exp_err | Error analysis: cost breakdown, misses, calibration, LR+GBM blend |
| exp_cost / exp_cost2 / exp_pin2 / exp_pin3 | Cost-model family, linear recalibration, pinball-loss tuning, bagging |
| cost_ens / cost_gp | Cost model: XGBoost/CatBoost ensembles, Gaussian process (no gain) |
| cal_exp | Calibration by logistic regression: Platt, quadratic, stacking |
| exp_shift / exp_shift2 / exp_prior | Why test predictions are higher: SHAP-style contributions, mixture and EM prior estimates |
| exp_sim / exp_simbk / exp_local | Simulated 5% depot: prior correction, cost breakdown, heteroscedastic residual pools |
| cmp_theirs / cmp_eval | Head-to-head with the team's earlier pipeline on the same folds (needs that folder's fleet_lib.py) |
| cmp_hist* | Whether the re-aligned histogram is trustworthy (per-bin AUC, rounding) and what it is worth to the cost model |
| test_fleet | Unit tests for the metric and the decision rule (checked against brute force) |
