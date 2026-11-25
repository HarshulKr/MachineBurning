# Viva notes: what every member must be able to explain

## 1. Problem and metric
- Cost per vehicle: if flagged, 10 + 0.20·(c−b)⁺ + 0.02·(b−c)⁺; if not flagged, c. Here c = repair cost if the vehicle fails, else 0.
- **Why not just a classifier?** Missing a failure costs its full repair (mean ≈ 492), while an inspection costs 10. A p > 0.5 threshold therefore misses almost every failure: V1 costs 53k against 15k for V3.

## 2. Decision theory: derive it on the board
- Not flagged: E[cost] = p·E[c | fail].
- Flagged with budget b: E[cost] = 10 + E[0.2(c−b)⁺ + 0.02(b−c)⁺], where c = 0 with probability 1−p.
- d/db: −0.2·P(c > b) + 0.02·P(c ≤ b) = 0, so **P(c ≤ b\*) = 0.909**. This is the pinball (quantile) loss.
- With the point mass at 0: b\* = 0 if p < 0.0909, else b\* = F⁻¹((p − 0.0909)/p).
- Flag iff E[cost | flagged, b\*] < p·E[c | fail].
- The rule needs **calibrated p** (Platt scaling: a 1-feature logistic regression on the model's log-odds, fitted on out-of-fold predictions) and a **predictive cost distribution** F: exp(μ̂(x) + residual), with the residuals taken from an inner CV.

## 3. Validation design
- Stratified 5-fold, repeated 3 times (about 400 positives). We report mean ± std and compare in a paired way.
- **Scoring uses only rows with followup_days = 365.** Censored non-failures have incomplete labels.
- Calibration is fitted on the *other* folds, and the cost residuals come from an *inner* CV, so neither is optimistic.
- A random split cannot see the depot shift, so we built a **simulated new depot**: resample the OOF rows so that failures make up 5%.
- The public LB is 30% of test (about 6k rows, about 300 failures) and noisy, so we never tune on it.

## 4. Depot-shift analysis (the data/error analysis factor)
- **Adversarial validation:** a classifier separates train from test with AUC ≈ 1.0.
- **Histogram s054–s063:** offset by one bin in test, so we re-align it. s057 is rounded to hundreds in test, so train is rounded to match. After that, each bin matches train as closely as the stable sensors do. The block is used only in the cost model, where dropping it costs about 8% in pinball loss. **s047:** missing in all of test, dropped. **s120/s122:** re-scaled (e.g. a constant +16M offset), dropped; CV shows dropping them costs nothing. **s055/s062:** rounded to tens in test, so train is rounded to match.
- **Prior shift:** after these fixes, the remaining shift is in the *upper tails* of many sensors while the medians match. Under label shift, P_test(x > t) = π′·P(x > t | fail) + (1−π′)·P(x > t | ok). Solving this per sensor gives π′ ≈ 0.05 for all 66 informative sensors.
- **EM cross-check (Saerens et al.):** treat the label as latent. E-step: re-weight the posteriors p′ ∝ (π′/π)·p. M-step: π′ = mean(p′). On train it recovers 0.0066 (sanity check); on test it gives 0.049.
- **Caveat:** the estimate drops as the threshold rises (0.08 at q90, 0.047 at q99.5). This means mild covariate drift in healthy vehicles too, so we use the high-threshold and EM values (π′ = 0.0477). The simulated cost is flat for π′ between about 0.04 and 0.06.

## 5. Each version and its gain (old-depot CV, cost per 20k)
| | Cost | Change |
|---|---|---|
| Flag nobody | 72,167 | |
| V1 | 52,998 | logistic regression (signed-log, standardised, C = 0.1), p > 0.5, median budget |
| | 22,749 | the same logistic regression + calibration + decision rule, which shows the decision rule matters most |
| V2 | 19,906 | LightGBM (400 trees, 7 leaves, min_child 50) + Platt + decision rule (constant cost distribution) |
| | 15,685 | + per-vehicle cost model (LightGBM on log cost, linear recalibration, inner-CV residuals) |
| | 15,121 | + censored negatives get weight 0 in training |
| | 14,612 | + drop s120/s122 |
| V3 | 14,256 | + mean of LightGBM/XGBoost/CatBoost on depot-stable sensors only (Platt-calibrated); then the prior correction for test (simulated 5% depot: 82.6k → 64.8k) |
| Earlier team pipeline | 16,662 | uncalibrated ensemble, histogram block dropped (weaker cost model), EM prior 3.6% (5% depot: 77.5k) |

## 6. Limitations of V3
- It assumes pure label shift, but there is some covariate drift.
- It assumes the cost distribution given a failure is unchanged in the new depot.
- Rush and return costs still dominate, because log-cost residual sd ≈ 0.9 is a noise floor (pinball loss 27 per failure, against 46 for a constant budget).
- Censoring is handled by weighting, not with a survival model.

## 7. Course theory to revise
Logistic regression as MLE, and Platt calibration as a 1-feature logistic regression; gradient boosting (additive trees fitted to the negative gradient); bias–variance (shallow trees, min_child_samples, early-stopping evidence); EM and latent variables; the quantile/pinball loss; adversarial validation; imbalanced metrics (AUC vs AP, why accuracy is useless at 0.7% positives).
