# Final Report: Fleet Maintenance Planning (ML Project 2)

**Team:** Machine Burning
**Competition:** Kaggle *ML Project 2*, Fleet Maintenance Planning
**Final submission:** `sub_v6.csv`, with a public score of **15,928.88**
**Our three best models:** **V5** (higher budgets), **V6** (final model) and **V7** (six classifiers)

From our earlier team pipeline to V6, the public cost came down by about **21%** (20,277 → 15,928.88).

---

## 1. The Problem

For each of the 20,000 test vehicles we have to decide two things, i.e. whether to inspect it (`flag`) and how much to set aside for spare parts (`budget`). The score is the total cost of these decisions, so a lower score is better.

| Situation | Cost |
|---|---|
| A vehicle is inspected | 10 |
| A failing vehicle is not inspected | the full `repair_cost` |
| The budget is too low | 0.20 for every missing unit |
| The budget is too high | 0.02 for every unused unit |

**About the data**

- **Training:** 56,000 vehicles from one depot, with `followup_days`, `failure` and `repair_cost`.
- **Test:** 20,000 vehicles from a **different depot**, with only the sensor readings.
- **Features:** 170 unnamed sensors with some missing values. We found seven groups of 10 sensors which behave like histograms.
- **Imbalance:** only **397** training vehicles failed, which is about 0.7%.
- **Follow-up:** a vehicle which looks healthy after a short follow-up may have failed later.

**How we evaluated our models:** we used cross-validation along with our own implementation of the competition cost, since plain accuracy tells us very little when failures are so rare. From V4 onwards we also tested on a simulated version of the new depot. The public leaderboard covers 30% of the test vehicles and the final ranking uses the other 70%.

---

## 2. Our Progress

| Version | Main idea | Test inspections | Local cost (per 20,000) | Public score |
|---|---|---:|---:|---:|
| Earlier pipeline | Ensemble with a simple cost estimate and EM | - | 16,662 (A) | 20,277 |
| V1 | Logistic regression used as a plain classifier | - | 52,998 (A) | not recorded |
| V2 | Calibrated LightGBM and a cost-based decision rule | 1,352 | 19,906 (A) | not recorded |
| V3a | Cost model for each vehicle and three classifiers | 1,940 | 14,256 (A) | 19,515 |
| V3b | Lower assumed failure rate and a new cost mix | - | - | 17,832 |
| V3c | A variation of V3b | - | - | slightly worse than V3b |
| V4a | All vehicles as reference and three cost models | 1,560 | 54,018 (B) | 17,407.88 |
| V4b | V4a with a lower odds multiplier | 1,444 | 54,546 (B) | 17,979.60 |
| **V5** | V4a with budgets × 1.15 | 1,560 | 55,163 (C) | **17,251.63** |
| **V6** | Seven histogram groups and a separate cost range per vehicle | 1,570 | 50,334 (C) | **15,928.88** |
| **V7** | V6 with six classifiers | 1,595 | - | **16,184.67** |

The local costs come from three different set-ups, so only values with the same letter can be compared:

- **(A)** cross-validation on the old depot;
- **(B)** a simulated new depot with 4% failures, on one set of data splits;
- **(C)** the same simulated depot on a different set of splits.

**Our journey in brief**

1. **Taking better decisions (V1 to V2):** choosing flags and budgets using the cost function.
2. **Understanding the new depot (V3 to V4):** correcting the sensor differences and the higher failure rate.
3. **Predicting repair costs better (V5 to V6):** adjusting the budgets and building much better cost features.

---

## 3. How We Got Here (V1 to V4)

### V1 and V2: the decisions mattered more than the model

V1 flagged a vehicle only when its failure probability was above 0.5 and gave every flagged vehicle the same budget. Its ranking of vehicles was already good (AUC 0.986), but it missed most of the failures. We then tried changing one thing at a time:

| Configuration | Local cost |
|---|---:|
| V1: logistic regression, 0.5 threshold, one budget | 52,998 |
| LightGBM with the same rule | 53,655 |
| Logistic regression with calibration and the cost-based rule | 22,749 |
| **V2: LightGBM with calibration and the cost-based rule** | **19,906** |

This clearly showed that most of the gain came from the decision rule and not from a stronger classifier.

### V3: a separate cost for each vehicle

V2 still assumed the same repair cost for everyone. We added the following one by one:

| Step | Local cost |
|---|---:|
| V2 | 19,906 |
| + a cost model for each vehicle | 15,685 |
| + zero weight for healthy vehicles with a short follow-up | 15,121 |
| + dropping the shifted sensors `s120` and `s122` | 14,612 |
| + LightGBM, XGBoost and CatBoost on stable sensors | **14,256** |

We also corrected for the higher failure rate of the new depot. In V3a we assumed about 4.8%, and in V3b we reduced it to 3.5% and changed the cost mix, which brought the public score from 19,515 to 17,832.

### V4: choosing the right reference

We found that vehicles with a short follow-up read much higher on the usage-like sensors and that the test depot looks like all the training vehicles together. Using all of them as the reference, the EM estimate gave an odds multiplier of 3.97 instead of about 5.4. V4a also combined three cost models. It scored 17,407.88, while V4b with a multiplier of 3.0 scored 17,979.60.

---

## 4. Selected Model 1: V5, Higher Budgets

**What we noticed in V4a:** a missing unit costs ten times more than an unused one, so a budget which is slightly too high is safer than one which is too low.

**Change:** we kept the inspections of V4a exactly the same and multiplied every budget by **1.15**, with a maximum of 8000.

**Results** (change in cost per 20,000 vehicles compared with V4a, on two sets of splits; negative is better):

| Assumed depot | Set 1 | Set 2 |
|---|---:|---:|
| Weighted over several assumed depots | −634 | −785 |
| Central assumption | −348 | −554 |
| Cheaper repairs and fewer failures | +377 | +398 |

**Observations**

- The total budget went up from about 875,000 to 1,001,000.
- The public score improved from 17,407.88 to **17,251.63**, which is a small gain of 156.
- It can make the cost worse if repairs are cheaper at the new depot.
- The weights of the assumed depots were estimated using our public scores, so this choice depends partly on the leaderboard.

**What was still left:** the budgets could only be as good as the repair cost predictions behind them.

---

## 5. Selected Model 2: V6, Better Repair Cost Predictions (Final Model)

**What was lacking in V5:** only one of the histogram-like sensor groups was being used as a histogram, and every vehicle shared the same range of cost errors.

### What V6 does

1. **Keeps the failure classifier:** the average of LightGBM, XGBoost and CatBoost, calibrated with Platt scaling, followed by the EM correction.
2. **Builds 163 histogram features** from the groups `s023–s032`, `s037–s046`, `s054–s063`, `s080–s089`, `s109–s118`, `s129–s138` and `s154–s163`.
3. **Averages three cost models:** RBF kernel ridge on 60 histogram features; Bayesian ridge with LightGBM learning its errors; and LightGBM on sensors, histogram features and 100 sensor ratios.
4. **Gives each vehicle its own range:** the range is wider when the three models disagree more.
5. **Decides using the cost function:** the budget is the 0.20/0.22 quantile of the predicted cost, allowing for the chance of no failure. A vehicle is inspected only if this lowers its expected cost. Finally the budgets are multiplied by 1.15.

### Results

**Simulated new depot** (cost per 20,000 vehicles, six splits not used for choosing the model):

| Failure rate | Price level | V5 | V6 | Difference |
|---|---|---:|---:|---:|
| 2.5% | 1.0 | 39,788 | **36,907** | −2,881 |
| 3.3% | 1.0 | 47,988 | **44,068** | −3,920 |
| 4.0% | 1.0 | 55,163 | **50,334** | −4,829 |
| 5.0% | 1.0 | 65,413 | **59,286** | −6,127 |
| 3.3% | 0.8 | 41,950 | **39,325** | −2,625 |
| 2.5% | 0.6 | 31,698 | **30,813** | −885 |
| 4.0% | 1.3 | 69,412 | **63,679** | −5,733 |

**Where the saving came from**

| Part of the cost | Change |
|---|---:|
| Budget too low | −4,181 |
| Missed failures | −1,435 |
| Budget too high | −885 |
| Inspection fees | +69 |

**Observations**

- V6 won on all six splits in every case above, and also on all eight splits of a further check.
- The R² on log repair cost went up from 0.775–0.791 to **0.825–0.835**.
- With the same linear model, the new features alone raised R² from about 0.72 to 0.79.
- Around three quarters of the saving came from the cost model and one quarter from the separate ranges.
- 1,570 vehicles are flagged, out of which 1,496 were also flagged by V5.
- The public score improved from 17,251.63 to **15,928.88**, i.e. by 7.7%.

---

## 6. Selected Model 3: V7, Six Classifiers

**Idea:** the classifier had not changed since V3, so we wanted to see whether adding shallower models would make it more reliable.

**Change:** we added three shallower versions of LightGBM, XGBoost and CatBoost and took the average of all six. The cost model is the same as V6.

**Results**

| Measure | Repeat 1 | Repeat 2 | Repeat 3 |
|---|---:|---:|---:|
| Average precision, V6 | 0.7710 | 0.7665 | 0.7788 |
| Average precision, V7 | 0.7745 | 0.7713 | 0.7803 |
| Simulated cost change from V6 | −719 | −12 | −1,030 |
| Share of vehicle resamples where V7 was better | 77% | 49% | 82% |

**Observations**

- V7 looked slightly better locally but did not reach the **90%** pass mark which we had set beforehand.
- Its public score was **16,184.67**, which is 256 worse than V6. This difference is small enough to be due to chance.
- We therefore kept V6 as our final model.

---

## 7. Other Ideas That Did Not Work

| What we tried | What happened |
|---|---|
| Increasing budgets beyond 1.15 | Our results did not support it |
| A cost-weighted mix of classifiers | It improved only one of three repeats |
| Stacking | It did not meet our pass mark |
| More cost models and bagging | R² did not improve |
| Histogram features in the classifier | No improvement |

---

## 8. How the Data Is Used

**Fitted on `train.csv` only:** all the V6 failure and repair cost models, along with the scaling, median filling and feature selection used by them.

**Calculated row by row for both files:** the histogram features.

**How `test.csv` is used**

- **Predictions:** the trained models are applied to the test vehicles.

- **Diagnostic:** our first analysis (`fleet_final_report.ipynb`) trained a classifier to tell train rows from test rows. No test labels were available.





---

## 9. Our Best Model (V6): Strengths and Limitations

**Strengths**

- The decisions follow directly from the competition costs.
- It predicts repair costs much better than our earlier versions.
- It accounts for being more unsure about some vehicles than others.
- It uses models of different types, which make different kinds of errors.

**Limitations**

- With only 397 failures our estimates are noisy, and much of the gain comes from a few expensive repairs.
- The same failures were used repeatedly during development, so offline results may be optimistic.
- Our simulations expected larger gains than we got, e.g. about 1,600 (median) for V6 against the actual 1,323.
- The repair prices of the new depot cannot be checked.
- The public score covers only 30% of the test vehicles, so the private score may differ.

---

## 10. Individual Contributions

| Member | Work |
|RIGZIN GYALPO| V4 notebook|
| HARSHUL KUMAR MAURYA| V2 and V7 notebook|
| DHRUV PESHIN |  V1 AND V3 Combined Notebook |
| HANSIKA SINGH | report and v6 notebook |

---

## 11. Repository and How to Run

```
fleet_final_report.ipynb      V1, V2 and V3a
notebooks/fleet_v4.ipynb      V4a and V4b
notebooks/fleet_v5.ipynb      V5
notebooks/fleet_v6.ipynb      final model V6
notebooks/fleet_v6_executed.ipynb   original saved run of V6
notebooks/fleet_v6_general.ipynb    V6 with automatic depot checks
notebooks/fleet_v7.ipynb      V7, which also creates V5 and V6
best_submissions/             sub_v5.csv, sub_v6.csv, sub_v7.csv
SUBMISSIONS.md                submission log
experiments/                  earlier exploratory scripts
```


