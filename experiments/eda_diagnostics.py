"""Quick diagnostics that drive the design decisions:
1. How separable is failure? (CV AUC / PR-AUC, LightGBM)
2. How different is the test depot? (adversarial validation)
3. Is repair cost predictable from sensors (given failure)?
Usage: python -I eda_diagnostics.py ../data
"""
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_predict
from sklearn.metrics import roc_auc_score, average_precision_score

D = sys.argv[1] if len(sys.argv) > 1 else "../data"
tr = pd.read_csv(f"{D}/train.csv")
te = pd.read_csv(f"{D}/test.csv")
S = [c for c in tr.columns if c.startswith("s")]
P = dict(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=20,
         subsample=0.8, subsample_freq=1, colsample_bytree=0.5, verbose=-1, n_jobs=2)

# 1. failure separability
y = tr.failure.values
oof = cross_val_predict(lgb.LGBMClassifier(**P), tr[S], y,
                        cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
print(f"failure  AUC={roc_auc_score(y, oof):.4f}  PR-AUC={average_precision_score(y, oof):.4f}  base={y.mean():.4f}")
full = tr.followup_days.values == 365
print(f"  on 365-day rows only AUC={roc_auc_score(y[full], oof[full]):.4f}")

# 2. adversarial validation (train vs test depot)
X = pd.concat([tr[S], te[S]], ignore_index=True)
z = np.r_[np.zeros(len(tr)), np.ones(len(te))]
adv = cross_val_predict(lgb.LGBMClassifier(**P), X, z,
                        cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
print(f"adversarial AUC={roc_auc_score(z, adv):.4f}")
m = lgb.LGBMClassifier(**P).fit(X, z)
imp = pd.Series(m.booster_.feature_importance("gain"), S).sort_values(ascending=False)
print("top shift features:", imp.head(10).round(0).to_dict())
# simple scale ratio of medians for top-shift features
for c in imp.index[:5]:
    print(f"  {c}: train median={tr[c].median():.3g} test median={te[c].median():.3g} "
          f"train NaN={tr[c].isna().mean():.2f} test NaN={te[c].isna().mean():.2f}")

# 3. repair cost predictability among failures
f = tr[tr.failure == 1]
lc = np.log1p(f.repair_cost.values)
pr = cross_val_predict(lgb.LGBMRegressor(**{**P, "n_estimators": 200, "min_child_samples": 10}),
                       f[S], lc, cv=KFold(5, shuffle=True, random_state=0))
r2 = 1 - ((lc - pr) ** 2).sum() / ((lc - lc.mean()) ** 2).sum()
print(f"log repair cost CV R2={r2:.3f} (n={len(f)})")
cors = f[S].apply(lambda s: s.corr(pd.Series(lc, index=f.index), method="spearman"))
print("top |spearman| with log cost:", cors.abs().sort_values(ascending=False).head(5).round(3).to_dict())
# oof failure score vs cost among failures: are 'easy' failures cheap or expensive?
print("spearman(oof p, log cost | fail) =",
      round(pd.Series(oof[y == 1]).corr(pd.Series(lc), method="spearman"), 3))
