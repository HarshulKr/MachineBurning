"""Is the depot shift marginal (per-column) only? Test two fixes via adversarial AUC:
 A) per-depot rank transform of every column
 B) quantile-map only the shifted columns of test onto train's marginal
Usage: python -I quantile_check.py ../data
"""
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from scipy.stats import ks_2samp

D = sys.argv[1] if len(sys.argv) > 1 else "../data"
tr = pd.read_csv(f"{D}/train.csv"); te = pd.read_csv(f"{D}/test.csv")
S = [c for c in tr.columns if c.startswith("s") and c != "s047"]
P = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, colsample_bytree=0.5,
         subsample=0.8, subsample_freq=1, verbose=-1, n_jobs=2)


def adv_auc(a, b):
    X = pd.concat([a, b], ignore_index=True); z = np.r_[np.zeros(len(a)), np.ones(len(b))]
    p = cross_val_predict(lgb.LGBMClassifier(**P), X, z, cv=StratifiedKFold(5, shuffle=True, random_state=0),
                          method="predict_proba")[:, 1]
    m = lgb.LGBMClassifier(**P).fit(X, z)
    imp = pd.Series(m.booster_.feature_importance("gain"), X.columns)
    return roc_auc_score(z, p), (imp / imp.sum()).sort_values(ascending=False).head(6).round(3).to_dict()


def qmap(src, ref):
    """Map src values onto ref's marginal by matching (average) ranks; NaNs kept."""
    out = src.copy(); m = src.notna()
    q = src[m].rank(pct=True, method="average").values
    out[m] = np.nanquantile(ref.dropna().values, np.clip(q, 0, 1))
    return out


print("A) rank all columns per depot:", adv_auc(tr[S].rank(pct=True), te[S].rank(pct=True)))
shifted = [c for c in S if ks_2samp(tr[c].dropna(), te[c].dropna()).statistic > 0.1]
print("shifted cols (KS>0.1):", shifted)
te_b = te[S].copy()
for c in shifted:
    te_b[c] = qmap(te[c], tr[c])
print("B) quantile-map shifted cols:", adv_auc(tr[S], te_b))
print("C) drop shifted cols:", adv_auc(tr[S].drop(columns=shifted), te[S].drop(columns=shifted)))
