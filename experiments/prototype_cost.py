"""End-to-end prototype: P(fail|x) + lognormal cost model -> expected-cost decisions.
Compares feature sets (all vs depot-stable) on out-of-fold total cost.
ASSUMED metric (verify on the competition Evaluation tab):
  flag=0: cost = repair_cost if failure else 0
  flag=1: cost = 10 + 0.20*max(0, c - b) + 0.02*max(0, b - c), with c = repair_cost if failure else 0
Usage: python -I prototype_cost.py ../data
"""
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import norm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score

INSPECT, SHORT, OVER = 10.0, 0.20, 0.02
CRIT = SHORT / (SHORT + OVER)              # newsvendor critical ratio = 0.909

D = sys.argv[1] if len(sys.argv) > 1 else "../data"
tr = pd.read_csv(f"{D}/train.csv")
S = [c for c in tr.columns if c.startswith("s")]
SHIFTED = ["s047"] + [f"s{i:03d}" for i in range(54, 64)] + ["s120", "s122"]
STABLE = [c for c in S if c not in SHIFTED]
y = tr.failure.values
c_true = tr.repair_cost.fillna(0).values

CLF = dict(n_estimators=500, learning_rate=0.02, num_leaves=15, min_child_samples=30,
           subsample=0.8, subsample_freq=1, colsample_bytree=0.4, reg_lambda=5, verbose=-1, n_jobs=2)
REG = dict(n_estimators=300, learning_rate=0.03, num_leaves=7, min_child_samples=10,
           subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=5, verbose=-1, n_jobs=2)


def realised_cost(flag, b, y, c):
    cc = np.where(y == 1, c, 0.0)
    insp = INSPECT + SHORT * np.maximum(cc - b, 0) + OVER * np.maximum(b - cc, 0)
    return np.where(flag == 1, insp, cc)


def decide(p, mu, sigma):
    """Bayes decision under lognormal cost given failure. Returns flag, budget."""
    # optimal budget: P(C<=b) = CRIT with C = 0 w.p. (1-p), LogN(mu, sigma) w.p. p
    q = np.clip((CRIT - (1 - p)) / np.maximum(p, 1e-12), 0, 1 - 1e-9)
    b = np.where(q > 0, np.exp(mu + sigma * norm.ppf(np.maximum(q, 1e-12))), 0.0)
    # expected cost of flagging with budget b (lognormal partial expectations)
    m = np.exp(mu + sigma ** 2 / 2)                                    # E[C|fail]
    lb = np.log(np.maximum(b, 1e-12))
    z1 = (lb - mu - sigma ** 2) / sigma; z0 = (lb - mu) / sigma
    e_short = np.where(b > 0, m * norm.sf(z1) - b * norm.sf(z0), m)     # E[(C-b)+|fail]
    e_over = np.where(b > 0, b * norm.cdf(z0) - m * norm.cdf(z1), 0.0)  # E[(b-C)+|fail]
    ec_flag = INSPECT + p * (SHORT * e_short + OVER * e_over) + (1 - p) * OVER * b
    ec_skip = p * m
    flag = (ec_flag < ec_skip).astype(int)
    return flag, np.where(flag == 1, b, 0.0)


def run(feats, label):
    oof_p = np.zeros(len(tr)); oof_mu = np.zeros(len(tr)); sig = []
    for k, (a, v) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(tr, y)):
        clf = lgb.LGBMClassifier(**CLF).fit(tr.iloc[a][feats], y[a])
        oof_p[v] = clf.predict_proba(tr.iloc[v][feats])[:, 1]
        fa = a[y[a] == 1]
        reg = lgb.LGBMRegressor(**REG).fit(tr.iloc[fa][feats], np.log1p(c_true[fa]))
        oof_mu[v] = reg.predict(tr.iloc[v][feats])
        # in-fold residual sigma would be optimistic; use held-out residuals of failures in v
        fv = v[y[v] == 1]; sig.append(np.log1p(c_true[fv]) - reg.predict(tr.iloc[fv][feats]))
    sigma = np.std(np.concatenate(sig))
    flag, b = decide(oof_p, oof_mu, sigma)
    cost = realised_cost(flag, b, y, c_true)
    print(f"[{label}] n_feats={len(feats)} AUC={roc_auc_score(y, oof_p):.4f} "
          f"PR-AUC={average_precision_score(y, oof_p):.4f} sigma_logcost={sigma:.3f}")
    print(f"   total cost={cost.sum():,.0f}  per-vehicle={cost.mean():.4f}  flagged={flag.sum()} "
          f"caught failures={(flag & y).sum()}/{y.sum()}  mean budget|flag={b[flag == 1].mean():.1f}")
    return oof_p


base = realised_cost(np.zeros(len(tr)), 0, y, c_true).sum()
print(f"baseline flag-nobody total cost = {base:,.0f} (per vehicle {base / len(tr):.4f})")
perfect = realised_cost(y, c_true, y, c_true)
print(f"oracle (flag failures with large enough cost, exact budget) = "
      f"{np.minimum(perfect, c_true * (y == 1)).sum():,.0f}")
run(S, "all 170")
run(STABLE, "depot-stable 157")
