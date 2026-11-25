"""Cross-validated experiments. All numbers are out-of-fold on the training depot.
Stage 1: censoring scheme (LightGBM).  Stage 2: model comparison + ensemble.
Stage 3: calibration.  Stage 4: tune sigma scale and flag margin on OOF cost.
Usage: python -I cv_experiments.py ../data
"""
import sys
import json
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

import fleet_lib as F

D = sys.argv[1] if len(sys.argv) > 1 else "../data"
tr, _ = F.load(D)
X = F.make_features(tr)
y = tr.failure.values
cost = tr.repair_cost.values
full = tr.followup_days.values == 365
folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y))


def oof_classifier(name, scheme):
    w = F.sample_weights(tr, scheme)
    oof = np.zeros(len(X))
    for a, v in folds:
        m = F.make_classifier(name)
        kw = {"logisticregression__sample_weight": w[a]} if name == "logreg" else {"sample_weight": w[a]}
        m.fit(X.iloc[a], y[a], **kw)
        oof[v] = m.predict_proba(X.iloc[v])[:, 1]
    return oof


def oof_cost_model():
    mu = np.zeros(len(X)); resid = []
    for a, v in folds:
        fa = a[y[a] == 1]
        preds = []
        for r in F.make_cost_regressors():
            r.fit(X.iloc[fa], np.log1p(cost[fa]))
            preds.append(r.predict(X.iloc[v]))
        mu[v] = np.mean(preds, axis=0)
    fl = y == 1
    resid = np.log1p(cost[fl]) - mu[fl]
    return mu, resid


def report(tag, p, mu, sigma, margin=1.0):
    flag, b = F.decide(p, mu, sigma, margin)
    c = F.realised_cost(flag, b, y, cost)
    out = dict(tag=tag, auc=roc_auc_score(y, p), prauc=average_precision_score(y, p),
               logloss=log_loss(y, np.clip(p, 1e-7, 1 - 1e-7)),
               cost_all=c.sum(), cost_365_per_veh=c[full].mean(), flagged=int(flag.sum()),
               caught=int((flag & y).sum()))
    print(f"{tag:<28} AUC={out['auc']:.4f} PR={out['prauc']:.4f} LL={out['logloss']:.5f} "
          f"cost={out['cost_all']:>8,.0f} cost/veh(365)={out['cost_365_per_veh']:.4f} "
          f"flag={out['flagged']} caught={out['caught']}/{y.sum()}", flush=True)
    return out


def cv_calibrate(p, method):
    """Out-of-fold calibration of OOF scores (so calibration is also evaluated honestly)."""
    out = np.zeros_like(p)
    for a, v in folds:
        if method == "isotonic":
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p[a], y[a])
            out[v] = iso.predict(p[v])
        else:  # platt on logit
            lg = lambda z: np.log(np.clip(z, 1e-7, 1 - 1e-7) / (1 - np.clip(z, 1e-7, 1 - 1e-7)))
            lr = LogisticRegression(C=1e4).fit(lg(p[a])[:, None], y[a])
            out[v] = lr.predict_proba(lg(p[v])[:, None])[:, 1]
    return out


t0 = time.time()
mu, resid = oof_cost_model()
sigma0 = resid.std()
r2 = 1 - (resid ** 2).sum() / ((np.log1p(cost[y == 1]) - np.log1p(cost[y == 1]).mean()) ** 2).sum()
print(f"cost model: OOF R2(log cost)={r2:.3f} sigma={sigma0:.3f}  [{time.time()-t0:.0f}s]")

print("\n== Stage 1: censoring scheme (LightGBM) ==")
res, oofs = [], {}
for scheme in ["none", "frac", "drop"]:
    oofs[("lgb", scheme)] = oof_classifier("lgb", scheme)
    res.append(report(f"lgb/{scheme}", oofs[("lgb", scheme)], mu, sigma0))
best_scheme = min(res, key=lambda r: r["cost_all"])["tag"].split("/")[1]
print(f"-> chosen censoring scheme: {best_scheme}  [{time.time()-t0:.0f}s]")

print("\n== Stage 2: models ==")
for name in ["xgb", "cat", "logreg"]:
    oofs[(name, best_scheme)] = oof_classifier(name, best_scheme)
    report(f"{name}/{best_scheme}", oofs[(name, best_scheme)], mu, sigma0)
    print(f"   [{time.time()-t0:.0f}s]", flush=True)
trees = np.mean([oofs[(n, best_scheme)] for n in ["lgb", "xgb", "cat"]], axis=0)
report("ens trees (mean)", trees, mu, sigma0)
ens4 = 0.85 * trees + 0.15 * oofs[("logreg", best_scheme)]
report("ens trees+logreg", ens4, mu, sigma0)

print("\n== Stage 3: calibration (applied to tree ensemble) ==")
cal = {"raw": trees, "platt": cv_calibrate(trees, "platt"), "isotonic": cv_calibrate(trees, "isotonic")}
for k, v in cal.items():
    report(f"trees/{k}", v, mu, sigma0)

print("\n== Stage 4: sigma scale x flag margin grid (tree ensemble, each calibration) ==")
grid = []
for k, p in cal.items():
    for s in [0.8, 1.0, 1.2, 1.4]:
        for mgn in [0.8, 0.9, 1.0, 1.1, 1.25]:
            f_, b_ = F.decide(p, mu, sigma0 * s, mgn)
            grid.append((k, s, mgn, F.realised_cost(f_, b_, y, cost).sum()))
g = pd.DataFrame(grid, columns=["cal", "sigma_scale", "margin", "cost"])
print(g.pivot_table(index=["cal", "sigma_scale"], columns="margin", values="cost").round(0).to_string())
best = g.loc[g.cost.idxmin()]
print(f"\nbest: {best.to_dict()}")
# prefer a robust choice: the cell whose 3x3 neighbourhood average is lowest
config = dict(scheme=best_scheme, calibration=best.cal, sigma=float(sigma0),
              sigma_scale=float(best.sigma_scale), margin=float(best.margin),
              train_prior=float(y.mean()))
json.dump(config, open("config.json", "w"), indent=2)
np.savez("oof.npz", trees=trees, mu=mu, y=y, cost=np.nan_to_num(cost))
print("saved config.json", config, f"[{time.time()-t0:.0f}s]")
