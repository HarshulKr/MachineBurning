"""Repeated stratified 5-fold CV (R repeats, new fold split + new model seeds each repeat)
for the final pipeline: mean of LGB/XGB/CAT with the `drop` censoring scheme, lognormal cost model.

Purpose: make the sigma-scale x flag-margin grid less noisy, and say whether any cell is
really better than pure Bayes (1.0, 1.0). Two kinds of noise are reported:
  * split/seed noise: std of total cost across repeats
  * label noise (only 397 failures): paired bootstrap over vehicles of cost(cell) - cost(1.0, 1.0)

Writes oof_repeated.npz (per-repeat OOF p and mu, reusable by later experiments) and
cv_repeated_grid.csv. Does not touch config.json.
Usage: py -3.12 cv_repeated.py data [R] [n_jobs]
"""
import sys
import time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import fleet_lib as F

D = sys.argv[1] if len(sys.argv) > 1 else "data"
R = int(sys.argv[2]) if len(sys.argv) > 2 else 3
NJ = int(sys.argv[3]) if len(sys.argv) > 3 else 8
SCHEME = "drop"
MODELS = ["lgb", "xgb", "cat"]
S_GRID = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
M_GRID = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
N_BOOT = 2000

tr, _ = F.load(D)
X = F.make_features(tr)
y = tr.failure.values
cost = tr.repair_cost.values
w = F.sample_weights(tr, SCHEME)


def fit_clf(name, seed, a, v):
    m = F.make_classifier(name, seed=seed)
    m.fit(X.iloc[a], y[a], sample_weight=w[a])
    return m.predict_proba(X.iloc[v])[:, 1]


def fit_cost(seed, a, v):
    fa = a[y[a] == 1]
    return np.mean([r.fit(X.iloc[fa], np.log1p(cost[fa])).predict(X.iloc[v])
                    for r in F.make_cost_regressors(seed=seed)], axis=0)


t0 = time.time()
P = np.zeros((R, len(X)))
MU = np.zeros((R, len(X)))
SIG = np.zeros(R)
for r in range(R):
    folds = list(StratifiedKFold(5, shuffle=True, random_state=1000 + r).split(X, y))
    jobs = [(n, k) for n in MODELS for k in range(5)]
    outs = Parallel(n_jobs=NJ, prefer="threads")(delayed(fit_clf)(n, r, *folds[k]) for n, k in jobs)
    per_model = {n: np.zeros(len(X)) for n in MODELS}
    for (n, k), o in zip(jobs, outs):
        per_model[n][folds[k][1]] = o
    P[r] = np.mean([per_model[n] for n in MODELS], axis=0)
    mus = Parallel(n_jobs=NJ, prefer="threads")(delayed(fit_cost)(r, *folds[k]) for k in range(5))
    for k in range(5):
        MU[r, folds[k][1]] = mus[k]
    fl = y == 1
    SIG[r] = (np.log1p(cost[fl]) - MU[r, fl]).std()
    aucs = " ".join(f"{n}={roc_auc_score(y, per_model[n]):.4f}" for n in MODELS)
    print(f"repeat {r}: {aucs} ens={roc_auc_score(y, P[r]):.4f} sigma={SIG[r]:.3f} "
          f"[{time.time()-t0:.0f}s]", flush=True)

np.savez("oof_repeated.npz", P=P, MU=MU, SIG=SIG, y=y, cost=np.nan_to_num(cost))

# ---------------------------------------------------------------- grid
# per-row cost averaged over repeats, for each cell -> (cells, n_rows)
cells = [(s, m) for s in S_GRID for m in M_GRID]
row_cost = np.zeros((len(cells), len(X)))
rep_tot = np.zeros((len(cells), R))
for i, (s, m) in enumerate(cells):
    for r in range(R):
        f_, b_ = F.decide(P[r], MU[r], SIG[r] * s, m)
        c = F.realised_cost(f_, b_, y, cost)
        rep_tot[i, r] = c.sum()
        row_cost[i] += c / R

base = cells.index((1.0, 1.0))
rng = np.random.default_rng(0)
diff = row_cost - row_cost[base]                       # (cells, rows)
# Poisson(1) bootstrap weights over vehicles, in chunks to bound memory
boot = np.concatenate([diff @ rng.poisson(1.0, size=(len(X), 200)) for _ in range(N_BOOT // 200)], axis=1)

g = pd.DataFrame(cells, columns=["sigma_scale", "margin"])
g["cost_mean"] = rep_tot.mean(axis=1)
g["cost_std_repeats"] = rep_tot.std(axis=1, ddof=1) if R > 1 else np.nan
g["diff_vs_bayes"] = diff.sum(axis=1)
g["diff_lo95"] = np.percentile(boot, 2.5, axis=1)
g["diff_hi95"] = np.percentile(boot, 97.5, axis=1)
g["p_better"] = (boot < 0).mean(axis=1)
g.to_csv("cv_repeated_grid.csv", index=False)

pd.set_option("display.width", 200)
print(f"\nMean OOF total cost over {R} repeats (rows: sigma_scale, cols: margin)")
print(g.pivot(index="sigma_scale", columns="margin", values="cost_mean").round(0).to_string())
print("\nStd across repeats")
print(g.pivot(index="sigma_scale", columns="margin", values="cost_std_repeats").round(0).to_string())
# 3x3 neighbourhood mean (flatness)
piv = g.pivot(index="sigma_scale", columns="margin", values="cost_mean")
nb = piv.copy()
A = piv.values
for i in range(A.shape[0]):
    for j in range(A.shape[1]):
        nb.iloc[i, j] = A[max(0, i-1):i+2, max(0, j-1):j+2].mean()
print("\n3x3 neighbourhood mean")
print(nb.round(0).to_string())
print("\nTop 8 cells vs pure Bayes (1.0,1.0); diff<0 is better; CI = paired bootstrap over vehicles")
print(g.sort_values("cost_mean").head(8).round(3).to_string(index=False))
i, j = np.unravel_index(np.argmin(nb.values), nb.shape)
print(f"\nflattest-best cell (lowest 3x3 mean): sigma_scale={nb.index[i]}, margin={nb.columns[j]}")
print(f"done [{time.time()-t0:.0f}s]")
