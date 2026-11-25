"""Cost-model experiments: predictive distribution of repair_cost given failure.

The decision rule needs, for each vehicle, a distribution of C | fail (for the budget quantile
and for E[C]). Candidates differ in the location model mu(x), the spread sigma(x) and the shape.

Folds are StratifiedKFold(5, shuffle, random_state=1000+r) on (X, y): the SAME splits as
cv_repeated.py, so the per-repeat OOF p in oof_repeated.npz can be combined with these OOF
cost predictions to score the real decision cost honestly.

Quality metrics (on the 397 failures, OOF):
  r2      R^2 of log1p(cost)
  nll     mean negative log-likelihood of log1p(cost) under N(mu, sigma)  (lower better)
  pin91   mean pinball loss at q=0.909 on the cost scale (what the budget pays for when p~1)
Decision metric (needs oof_repeated.npz): OOF total cost with decide(), sigma scale 1, margin 1.

Usage: py -3.12 cost_model_experiments.py data [R]
"""
import os
import sys
import time
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.linear_model import RidgeCV, GammaRegressor
import lightgbm as lgb

import fleet_lib as F

D = sys.argv[1] if len(sys.argv) > 1 else "data"
R = int(sys.argv[2]) if len(sys.argv) > 2 else 5
tr, _ = F.load(D)
X = F.make_features(tr)
y = tr.failure.values
cost = np.nan_to_num(tr.repair_cost.values)
fl = y == 1
lc = np.log1p(cost)


def ridge():
    return make_pipeline(SimpleImputer(strategy="median"), FunctionTransformer(np.log1p),
                         StandardScaler(), RidgeCV(alphas=np.logspace(-1, 4, 30)))


def lgbr(seed, **kw):
    p = dict(F.LGB_REG); p.update(kw)
    return lgb.LGBMRegressor(**p, random_state=seed)


def lgbq(seed, alpha):
    p = dict(F.LGB_REG); p.update(objective="quantile", alpha=alpha)
    return lgb.LGBMRegressor(**p, random_state=seed)


def gamma_glm():
    return make_pipeline(SimpleImputer(strategy="median"), FunctionTransformer(np.log1p),
                         StandardScaler(), GammaRegressor(alpha=1.0, max_iter=2000))


# ------------------------------------------------------------------ location models (log scale)
def loc_base(Xa, la, Xv, seed):          # current pipeline: mean(LGB, Ridge)
    return np.mean([lgbr(seed).fit(Xa, la).predict(Xv), ridge().fit(Xa, la).predict(Xv)], axis=0)


def loc_ridge(Xa, la, Xv, seed):
    return ridge().fit(Xa, la).predict(Xv)


def loc_lgb(Xa, la, Xv, seed):
    return lgbr(seed).fit(Xa, la).predict(Xv)


def loc_lgb_slow(Xa, la, Xv, seed):      # more regularised LGB
    return lgbr(seed, n_estimators=600, learning_rate=0.015, num_leaves=4,
                min_child_samples=15, colsample_bytree=0.3).fit(Xa, la).predict(Xv)


def loc_base3(Xa, la, Xv, seed):         # mean(LGB, Ridge, Gamma GLM in log space)
    g = gamma_glm().fit(Xa, np.expm1(la)).predict(Xv)
    return np.mean([lgbr(seed).fit(Xa, la).predict(Xv), ridge().fit(Xa, la).predict(Xv),
                    np.log1p(g)], axis=0)


def loc_slow_ridge(Xa, la, Xv, seed):
    return 0.5 * loc_lgb_slow(Xa, la, Xv, seed) + 0.5 * loc_ridge(Xa, la, Xv, seed)


def inner_oof(loc_fn, Xa, la, seed):
    """OOF location predictions inside the training part (for residual-based sigma models)."""
    o = np.zeros(len(la))
    for a, v in KFold(5, shuffle=True, random_state=seed).split(Xa):
        o[v] = loc_fn(Xa.iloc[a], la[a], Xa.iloc[v], seed)
    return o


# ------------------------------------------------------------------ candidates
# each returns (mu_v, sigma_v) for the validation rows; sigma may be scalar or per-row
def make_const(loc_fn):
    def f(Xa, la, Xv, seed):
        r = la - inner_oof(loc_fn, Xa, la, seed)
        return loc_fn(Xa, la, Xv, seed), r.std()
    return f


def make_hetero(loc_fn, kind):
    """sigma(x): regress |inner-OOF residual| on x, sigma = sqrt(pi/2) * E|r|."""
    def f(Xa, la, Xv, seed):
        r = la - inner_oof(loc_fn, Xa, la, seed)
        if kind == "ridge":
            m = ridge().fit(Xa, np.abs(r))
        else:
            m = lgbr(seed, n_estimators=200, num_leaves=4, min_child_samples=20).fit(Xa, np.abs(r))
        s = np.sqrt(np.pi / 2) * m.predict(Xv)
        s = np.clip(s, 0.5 * r.std(), 1.6 * r.std())
        return loc_fn(Xa, la, Xv, seed), s
    return f


CANDS = {
    "base (lgb+ridge, const s)": make_const(loc_base),
    "ridge only": make_const(loc_ridge),
    "lgb only": make_const(loc_lgb),
    "lgb slow": make_const(loc_lgb_slow),
    "slow lgb + ridge": make_const(loc_slow_ridge),
    "lgb+ridge+gamma": make_const(loc_base3),
    "base + hetero s (ridge)": make_hetero(loc_base, "ridge"),
    "base + hetero s (lgb)": make_hetero(loc_base, "lgb"),
}


def pinball(c, b, q=F.CRIT):
    d = c - b
    return np.mean(np.maximum(q * d, (q - 1) * d))


t0 = time.time()
MU = {k: np.zeros((R, len(X))) for k in CANDS}
SG = {k: np.zeros((R, len(X))) for k in CANDS}
QB = np.zeros((R, len(X)))        # direct LGB quantile regression at q=CRIT (cost scale via log)
for r in range(R):
    folds = list(StratifiedKFold(5, shuffle=True, random_state=1000 + r).split(X, y))
    for a, v in folds:
        fa = a[fl[a]]
        Xa, la, Xv = X.iloc[fa], lc[fa], X.iloc[v]
        for k, fn in CANDS.items():
            mu, s = fn(Xa, la, Xv, r)
            MU[k][r, v] = mu
            SG[k][r, v] = s
        QB[r, v] = lgbq(r, F.CRIT).fit(Xa, la).predict(Xv)
    print(f"repeat {r} done [{time.time()-t0:.0f}s]", flush=True)

np.savez("oof_costmodels.npz", **{f"mu|{k}": MU[k] for k in CANDS},
         **{f"sg|{k}": SG[k] for k in CANDS}, QB=QB)

have_p = os.path.exists("oof_repeated.npz")
if have_p:
    Z = np.load("oof_repeated.npz")
    P = Z["P"][:R]
    assert np.array_equal(Z["y"], y)

rows = []
for k in CANDS:
    r2s, nlls, pins, dec = [], [], [], []
    for r in range(R):
        mu, s = MU[k][r], SG[k][r]
        res = lc[fl] - mu[fl]
        r2s.append(1 - (res ** 2).sum() / ((lc[fl] - lc[fl].mean()) ** 2).sum())
        nlls.append(np.mean(-norm.logpdf(lc[fl], mu[fl], s[fl])))
        b91 = np.expm1(mu[fl] + s[fl] * norm.ppf(F.CRIT))
        pins.append(pinball(cost[fl], b91))
        if have_p:
            f_, b_ = F.decide(P[r], mu, s, 1.0)
            dec.append(F.realised_cost(f_, b_, y, cost).sum())
    rows.append(dict(model=k, r2=np.mean(r2s), nll=np.mean(nlls), pin91=np.mean(pins),
                     dec_cost=np.mean(dec) if dec else np.nan,
                     dec_std=np.std(dec, ddof=1) if len(dec) > 1 else np.nan))
qb_pin = np.mean([pinball(cost[fl], np.expm1(QB[r, fl])) for r in range(R)])
rows.append(dict(model="lgb quantile q=0.909 (direct)", r2=np.nan, nll=np.nan, pin91=qb_pin,
                 dec_cost=np.nan, dec_std=np.nan))

out = pd.DataFrame(rows)
out.to_csv("cost_model_results.csv", index=False)
pd.set_option("display.width", 200)
print(f"\nOOF over {R} repeats (397 failures). dec_cost = total decision cost, sigma scale 1, margin 1")
print(out.round(4).to_string(index=False))
if not have_p:
    print("\n(oof_repeated.npz not found: run cv_repeated.py first for the decision-cost column)")
print(f"done [{time.time()-t0:.0f}s]")
