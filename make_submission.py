"""Train on all training data and write submission.csv.
Usage: python make_submission.py ../data [--no-em]
Reads config.json written by cv_experiments.py (falls back to defaults).
"""
import sys
import json
import os
import numpy as np
import pandas as pd

import fleet_lib as F

D = sys.argv[1] if len(sys.argv) > 1 else "../data"
USE_EM = "--no-em" not in sys.argv
SEEDS = [0, 1, 2]

cfg = dict(scheme="drop", sigma=0.955, sigma_scale=1.0, margin=1.0, train_prior=0.00709)
if os.path.exists("config.json"):
    cfg.update(json.load(open("config.json")))
# The CV grid over sigma_scale/margin is noisy (+-1.5k between neighbours); use pure Bayes
# (1.0, 1.0) unless a seed-averaged CV says otherwise.
cfg["sigma_scale"], cfg["margin"] = 1.0, 1.0
print("config:", cfg)

tr, te = F.load(D)
Xtr, Xte = F.make_features(tr), F.make_features(te)[F.make_features(tr).columns]
y = tr.failure.values
w = F.sample_weights(tr, cfg["scheme"])

# --- failure probability: mean of LightGBM, XGBoost, CatBoost over seeds
preds = []
for name in ["lgb", "xgb", "cat"]:
    for s in SEEDS:
        m = F.make_classifier(name, seed=s).fit(Xtr, y, sample_weight=w)
        preds.append(m.predict_proba(Xte)[:, 1])
        print(f"  fitted {name} seed {s}", flush=True)
p = np.mean(preds, axis=0)

# --- test-depot prior shift via EM
if USE_EM:
    p_adj, pi_test = F.em_prior_adjust(p, cfg["train_prior"])
    print(f"EM: train prior={cfg['train_prior']:.4%}, estimated test prior={pi_test:.4%}, "
          f"mean raw p={p.mean():.4%}")
    p = p_adj

# --- repair cost given failure (log scale)
fl = y == 1
mus = []
for s in SEEDS:
    for r in F.make_cost_regressors(seed=s):
        r.fit(Xtr[fl], np.log1p(tr.repair_cost.values[fl]))
        mus.append(r.predict(Xte))
mu = np.mean(mus, axis=0)

flag, budget = F.decide(p, mu, cfg["sigma"] * cfg["sigma_scale"], cfg["margin"])
sub = pd.DataFrame({"id": te["id"], "flag": flag.astype(int), "budget": np.round(budget, 2)})

# --- format checks against sample_submission
ss = pd.read_csv(f"{D}/sample_submission.csv")
assert list(sub.columns) == list(ss.columns), "column mismatch"
assert len(sub) == len(ss) and (sub["id"].values == ss["id"].values).all(), "id mismatch/order"
assert sub["flag"].isin([0, 1]).all() and (sub["budget"] >= 0).all() and sub.notna().all().all()
assert (sub.loc[sub.flag == 0, "budget"] == 0).all()
sub.to_csv("submission.csv", index=False)
np.savez("test_preds.npz", p=p, mu=mu)
print(f"wrote submission.csv: flagged={flag.sum()} ({flag.mean():.2%}), "
      f"with budget>0={(budget > 0).sum()}, mean budget|budget>0="
      f"{budget[budget > 0].mean() if (budget > 0).any() else 0:.1f}")
print(f"expected-cost sanity: sum p*E[c] if nobody flagged = "
      f"{(p * np.exp(mu + (cfg['sigma'] ** 2) / 2)).sum():,.0f}")
