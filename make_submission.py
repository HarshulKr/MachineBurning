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
