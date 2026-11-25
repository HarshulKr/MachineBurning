"""Shared pipeline: data loading, depot-stable features, cost metric, Bayes decision rule, models.

Only permitted libraries: numpy, pandas, scipy, scikit-learn, lightgbm, xgboost, catboost.

ASSUMED metric (check the competition's Evaluation tab; change realised_cost / decide if it differs):
  flag=0: cost = repair_cost if the vehicle fails, else 0
  flag=1: cost = 10 + 0.20*max(0, c - b) + 0.02*max(0, b - c),  c = repair_cost if fails else 0
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

INSPECT, SHORT, OVER = 10.0, 0.20, 0.02
CRIT = SHORT / (SHORT + OVER)  # newsvendor critical ratio ~0.909

# Columns whose distribution differs between depots (found by adversarial validation).
# s047 is entirely missing in test; s054-s063, s120, s122 are shifted/rescaled.
# Dropping these makes train and test indistinguishable (adversarial AUC 1.00 -> 0.51).
SHIFTED = ["s047"] + [f"s{i:03d}" for i in range(54, 64)] + ["s120", "s122"]
CONSTANT = ["s146"]


# ----------------------------------------------------------------------------- data
def load(data_dir):
    tr = pd.read_csv(f"{data_dir}/train.csv")
    te = pd.read_csv(f"{data_dir}/test.csv")
    return tr, te


def sensor_cols(df):
    return [c for c in df.columns if c.startswith("s") and c not in SHIFTED + CONSTANT]


def make_features(df):
    """Depot-stable sensors + simple row summaries (all computed only from stable columns)."""
    S = sensor_cols(df)
    X = df[S].copy()
    X["n_missing"] = df[S].isna().sum(axis=1)
    X["n_zero"] = (df[S] == 0).sum(axis=1)
    X["log_row_sum"] = np.log1p(df[S].fillna(0).sum(axis=1))
    return X


def sample_weights(df, scheme):
    """How to treat censored negatives (follow-up < 365 days, no failure observed).
    none : weight 1 for everyone
    frac : negatives weighted by followup/365 (less evidence of 'no failure')
    drop : negatives with followup < 180 days get weight 0
    """
    w = np.ones(len(df))
    neg = df["failure"].values == 0
    fu = df["followup_days"].values
    if scheme == "frac":
        w[neg] = fu[neg] / 365.0
    elif scheme == "drop":
        w[neg & (fu < 180)] = 0.0
    return w


# ----------------------------------------------------------------------------- metric
def realised_cost(flag, budget, failure, repair_cost):
    c = np.where(failure == 1, np.nan_to_num(repair_cost), 0.0)
    insp = INSPECT + SHORT * np.maximum(c - budget, 0) + OVER * np.maximum(budget - c, 0)
    return np.where(flag == 1, insp, c)
