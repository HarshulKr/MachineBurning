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

# ----------------------------------------------------------------------------- decisions
def decide(p, mu, sigma, flag_margin=1.0):
    """Bayes-optimal flag & budget for each vehicle.

    p      : calibrated P(failure | x)
    mu     : predicted mean of log(repair_cost) given failure
    sigma  : std of log cost given failure (lognormal predictive distribution)
    flag_margin : flag iff E[cost|skip] > flag_margin * E[cost|inspect]  (1.0 = pure Bayes)

    Budget: the cost distribution is a mixture (0 w.p. 1-p, LogN(mu, sigma) w.p. p).
    Newsvendor optimum is its CRIT-quantile, which is 0 whenever p < 1 - CRIT (~9.1%).
    """
    p = np.clip(p, 1e-9, 1 - 1e-9)
    sigma = np.broadcast_to(np.asarray(sigma, float), p.shape)
    q = (CRIT - (1 - p)) / p
    pos = q > 0
    b = np.zeros_like(p)
    b[pos] = np.exp(mu[pos] + sigma[pos] * norm.ppf(np.clip(q[pos], 1e-12, 1 - 1e-9)))

    m = np.exp(mu + sigma ** 2 / 2)  # E[C | fail]
    lb = np.log(np.maximum(b, 1e-300))
    z0 = (lb - mu) / sigma
    z1 = z0 - sigma
    e_short = np.where(pos, m * norm.sf(z1) - b * norm.sf(z0), m)       # E[(C-b)+ | fail]
    e_over = np.where(pos, b * norm.cdf(z0) - m * norm.cdf(z1), 0.0)    # E[(b-C)+ | fail]
    ec_inspect = INSPECT + p * (SHORT * e_short + OVER * e_over) + (1 - p) * OVER * b
    ec_skip = p * m
    flag = (ec_skip > flag_margin * ec_inspect).astype(int)
    return flag, np.where(flag == 1, b, 0.0)


# ----------------------------------------------------------------------------- prior shift (EM)
def em_prior_adjust(p, train_prior, n_iter=200, tol=1e-8):
    """Saerens-Latinne-Decaestecker EM: re-estimate the class prior on an unlabeled set
    and rescale posteriors accordingly. Returns (adjusted p, estimated prior)."""
    p = np.clip(p, 1e-9, 1 - 1e-9)
    pi = train_prior
    for _ in range(n_iter):
        a = p * pi / train_prior
        bneg = (1 - p) * (1 - pi) / (1 - train_prior)
        new = a / (a + bneg)
        pi_new = new.mean()
        if abs(pi_new - pi) < tol:
            pi = pi_new
            break
        pi = pi_new
    a = p * pi / train_prior
    bneg = (1 - p) * (1 - pi) / (1 - train_prior)
    return a / (a + bneg), pi


# ----------------------------------------------------------------------------- models
LGB_CLF = dict(n_estimators=700, learning_rate=0.02, num_leaves=15, min_child_samples=30,
               subsample=0.8, subsample_freq=1, colsample_bytree=0.4, reg_lambda=5.0,
               verbose=-1, n_jobs=2)
XGB_CLF = dict(n_estimators=700, learning_rate=0.02, max_depth=5, min_child_weight=3,
               subsample=0.8, colsample_bytree=0.4, reg_lambda=5.0, tree_method="hist",
               eval_metric="logloss", n_jobs=2, verbosity=0)
CAT_CLF = dict(iterations=800, learning_rate=0.04, depth=6, l2_leaf_reg=5.0, rsm=0.5,
               verbose=0, thread_count=2, allow_writing_files=False)
LGB_REG = dict(n_estimators=300, learning_rate=0.03, num_leaves=7, min_child_samples=10,
               subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=5.0,
               verbose=-1, n_jobs=2)


def make_classifier(name, seed=0):
    if name == "lgb":
        import lightgbm as lgb
        return lgb.LGBMClassifier(**LGB_CLF, random_state=seed)
    if name == "xgb":
        import xgboost as xgb
        return xgb.XGBClassifier(**XGB_CLF, random_state=seed)
    if name == "cat":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(**CAT_CLF, random_seed=seed)
    if name == "logreg":
        from sklearn.pipeline import make_pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import FunctionTransformer, StandardScaler
        from sklearn.linear_model import LogisticRegression
        return make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                             FunctionTransformer(np.log1p), StandardScaler(),
                             LogisticRegression(C=0.05, max_iter=3000))
    raise ValueError(name)


def make_cost_regressors(seed=0):
    """Regressors for log1p(repair_cost | failure); predictions are averaged.
    LightGBM only: in cost_model_experiments.py (5x5 CV) it beat LGB+Ridge on R2 (0.639 vs 0.624),
    NLL and pinball@0.909; Ridge alone is much weaker (R2 0.538) and dragged the average down.
    OOF residual sigma for this model is ~0.955 (use as `sigma` in config.json)."""
    import lightgbm as lgb
    return [lgb.LGBMRegressor(**LGB_REG, random_state=seed)]
