"""Model fitting / cross-validation helpers (course models only)."""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.linear_model import LogisticRegression, RidgeCV, LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer

from fleet import decide, fleet_cost, SEED

LGB_CLF = dict(n_estimators=400, learning_rate=0.02, num_leaves=7, min_child_samples=50,
               subsample=0.8, subsample_freq=1, colsample_bytree=0.4, reg_lambda=2.0,
               verbose=-1, n_jobs=2)
LGB_COST = dict(n_estimators=250, learning_rate=0.03, num_leaves=7, min_child_samples=15,
                subsample=0.8, subsample_freq=1, colsample_bytree=0.5, verbose=-1, n_jobs=2)


def signed_log(X):
    X = np.asarray(X, float)
    return np.sign(X) * np.log1p(np.abs(X))


def make_logreg(C=0.1):
    return make_pipeline(SimpleImputer(strategy="median"), FunctionTransformer(signed_log),
                         StandardScaler(), LogisticRegression(C=C, max_iter=2000))


def make_lgb(seed=SEED, **kw):
    return lgb.LGBMClassifier(random_state=seed, **{**LGB_CLF, **kw})


def make_cost_model(seed=SEED):
    return lgb.LGBMRegressor(random_state=seed, **LGB_COST)


def folds(y, n_splits=5, repeats=3):
    out = []
    for r in range(repeats):
        skf = StratifiedKFold(n_splits, shuffle=True, random_state=SEED + r)
        out.append(list(skf.split(np.zeros(len(y)), y)))
    return out


def oof_proba(model_fn, X, y, fold_sets, w=None, predict_X=None):
    """Raw out-of-fold P(fail) for each repeat: array (repeats, n).

    predict_X lets us predict a modified copy of the rows (e.g. followup_days set to 365).
    """
    predict_X = X if predict_X is None else predict_X
    P = np.zeros((len(fold_sets), len(y)))
    for r, fs in enumerate(fold_sets):
        for tr_idx, va_idx in fs:
            m = model_fn(r)
            kw = {} if w is None else {"sample_weight" if not hasattr(m, "steps") else
                                       m.steps[-1][0] + "__sample_weight": w[tr_idx]}
            m.fit(X.iloc[tr_idx], y[tr_idx], **kw)
            P[r, va_idx] = m.predict_proba(predict_X.iloc[va_idx])[:, 1]
    return P


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))[:, None]


class Platt:
    """Platt scaling: 1-feature logistic regression on the model's log-odds."""
    def fit(self, p, y):
        self.m = LogisticRegression(C=1e4, max_iter=5000).fit(_logit(p), y)
        return self

    def predict(self, p):
        return self.m.predict_proba(_logit(p))[:, 1]


def cross_calibrate(p_raw, y, fold_list, mask):
    """Platt calibration fitted on the *other* folds' OOF predictions (no optimism)."""
    p = np.zeros_like(p_raw)
    for tr_idx, va_idx in fold_list:
        t = tr_idx[mask[tr_idx]]
        p[va_idx] = Platt().fit(p_raw[t], y[t]).predict(p_raw[va_idx])
    return p


def fit_cost_model(Xf, yf, model_fn=make_cost_model, seed_offset=0, inner=5, recal=True):
    """Fit log-cost model on failures; return (predict_fn, residual pool).

    Inner-CV predictions give honest residuals. With recal=True a 1-D linear regression of
    y on the inner OOF prediction undoes the shrinkage of boosted predictions toward the mean.
    """
    yf = np.asarray(yf, float)
    inner_pred = np.zeros(len(yf))
    for a, b in KFold(inner, shuffle=True, random_state=SEED + seed_offset).split(Xf):
        inner_pred[b] = model_fn(seed_offset).fit(Xf.iloc[a], yf[a]).predict(Xf.iloc[b])
    if recal:
        lin = LinearRegression().fit(inner_pred[:, None], yf)
        a1, a0 = float(lin.coef_[0]), float(lin.intercept_)
    else:
        a1, a0 = 1.0, 0.0
    m = model_fn(seed_offset).fit(Xf, yf)
    res = yf - (a0 + a1 * inner_pred)
    return (lambda Xn: a0 + a1 * m.predict(Xn)), res


def oof_cost_model(X, log_cost, fail, fold_sets, model_fn=make_cost_model, inner=5, recal=False):
    """OOF predicted log cost (mu) for every row + per-fold residual pools from inner CV."""
    MU = np.zeros((len(fold_sets), len(fail)))
    RES = []
    for r, fs in enumerate(fold_sets):
        res_r = []
        for tr_idx, va_idx in fs:
            fi = tr_idx[fail[tr_idx] == 1]
            pred, res = fit_cost_model(X.iloc[fi], log_cost[fi], model_fn, r, inner, recal)
            MU[r, va_idx] = pred(X.iloc[va_idx])
            res_r.append((va_idx, res))
        RES.append(res_r)
    return MU, RES


def evaluate(P_raw, y, cost, fold_sets, eval_mask, policy, MU=None, RES=None, scale_to=20000):
    """Simulated total cost (per `scale_to` vehicles) on eval rows, mean & std over repeats.

    policy(p_cal, mu, resid, tr_idx, va_idx) -> (flag, budget) for va rows.
    """
    totals, parts = [], []
    for r, fs in enumerate(fold_sets):
        p_cal = cross_calibrate(P_raw[r], y, fs, eval_mask)
        flag = np.zeros(len(y), int); bud = np.zeros(len(y))
        for k, (tr_idx, va_idx) in enumerate(fs):
            mu = None if MU is None else MU[r, va_idx]
            res = None if RES is None else RES[r][k][1]
            f, b = policy(p_cal[va_idx], mu, res, tr_idx, va_idx)
            flag[va_idx], bud[va_idx] = f, b
        c = fleet_cost(flag, bud, y, cost)[eval_mask]
        totals.append(c.sum() * scale_to / eval_mask.sum())
        parts.append((flag, bud, p_cal))
    return np.mean(totals), np.std(totals), parts
