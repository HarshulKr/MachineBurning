"""Fleet Maintenance Planning - shared pipeline code.

Everything here uses only course-allowed tools: numpy, pandas, scikit-learn,
lightgbm (gradient boosting), Platt calibration (logistic regression), and the decision-theory
layer derived in the report (expected cost + pinball/quantile optimal budget).
"""
import numpy as np
import pandas as pd

SENSORS = [f"s{i:03d}" for i in range(170)]
HIST = [f"s{i:03d}" for i in range(54, 64)]       # 10-bin histogram that is shifted by one bin in the test depot
INSPECT, RUSH, RETURN = 10.0, 0.20, 0.02          # metric constants
Q_STAR = RUSH / (RUSH + RETURN)                   # 0.90909: optimal P(c <= b)
SEED = 2024


# ----------------------------------------------------------------------------- data
def load(data_dir):
    tr = pd.read_csv(f"{data_dir}/train.csv")
    te = pd.read_csv(f"{data_dir}/test.csv", dtype={"id": str})
    tr["id"] = tr["id"].astype(str)
    return tr, te


def align_test_depot(te):
    """Undo the one-bin offset of histogram s054..s063 found by adversarial validation.

    In the new depot, test s(k+1) has the distribution of train s(k) for k = 54..62 and
    test s054 is always 0, so we shift the bins back by one. Train s063 is ~always 0,
    so the vacated top bin is set to 0.
    """
    te = te.copy()
    old = te[HIST].copy()
    for k in range(54, 63):
        te[f"s{k:03d}"] = old[f"s{k + 1:03d}"].values
    te["s063"] = np.where(old["s054"].isna(), np.nan, 0.0)
    return te


def harmonize_resolution(tr, te, cols):
    """The new depot reports some sensors rounded to multiples of 10; round train the same way."""
    tr = tr.copy()
    for c in cols:
        tr[c] = np.round(tr[c] / 10.0) * 10.0
    return tr


def quantile_map(tr, te, cols):
    """Map each test value of `cols` to the train value at the same depot-specific quantile."""
    te = te.copy()
    for c in cols:
        ref = np.sort(tr[c].dropna().values)
        r = pd.Series(te[c]).rank(pct=True, method="average").values
        mapped = np.quantile(ref, np.clip(r, 0, 1))
        te[c] = np.where(te[c].isna(), np.nan, mapped)
    return te


def add_features(df, cols):
    X = df[cols].copy()
    X["n_nan"] = df[cols].isna().sum(axis=1)
    X["n_zero"] = (df[cols] == 0).sum(axis=1)
    return X


# ----------------------------------------------------------------------------- metric
def fleet_cost(flag, budget, failure, repair_cost):
    """Exact competition cost per vehicle (lower is better)."""
    flag = np.asarray(flag).astype(int)
    b = np.maximum(np.asarray(budget, float), 0.0)
    c = np.where(np.asarray(failure) == 1, np.nan_to_num(np.asarray(repair_cost, float)), 0.0)
    flagged = INSPECT + RUSH * np.maximum(c - b, 0) + RETURN * np.maximum(b - c, 0)
    return np.where(flag == 1, flagged, c)


def total_cost(flag, budget, failure, repair_cost):
    return float(fleet_cost(flag, budget, failure, repair_cost).sum())


# ----------------------------------------------------------------------------- decisions
def decide(p, mu, resid):
    """Bayes-optimal flag and budget.

    p     : (n,) calibrated P(fail | x)
    mu    : (n,) predicted log repair cost given failure (use 0 for a constant model)
    resid : (R,) residuals of log cost; cost | fail ~ exp(mu_i + resid_j), j uniform

    Predictive distribution of c: point mass 1-p at 0, mass p/R at each exp(mu_i + r_j).
    Pinball-loss optimum: P(c <= b) = 0.2/0.22  =>  b* = F_{c|fail}^{-1}((p - 0.0909)/p).
    Flag iff the expected cost when flagged (with b*) is below p * E[c | fail].
    """
    p = np.clip(np.asarray(p, float), 0, 1)
    r = np.sort(np.asarray(resid, float), axis=-1)
    R = r.shape[-1]
    r = r[None, :] if r.ndim == 1 else r                              # shared pool or one pool per vehicle
    C = np.exp(np.asarray(mu, float)[:, None] + r)                    # (n, R), each row sorted
    level = np.where(p > 1 - Q_STAR, (p - (1 - Q_STAR)) / np.maximum(p, 1e-12), 0.0)
    idx = np.clip(np.ceil(level * R).astype(int) - 1, 0, R - 1)       # inverted-CDF quantile
    b = np.where(level > 0, C[np.arange(len(p)), idx], 0.0)
    over = np.maximum(C - b[:, None], 0).mean(1)
    under = np.maximum(b[:, None] - C, 0).mean(1)
    cost_flag = INSPECT + p * (RUSH * over + RETURN * under) + (1 - p) * RETURN * b
    cost_skip = p * C.mean(1)
    flag = (cost_flag < cost_skip).astype(int)
    return flag, np.where(flag == 1, np.round(b, 2), 0.0), cost_flag, cost_skip


def cost_breakdown(flag, budget, failure, repair_cost):
    c = np.where(failure == 1, np.nan_to_num(repair_cost), 0.0)
    f = flag == 1
    return {
        "inspection": INSPECT * f.sum(),
        "missed_failures": c[~f].sum(),
        "rush (under-budget)": (RUSH * np.maximum(c - budget, 0))[f].sum(),
        "return (over-budget)": (RETURN * np.maximum(budget - c, 0))[f].sum(),
        "n_flagged": int(f.sum()),
        "failures_caught": int((f & (failure == 1)).sum()),
        "failures_missed": int((~f & (failure == 1)).sum()),
    }


def write_submission(ids, flag, budget, path):
    sub = pd.DataFrame({"id": ids, "flag": np.asarray(flag).astype(int),
                        "budget": np.round(np.where(np.asarray(flag) == 1, budget, 0.0), 2)})
    assert sub["budget"].ge(0).all() and np.isfinite(sub["budget"]).all()
    assert set(sub["flag"].unique()) <= {0, 1}
    sub.to_csv(path, index=False)
    return sub


# ----------------------------------------------------------------------------- label (prior) shift
def adjust_prior(p, pi_train, pi_test):
    """Bayes-rule correction of calibrated posteriors for a change in the failure prior,
    assuming P(x | failure) and P(x | no failure) are the same in both depots."""
    r1, r0 = pi_test / pi_train, (1 - pi_test) / (1 - pi_train)
    return r1 * p / (r1 * p + r0 * (1 - p))


def em_prior(p, pi_train, tol=1e-8, max_iter=500):
    """Saerens-Latinne-Decaestecker EM: latent label = E-step, prior = M-step."""
    pi = pi_train
    for _ in range(max_iter):
        new = adjust_prior(p, pi_train, pi).mean()
        if abs(new - pi) < tol:
            break
        pi = new
    return pi


def mixture_prior(train_neg, train_pos, test, cols, q=0.99):
    """Model-free prior estimate per sensor: fraction of test rows above the train-negative
    q-quantile is a mixture  f = pi*P(>t|fail) + (1-pi)*P(>t|ok)  -> solve for pi."""
    out = {}
    for c in cols:
        t = train_neg[c].quantile(q)
        a, b, f = (train_neg[c] > t).mean(), (train_pos[c] > t).mean(), (test[c] > t).mean()
        if b - a > 0.2:
            out[c] = (f - a) / (b - a)
    return pd.Series(out)
