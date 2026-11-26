"""Paired evaluation of repair-cost-model candidates with the V3b classifier held fixed.

Score = exact metric. Main criterion = deterministic prior-weighted cost per 20k at true
failure prior 3/4/5% (decisions always made with the V3b assumption pi'=0.035), plus the
old-depot score. Per-vehicle losses are averaged over the 3 CV repeats for paired SEs.
"""
import sys, itertools, time
sys.path.insert(0, 'src')
import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
from fleet import *
from models import folds, Platt, cross_calibrate, signed_log
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

tr, te = load('../data')
for c, m in {'s055': 10, 's057': 100, 's062': 10}.items():
    tr[c] = np.round(tr[c] / m) * m
y = tr.failure.values; fu = tr.followup_days.values; cost = tr.repair_cost.values
FS = folds(y, 5, 3); M = fu == 365; PI0 = y[M].mean(); PI_A = 0.035
LOGC = np.log(np.nan_to_num(cost, nan=1.0))
T = np.load('theirs_oof.npz'); PT = (T['lgb'] + T['xgb'] + T['cat']) / 3
PCAL = [adjust_prior(cross_calibrate(PT[r], y, FS[r], M), PI0, PI_A) for r in range(3)]

F_ROB = [c for c in SENSORS if c not in ('s047', 's120', 's122')]
F_COST = [c for c in F_ROB if c not in ('s063', 's146')]
HB = [h for h in HIST if h != 's063']
L1P = lambda d: np.log1p(d.clip(lower=0))
COST_NB = dict(n_estimators=400, learning_rate=0.03, num_leaves=7, min_child_samples=8,
               subsample_freq=1, colsample_bytree=0.5, verbose=-1, n_jobs=2)            # as in notebook (subsample commented out)
COST_SUB = dict(COST_NB, subsample=0.8)                                                # as intended / as tuned


def add_feats(df, cols):
    X = df[cols].copy(); X['n_nan'] = df[cols].isna().sum(1); X['n_zero'] = (df[cols] == 0).sum(1); return X


class CostModel:
    """V3b cost model; options switch on candidate changes."""
    def __init__(self, seed, params=COST_NB, w_ridge=0.4, k=12, kr=40, hist_desc=False, extra=None, sample_weight_gamma=0.0):
        self.seed, self.p, self.w, self.k, self.kr, self.hd, self.extra, self.g = seed, params, w_ridge, k, kr, hist_desc, extra, sample_weight_gamma
    def _feats(self, df):
        X = add_feats(df, F_COST); H = df[HB].values; tot = np.nansum(H, 1); W = H / np.maximum(tot, 1)[:, None]
        for i, h in enumerate(HB): X[h + '_share'] = W[:, i]
        X['hist_total'] = np.log1p(tot)
        if self.hd:
            j = np.arange(len(HB)); mean = np.nansum(W * j, 1)
            X['h_mean'] = mean; X['h_var'] = np.nansum(W * (j - mean[:, None]) ** 2, 1)
            X['h_entropy'] = -np.nansum(np.where(W > 0, W * np.log(np.where(W > 0, W, 1)), 0), 1)
            X['h_peak'] = np.nanmax(W, 1); X['h_upper'] = np.nansum(W[:, 5:], 1)
        for a, b in itertools.combinations(self.top, 2): X[f'lr_{a}_{b}'] = L1P(df[a]) - L1P(df[b])
        if self.extra is not None: X = pd.concat([X, self.extra(df)], axis=1)
        return X
    def fit(self, df, yy):
        sw = None if self.g == 0 else np.minimum((np.exp(yy) / np.median(np.exp(yy))) ** self.g, 5)
        base = add_feats(df, F_COST)
        imp = pd.Series(lgb.LGBMRegressor(random_state=self.seed, **self.p).fit(base, yy).booster_.feature_importance('gain'), base.columns)
        self.top = [c for c in imp.sort_values(ascending=False).index if c.startswith('s')][:self.k]
        self.gm = lgb.LGBMRegressor(random_state=self.seed, **self.p).fit(self._feats(df), yy, sample_weight=sw)
        if self.w > 0:
            Lg = L1P(df[F_COST]); self.med = Lg.median(); Lg = Lg.fillna(self.med)
            corr = pd.Series({c: abs(np.corrcoef(Lg[c], yy)[0, 1]) for c in F_COST}).fillna(0).sort_values(ascending=False)
            self.rc = list(corr.index[:self.kr])
            self.r = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 30))).fit(Lg[self.rc], yy, ridgecv__sample_weight=sw)
        return self
    def predict(self, df):
        p = self.gm.predict(self._feats(df))
        if self.w > 0: p = (1 - self.w) * p + self.w * self.r.predict(L1P(df[self.rc]).fillna(self.med))
        return p


def fit_pipeline(make, D, yf, seed, inner=5, crossfit_recal=False):
    """mu predictor + residual pool (inner-CV, linear recalibration)."""
    yf = np.asarray(yf, float); ip = np.zeros(len(yf)); folds_in = list(KFold(inner, shuffle=True, random_state=seed).split(D))
    for a, b in folds_in: ip[b] = make(seed).fit(D.iloc[a], yf[a]).predict(D.iloc[b])
    lin = LinearRegression().fit(ip[:, None], yf)
    if crossfit_recal:
        res = np.zeros(len(yf))
        for a, b in folds_in: res[b] = yf[b] - LinearRegression().fit(ip[a, None], yf[a]).predict(ip[b, None])
    else:
        res = yf - lin.predict(ip[:, None])
    m = make(seed).fit(D, yf)
    return (lambda Dn: lin.predict(m.predict(Dn)[:, None])), res


def cv_samples(make, R=200, floor=None, spread=1.0, crossfit_recal=False):
    """Per outer fold: (val_idx, log-cost sample matrix (n_val, R)). floor = callable(seed)->classifier for P(c=5)."""
    out = []
    qs = (np.arange(R) + 0.5) / R
    for r, fs in enumerate(FOLDS_):
        rr = []
        for k, (a, b) in enumerate(fs):
            fi = a[y[a] == 1]; D = tr.iloc[fi]; yy = LOGC[fi]; seed = 2024 + 10 * r + k
            if floor is None:
                pred, res = fit_pipeline(make, D, yy, seed, crossfit_recal=crossfit_recal)
                mu = pred(tr.iloc[b]); S = mu[:, None] + spread * np.quantile(res, qs)[None, :]
            else:
                isf = yy <= np.log(5) + 1e-9
                pred, res = fit_pipeline(make, D.iloc[np.where(~isf)[0]], yy[~isf], seed, crossfit_recal=crossfit_recal)
                mu = pred(tr.iloc[b]); q = floor(seed, D, isf, tr.iloc[b])
                n5 = np.round(q * R).astype(int)[:, None]; j = np.arange(R)[None, :]
                u = np.clip((j - n5 + 0.5) / np.maximum(R - n5, 1), 0, 1)
                rs = np.sort(res); cont = mu[:, None] + spread * np.interp(u, (np.arange(len(rs)) + 0.5) / len(rs), rs)
                S = np.where(j < n5, np.log(5.0), np.maximum(cont, np.log(5.0)))
            rr.append((b, S))
        out.append(rr)
    return out


FOLDS_ = FS


def decisions(samples):
    """flag/budget per repeat from sample matrices (each row = log-cost draws, uniform weights)."""
    res = []
    for r in range(3):
        fl = np.zeros(len(y), int); bd = np.zeros(len(y))
        for b, S in samples[r]:
            f_, b_, _, _ = decide(PCAL[r][b], np.zeros(len(b)), S)
            fl[b], bd[b] = f_, b_
        res.append((fl, bd))
    return res


def losses(dec):
    """per-vehicle exact loss averaged over repeats (label-complete rows)."""
    return np.mean([fleet_cost(f, b, y, cost) for f, b in dec], 0)


def report(L, name, base=None):
    pos, neg = M & (y == 1), M & (y == 0)
    out = {'exp': name}
    for pi in (0.03, 0.04, 0.05):
        out[f'{int(pi*100)}%'] = 20000 * (pi * L[pos].mean() + (1 - pi) * L[neg].mean())
    if base is not None:
        d = L - base
        for pi in (0.03, 0.04, 0.05):
            mean = 20000 * (pi * d[pos].mean() + (1 - pi) * d[neg].mean())
            se = 20000 * np.sqrt((pi ** 2) * d[pos].var() / pos.sum() + ((1 - pi) ** 2) * d[neg].var() / neg.sum())
            out[f'Δ{int(pi*100)}%'] = f'{mean:+.0f}±{se:.0f}'
        top = np.sort(d[pos])
        out['largest single-failure Δ'] = f'{top[0]:+.0f}/{top[-1]:+.0f}'
    return out


# ---- extra candidate pieces -------------------------------------------------------------
from sklearn.linear_model import GammaRegressor, LogisticRegression
from sklearn.decomposition import PCA
_teA = align_test_depot(te)
_all = pd.concat([tr[F_COST], _teA[F_COST]], ignore_index=True)
_Lall = L1P(_all); _med_all = _Lall.median(); _Lall = _Lall.fillna(_med_all)
_sc = StandardScaler().fit(_Lall); _pca = PCA(15, random_state=0).fit(_sc.transform(_Lall))
def pca_feats(df):
    Z = _pca.transform(_sc.transform(L1P(df[F_COST]).fillna(_med_all)))
    return pd.DataFrame(Z, columns=[f'pc{i}' for i in range(Z.shape[1])], index=df.index)


class GammaBlend(CostModel):
    """V3b with the ridge part replaced by a Gamma GLM (log link) on the same 40 log1p sensors."""
    def fit(self, df, yy):
        super().fit(df, yy)
        Lg = L1P(df[self.rc]).fillna(self.med)
        self.gam = make_pipeline(StandardScaler(), GammaRegressor(alpha=1.0, max_iter=3000)).fit(Lg, np.exp(yy))
        return self
    def predict(self, df):
        return (1 - self.w) * self.gm.predict(self._feats(df)) + self.w * np.log(self.gam.predict(L1P(df[self.rc]).fillna(self.med)))


class RidgeThenBoost:
    """ridge on 40 log1p sensors, then a small LightGBM on its inner-OOF residuals (stacking)."""
    def __init__(self, seed, lam=0.5, params=None):
        self.seed, self.lam = seed, lam
        self.p = params or dict(n_estimators=100, learning_rate=0.03, num_leaves=7, min_child_samples=15, subsample=0.8,
                                subsample_freq=1, colsample_bytree=0.5, verbose=-1, n_jobs=2)
        self.base = CostModel(seed, params=COST_SUB)
    def fit(self, df, yy):
        Lg = L1P(df[F_COST]); self.med = Lg.median(); Lg = Lg.fillna(self.med)
        corr = pd.Series({c: abs(np.corrcoef(Lg[c], yy)[0, 1]) for c in F_COST}).fillna(0).sort_values(ascending=False)
        self.rc = list(corr.index[:40]); mk = lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 30)))
        oof = np.zeros(len(yy))
        for a, b in KFold(5, shuffle=True, random_state=self.seed).split(df): oof[b] = mk().fit(Lg[self.rc].iloc[a], yy[a]).predict(Lg[self.rc].iloc[b])
        self.r = mk().fit(Lg[self.rc], yy)
        self.base.top = []; self.base.top = [c for c in self.rc[:12]]
        self.g = lgb.LGBMRegressor(random_state=self.seed, **self.p).fit(self.base._feats(df), yy - oof)
        return self
    def predict(self, df):
        return self.r.predict(L1P(df[self.rc]).fillna(self.med)) + self.lam * self.g.predict(self.base._feats(df))


def logistic_floor(seed, D, isf, Dv, C=0.1, k=10):
    Lg = L1P(D[F_COST]); med = Lg.median(); Lg = Lg.fillna(med)
    corr = pd.Series({c: abs(np.corrcoef(Lg[c], isf.astype(float))[0, 1]) for c in F_COST}).fillna(0).sort_values(ascending=False)
    cols = list(corr.index[:k])
    m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000)).fit(Lg[cols], isf.astype(int))
    return m.predict_proba(L1P(Dv[cols]).fillna(med))[:, 1]
