import sys,itertools,time; sys.path.insert(0,'src'); import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from fleet import *; from models import *
from sklearn.linear_model import RidgeCV, LinearRegression
from scipy.stats import spearmanr
tr,te=load('../data')
for c,m in {'s055':10,'s057':100,'s062':10}.items(): tr[c]=np.round(tr[c]/m)*m
F=[c for c in SENSORS if c not in ('s047','s120','s122','s146')]
y=tr.failure.values; f=tr[y==1].reset_index(drop=True); c=f.repair_cost.values; yc=np.log(c)
P8=dict(LGB_COST, n_estimators=400, min_child_samples=8)
def pin(b,c): return np.mean(0.2*np.maximum(c-b,0)+0.02*np.maximum(b-c,0))
L=lambda s: np.log1p(s.clip(lower=0))
class CostFE:
    """fit-time feature selection: top-k ratios by LGB importance, hist fractions; optional ridge blend"""
    def __init__(s,seed,ratios=True,histfrac=True,ridge_w=0.0,k=12,kr=40): s.seed=seed; s.ratios=ratios; s.histfrac=histfrac; s.rw=ridge_w; s.k=k; s.kr=kr
    def feats(s,df):
        X=add_features(df,F)
        if s.histfrac:
            H=df[HIST].values; tot=np.nansum(H,1)
            for i,h in enumerate(HIST): X[h+'_frac']=H[:,i]/np.maximum(tot,1)
            X['hist_tot']=np.log1p(tot)
        if s.ratios:
            for a,b in itertools.combinations(s.top,2): X[f'r_{a}_{b}']=L(df[a])-L(df[b])
        return X
    def fit(s,df,yy):
        base=add_features(df,F); imp=pd.Series(lgb.LGBMRegressor(random_state=s.seed,**P8).fit(base,yy).booster_.feature_importance('gain'),base.columns)
        s.top=[t for t in imp.sort_values(ascending=False).index if t.startswith('s')][:s.k]
        s.m=lgb.LGBMRegressor(random_state=s.seed,**P8).fit(s.feats(df),yy)
        if s.rw>0:
            Lg=L(df[F]); s.med=Lg.median(); Lg=Lg.fillna(s.med)
            sp=pd.Series({cc:abs(spearmanr(Lg[cc],yy)[0]) for cc in F}).fillna(0).sort_values(ascending=False); s.rc=list(sp.index[:s.kr])
            s.r=make_pipeline(StandardScaler(),RidgeCV(alphas=np.logspace(-2,4,30))).fit(Lg[s.rc],yy)
        return s
    def predict(s,df):
        p=s.m.predict(s.feats(df))
        if s.rw>0: p=(1-s.rw)*p+s.rw*s.r.predict(L(df[s.rc]).fillna(s.med))
        return p
def evaluate(make,reps=3):
    Ls=[];Rs=[]
    for seed in range(reps):
        b=np.zeros(len(c)); mu_all=np.zeros(len(c))
        for a,bb in KFold(5,shuffle=True,random_state=100+seed).split(f):
            fa,ya=f.iloc[a],yc[a]; ip=np.zeros(len(ya))
            for aa,b2 in KFold(5,shuffle=True,random_state=seed).split(fa): ip[b2]=make(seed).fit(fa.iloc[aa],ya[aa]).predict(fa.iloc[b2])
            lr=LinearRegression().fit(ip[:,None],ya); mu=lr.predict(make(seed).fit(fa,ya).predict(f.iloc[bb])[:,None]); mu_all[bb]=mu
            b[bb]=np.exp(mu+np.quantile(ya-lr.predict(ip[:,None]),Q_STAR))
        Ls.append(pin(b,c)); Rs.append(1-((yc-mu_all)**2).mean()/yc.var())
    return round(np.mean(Ls),2),round(np.std(Ls),2),round(np.mean(Rs),3)
for name,mk in [('ratios+histfrac+ridge0.2',lambda s: CostFE(s,True,True,0.2)),('ratios+histfrac+ridge0.4',lambda s: CostFE(s,True,True,0.4)),('ratios+histfrac+ridge0.5',lambda s: CostFE(s,True,True,0.5)),('ridge0.3 kr60',lambda s: CostFE(s,True,True,0.3,12,60))]:
    t=time.time(); print(name,evaluate(mk),'%.0fs'%(time.time()-t),flush=True)
