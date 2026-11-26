import sys; sys.path.insert(0,'src'); import numpy as np, pandas as pd, warnings, time; warnings.filterwarnings('ignore')
from fleet import *; from models import *
from catboost import CatBoostRegressor; from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
tr,te=load('../data')
for c,m in {'s055':10,'s057':100,'s062':10}.items(): tr[c]=np.round(tr[c]/m)*m
F=[c for c in SENSORS if c not in ('s047','s120','s122')]
f=tr[tr.failure==1].reset_index(drop=True); c=f.repair_cost.values; yc=np.log(c); X=add_features(f,F)
P8=dict(LGB_COST, n_estimators=400, min_child_samples=8)
mk={'lgb':lambda s: lgb.LGBMRegressor(random_state=s,**P8),
    'cat':lambda s: CatBoostRegressor(iterations=600,learning_rate=0.03,depth=4,l2_leaf_reg=3,rsm=0.5,verbose=0,random_seed=s,thread_count=2,allow_writing_files=False),
    'xgb':lambda s: XGBRegressor(n_estimators=400,learning_rate=0.03,max_depth=3,min_child_weight=3,subsample=0.8,colsample_bytree=0.5,n_jobs=2,random_state=s)}
def pin(b,c): return np.mean(0.2*np.maximum(c-b,0)+0.02*np.maximum(b-c,0))
def run(names,seed):
    def fitpred(Xa,ya,Xb): return np.mean([mk[n](seed).fit(Xa,ya).predict(Xb) for n in names],0)
    b=np.zeros(len(c)); mu_all=np.zeros(len(c))
    for a,bb in KFold(5,shuffle=True,random_state=100+seed).split(X):
        Xa,ya=X.iloc[a],yc[a]; ip=np.zeros(len(ya))
        for aa,b2 in KFold(5,shuffle=True,random_state=seed).split(Xa): ip[b2]=fitpred(Xa.iloc[aa],ya[aa],Xa.iloc[b2])
        lr=LinearRegression().fit(ip[:,None],ya); mu=lr.predict(fitpred(Xa,ya,X.iloc[bb])[:,None]); mu_all[bb]=mu
        b[bb]=np.exp(mu+np.quantile(ya-lr.predict(ip[:,None]),Q_STAR))
    return pin(b,c), 1-((yc-mu_all)**2).mean()/yc.var()
for names in [['lgb'],['cat'],['xgb'],['lgb','cat'],['lgb','xgb','cat']]:
    t=time.time(); R=[run(names,s) for s in range(3)]
    print(names,'pinball %.2f ± %.2f  R2 %.3f  %.0fs'%(np.mean([r[0] for r in R]),np.std([r[0] for r in R]),np.mean([r[1] for r in R]),time.time()-t),flush=True)
