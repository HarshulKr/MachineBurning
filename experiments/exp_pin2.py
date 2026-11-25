from sklearn.linear_model import LinearRegression
import sys; sys.path.insert(0,'src'); import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from fleet import *; from models import *
tr,te=load('../data'); F=[c for c in SENSORS if c not in ('s047','s120','s122')]
f=tr[tr.failure==1].reset_index(drop=True); c=f.repair_cost.values; yc=np.log(c); X=add_features(f,F)
def pin(b,c): return np.mean(0.2*np.maximum(c-b,0)+0.02*np.maximum(b-c,0))
print('constant budget (90.9% quantile of all costs):', round(pin(np.quantile(c,Q_STAR),c),2), '| b=0:', round(pin(0,c),2))
def logmodel(params, recal=True):
    def run(Xa,ya,Xb,seed):
        ip=np.zeros(len(ya))
        for a,b in KFold(5,shuffle=True,random_state=seed).split(Xa): ip[b]=lgb.LGBMRegressor(random_state=seed,**params).fit(Xa.iloc[a],ya[a]).predict(Xa.iloc[b])
        _l=LinearRegression().fit(ip[:,None],ya); a1,a0=float(_l.coef_[0]),float(_l.intercept_) if recal else (1,0)
        mu=a0+a1*lgb.LGBMRegressor(random_state=seed,**params).fit(Xa,ya).predict(Xb); res=ya-(a0+a1*ip)
        return np.exp(mu+np.quantile(res,Q_STAR))
    return run
def quantmodel(params):
    def run(Xa,ya,Xb,seed):
        return np.exp(lgb.LGBMRegressor(random_state=seed,objective='quantile',alpha=Q_STAR,**params).fit(Xa,ya).predict(Xb))
    return run
base=dict(LGB_COST)
print('--- sweep 2')
cands={
 'mcs8 n600':logmodel({**base,'n_estimators':600,'min_child_samples':8}),
 'mcs5 n400':logmodel({**base,'n_estimators':400,'min_child_samples':5}),
 'mcs8 n400 col.3':logmodel({**base,'n_estimators':400,'min_child_samples':8,'colsample_bytree':0.3}),
 'mcs8 n800 lr.015':logmodel({**base,'n_estimators':800,'min_child_samples':8,'learning_rate':0.015}),
 'mcs10 n400 l5':logmodel({**base,'n_estimators':400,'min_child_samples':10,'num_leaves':5}),
}
for name,fn in cands.items():
    L=[]
    for s in range(3):
        b=np.zeros(len(c))
        for a,bb in KFold(5,shuffle=True,random_state=100+s).split(X): b[bb]=fn(X.iloc[a],yc[a],X.iloc[bb],SEED+s)
        L.append(pin(b,c))
    print(f'{name:18s} pinball/failure {np.mean(L):.2f} ± {np.std(L):.2f}',flush=True)
