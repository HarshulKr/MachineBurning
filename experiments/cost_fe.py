import sys,itertools; sys.path.insert(0,'src'); import numpy as np, pandas as pd, warnings, time; warnings.filterwarnings('ignore')
from fleet import *; from models import *
from sklearn.linear_model import LinearRegression
tr,te=load('../data')
for c,m in {'s055':10,'s057':100,'s062':10}.items(): tr[c]=np.round(tr[c]/m)*m
F=[c for c in SENSORS if c not in ('s047','s120','s122')]
y=tr.failure.values; T=np.load('theirs_oof.npz'); poof=((T['lgb']+T['xgb']+T['cat'])/3).mean(0)
f=tr[y==1].reset_index(drop=True); c=f.repair_cost.values; yc=np.log(c); X0=add_features(f,F)
P8=dict(LGB_COST, n_estimators=400, min_child_samples=8)
def pin(b,c): return np.mean(0.2*np.maximum(c-b,0)+0.02*np.maximum(b-c,0))
def evalX(X,reps=3):
    L=[];R=[]
    for seed in range(reps):
        b=np.zeros(len(c)); mu_all=np.zeros(len(c))
        for a,bb in KFold(5,shuffle=True,random_state=100+seed).split(X):
            Xa,ya=X.iloc[a],yc[a]; ip=np.zeros(len(ya))
            for aa,b2 in KFold(5,shuffle=True,random_state=seed).split(Xa): ip[b2]=lgb.LGBMRegressor(random_state=seed,**P8).fit(Xa.iloc[aa],ya[aa]).predict(Xa.iloc[b2])
            lr=LinearRegression().fit(ip[:,None],ya); mu=lr.predict(lgb.LGBMRegressor(random_state=seed,**P8).fit(Xa,ya).predict(X.iloc[bb])[:,None]); mu_all[bb]=mu
            b[bb]=np.exp(mu+np.quantile(ya-lr.predict(ip[:,None]),Q_STAR))
        L.append(pin(b,c)); R.append(1-((yc-mu_all)**2).mean()/yc.var())
    return np.mean(L),np.std(L),np.mean(R)
#
#
imp=pd.Series(lgb.LGBMRegressor(random_state=0,**P8).fit(X0,yc).booster_.feature_importance('gain'),X0.columns).sort_values(ascending=False)
top=[t for t in imp.index[:12] if t.startswith('s')]
R={f'r_{a}_{b}':np.log1p(f[a].clip(lower=0))-np.log1p(f[b].clip(lower=0)) for a,b in itertools.combinations(top,2)}
X2=pd.concat([X0,pd.DataFrame(R)],axis=1)
H=f[HIST].values; tot=np.nansum(H,1); Hf=pd.DataFrame(H/np.maximum(tot,1)[:,None],columns=[h+'_frac' for h in HIST]); Hf['hist_tot']=np.log1p(tot)
X3=pd.concat([X0,Hf],axis=1)
#
m=lgb.LGBMRegressor(random_state=0,**P8).fit(X2,yc); print(pd.Series(m.booster_.feature_importance('gain'),X2.columns).sort_values(ascending=False).head(12).round(0))
m=lgb.LGBMRegressor(random_state=0,**P8).fit(X3,yc); print(pd.Series(m.booster_.feature_importance('gain'),X3.columns).sort_values(ascending=False).head(12).round(0))
print('top12',top)
