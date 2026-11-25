from sklearn.linear_model import LinearRegression
import sys; sys.path.insert(0,'src'); import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from fleet import *; from models import *
tr,te=load('../data')
F=[c for c in SENSORS if c not in ('s047','s120','s122')]; FS_=[c for c in F if c not in HIST]
f=tr[tr.failure==1].reset_index(drop=True); c=f.repair_cost.values; yc=np.log(c)
P8=dict(LGB_COST, n_estimators=400, min_child_samples=8)
def pin(b,c): return np.mean(0.2*np.maximum(c-b,0)+0.02*np.maximum(b-c,0))
for name,cols in [('with hist',F),('without hist',FS_)]:
    X=add_features(f,cols); L=[];R=[]
    for s in range(3):
        b=np.zeros(len(c)); mu_all=np.zeros(len(c))
        for a,bb in KFold(5,shuffle=True,random_state=100+s).split(X):
            Xa,ya=X.iloc[a],yc[a]; ip=np.zeros(len(ya))
            for aa,b2 in KFold(5,shuffle=True,random_state=s).split(Xa): ip[b2]=lgb.LGBMRegressor(random_state=s,**P8).fit(Xa.iloc[aa],ya[aa]).predict(Xa.iloc[b2])
            _l=LinearRegression().fit(ip[:,None],ya); a1,a0=float(_l.coef_[0]),float(_l.intercept_); mu=a0+a1*lgb.LGBMRegressor(random_state=s,**P8).fit(Xa,ya).predict(X.iloc[bb]); mu_all[bb]=mu
            b[bb]=np.exp(mu+np.quantile(ya-(a0+a1*ip),Q_STAR))
        L.append(pin(b,c)); R.append(1-((yc-mu_all)**2).mean()/yc.var())
    print(name,'pinball',round(np.mean(L),2),'±',round(np.std(L),2),'R2',round(np.mean(R),3))
