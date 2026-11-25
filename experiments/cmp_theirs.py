import sys,time,pickle; sys.path.insert(0,'src'); sys.path.insert(0,'/home/claude/theirs/fleet_maintenance')
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
import fleet_lib as TL
from fleet import load, SENSORS
from models import folds
tr,te=load('../data'); y=tr.failure.values
FS=folds(y,5,3)
X=TL.make_features(tr); w=TL.sample_weights(tr,'drop')
P={n:np.zeros((3,len(y))) for n in ['lgb','xgb','cat']}; MU=np.zeros((3,len(y)))
fl=y==1; lc=np.log1p(tr.repair_cost.values)
t=time.time()
for r,fs in enumerate(FS):
    for k,(a,b) in enumerate(fs):
        for n in P:
            m=TL.make_classifier(n,seed=r).fit(X.iloc[a],y[a],sample_weight=w[a]); P[n][r,b]=m.predict_proba(X.iloc[b])[:,1]
        fa=a[fl[a]]
        MU[r,b]=TL.make_cost_regressors(seed=r)[0].fit(X.iloc[fa],lc[fa]).predict(X.iloc[b])
        print(r,k,'%.0fs'%(time.time()-t),flush=True)
np.savez('theirs_oof.npz',MU=MU,**P)
