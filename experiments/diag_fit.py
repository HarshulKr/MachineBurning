# fit final components once and cache test-side quantities for diagnostics
import sys,time; sys.path.insert(0,'src'); sys.path.insert(0,'/home/claude/theirs/fleet_maintenance')
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
import fleet_lib as TL
from fleet import *; from models import *
import models; models.LGB_COST.update(n_estimators=400, min_child_samples=8)
tr,te=load('../data'); y=tr.failure.values; fu=tr.followup_days.values; cost=tr.repair_cost.values
XS=TL.make_features(tr); XST=TL.make_features(te)[XS.columns]; w=TL.sample_weights(tr,'drop')
t=time.time(); comp={}
for n in ['lgb','xgb','cat']:
    comp[n]=np.mean([TL.make_classifier(n,seed=s).fit(XS,y,sample_weight=w).predict_proba(XST)[:,1] for s in [0,1]],0); print(n,'%.0fs'%(time.time()-t),flush=True)
teA=align_test_depot(te); trH=tr.copy()
for c,m in {'s055':10,'s057':100,'s062':10}.items(): trH[c]=np.round(trH[c]/m)*m
F=[c for c in SENSORS if c not in ('s047','s120','s122')]
X3=add_features(trH,F); X3T=add_features(teA,F); fm=y==1; logc=np.log(np.nan_to_num(cost,nan=1.0))
pred,res=fit_cost_model(X3.iloc[np.where(fm)[0]],logc[fm],models.make_cost_model,0,5,True)
np.savez('diag_test.npz',lgb=comp['lgb'],xgb=comp['xgb'],cat=comp['cat'],mu=pred(X3T),res=res)
print('done')
