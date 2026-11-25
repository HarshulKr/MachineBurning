import sys; sys.path.insert(0,'src'); import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from fleet import *; from models import *
from models import Platt
tr,te=load('../data'); y=tr.failure.values; fu=tr.followup_days.values
w=np.where((y==0)&(fu<365),0.0,1.0); teA=align_test_depot(te)
F=[c for c in SENSORS if c not in ('s047','s120','s122')]
# 1) mixture-quantile check: fraction of test rows above the train-negative 99th pct, vs mixture prediction
M=fu==365; neg=tr[M&(y==0)]; pos=tr[y==1]
print('feature  P(x>q99neg|neg) P(..|pos)  test_frac  implied_pi')
for c in ['s123','s149','s034','s053','s071','s111','s097','s012','s088']:
    t=neg[c].quantile(0.99); a=(neg[c]>t).mean(); b=(pos[c]>t).mean(); f=(teA[c]>t).mean()
    print(c, round(a,4), round(b,3), round(f,4), round((f-a)/(b-a),4))
# 2) Saerens EM prior estimate from OOF-calibrated model
X=add_features(tr,F); XT=add_features(teA,F)
fs=folds(y,5,1)[0]; oof=np.zeros(len(y)); pt=np.zeros(len(te))
for a,b in fs:
    m=make_lgb().fit(X.iloc[a],y[a],sample_weight=w[a]); oof[b]=m.predict_proba(X.iloc[b])[:,1]; pt+=m.predict_proba(XT)[:,1]/5
iso=Platt().fit(oof[M],y[M]); pc=iso.predict(pt)
pi0=y[M].mean(); pi=pi0
for it in range(200):
    r1=pi/pi0; r0=(1-pi)/(1-pi0); q=r1*pc/(r1*pc+r0*(1-pc)); new=q.mean()
    if abs(new-pi)<1e-7: break
    pi=new
print('train prior',round(pi0,4),'mean calibrated test p',round(pc.mean(),4),'EM test prior',round(pi,4),'iters',it)
# sanity: EM on a train holdout subsample should recover train prior
pcv=iso.predict(oof[M]); pi=pi0
for it in range(200):
    r1=pi/pi0; r0=(1-pi)/(1-pi0); q=r1*pcv/(r1*pcv+r0*(1-pcv)); pi=q.mean()
print('EM on train OOF (should ~ train prior):',round(pi,4))
np.save('test_pcal_quick.npy',pc)
