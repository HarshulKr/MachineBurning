import sys; sys.path.insert(0,'src'); sys.path.insert(0,'/home/claude/theirs/fleet_maintenance')
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
import fleet_lib as TL
from fleet import *; from models import *
import models; models.LGB_COST.update(n_estimators=400, min_child_samples=8)
tr,te=load('../data'); y=tr.failure.values; fu=tr.followup_days.values; M=fu==365; pi0=y[M].mean(); cost=tr.repair_cost.values
T=np.load('theirs_oof.npz'); D=np.load('diag_test.npz')
oof=((T['lgb']+T['xgb']+T['cat'])/3).mean(0); praw=(D['lgb']+D['xgb']+D['cat'])/3
cal=Platt().fit(oof[M],y[M]); pc_tr=cal.predict(oof); pc=adjust_prior(cal.predict(praw),pi0,0.03)
mu=D['mu']
# cost model without hist block, fit on train failures, predict test
fm=y==1; logc=np.log(np.nan_to_num(cost,nan=1.0))
XS=TL.make_features(tr); XST=TL.make_features(te)[XS.columns]
pred2,res2=fit_cost_model(XS.iloc[np.where(fm)[0]],logc[fm],models.make_cost_model,0,5,True)
mu2=pred2(XST)
# weighted by p: expected log cost among likely failures
for name,m in [('with hist (V3)',mu),('stable only',mu2)]:
    w=pc; print(name,'p-weighted mean mu on test %.3f'%(np.sum(w*m)/w.sum()),'| high-p (>0.5) mean %.3f'%m[pc>0.5].mean())
print('train failures: mean log cost %.3f'%logc[fm].mean())
# in train, OOF-ish: p-weighted mu among train with high p? use in-sample predictions of cost models on train failures
print('train failures mean pred (in-sample) hist %.3f stable %.3f'%(pred2(XS.iloc[np.where(fm)[0]]).mean(), pred2(XS.iloc[np.where(fm)[0]]).mean()))
print('corr mu vs mu2 on test high-p', np.corrcoef(mu[pc>0.5],mu2[pc>0.5])[0,1].round(3), 'mean diff', (mu-mu2)[pc>0.5].mean().round(3))
# per-feature: does test have more 'high-cost' sensor values? compare top cost features distributions for high-p test vs train failures
f=tr[fm]; tA=align_test_depot(te)[pc>0.5]
for c in ['s056','s082','s081','s088','s083','s084','s058','s038']:
    print(c,'train fail median %.3g  test high-p median %.3g'%(f[c].median(), tA[c].median()))
