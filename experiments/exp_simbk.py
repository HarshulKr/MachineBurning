import sys,pickle; sys.path.insert(0,'src'); import numpy as np, warnings; warnings.filterwarnings('ignore')
from fleet import *; from models import *
tr,te=load('../data'); y=tr.failure.values; fu=tr.followup_days.values; cost=tr.repair_cost.values
FS=folds(y,5,3); M=fu==365; O=dict(np.load('oof_cache.npz'))
MU=np.load('cost_cache_recal1.npz')['MU']; RES=pickle.load(open('res_cache_recal1.pkl','rb'))
P=O['lgb_dropcens']; pi0=y[M].mean(); r=0
pc=cross_calibrate(P[r],y,FS[r],M); fk=np.zeros(len(y),int)
for k,(a,b) in enumerate(FS[r]): fk[b]=k
rng=np.random.default_rng(0); posi=np.where(M&(y==1))[0]; negi=np.where(M&(y==0))[0]
idx=np.r_[posi, rng.choice(negi,int(len(posi)*19),replace=False)]
p=adjust_prior(pc[idx],pi0,0.048); fl=np.zeros(len(idx),int); bd=np.zeros(len(idx))
for k in range(5):
    mk=fk[idx]==k; f,b,_,_=decide(p[mk],MU[r,idx[mk]],RES[r][k][1]); fl[mk]=f; bd[mk]=b
bk=cost_breakdown(fl,bd,y[idx],cost[idx]); sc=20000/len(idx)
print({k:round(v*sc) for k,v in bk.items()})
# oracle analyses: perfect cost knowledge (budget = c for caught failures)
c=np.where(y[idx]==1,cost[idx],0)
tot=fleet_cost(fl,bd,y[idx],cost[idx]).sum()*sc
bd_or=np.where(y[idx]==1,cost[idx],bd)
print('actual',round(tot),'| oracle budget for failures:',round(fleet_cost(fl,np.where(fl==1,bd_or,0),y[idx],cost[idx]).sum()*sc))
# oracle classifier: flag exactly failures with cost>12.5 using our budgets
