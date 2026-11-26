import sys,pickle; sys.path.insert(0,'src'); import numpy as np, warnings; warnings.filterwarnings('ignore')
from fleet import *; from models import *
tr,te=load('../data'); y=tr.failure.values; fu=tr.followup_days.values; cost=tr.repair_cost.values
FS=folds(y,5,3); M=fu==365; O=dict(np.load('oof_cache.npz'))
MU=np.load('cost_cache_recal1.npz')['MU']; RES=pickle.load(open('res_cache_recal1.pkl','rb'))
P=O['lgb_dropcens']; pi0=y[M].mean()
# OOF calibrated p, budgets prepared per repeat
cal=[]
for r,fs in enumerate(FS):
    pc=cross_calibrate(P[r],y,fs,M)
    resid=np.zeros((len(y),len(RES[r][0][1])),dtype=float)  # pool per row (by fold)
    pools={}
    for k,(a,b) in enumerate(fs): pools.update({i:k for i in b})
    cal.append((pc,np.array([pools[i] for i in range(len(y))])))
rng=np.random.default_rng(0); posi=np.where(M&(y==1))[0]; negi=np.where(M&(y==0))[0]
def sim_cost(pi_true, pi_assumed, n_sims=20):
    tots=[]
    for s in range(n_sims):
        r=s%3; pc,fk=cal[r]
        n_pos=len(posi); n_neg=int(n_pos*(1-pi_true)/pi_true)
        idx=np.r_[posi, rng.choice(negi,min(n_neg,len(negi)),replace=False)]
        p=adjust_prior(pc[idx],pi0,pi_assumed)
        fl=np.zeros(len(idx),int); bd=np.zeros(len(idx))
        for k in range(5):
            mk=fk[idx]==k; f,b,_,_=decide(p[mk],MU[r,idx[mk]],RES[r][k][1]); fl[mk]=f; bd[mk]=b
        tots.append(fleet_cost(fl,bd,y[idx],cost[idx]).sum()*20000/len(idx))
    return np.mean(tots),np.std(tots)
for pi_true in [0.0066,0.05]:
    for pa in [pi0,0.02,0.035,0.045,0.05,0.06,0.08]:
        m,s=sim_cost(pi_true,pa); print(f'true prior {pi_true:.4f} assumed {pa:.4f}: cost/20k {m:.0f} ± {s:.0f}',flush=True)
