import sys,pickle; sys.path.insert(0,'src'); import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from fleet import *; from models import *
tr,te=load('../data'); y=tr.failure.values; fu=tr.followup_days.values; cost=tr.repair_cost.values
FS=folds(y,5,3); M=fu==365; pi0=y[M].mean()
MUn,RESn,MUo,RESo=pickle.load(open('full_cmp_cost.pkl','rb'))
RESo=[[x[1] if isinstance(x,tuple) else x for x in rr] for rr in RESo]
T=np.load('theirs_oof.npz'); PT=(T['lgb']+T['xgb']+T['cat'])/3
fk=[np.zeros(len(y),int) for _ in range(3)]
for r in range(3):
    for k,(a,b) in enumerate(FS[r]): fk[r][b]=k
cal=[cross_calibrate(PT[r],y,FS[r],M) for r in range(3)]
posi=np.where(M&(y==1))[0]; negi=np.where(M&(y==0))[0]
def scores(MU,RES,pi_true,pi_as,n=30):
    rng=np.random.default_rng(7); tots=[]
    for s_ in range(n):
        r=s_%3; idx=np.r_[posi,rng.choice(negi,int(len(posi)*(1-pi_true)/pi_true),replace=False)]
        pc=adjust_prior(cal[r][idx],pi0,pi_as); fl=np.zeros(len(idx),int); bd=np.zeros(len(idx))
        for k in range(5):
            mk=fk[r][idx]==k; f_,b_,_,_=decide(pc[mk],MU[r,idx[mk]],RES[r][k]); fl[mk]=f_; bd[mk]=b_
        tots.append(fleet_cost(fl,bd,y[idx],cost[idx]).sum()*20000/len(idx))
    return np.array(tots)
MUa=(MUn+MUo)/2; RESa=RESn  # rough: avg mu with new residual pools
rows=[]
for pt in [0.03,0.04]:
    for pa in [0.03,0.035,0.0477]:
        o=scores(MUo,RESo,pt,pa); n=scores(MUn,RESn,pt,pa)
        d=n-o; rows.append((pt,pa,o.mean(),n.mean(),d.mean(),d.std()/np.sqrt(len(d)),(d<0).mean()))
print(pd.DataFrame(rows,columns=['true pi','assumed pi','current','new','diff','se','P(new better)']).round(1).to_string(index=False))
