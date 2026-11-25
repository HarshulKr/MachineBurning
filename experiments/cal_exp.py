import sys; sys.path.insert(0,'src'); import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from fleet import *; from models import *
import models; models.LGB_COST.update(n_estimators=400, min_child_samples=8)
from sklearn.linear_model import LogisticRegression
tr,te=load('../data'); y=tr.failure.values; fu=tr.followup_days.values; cost=tr.repair_cost.values
FS=folds(y,5,3); M=fu==365; logc=np.log(np.nan_to_num(cost,nan=1.0)); pi0=y[M].mean()
for c,m in {'s055':10,'s057':100,'s062':10}.items(): tr[c]=np.round(tr[c]/m)*m
F=[c for c in SENSORS if c not in ('s047','s120','s122')]
X=add_features(tr,F); MU,RES=oof_cost_model(X,logc,y,FS,recal=True)
T=np.load('theirs_oof.npz'); comps=np.stack([T['lgb'],T['xgb'],T['cat']],-1)  # (3,n,3)
def lg(p): p=np.clip(p,1e-6,1-1e-6); return np.log(p/(1-p))
def feats(kind,r):
    pm=comps[r].mean(1)
    if kind=='platt': return lg(pm)[:,None]
    if kind=='platt2': z=lg(pm); return np.c_[z,z**2]
    if kind=='beta': q=np.clip(pm,1e-6,1-1e-6); return np.c_[np.log(q),-np.log(1-q)]
    if kind=='stack': return lg(comps[r])
    if kind=='stack2': Z=lg(comps[r]); return np.c_[Z,Z.mean(1)**2]
def calib(kind,r,C=1e4):
    Z=feats(kind,r); p=np.zeros(len(y))
    for a,b in FS[r]:
        a=a[M[a]]; m=LogisticRegression(C=C,max_iter=5000).fit(Z[a],y[a]); p[b]=m.predict_proba(Z[b])[:,1]
    return p
fk=[np.zeros(len(y),int) for _ in range(3)]
for r in range(3):
    for k,(a,b) in enumerate(FS[r]): fk[r][b]=k
posi=np.where(M&(y==1))[0]; negi=np.where(M&(y==0))[0]
def score(PC, pi_true=None, pi_as=None, n=12):
    rng=np.random.default_rng(1); tots=[]
    for s in range(3 if pi_true is None else n):
        r=s%3
        idx=np.where(M)[0] if pi_true is None else np.r_[posi,rng.choice(negi,int(len(posi)*(1-pi_true)/pi_true),replace=False)]
        pc=PC[r][idx]
        if pi_as: pc=adjust_prior(pc,pi0,pi_as)
        fl=np.zeros(len(idx),int); bd=np.zeros(len(idx))
        for k in range(5):
            mk=fk[r][idx]==k; f,b,_,_=decide(pc[mk],MU[r,idx[mk]],RES[r][k][1]); fl[mk]=f; bd[mk]=b
        tots.append(fleet_cost(fl,bd,y[idx],cost[idx]).sum()*20000/len(idx))
    return np.array(tots)
rows=[]
for name,fn in [('platt',lambda r: calib('platt',r)),('platt quadratic',lambda r: calib('platt2',r)),
                ('LR on log p, log(1-p)',lambda r: calib('beta',r)),('LR stacking 3 logits',lambda r: calib('stack',r)),('LR stacking + quad',lambda r: calib('stack2',r))]:
    PC=[fn(r) for r in range(3)]
    # calibration table check: mean p vs rate on M
    nat=score(PC); s5=score(PC,0.05,0.0475); s36=score(PC,0.036,0.0475)
    rows.append((name,nat.mean(),nat.std(),s5.mean(),s36.mean(), np.mean([PC[r][M].mean() for r in range(3)])))
    print(rows[-1],flush=True)
print(pd.DataFrame(rows,columns=['calibration','old depot','std','5% depot','3.6% depot','mean p (365 rows)']).round(1).to_string(index=False))
print('true rate',pi0)
