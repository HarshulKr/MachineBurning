import sys,time; sys.path.insert(0,'src'); sys.path.insert(0,'/home/claude/theirs/fleet_maintenance')
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
import fleet_lib as TL
from fleet import *; from models import *
import models
from models import Platt
models.LGB_COST.update(n_estimators=400, min_child_samples=8)
tr,te=load('../data'); y=tr.failure.values; fu=tr.followup_days.values; cost=tr.repair_cost.values
FS=folds(y,5,3); M=fu==365; logc=np.log(np.nan_to_num(cost,nan=1.0)); pi0=y[M].mean()
teA=align_test_depot(te)
for c,m in {'s055':10,'s057':100,'s062':10}.items(): tr[c]=np.round(tr[c]/m)*m
F=[c for c in SENSORS if c not in ('s047','s120','s122')]
X=add_features(tr,F); W=np.where((y==0)&(fu<365),0.0,1.0)
t=time.time(); PM=oof_proba(lambda r: make_lgb(SEED+r),X,y,FS,w=W); print('mine clf %.0fs'%(time.time()-t),flush=True)
MU,RES=oof_cost_model(X,logc,y,FS,recal=True)
T=np.load('theirs_oof.npz'); PT=(T['lgb']+T['xgb']+T['cat'])/3; MUT=T['MU']
from sklearn.metrics import roc_auc_score, average_precision_score
for n,P in [('mine lgb',PM),('theirs ens',PT),('their lgb',T['lgb']),('their xgb',T['xgb']),('their cat',T['cat']),('avg mine+ens',(PM+PT)/2)]:
    print(n,'AUC %.4f AP %.4f'%(np.mean([roc_auc_score(y[M],p[M]) for p in P]),np.mean([average_precision_score(y[M],p[M]) for p in P])))
fk=[np.zeros(len(y),int) for _ in range(3)]
for r in range(3):
    for k,(a,b) in enumerate(FS[r]): fk[r][b]=k
posi=np.where(M&(y==1))[0]; negi=np.where(M&(y==0))[0]
def score(pipe, P, pi_true=None, pi_as=None, n=12):
    rng=np.random.default_rng(1); tots=[]
    for s in range(3 if pi_true is None else n):
        r=s%3
        idx=np.where(M)[0] if pi_true is None else np.r_[posi,rng.choice(negi,int(len(posi)*(1-pi_true)/pi_true),replace=False)]
        fl=np.zeros(len(idx),int); bd=np.zeros(len(idx))
        if pipe=='theirs':
            p=P[r][idx]
            if pi_as: p=adjust_prior(p,0.00709,pi_as)
            f,b=TL.decide(p,MUT[r,idx],0.955); fl,bd=f,b
        else:
            pc=cross_calibrate(P[r],y,FS[r],M)[idx]
            if pi_as: pc=adjust_prior(pc,pi0,pi_as)
            for k in range(5):
                mk=fk[r][idx]==k; f,b,_,_=decide(pc[mk],MU[r,idx[mk]],RES[r][k][1]); fl[mk]=f; bd[mk]=b
        tots.append(fleet_cost(fl,bd,y[idx],cost[idx]).sum()*20000/len(idx))
    return np.array(tots)
rows=[]
cfgs=[('THEIRS (ens raw, lognormal, no hist)','theirs',PT),('MINE V3 (lgb Platt, emp. resid, hist)','mine',PM),
      ('MINE + their ensemble probs','mine',PT),('MINE + avg(mine, their ens)','mine',(PM+PT)/2)]
base=None
for name,pipe,P in cfgs:
    nat=score(pipe,P)
    s36=score(pipe,P,0.05,0.036); s48=score(pipe,P,0.05,0.0475)
    s36t=score(pipe,P,0.036,0.036); s48t=score(pipe,P,0.036,0.0475)
    rows.append((name,nat.mean(),nat.std(),s36.mean(),s48.mean(),s36t.mean(),s48t.mean()))
    print(rows[-1],flush=True)
d=pd.DataFrame(rows,columns=['pipeline','old depot','std','5% depot, assume 3.6%','5% depot, assume 4.8%','3.6% depot, assume 3.6%','3.6% depot, assume 4.8%']).round(0)
print(d.to_string(index=False)); d.to_csv('cmp_results.csv',index=False)
