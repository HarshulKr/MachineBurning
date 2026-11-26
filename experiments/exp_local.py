from sklearn.linear_model import LinearRegression
import sys,pickle; sys.path.insert(0,'src'); import numpy as np, warnings; warnings.filterwarnings('ignore')
from fleet import *; from models import *
tr,te=load('../data'); F=[c for c in SENSORS if c not in ('s047','s120','s122')]
y=tr.failure.values; fu=tr.followup_days.values; cost=tr.repair_cost.values
X=add_features(tr,F); FS=folds(y,5,3); M=fu==365; O=dict(np.load('oof_cache.npz')); P=O['lgb_dropcens']; pi0=y[M].mean()
logc=np.log(np.nan_to_num(cost,nan=1.0))
# cost model with inner preds kept
MU=np.zeros((3,len(y))); POOL=[]
for r,fs in enumerate(FS):
    pr=[]
    for k,(a,b) in enumerate(fs):
        fi=a[y[a]==1]; Xf=X.iloc[fi]; yf=logc[fi]; ip=np.zeros(len(fi))
        for aa,bb in KFold(5,shuffle=True,random_state=SEED+r).split(Xf): ip[bb]=make_cost_model(r).fit(Xf.iloc[aa],yf[aa]).predict(Xf.iloc[bb])
        _l=LinearRegression().fit(ip[:,None],yf); a1,a0=float(_l.coef_[0]),float(_l.intercept_); m=make_cost_model(r).fit(Xf,yf); MU[r,b]=a0+a1*m.predict(X.iloc[b]); ipc=a0+a1*ip
        pr.append((ipc,yf-ipc))
    POOL.append(pr)
def pool_for(mu,ipc,res,K):
    if K is None: return res
    d=np.abs(mu[:,None]-ipc[None,:]); nn=np.argsort(d,1)[:,:K]; return res[nn]
cal=[cross_calibrate(P[r],y,FS[r],M) for r in range(3)]
fk=[np.zeros(len(y),int) for _ in range(3)]
for r in range(3):
    for k,(a,b) in enumerate(FS[r]): fk[r][b]=k
posi=np.where(M&(y==1))[0]; negi=np.where(M&(y==0))[0]
def run(K,pi_true,pi_as,n=12):
    rng=np.random.default_rng(1); tots=[]; cov=[]
    for s in range(n):
        r=s%3
        idx=np.where(M)[0] if pi_true is None else np.r_[posi,rng.choice(negi,int(len(posi)*(1-pi_true)/pi_true),replace=False)]
        p=cal[r][idx] if pi_as is None else adjust_prior(cal[r][idx],pi0,pi_as)
        fl=np.zeros(len(idx),int); bd=np.zeros(len(idx))
        for k in range(5):
            mk=fk[r][idx]==k; ipc,res=POOL[r][k]; mu=MU[r,idx[mk]]
            f,b,_,_=decide(p[mk],mu,pool_for(mu,ipc,res,K)); fl[mk]=f; bd[mk]=b
        tots.append(fleet_cost(fl,bd,y[idx],cost[idx]).sum()*20000/len(idx))
        cf=(fl==1)&(y[idx]==1)&(p>0.8); cov.append((cost[idx][cf]<=bd[cf]).mean())
        if pi_true is None and s==2: break
    return np.mean(tots),np.std(tots),np.mean(cov)
for K in [None,60,120,200]:
    print('K',K,'old depot',np.round(run(K,None,None),3),'| 5% depot',np.round(run(K,0.05,0.048),3),flush=True)
