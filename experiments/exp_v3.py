import sys,pickle,time; sys.path.insert(0,'src'); import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from fleet import *; from models import *
tr,te=load('../data'); F=[c for c in SENSORS if c not in ('s047','s120','s122')]
y=tr.failure.values; fu=tr.followup_days.values; cost=tr.repair_cost.values
FS=folds(y,5,3); M=fu==365
MU=np.load('cost_cache_recal1.npz')['MU']; RES=pickle.load(open('res_cache_recal1.pkl','rb'))
def pol(p,mu,res,tri,vai): f,b,_,_=decide(p,mu,res); return f,b
w=np.where((y==0)&(fu<365),0.0,1.0)
def hist_feats(df):
    H=df[HIST].values; tot=np.nansum(H,1); P=H/np.maximum(tot,1)[:,None]
    out=pd.DataFrame(P,columns=[c+'_frac' for c in HIST],index=df.index); out['hist_total']=tot; out['hist_meanbin']=(P*np.arange(10)).sum(1)
    return out
X=add_features(tr,F); Xh=pd.concat([X,hist_feats(tr)],axis=1)
O={}
def bag(XX,seeds):
    class B:
        def __init__(s,r): s.ms=[make_lgb(SEED+100*r+k) for k in range(seeds)]
        def fit(s,X,y,sample_weight=None): [m.fit(X,y,sample_weight=sample_weight) for m in s.ms]; return s
        def predict_proba(s,X): return np.mean([m.predict_proba(X) for m in s.ms],0)
    return B
for name,XX,fn in [('v3feat',X,lambda r: make_lgb(SEED+r)),('v3feat_hist',Xh,lambda r: make_lgb(SEED+r)),('v3feat_bag3',X,bag(X,3))]:
    t=time.time(); P=oof_proba(fn,XX,y,FS,w=w); O[name]=P
    m,s,parts=evaluate(P,y,cost,FS,M,pol,MU,RES); tots=[fleet_cost(f,b,y,cost)[M].sum()*20000/M.sum() for f,b,_ in parts]
    print(name,round(m),round(s),np.round(tots),'%.0fs'%(time.time()-t),flush=True)
np.savez('oof_v3.npz',**O)
