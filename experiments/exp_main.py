import sys,time; sys.path.insert(0,'src'); import numpy as np, warnings; warnings.filterwarnings('ignore')
from fleet import *; from models import *
tr,te=load('../data'); F=[c for c in SENSORS if c!='s047']
y=tr.failure.values; fu=tr.followup_days.values
X=add_features(tr,F); Xfu=X.assign(followup_days=fu); Xfu365=Xfu.assign(followup_days=365)
FS=folds(y,5,3); out={}
def run(name,fn,XX,w=None,PX=None):
    t=time.time(); out[name]=oof_proba(fn,XX,y,FS,w=w,predict_X=PX); print(name,'%.0fs'%(time.time()-t),flush=True)
run('lr',lambda r: make_logreg(0.1),X)
run('lgb_all',lambda r: make_lgb(SEED+r),X)
run('lgb_fu',lambda r: make_lgb(SEED+r),Xfu,PX=Xfu365)
w=np.where((y==0)&(fu<365),fu/365,1.0)
run('lgb_w',lambda r: make_lgb(SEED+r),X,w=w)
w2=np.where((y==0)&(fu<365),0.0,1.0)
run('lgb_dropcens',lambda r: make_lgb(SEED+r),X,w=w2)
F2=[c for c in F if c not in ('s120','s122')]
run('lgb_all_no120',lambda r: make_lgb(SEED+r),add_features(tr,F2))
np.savez('oof_cache.npz',**out)
