import sys,pickle; sys.path.insert(0,'src'); import numpy as np, warnings; warnings.filterwarnings('ignore')
from fleet import *; from models import *
tr,te=load('../data'); F=[c for c in SENSORS if c not in ('s047','s120','s122')]
y=tr.failure.values; fu=tr.followup_days.values; cost=tr.repair_cost.values
X=add_features(tr,F); FS=folds(y,5,3); O=dict(np.load('oof_cache.npz'))
M=fu==365; logc=np.log(np.nan_to_num(cost,nan=1.0))
def pol(p,mu,res,tri,vai): f,b,_,_=decide(p,mu,res); return f,b
P=O['lgb_dropcens']; R={}
for recal in [False,True]:
    MU,RES=oof_cost_model(X,logc,y,FS,recal=recal)
    m,s,parts=evaluate(P,y,cost,FS,M,pol,MU,RES)
    tots=[fleet_cost(f,b,y,cost)[M].sum()*20000/M.sum() for f,b,_ in parts]; R[recal]=np.array(tots)
    print('recal',recal,round(m),round(s), np.round(tots))
    np.savez(f'cost_cache_recal{int(recal)}.npz',MU=MU); pickle.dump(RES,open(f'res_cache_recal{int(recal)}.pkl','wb'))
print('paired diff', R[True]-R[False])
