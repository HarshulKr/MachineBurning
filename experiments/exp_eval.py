import sys,time; sys.path.insert(0,'src'); import numpy as np, warnings; warnings.filterwarnings('ignore')
from fleet import *; from models import *
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss
tr,te=load('../data'); F=[c for c in SENSORS if c!='s047']
y=tr.failure.values; fu=tr.followup_days.values; cost=tr.repair_cost.values
X=add_features(tr,F); FS=folds(y,5,3); O=dict(np.load('oof_cache.npz'))
M=(fu==365); logc=np.log(np.nan_to_num(cost,nan=1.0))
sc=20000/M.sum()
c0=np.where(y==1,cost,0)
print('flag nobody %.0f'%(c0[M].sum()*sc))
print('flag all b=0 %.0f'%(fleet_cost(np.ones(len(y)),0,y,cost)[M].sum()*sc))
best=min((fleet_cost(np.ones(len(y)),b,y,cost)[M].sum()*sc,b) for b in [0,1,2,5,10]); print('flag all best const',best)
t=time.time(); MU,RES=oof_cost_model(X,logc,y,FS); print('cost model %.0fs'%(time.time()-t))
fm=(y==1); print('cost-model OOF R2 (log):', [round(1-((logc[fm]-MU[r,fm])**2).mean()/logc[fm].var(),3) for r in range(3)])
np.savez('cost_cache.npz',MU=MU)
import pickle; pickle.dump(RES,open('res_cache.pkl','wb'))
med=np.nanmedian(cost)
def pol_v1(p,mu,res,tri,vai):  # uses raw p stashed globally
    return (PRAW[vai]>0.5).astype(int), np.full(len(vai),med)
def pol_const(p,mu,res,tri,vai):
    r=logc[tri][y[tri]==1]; f,b,_,_=decide(p,np.zeros(len(p)),r); return f,b
def pol_model(p,mu,res,tri,vai):
    f,b,_,_=decide(p,mu,res); return f,b
for name,P in O.items():
    m=[roc_auc_score(y[M],P[r][M]) for r in range(3)]; ap=[average_precision_score(y[M],P[r][M]) for r in range(3)]
    line=f'{name:15s} AUC {np.mean(m):.4f} AP {np.mean(ap):.4f} | '
    if name=='lr':
        tots=[]
        for r in range(3):
            PRAW=P[r]; 
            fl=(P[r]>0.5).astype(int); tots.append(fleet_cost(fl,np.where(fl==1,med,0),y,cost)[M].sum()*sc)
        line+=f'p>0.5,b=median {np.mean(tots):.0f}±{np.std(tots):.0f} | '
    a=evaluate(P,y,cost,FS,M,pol_const); b=evaluate(P,y,cost,FS,M,pol_model,MU,RES)
    line+=f'decide const-cost {a[0]:.0f}±{a[1]:.0f} | decide cost-model {b[0]:.0f}±{b[1]:.0f}'
    print(line,flush=True)
