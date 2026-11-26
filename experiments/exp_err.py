import sys,pickle; sys.path.insert(0,'src'); import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from fleet import *; from models import *
tr,te=load('../data'); y=tr.failure.values; fu=tr.followup_days.values; cost=tr.repair_cost.values
FS=folds(y,5,3); O=dict(np.load('oof_cache.npz')); MU=np.load('cost_cache.npz')['MU']; RES=pickle.load(open('res_cache.pkl','rb'))
M=(fu==365); logc=np.log(np.nan_to_num(cost,nan=1.0)); sc=20000/M.sum()
def pol_model(p,mu,res,tri,vai):
    f,b,_,_=decide(p,mu,res); return f,b
for name in ['lgb_all','lgb_dropcens']:
    m,s,parts=evaluate(O[name],y,cost,FS,M,pol_model,MU,RES)
    fl,bd,pc=parts[0]
    bk=cost_breakdown(fl[M],bd[M],y[M],cost[M]); print(name,{k:round(v*sc if not k.startswith(('n_','fail')) else v) for k,v in bk.items()})
    # missed failures: their p and cost
    miss=M&(y==1)&(fl==0); print(' missed: n',miss.sum(),'p median',np.median(pc[miss]).round(4),'cost sum',cost[miss].sum(),'cost median',np.median(cost[miss]))
    print(' missed cost quantiles', np.percentile(cost[miss],[25,50,75,90,100]))
    caught=M&(y==1)&(fl==1); r=bd[caught]/cost[caught]; print(' caught budget/cost ratio quantiles',np.percentile(r,[10,25,50,75,90]).round(2))
    fp=M&(y==0)&(fl==1); print(' false flags',fp.sum(),'mean budget on them',bd[fp].mean().round(1))
    # calibration table
    d=pd.DataFrame({'p':pc[M],'y':y[M]}); d['bin']=pd.cut(d.p,[-1,0.001,0.01,0.03,0.1,0.3,0.6,1.01])
    print(d.groupby('bin').agg(n=('y','size'),mean_p=('p','mean'),rate=('y','mean')).round(4))
# blend LR + LGB
for wl in [0.0,0.2,0.3,0.5]:
    P=(1-wl)*O['lgb_dropcens']+wl*O['lr']; m,s,_=evaluate(P,y,cost,FS,M,pol_model,MU,RES); print('blend lr w',wl,round(m),round(s))
# rank-average
from scipy.stats import rankdata
