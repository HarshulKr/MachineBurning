import sys; sys.path.insert(0,'src'); import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from fleet import *; from models import *
tr,te=load('../data'); y=tr.failure.values; fu=tr.followup_days.values; M=fu==365; pi0=y[M].mean()
T=np.load('theirs_oof.npz'); D=np.load('diag_test.npz')
oof=((T['lgb']+T['xgb']+T['cat'])/3).mean(0); praw=(D['lgb']+D['xgb']+D['cat'])/3
pc0=Platt().fit(oof[M],y[M]).predict(praw); mu=D['mu']; r=np.sort(D['res'])
subs={'old':pd.read_csv('/home/claude/theirs/fleet_maintenance/submission_em.csv'),'v3':pd.read_csv('run/submissions/sub_v3.csv')}
LB={'old':20277.3,'v3':19515.0}
def ev(sub,p,k):
    C=k*np.exp(mu[:,None]+r[None,:]); f=sub.flag.values==1; b=sub.budget.values
    over=np.maximum(C-b[:,None],0).mean(1); under=np.maximum(b[:,None]-C,0).mean(1)
    return np.where(f,10+p*(0.2*over+0.02*under)+(1-p)*0.02*b,p*C.mean(1)).sum()*0.3
rows=[]
for pi in [0.02,0.025,0.03,0.035,0.04,0.0477,0.055]:
    p=adjust_prior(pc0,pi0,pi)
    for k in [0.6,0.7,0.8,0.9,1.0,1.1]:
        e={n:ev(s,p,k) for n,s in subs.items()}
        err=np.sqrt(np.mean([(e[n]-LB[n])**2 for n in LB]))
        f,b,cf,cs=decide(p,mu+np.log(k),r)
        rows.append((pi,k,round(e['old']),round(e['v3']),round(err),round((p*k*np.exp(mu[:,None]+r).mean(1)).sum()*0.3),f.sum()))
d=pd.DataFrame(rows,columns=['pi','cost scale','E old','E v3','rmse vs LB','E flag-nobody','#flag optimal'])
print(d.sort_values('rmse vs LB').head(12).to_string(index=False))
