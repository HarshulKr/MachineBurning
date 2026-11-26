import sys; sys.path.insert(0,'src'); import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from fleet import *; from models import *
tr,te=load('../data'); y=tr.failure.values; fu=tr.followup_days.values; M=fu==365; pi0=y[M].mean()
T=np.load('theirs_oof.npz'); D=np.load('diag_test.npz')
oof=((T['lgb']+T['xgb']+T['cat'])/3).mean(0); praw=(D['lgb']+D['xgb']+D['cat'])/3
cal=Platt().fit(oof[M],y[M]); pc0=cal.predict(praw)
mu=D['mu']; r=np.sort(D['res']); C=np.exp(mu[:,None]+r[None,:]); EC=C.mean(1)
def parts(flag,b,p):
    f=flag==1; over=np.maximum(C-b[:,None],0).mean(1); under=np.maximum(b[:,None]-C,0).mean(1)
    return {'inspect':10*f.sum()*.3,'rush':(p*0.2*over)[f].sum()*.3,'return':(p*0.02*under+(1-p)*0.02*b)[f].sum()*.3,'missed':(p*EC)[~f].sum()*.3}
sub=pd.read_csv('run/submissions/sub_v3.csv')
for pt in [0.03]:
    p=adjust_prior(pc0,pi0,pt); d=parts(sub.flag.values,sub.budget.values,p); print('V3 actual decisions, world pi',pt,{k:round(v) for k,v in d.items()},'total',round(sum(d.values())))
    rows=[]
    for pa in [0.015,0.02,0.025,0.03,0.035,0.04,0.0477]:
        f,b,_,_=decide(adjust_prior(pc0,pi0,pa),mu,r); 
        row=[pa,f.sum()]
        for pt2 in [0.02,0.03,0.04,0.05]:
            row.append(round(sum(parts(f,b,adjust_prior(pc0,pi0,pt2)).values())))
        rows.append(row)
    print(pd.DataFrame(rows,columns=['decide with pi','#flag','world .02','world .03','world .04','world .05']).to_string(index=False))
