import sys; sys.path.insert(0,'src'); import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from fleet import *; from models import *
tr,te=load('../data'); y=tr.failure.values; fu=tr.followup_days.values; M=fu==365; pi0=y[M].mean()
T=np.load('theirs_oof.npz'); D=np.load('diag_test.npz')
oof=((T['lgb']+T['xgb']+T['cat'])/3).mean(0); praw=(D['lgb']+D['xgb']+D['cat'])/3
pc=adjust_prior(Platt().fit(oof[M],y[M]).predict(praw),pi0,0.03); mu=D['mu']; r=np.sort(D['res'])
def exp_cost(p,mu,r):
    f,b,cf,cs=decide(p,mu,r); return (np.where(f==1,cf,cs)).sum()*0.3
print('model as is', round(exp_cost(pc,mu,r)))
for sc in [0.75,0.5,0.25,0.0]:
    rr=r*sc if sc>0 else np.zeros(5)
    mu2=mu+ (np.log(np.mean(np.exp(r))) - np.log(np.mean(np.exp(rr))) )  # keep E[c] fixed
    print('residual sd x',sc, round(exp_cost(pc,mu2,rr)))
# perfect classifier in expectation: sharpen p
for t in [2,4]:
    z=np.log(pc/(1-pc)); ps=1/(1+np.exp(-t*(z-np.log(0.03/0.97))-np.log(0.03/0.97)))
    print('sharper p x',t,'mean p %.4f'%ps.mean(), round(exp_cost(ps,mu,r)))
