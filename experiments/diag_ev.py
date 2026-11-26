import sys; sys.path.insert(0,'src'); import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from fleet import *; from models import *
tr,te=load('../data'); y=tr.failure.values; fu=tr.followup_days.values; M=fu==365; pi0=y[M].mean()
T=np.load('theirs_oof.npz'); D=np.load('diag_test.npz')
oof=((T['lgb']+T['xgb']+T['cat'])/3).mean(0); praw=(D['lgb']+D['xgb']+D['cat'])/3
cal=Platt().fit(oof[M],y[M]); pc=cal.predict(praw)
mu=D['mu']; r=np.sort(D['res']); C=np.exp(mu[:,None]+r[None,:]); EC=C.mean(1)
def expected(sub,p):
    f=sub.flag.values==1; b=sub.budget.values
    over=np.maximum(C-b[:,None],0).mean(1); under=np.maximum(b[:,None]-C,0).mean(1)
    e=np.where(f,10+p*(0.2*over+0.02*under)+(1-p)*0.02*b, p*EC)
    return e.sum()*0.3
subs={'old submission_em (LB 20277)':pd.read_csv('/home/claude/theirs/fleet_maintenance/submission_em.csv'),
      'old submission_noem':pd.read_csv('/home/claude/theirs/fleet_maintenance/submission_noem.csv'),
      'V3 platt (LB 19515)':pd.read_csv('run/submissions/sub_v3.csv'),
      'flag nobody':pd.DataFrame({'flag':np.zeros(20000,int),'budget':np.zeros(20000)})}
rows=[]
for pi in [0.0066,0.02,0.03,0.04,0.0477,0.06,0.08]:
    p=adjust_prior(pc,pi0,pi); rows.append([pi,p.mean()]+[round(expected(s,p)) for s in subs.values()])
print(pd.DataFrame(rows,columns=['assumed pi','mean p']+list(subs)).to_string(index=False))
p=adjust_prior(pc,pi0,0.0477)
print('test p quantiles', np.percentile(p,[50,90,95,98,99,99.5]).round(3), 'n p>0.5', (p>0.5).sum())
print('mean E[c|fail] test', EC.mean().round(1), 'train failures mean cost', np.nanmean(tr.repair_cost).round(1), 'median exp(mu) test', np.median(np.exp(mu)).round(1), 'train median exp(mu) on failures?')
