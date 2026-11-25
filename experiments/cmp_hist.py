import sys; sys.path.insert(0,'src'); import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from fleet import *; from models import *
from sklearn.metrics import roc_auc_score
tr,te=load('../data'); teA=align_test_depot(te); y=tr.failure.values; fu=tr.followup_days.values
for c in ['s055','s062']: tr[c]=np.round(tr[c]/10)*10
# single-feature adversarial AUC (rank-based, NaN->-1): train vs test
def auc1(c, T):
    a=tr[c].fillna(-1).values; b=T[c].fillna(-1).values
    return roc_auc_score(np.r_[np.zeros(len(a)),np.ones(len(b))], np.r_[a,b])
rows=[(c, auc1(c,teA)) for c in HIST+['s123','s149','s034','s053','s071','s097','s012']]
print(pd.DataFrame(rows,columns=['col','AUC aligned test vs train']).round(3).to_string(index=False))
# expected AUC from pure prior shift: mix train pos with weight to make 5%
pos=tr[y==1]; neg=tr[(y==0)]
rng=np.random.default_rng(0); n=len(neg); k=int(n*0.05/0.95)
mixdf=pd.concat([neg, pos.sample(k,replace=True,random_state=0)])
print('expected AUC under pure 5% prior shift (train neg vs synthetic mixture):')
print({c: round(roc_auc_score(np.r_[np.zeros(len(neg)),np.ones(len(mixdf))], np.r_[neg[c].fillna(-1),mixdf[c].fillna(-1)]),3) for c in ['s057','s058','s056','s123','s149']})
