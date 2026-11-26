import sys; sys.path.insert(0,'src'); import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from fleet import *; from models import *
from sklearn.metrics import roc_auc_score
tr,te=load('../data'); teA=align_test_depot(te)
F=[c for c in SENSORS if c not in ('s047','s120','s122')]
res={}
for c in F:
    a=tr[c].dropna(); b=teA[c].dropna()
    for m in [1000,100,50,20,10]:
        if ((b%m)==0).mean()>0.995 and ((a%m)==0).mean()<0.95: res[c]=m; break
print(res)
for c,m in res.items(): tr[c]=np.round(tr[c]/m)*m
def adv(cols):
    X=pd.concat([tr[cols],teA[cols]]); z=np.r_[np.zeros(len(tr)),np.ones(len(te))]; oof=np.zeros(len(z))
    for a,b in StratifiedKFold(5,shuffle=True,random_state=0).split(X,z):
        oof[b]=lgb.LGBMClassifier(n_estimators=150,learning_rate=0.05,verbose=-1).fit(X.iloc[a],z[a]).predict_proba(X.iloc[b])[:,1]
    return round(roc_auc_score(z,oof),3)
print('hist block after resolution harmonization', adv(HIST))
print('all robust features', adv(F))
STABLE=[c for c in F if c not in HIST]
print('their stable set only', adv(STABLE))
