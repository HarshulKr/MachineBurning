import sys; sys.path.insert(0,'src'); import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from fleet import *; from models import *
from sklearn.metrics import roc_auc_score
tr,te=load('../data'); teA=align_test_depot(te); y=tr.failure.values
for c in ['s055','s062']: tr[c]=np.round(tr[c]/10)*10
def adv(cols):
    X=pd.concat([tr[cols],teA[cols]]); z=np.r_[np.zeros(len(tr)),np.ones(len(te))]; oof=np.zeros(len(z))
    for a,b in StratifiedKFold(5,shuffle=True,random_state=0).split(X,z):
        oof[b]=lgb.LGBMClassifier(n_estimators=150,learning_rate=0.05,verbose=-1).fit(X.iloc[a],z[a]).predict_proba(X.iloc[b])[:,1]
    return round(roc_auc_score(z,oof),3)
print('hist block only', adv(HIST))
print('hist without s063', adv(HIST[:-1]))
for c in HIST: print(c, 'alone', adv([c]))
# ratio features
H=HIST[:-1]
r_tr=tr[H].div(tr[H].sum(1)+1,axis=0); r_te=teA[H].div(teA[H].sum(1)+1,axis=0)
print('hist totals: train q', tr[HIST].sum(1).quantile([.1,.5,.9]).values, 'aligned test', teA[HIST].sum(1).quantile([.1,.5,.9]).values)
