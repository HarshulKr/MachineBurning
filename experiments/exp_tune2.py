import sys,time; sys.path.insert(0,'src'); import numpy as np, warnings; warnings.filterwarnings('ignore')
from fleet import *; from models import *
from sklearn.metrics import log_loss, roc_auc_score, average_precision_score
tr,te=load('../data'); F=[c for c in SENSORS if c!='s047']
X=add_features(tr,F); y=tr.failure.values
fs=folds(y,5,1)
def rep(name,P): print(name,'AUC %.4f AP %.4f LL %.5f'%(roc_auc_score(y,P),average_precision_score(y,P),log_loss(y,np.clip(P,1e-6,1-1e-6))),flush=True)
for C in [0.01,0.03,0.1,0.3]:
    rep(f'LR C={C}', oof_proba(lambda r: make_logreg(C),X,y,fs)[0])
for kw in [dict(n_estimators=800,learning_rate=0.01,num_leaves=4,min_child_samples=100),
           dict(n_estimators=600,learning_rate=0.02,num_leaves=7,min_child_samples=50,reg_lambda=10,min_split_gain=0.0),
           dict(n_estimators=1000,learning_rate=0.02,num_leaves=7,min_child_samples=30,extra_trees=True)]:
    rep(str(kw), oof_proba(lambda r: make_lgb(**kw),X,y,fs)[0])
