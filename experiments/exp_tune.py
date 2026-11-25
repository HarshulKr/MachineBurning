import sys,time; sys.path.insert(0,'src'); import numpy as np, warnings; warnings.filterwarnings('ignore')
from fleet import *; from models import *
from sklearn.metrics import log_loss, roc_auc_score, average_precision_score
tr,te=load('../data'); F=[c for c in SENSORS if c!='s047']
X=add_features(tr,F); y=tr.failure.values
fs=folds(y,5,1)
for kw in [dict(n_estimators=300), dict(n_estimators=300,num_leaves=7), dict(n_estimators=400,num_leaves=7,min_child_samples=50),
           dict(n_estimators=300,colsample_bytree=0.2), dict(n_estimators=500,learning_rate=0.01,num_leaves=31,min_child_samples=40)]:
    t=time.time(); P=oof_proba(lambda r: make_lgb(**kw),X,y,fs)[0]
    print(kw, 'AUC %.4f AP %.4f LL %.5f'%(roc_auc_score(y,P),average_precision_score(y,P),log_loss(y,P)), '%.0fs'%(time.time()-t))
