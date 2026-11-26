import sys; sys.path.insert(0,'src'); import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from fleet import *; from models import *
tr,te=load('../data'); y=tr.failure.values; fu=tr.followup_days.values
w=np.where((y==0)&(fu<365),0.0,1.0)
teA=align_test_depot(te)
for label,F in [('with s120/s122',[c for c in SENSORS if c!='s047']),('no s120/s122',[c for c in SENSORS if c not in ('s047','s120','s122')])]:
    X=add_features(tr,F); m=make_lgb().fit(X,y,sample_weight=w)
    ptr=m.predict_proba(X[fu==365])[:,1]
    for nm,T in [('raw test',te),('aligned test',teA)]:
        pt=m.predict_proba(add_features(T,F))[:,1]
        print(label,nm,'mean p test %.4f'%pt.mean(),'frac p>0.03 %.4f'%(pt>0.03).mean(),'| train(in-sample 365) mean p %.4f frac>0.03 %.4f'%(ptr.mean(),(ptr>0.03).mean()))
print('train failure rate (365 rows)', y[fu==365].mean())
