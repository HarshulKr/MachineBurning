import sys; sys.path.insert(0,'src'); import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from fleet import *; from models import *
tr,te=load('../data'); y=tr.failure.values; fu=tr.followup_days.values
w=np.where((y==0)&(fu<365),0.0,1.0); teA=align_test_depot(te)
F=[c for c in SENSORS if c not in ('s047','s120','s122')]
X=add_features(tr,F); XT=add_features(teA,F); m=make_lgb().fit(X,y,sample_weight=w)
ctr=m.predict_proba(X,pred_contrib=True)[fu==365].mean(0); cte=m.predict_proba(XT,pred_contrib=True).mean(0)
d=pd.Series(cte-ctr,index=list(X.columns)+['bias']).sort_values(ascending=False); print(d.head(12).round(3)); print(d.tail(5).round(3))
q=[.5,.9,.95,.99]
for c in d.index[:6]:
    if c in tr: print(c,'TR',tr[c].quantile(q).values,'NA',tr[c].isna().mean().round(3),'| TE',teA[c].quantile(q).values,'NA',teA[c].isna().mean().round(3))
