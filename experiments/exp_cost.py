import sys; sys.path.insert(0,'src'); import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from fleet import *; from models import *
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
tr,te=load('../data'); F=[c for c in SENSORS if c not in ('s047','s120','s122')]
f=tr[tr.failure==1].reset_index(drop=True); yc=np.log(f.repair_cost.values); X=add_features(f,F)
def pre(m): return make_pipeline(SimpleImputer(strategy='median'),FunctionTransformer(signed_log),StandardScaler(),m)
cands={
 'lgb_base':make_cost_model(),
 'lgb_l4_500':lgb.LGBMRegressor(n_estimators=500,learning_rate=0.02,num_leaves=4,min_child_samples=10,subsample=0.8,subsample_freq=1,colsample_bytree=0.5,verbose=-1),
 'lgb_l7_400':lgb.LGBMRegressor(n_estimators=400,learning_rate=0.02,num_leaves=7,min_child_samples=10,subsample=0.8,subsample_freq=1,colsample_bytree=0.3,verbose=-1),
 'rf':make_pipeline(SimpleImputer(strategy='median'),RandomForestRegressor(500,min_samples_leaf=3,max_features=0.3,n_jobs=2,random_state=0)),
 'ridge':pre(RidgeCV(alphas=np.logspace(-1,4,30))),
 'svr':pre(SVR(C=3,epsilon=0.1,gamma='scale')),
 'knn':pre(KNeighborsRegressor(10,weights='distance')),
}
preds={}
for k,m in cands.items():
    r2=[];ps=[]
    for s in range(3):
        p=cross_val_predict(m,X,yc,cv=KFold(5,shuffle=True,random_state=s)); ps.append(p); r2.append(1-((yc-p)**2).mean()/yc.var())
    preds[k]=np.array(ps); print(k,np.round(np.mean(r2),3),np.round(np.std(r2),3),flush=True)
for combo in [('lgb_base','svr'),('lgb_base','rf'),('lgb_base','ridge','svr')]:
    p=np.mean([preds[c] for c in combo],0); print(combo,[round(1-((yc-pp)**2).mean()/yc.var(),3) for pp in p])
