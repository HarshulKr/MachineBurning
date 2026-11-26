import sys,itertools; sys.path.insert(0,'src'); import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from fleet import *; from models import *
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.model_selection import cross_val_predict
from scipy.stats import spearmanr
tr,te=load('../data')
for c,m in {'s055':10,'s057':100,'s062':10}.items(): tr[c]=np.round(tr[c]/m)*m
F=[c for c in SENSORS if c not in ('s047','s120','s122','s146')]
y=tr.failure.values; f=tr[y==1].reset_index(drop=True); yc=np.log(f.repair_cost.values)
Lg=np.log1p(f[F].clip(lower=0)); Lg=Lg.fillna(Lg.median())
sp=pd.Series({c:spearmanr(Lg[c],yc)[0] for c in F}).abs().sort_values(ascending=False)
def r2cv(X,model):
    out=[]
    for s in range(3):
        p=cross_val_predict(model,X,yc,cv=KFold(5,shuffle=True,random_state=s)); out.append(1-((yc-p)**2).mean()/yc.var())
    return round(np.mean(out),3)
for k in [10,20,40,len(F)]:
    cols=sp.index[:k]
    print(k,'ridge log1p',r2cv(Lg[cols],make_pipeline(StandardScaler(),RidgeCV(alphas=np.logspace(-2,4,30)))),
          'lasso',r2cv(Lg[cols],make_pipeline(StandardScaler(),LassoCV(cv=5,max_iter=20000))))
# zeros indicator
Z=(f[F]==0).astype(float).add_suffix('_z')
cols=sp.index[:40]
print('ridge log1p + zero ind top40', r2cv(pd.concat([Lg[cols],Z[[c+'_z' for c in cols]]],axis=1),make_pipeline(StandardScaler(),RidgeCV(alphas=np.logspace(-2,4,30)))))
m=make_pipeline(StandardScaler(),LassoCV(cv=5,max_iter=20000)).fit(Lg[sp.index[:40]],yc)
coef=pd.Series(m[-1].coef_/m[0].scale_,sp.index[:40]); print(coef[coef.abs()>0].sort_values().round(3))
