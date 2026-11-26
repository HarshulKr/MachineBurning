import sys,itertools; sys.path.insert(0,'src'); import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from fleet import *
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
tr,te=load('../data'); y=tr.failure.values; f=tr[y==1].reset_index(drop=True); yc=np.log(f.repair_cost.values)
F=[s for s in SENSORS if s not in ('s047','s120','s122','s146','s063')]
Lg=np.log1p(f[F].clip(lower=0)); Lg=Lg.fillna(Lg.median())
H=f[[h for h in HIST if h!='s063']].values; tot=np.nansum(H,1)
Hf=pd.DataFrame(np.log((H+1)/(tot[:,None]+10)),columns=[h+'_lf' for h in HIST if h!='s063']).fillna(0)
def r2(X,m):
    o=[]
    for s in range(3):
        p=cross_val_predict(m,X,yc,cv=KFold(5,shuffle=True,random_state=s)); o.append(1-((yc-p)**2).mean()/yc.var())
    return round(np.mean(o),3)
las=lambda: make_pipeline(StandardScaler(),LassoCV(cv=5,max_iter=50000,n_jobs=2))
print('lasso log1p all', r2(Lg,las()))
X=pd.concat([Lg,Hf],axis=1); print('lasso log1p + log hist fracs', r2(X,las()))
Z=(f[F].fillna(0)==0).astype(float).add_suffix('_z'); X2=pd.concat([Lg,Hf,Z],axis=1); print('+ zero indicators', r2(X2,las()))
# quadratic terms of top 20 by spearman
sp=pd.Series({c:abs(spearmanr(Lg[c],yc)[0]) for c in Lg}).fillna(0).sort_values(ascending=False); top=list(sp.index[:20])
Q=pd.DataFrame({f'{a}*{b}':Lg[a]*Lg[b] for a,b in itertools.combinations_with_replacement(top,2)})
X3=pd.concat([X2,Q],axis=1); print('+ quadratic top20', r2(X3,las()))
