import sys; sys.path.insert(0,'src'); import numpy as np, warnings, time; warnings.filterwarnings('ignore')
from fleet import *; from models import *
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as Ck
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
tr,te=load('../data')
for c,m in {'s055':10,'s057':100,'s062':10}.items(): tr[c]=np.round(tr[c]/m)*m
F=[c for c in SENSORS if c not in ('s047','s120','s122','s146')]
f=tr[tr.failure==1].reset_index(drop=True); c=f.repair_cost.values; yc=np.log(c); X=add_features(f,F)
P8=dict(LGB_COST, n_estimators=400, min_child_samples=8)
def pin(b,c): return np.mean(0.2*np.maximum(c-b,0)+0.02*np.maximum(b-c,0))
def gp(nc): return make_pipeline(SimpleImputer(strategy='median'),FunctionTransformer(signed_log),StandardScaler(),PCA(nc,random_state=0),
                          GaussianProcessRegressor(Ck(1.0)*RBF(np.ones(nc)*3)+WhiteKernel(0.5),normalize_y=True,n_restarts_optimizer=1,random_state=0))
mk={'gp10':lambda s: gp(10),'gp20':lambda s: gp(20),'lgb':lambda s: lgb.LGBMRegressor(random_state=s,**P8)}
def run(names,seed):
    def fp(Xa,ya,Xb): return np.mean([mk[n](seed).fit(Xa,ya).predict(Xb) for n in names],0)
    b=np.zeros(len(c)); mu_all=np.zeros(len(c))
    for a,bb in KFold(5,shuffle=True,random_state=100+seed).split(X):
        Xa,ya=X.iloc[a],yc[a]; ip=np.zeros(len(ya))
        for aa,b2 in KFold(5,shuffle=True,random_state=seed).split(Xa): ip[b2]=fp(Xa.iloc[aa],ya[aa],Xa.iloc[b2])
        lr=LinearRegression().fit(ip[:,None],ya); mu=lr.predict(fp(Xa,ya,X.iloc[bb])[:,None]); mu_all[bb]=mu
        b[bb]=np.exp(mu+np.quantile(ya-lr.predict(ip[:,None]),Q_STAR))
    return pin(b,c), 1-((yc-mu_all)**2).mean()/yc.var()
for names in [['gp10'],['gp20'],['lgb','gp20']]:
    t=time.time(); R=[run(names,s) for s in range(3)]
    print(names,'pinball %.2f ± %.2f  R2 %.3f  %.0fs'%(np.mean([r[0] for r in R]),np.std([r[0] for r in R]),np.mean([r[1] for r in R]),time.time()-t),flush=True)
