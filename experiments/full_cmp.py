import sys,itertools,time,pickle; sys.path.insert(0,'src'); import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from fleet import *; from models import *
import models; models.LGB_COST.update(n_estimators=400, min_child_samples=8)
exec(open('cost_nested.py').read().split("def evaluate(")[0].split("tr,te=load('../data')")[1].split("y=tr.failure.values")[1].split("P8=")[0]) if False else None
from sklearn.linear_model import RidgeCV, LinearRegression
from scipy.stats import spearmanr
tr,te=load('../data'); y=tr.failure.values; fu=tr.followup_days.values; cost=tr.repair_cost.values
for c,m in {'s055':10,'s057':100,'s062':10}.items(): tr[c]=np.round(tr[c]/m)*m
FS=folds(y,5,3); M=fu==365; logc=np.log(np.nan_to_num(cost,nan=1.0)); pi0=y[M].mean()
F=[c for c in SENSORS if c not in ('s047','s120','s122','s146','s063')]
P8=dict(models.LGB_COST)
L=lambda s: np.log1p(s.clip(lower=0))
HB=[h for h in HIST if h!='s063']
class CostFE:
    def __init__(s,seed,rw=0.4,k=12,kr=40): s.seed=seed; s.rw=rw; s.k=k; s.kr=kr
    def feats(s,df):
        X=add_features(df,F); H=df[HB].values; tot=np.nansum(H,1)
        for i,h in enumerate(HB): X[h+'_frac']=H[:,i]/np.maximum(tot,1)
        X['hist_tot']=np.log1p(tot)
        for a,b in itertools.combinations(s.top,2): X[f'r_{a}_{b}']=L(df[a])-L(df[b])
        return X
    def fit(s,df,yy):
        base=add_features(df,F); imp=pd.Series(lgb.LGBMRegressor(random_state=s.seed,**P8).fit(base,yy).booster_.feature_importance('gain'),base.columns)
        s.top=[t for t in imp.sort_values(ascending=False).index if t.startswith('s')][:s.k]
        s.m=lgb.LGBMRegressor(random_state=s.seed,**P8).fit(s.feats(df),yy)
        Lg=L(df[F]); s.med=Lg.median(); Lg=Lg.fillna(s.med)
        sp=pd.Series({cc:abs(spearmanr(Lg[cc],yy)[0]) for cc in F}).fillna(0).sort_values(ascending=False); s.rc=list(sp.index[:s.kr])
        s.r=make_pipeline(StandardScaler(),RidgeCV(alphas=np.logspace(-2,4,30))).fit(Lg[s.rc],yy); return s
    def predict(s,df): return (1-s.rw)*s.m.predict(s.feats(df))+s.rw*s.r.predict(L(df[s.rc]).fillna(s.med))
def cv_cost(make):
    MU=np.zeros((3,len(y))); RES=[]
    for r,fs in enumerate(FS):
        rr=[]
        for k,(a,b) in enumerate(fs):
            fi=a[y[a]==1]; df=tr.iloc[fi]; yy=logc[fi]; ip=np.zeros(len(fi))
            for aa,b2 in KFold(5,shuffle=True,random_state=r).split(df): ip[b2]=make(r).fit(df.iloc[aa],yy[aa]).predict(df.iloc[b2])
            lr=LinearRegression().fit(ip[:,None],yy); MU[r,b]=lr.predict(make(r).fit(df,yy).predict(tr.iloc[b])[:,None]); rr.append(yy-lr.predict(ip[:,None]))
        RES.append(rr); print('rep',r,flush=True)
    return MU,RES
t=time.time(); MUn,RESn=cv_cost(lambda s: CostFE(s)); print('new cost %.0fs'%(time.time()-t))
X=add_features(tr,F); MUo,RESo=oof_cost_model(X,logc,y,FS,recal=True)
pickle.dump((MUn,RESn,MUo,RESo),open('full_cmp_cost.pkl','wb'))
T=np.load('theirs_oof.npz'); PT=(T['lgb']+T['xgb']+T['cat'])/3
fk=[np.zeros(len(y),int) for _ in range(3)]
for r in range(3):
    for k,(a,b) in enumerate(FS[r]): fk[r][b]=k
cal=[cross_calibrate(PT[r],y,FS[r],M) for r in range(3)]
posi=np.where(M&(y==1))[0]; negi=np.where(M&(y==0))[0]
def score(MU,RES,pi_true=None,pi_as=None,n=12):
    rng=np.random.default_rng(1); tots=[]
    for s_ in range(3 if pi_true is None else n):
        r=s_%3; idx=np.where(M)[0] if pi_true is None else np.r_[posi,rng.choice(negi,int(len(posi)*(1-pi_true)/pi_true),replace=False)]
        pc=cal[r][idx]; pc=adjust_prior(pc,pi0,pi_as) if pi_as else pc
        fl=np.zeros(len(idx),int); bd=np.zeros(len(idx))
        for k in range(5):
            mk=fk[r][idx]==k; f_,b_,_,_=decide(pc[mk],MU[r,idx[mk]],RES[r][k]); fl[mk]=f_; bd[mk]=b_
        tots.append(fleet_cost(fl,bd,y[idx],cost[idx]).sum()*20000/len(idx))
    return np.mean(tots)
rows=[]
for name,MU,RES in [('current cost model',MUo,[[x[1] for x in rr] for rr in RESo] if isinstance(RESo[0][0],tuple) else RESo),('new cost model',MUn,RESn)]:
    rows.append([name,score(MU,RES)]+[score(MU,RES,pt,pa) for pt,pa in [(0.03,0.03),(0.03,0.035),(0.03,0.0477),(0.05,0.035),(0.05,0.0477)]])
print(pd.DataFrame(rows,columns=['cost model','old depot','3% world/assume 3%','3%/3.5%','3%/4.77%','5%/3.5%','5%/4.77%']).round(0).to_string(index=False))
