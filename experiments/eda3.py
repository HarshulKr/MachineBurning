import pandas as pd, numpy as np, lightgbm as lgb, warnings; warnings.filterwarnings('ignore')
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
tr=pd.read_csv('../data/train.csv'); te=pd.read_csv('../data/test.csv')
S=[c for c in te.columns if c.startswith('s') and c!='s047']
for c,ct in [('s055','s056'),('s062','s063'),('s057','s058')]:
    a=tr[c].dropna(); b=te[ct].dropna()
    print(c,'mod10 train',((a%10)==0).mean().round(3),'test',((b%10)==0).mean().round(3))
f=tr[tr.failure==1].reset_index(drop=True); y=np.log(f.repair_cost)
X=np.log1p(f[S].clip(lower=0))
for name,m in [('ridge',make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),RidgeCV(alphas=np.logspace(-1,4,20)))),
               ('lgb',lgb.LGBMRegressor(n_estimators=200,learning_rate=0.03,num_leaves=7,min_child_samples=15,subsample=0.8,subsample_freq=1,colsample_bytree=0.5,verbose=-1))]:
    r2=[]
    for s in range(3):
        p=cross_val_predict(m,X,y,cv=KFold(5,shuffle=True,random_state=s)); r2.append(1-((y-p)**2).mean()/y.var())
    print(name,'R2 log cost',np.round(r2,3))
print('log cost sd',y.std(), 'skew', pd.Series(y).skew())
print(np.corrcoef(f.followup_days,y)[0,1])
# cost by followup<365
print(f.groupby(f.followup_days<365).repair_cost.describe())
