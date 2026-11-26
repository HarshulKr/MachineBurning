import pandas as pd, numpy as np, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
tr=pd.read_csv('../data/train.csv'); te=pd.read_csv('../data/test.csv')
S=[c for c in te.columns if c.startswith('s')]
f=tr[tr.failure==1]; n=tr[tr.failure==0]
print(pd.crosstab(tr.followup_days.lt(365), tr.failure))
print(np.percentile(n.followup_days[n.followup_days<365],[5,25,50,75,95]))
print(np.percentile(f.followup_days[f.followup_days<365],[5,25,50,75,95]))
# adversarial validation
X=pd.concat([tr[S],te[S]]).drop(columns=['s047']); y=np.r_[np.zeros(len(tr)),np.ones(len(te))]
oof=np.zeros(len(y)); imp=np.zeros(X.shape[1])
for a,b in StratifiedKFold(5,shuffle=True,random_state=0).split(X,y):
    m=lgb.LGBMClassifier(n_estimators=200,learning_rate=0.05,num_leaves=31,verbose=-1).fit(X.iloc[a],y[a])
    oof[b]=m.predict_proba(X.iloc[b])[:,1]; imp+=m.booster_.feature_importance('gain')
print('adv AUC',roc_auc_score(y,oof))
print(pd.Series(imp,X.columns).sort_values(ascending=False).head(15))
# per-feature KS
# per-feature shift: relative change of the median
md={c:abs(te[c].median()-tr[c].median())/(abs(tr[c].median())+1) for c in S if te[c].notna().sum()>100}
print(pd.Series(md).sort_values(ascending=False).head(15))
# NaN pattern shift
print('row nan train',tr[S].isna().sum(1).mean(),'test',te[S].isna().sum(1).mean())
