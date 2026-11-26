import pandas as pd, numpy as np, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
tr=pd.read_csv('../data/train.csv'); te=pd.read_csv('../data/test.csv')
S=[c for c in te.columns if c.startswith('s')]
H=[f's{i:03d}' for i in range(54,64)]
print('train hist bin sums', tr[H].sum(1).describe()[['50%']].values, 'test', te[H].sum(1).describe()[['50%']].values)
# check other groups that sum equal: find columns whose row-sum equals
tot_tr=tr[H].sum(1)
cand=[c for c in S if np.nanmedian(np.abs(tr[c]-tot_tr)/(tot_tr+1))<0.05]; print('cols ~ hist total',cand)
te2=te.copy()
for k in range(54,63): te2[f's{k:03d}']=te[f's{k+1:03d}']
te2['s063']=0.0
def adv(trX,teX,label):
    X=pd.concat([trX,teX]); y=np.r_[np.zeros(len(trX)),np.ones(len(teX))]
    oof=np.zeros(len(y)); imp=np.zeros(X.shape[1])
    for a,b in StratifiedKFold(5,shuffle=True,random_state=0).split(X,y):
        m=lgb.LGBMClassifier(n_estimators=200,learning_rate=0.05,verbose=-1).fit(X.iloc[a],y[a])
        oof[b]=m.predict_proba(X.iloc[b])[:,1]; imp+=m.booster_.feature_importance('gain')
    print(label,'adv AUC',round(roc_auc_score(y,oof),4)); print(pd.Series(imp/imp.sum(),X.columns).sort_values(ascending=False).head(8).round(3).to_dict())
F=[c for c in S if c!='s047']
adv(tr[F],te2[F],'realigned')
ks={c:abs(te2[c].median()-tr[c].median())/(abs(tr[c].median())+1) for c in F}
print(pd.Series(ks).sort_values(ascending=False).head(12).round(3).to_dict())
F2=[c for c in F if c not in ('s120','s122')]
adv(tr[F2],te2[F2],'realigned - s120,s122')
# s120 vs s122 relation
print(np.corrcoef(np.log1p(tr.s120.fillna(0)),np.log1p(tr.s122.fillna(0)))[0,1])
for c in ['s120','s122']:
    for q in [.05,.3,.5,.7,.95]: print(c,q,tr[c].quantile(q),te[c].quantile(q), te[c].quantile(q)-tr[c].quantile(q), te[c].quantile(q)/max(tr[c].quantile(q),1))
