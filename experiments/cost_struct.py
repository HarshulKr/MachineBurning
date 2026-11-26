import sys,itertools; sys.path.insert(0,'src'); import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from fleet import *
from scipy.stats import spearmanr, rankdata
tr,te=load('../data'); y=tr.failure.values; f=tr[y==1].reset_index(drop=True); c=f.repair_cost.values; yc=np.log(c)
print('integer?',np.all(c==np.round(c)),'n at 5:',(c==5).sum(),'n at 8000:',(c==8000).sum(),'top values',np.sort(c)[-8:])
F=[s for s in SENSORS if s not in ('s047','s120','s122','s146','s063')]
Lg=np.log1p(f[F].clip(lower=0))
ry=rankdata(yc)
def sp(v):
    m=~np.isnan(v); 
    return abs(np.corrcoef(rankdata(v[m]),rankdata(yc[m]))[0,1]) if m.sum()>300 else 0
single=pd.Series({s:sp(Lg[s].values) for s in F}).sort_values(ascending=False)
print(single.head(10).round(3).to_dict())
cand=list(single.index[:70])+[s for s in F if s not in single.index[:70]][:0]
res={}
A=Lg[cand].values
for i,j in itertools.combinations(range(len(cand)),2):
    res[(cand[i],cand[j],'-')]=sp(A[:,i]-A[:,j]); res[(cand[i],cand[j],'+')]=sp(A[:,i]+A[:,j])
S=pd.Series(res).sort_values(ascending=False); print(S.head(20).round(3))
