import harness as H
from harness import *
from sklearn.model_selection import StratifiedKFold
pos,neg=M&(y==1),M&(y==0)
rows=[]
for fold_name,seed0 in [('original folds',None),('fresh folds',777)]:
    if seed0: H.FOLDS_=[list(StratifiedKFold(5,shuffle=True,random_state=seed0+r).split(np.zeros(len(y)),y)) for r in range(3)]
    else: H.FOLDS_=FS
    decs={n:decisions(cv_samples(mk)) for n,mk in [('V3b',lambda s: CostModel(s)),('V4',lambda s: CostModel(s,params=COST_SUB,hist_desc=True,sample_weight_gamma=0.5))]}
    for k in [0.8,1.0,1.2]:
        Ls={n:np.mean([fleet_cost(f,b,y,cost*k) for f,b in d],0) for n,d in decs.items()}
        for pi in [0.03,0.04,0.05]:
            s={n:20000*(pi*L[pos].mean()+(1-pi)*L[neg].mean()) for n,L in Ls.items()}
            rows.append((fold_name,pi,k,round(s['V3b']),round(s['V4']),round(s['V4']-s['V3b']),f"{100*(s['V4']-s['V3b'])/s['V3b']:+.1f}%"))
    print(fold_name,'done',flush=True)
d=pd.DataFrame(rows,columns=['folds','true prior','cost x','V3b','V4','diff','rel']); print(d.to_string(index=False)); d.to_csv('stress.csv',index=False)
