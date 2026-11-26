exec(open('cost_lasso.py').read().split("las=lambda")[0])
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.ensemble import BaggingRegressor
import time
sp=pd.Series({c:abs(spearmanr(Lg[c],yc)[0]) for c in Lg}).fillna(0).sort_values(ascending=False)
X=pd.concat([Lg,Hf],axis=1)
for k in [30,60]:
    cols=list(sp.index[:k])+list(Hf.columns); Xk=X[cols]
    t=time.time()
    print(k,'SVR C3', r2(Xk,make_pipeline(StandardScaler(),SVR(C=3,epsilon=0.1,gamma='scale'))), 'SVR C10',r2(Xk,make_pipeline(StandardScaler(),SVR(C=10,epsilon=0.1,gamma='scale'))),'%.0fs'%(time.time()-t),flush=True)
    for h,a in [((32,),1.0),((64,32),3.0),((16,),3.0)]:
        m=make_pipeline(StandardScaler(),BaggingRegressor(MLPRegressor(hidden_layer_sizes=h,alpha=a,max_iter=3000,early_stopping=False,random_state=0),n_estimators=10,random_state=0,n_jobs=2))
        t=time.time(); print(k,'MLP',h,a,r2(Xk,m),'%.0fs'%(time.time()-t),flush=True)
