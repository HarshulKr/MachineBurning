from sklearn.linear_model import LinearRegression
exec(open('exp_pin2.py').read().split("print('--- sweep 2')")[0])
P8={**base,'n_estimators':400,'min_child_samples':8}
class Bag:
    def __init__(s,seed,n): s.ms=[lgb.LGBMRegressor(random_state=seed+31*k,**P8) for k in range(n)]
    def fit(s,X,y): [m.fit(X,y) for m in s.ms]; return s
    def predict(s,X): return np.mean([m.predict(X) for m in s.ms],0)
def bagmodel(n):
    def run(Xa,ya,Xb,seed):
        ip=np.zeros(len(ya))
        for a,b in KFold(5,shuffle=True,random_state=seed).split(Xa): ip[b]=Bag(seed,n).fit(Xa.iloc[a],ya[a]).predict(Xa.iloc[b])
        _l=LinearRegression().fit(ip[:,None],ya); a1,a0=float(_l.coef_[0]),float(_l.intercept_); mu=a0+a1*Bag(seed,n).fit(Xa,ya).predict(Xb); res=ya-(a0+a1*ip)
        return np.exp(mu+np.quantile(res,Q_STAR))
    return run
for n in [1,5]:
    L=[]
    for s in range(3):
        b=np.zeros(len(c))
        for a,bb in KFold(5,shuffle=True,random_state=100+s).split(X): b[bb]=bagmodel(n)(X.iloc[a],yc[a],X.iloc[bb],SEED+s)
        L.append(pin(b,c))
    print(n,'seeds',round(np.mean(L),2),round(np.std(L),2),flush=True)
