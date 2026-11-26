exec(open('cost_fe.py').read().split("print('base'")[0].split("#")[0] if False else open('cost_fe.py').read().split("m=lgb.LGBMRegressor(random_state=0,**P8).fit(X2")[0])
from scipy.stats import spearmanr
L=lambda s: np.log1p(s.clip(lower=0))
for a in ['s056','s058','s082','s088','s084']:
    for b in ['s042','s043','s037','hist_tot']:
        v=(L(f[a])-(Hf['hist_tot'] if b=='hist_tot' else L(f[b])))
        print(a,b,'spearman %.3f'%spearmanr(v,yc,nan_policy='omit')[0], end=' | ')
    print('alone %.3f'%spearmanr(L(f[a]),yc,nan_policy='omit')[0])
print('s042 vs s043 corr', spearmanr(f.s042,f.s043,nan_policy='omit')[0], 's042 vs hist_tot', spearmanr(L(f.s042),Hf.hist_tot,nan_policy='omit')[0])
# combos
Rn={f'n_{a}_{b}':L(f[a])-L(f[b]) for a in [t for t in top if t not in ('s042','s043')] for b in ['s042','s043']}
XA=pd.concat([X0,Hf,pd.DataFrame(Rn)],axis=1); print('X0+histfrac+norm by s042/s043',evalX(XA))
XB=pd.concat([X0,pd.DataFrame(R),Hf],axis=1); print('X0+ratios+histfrac',evalX(XB))
XC=pd.concat([X0,pd.DataFrame(Rn)],axis=1); print('X0+norm by s042/s043',evalX(XC))
