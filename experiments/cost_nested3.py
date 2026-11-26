exec(open('cost_nested.py').read().split("for name,mk in [")[0])
from sklearn.svm import SVR
F2=[x for x in F if x!='s063']; HB=[h for h in HIST if h!='s063']
def hist_lf(df):
    H=df[HB].values; tot=np.nansum(H,1); return pd.DataFrame(np.log((H+1)/(tot[:,None]+10)),columns=[h+'_lf' for h in HB],index=df.index).fillna(0)
class Blend:
    def __init__(s,seed,w=(0.5,0.2,0.3),ks=30,C=3,k=12,kr=40): s.seed=seed; s.w=w; s.ks=ks; s.C=C; s.k=k; s.kr=kr
    def fit(s,df,yy):
        s.g=CostFE(s.seed,True,True,0.0,s.k).fit(df,yy)
        Lg=L(df[F2]); s.med=Lg.median(); Lg=Lg.fillna(s.med)
        sp=pd.Series({cc:abs(spearmanr(Lg[cc],yy)[0]) for cc in F2}).fillna(0).sort_values(ascending=False)
        s.rc=list(sp.index[:s.kr]); s.sc=list(sp.index[:s.ks])
        s.r=make_pipeline(StandardScaler(),RidgeCV(alphas=np.logspace(-2,4,30))).fit(Lg[s.rc],yy)
        s.v=make_pipeline(StandardScaler(),SVR(C=s.C,epsilon=0.1,gamma='scale')).fit(pd.concat([Lg[s.sc],hist_lf(df)],axis=1),yy)
        return s
    def predict(s,df):
        Lg=L(df[F2]).fillna(s.med)
        parts=[s.g.predict(df), s.r.predict(Lg[s.rc]), s.v.predict(pd.concat([Lg[s.sc],hist_lf(df)],axis=1))]
        return sum(w*p for w,p in zip(s.w,parts))
for name,mk in [('LGB.6+ridge.4 (prev best)',lambda s: Blend(s,(0.6,0.4,0.0))),('SVR only',lambda s: Blend(s,(0,0,1))),
                ('LGB.5+SVR.5',lambda s: Blend(s,(0.5,0,0.5))),('LGB.4+ridge.2+SVR.4',lambda s: Blend(s,(0.4,0.2,0.4))),('LGB.34+ridge.33+SVR.33',lambda s: Blend(s,(0.34,0.33,0.33)))]:
    t=time.time(); print(name,evaluate(mk),'%.0fs'%(time.time()-t),flush=True)
