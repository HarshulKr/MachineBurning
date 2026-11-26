exec(open('diag_grid.py').read().split("rows=[]")[0])
worlds=[(0.03,1.0),(0.035,0.9),(0.0477,0.8),(0.055,0.7)]
rows=[]
for pa in [0.025,0.03,0.035,0.04,0.0477]:
    f,b,_,_=decide(adjust_prior(pc0,pi0,pa),mu,r); sub=pd.DataFrame({'flag':f,'budget':b})
    rows.append([pa,f.sum()]+[round(ev(sub,adjust_prior(pc0,pi0,pw),k)) for pw,k in worlds])
d=pd.DataFrame(rows,columns=['decide pi','#flag']+[f'world {pw}/{k}' for pw,k in worlds])
best=d.iloc[:,2:].min(); d['max regret']=(d.iloc[:,2:]-best).max(axis=1)
print(d.to_string(index=False))
