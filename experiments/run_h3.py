import harness as H
from harness import *
from sklearn.model_selection import StratifiedKFold
# independent confirmation: new cost-model fold seeds (classifier OOF unchanged)
H.FOLDS_=[list(StratifiedKFold(5,shuffle=True,random_state=777+r).split(np.zeros(len(y)),y)) for r in range(3)]
t=time.time(); base=losses(decisions(cv_samples(lambda s: CostModel(s)))); print('base',time.time()-t,flush=True)
rows=[report(base,'V3b (notebook)')]
def run(name,**kw):
    L=losses(decisions(cv_samples(**kw))); rows.append(report(L,name,base)); np.save(f'h3_{len(rows)}.npy',L); print(rows[-1],flush=True)
run('C0: subsample fix', make=lambda s: CostModel(s,params=COST_SUB))
run('C1: subsample + hist descriptors', make=lambda s: CostModel(s,params=COST_SUB,hist_desc=True))
run('C2: C1 + PCA features', make=lambda s: CostModel(s,params=COST_SUB,hist_desc=True,extra=pca_feats))
run('C3: C2 + cost weights .5', make=lambda s: CostModel(s,params=COST_SUB,hist_desc=True,extra=pca_feats,sample_weight_gamma=0.5))
run('C4: C1 + cost weights .5', make=lambda s: CostModel(s,params=COST_SUB,hist_desc=True,sample_weight_gamma=0.5))
run('C5: C2 + cost weights .25', make=lambda s: CostModel(s,params=COST_SUB,hist_desc=True,extra=pca_feats,sample_weight_gamma=0.25))
print(pd.DataFrame(rows).to_string(index=False)); pd.DataFrame(rows).to_csv('h3_results.csv',index=False)
