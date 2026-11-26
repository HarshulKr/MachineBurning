from harness import *
base=np.load('h_base.npy'); rows=[]
def run(name,**kw):
    t=time.time(); L=losses(decisions(cv_samples(**kw))); rows.append(report(L,name,base)); np.save(f'h2_{len(rows)}.npy',L)
    print(rows[-1],'%.0fs'%(time.time()-t),flush=True)
run('A: subsample + hist descriptors + cost weights .5', make=lambda s: CostModel(s,params=COST_SUB,hist_desc=True,sample_weight_gamma=0.5))
run('B: A + spread x1.1', make=lambda s: CostModel(s,params=COST_SUB,hist_desc=True,sample_weight_gamma=0.5), spread=1.1)
run('subsample + cost weights .75', make=lambda s: CostModel(s,params=COST_SUB,sample_weight_gamma=0.75))
run('subsample + cost weights 1.0', make=lambda s: CostModel(s,params=COST_SUB,sample_weight_gamma=1.0))
run('subsample + PCA(15, train+test) features', make=lambda s: CostModel(s,params=COST_SUB,extra=pca_feats))
run('subsample + Gamma GLM instead of ridge', make=lambda s: GammaBlend(s,params=COST_SUB))
run('ridge then LightGBM residual boosting', make=lambda s: RidgeThenBoost(s))
run('subsample + logistic floor model', make=lambda s: CostModel(s,params=COST_SUB), floor=logistic_floor)
print(pd.DataFrame(rows).to_string(index=False)); pd.DataFrame(rows).to_csv('h2_results.csv',index=False)
