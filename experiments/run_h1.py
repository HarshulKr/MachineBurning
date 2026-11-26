from harness import *
import pickle
t=time.time()
base=losses(decisions(cv_samples(lambda s: CostModel(s)))); print('base %.0fs'%(time.time()-t),flush=True)
np.save('h_base.npy',base)
rows=[report(base,'V3b as in notebook (no row subsampling)')]
def run(name,**kw):
    t=time.time(); L=losses(decisions(cv_samples(**kw))); rows.append(report(L,name,base)); np.save(f'h_{name[:20].replace(" ","_")}.npy',L)
    print(rows[-1],'%.0fs'%(time.time()-t),flush=True)
run('subsample=0.8 fix', make=lambda s: CostModel(s,params=COST_SUB))
run('cross-fitted recalibration residuals', make=lambda s: CostModel(s), crossfit_recal=True)
run('residual spread x1.1', make=lambda s: CostModel(s), spread=1.1)
run('residual spread x0.9', make=lambda s: CostModel(s), spread=0.9)
run('histogram shape descriptors', make=lambda s: CostModel(s,hist_desc=True))
run('cost weights gamma .25', make=lambda s: CostModel(s,sample_weight_gamma=0.25))
run('cost weights gamma .5', make=lambda s: CostModel(s,sample_weight_gamma=0.5))
const_floor=lambda seed,D,isf,Dv: np.full(len(Dv), isf.mean())
run('floor mixture, constant P(c=5)', make=lambda s: CostModel(s), floor=const_floor)
print(pd.DataFrame(rows).to_string(index=False))
pd.DataFrame(rows).to_csv('h1_results.csv',index=False)
