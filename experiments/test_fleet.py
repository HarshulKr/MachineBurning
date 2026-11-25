import sys; sys.path.insert(0,'src'); import numpy as np
from fleet import *
# metric worked checks
assert np.allclose(fleet_cost([1],[100],[1],[300]), 10+0.2*200)
assert np.allclose(fleet_cost([1],[300],[1],[100]), 10+0.02*200)
assert np.allclose(fleet_cost([0],[999],[1],[300]), 300)
assert np.allclose(fleet_cost([1],[50],[0],[np.nan]), 10+0.02*50)
assert np.allclose(fleet_cost([0],[0],[0],[np.nan]), 0)
# decision vs brute force
rng=np.random.default_rng(0); r=rng.normal(0,1.5,400)
for p in [0.01,0.05,0.1,0.3,0.9]:
    mu=np.log(200); f,b,cf,cs=decide(np.array([p]),np.array([mu]),r)
    C=np.exp(mu+r); grid=np.linspace(0,3000,30001)
    ec=[10+p*np.mean(0.2*np.maximum(C-g,0)+0.02*np.maximum(g-C,0))+(1-p)*0.02*g for g in grid]
    g=grid[int(np.argmin(ec))]; print(p, 'b*',round(b[0] if f[0] else -1,2),'grid',g,'flag',f[0], round(cf[0],2), round(min(ec),2), round(cs[0],2))
    assert abs(cf[0]-min(ec))<0.05
print('ok')
