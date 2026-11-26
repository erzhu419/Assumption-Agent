import sys,time
sys.path.insert(0,'/tmp')
from test_eq02 import eq,check,design
import numpy as np
for seed,p in [(36,19),(29,19),(33,18),(31,19),(34,18),(35,19)]:
    rng=np.random.default_rng(seed)
    a=design(rng,120,(p+1)//2,.10)
    x=np.repeat(a,2,axis=1)[:,:p]+.005*rng.normal(size=(120,p))
    y=a@(rng.choice([-1.,1.],a.shape[1])*rng.uniform(.47,.59,a.shape[1]))+.04*rng.normal(size=120)
    t=time.perf_counter()
    try:
        out=eq._Selector(x,y,.035,.35,.52).solve()
        dt=time.perf_counter()-t
        print(seed,p,round(dt,3),check(x,y,.035,.35,.52,out),flush=True)
    except RuntimeError as e:
        print(seed,p,round(time.perf_counter()-t,3),'FAIL',e,flush=True)
