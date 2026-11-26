from test_eq02 import eq, check, design
import time
import numpy as np

for seed,p,kind in [(17,19,'near-threshold'),(18,18,'near-threshold'),(29,19,'paired'),(33,18,'paired'),(51,19,'cancellation')]:
    rng=np.random.default_rng(seed); n=120; lam0=.35; lam2=.035; m=.52
    if kind=='near-threshold':
        x=design(rng,n,p,.30)
        g=x.T@x
        z=rng.choice([-1.,1.],p)*np.sqrt(2*lam0*g.diagonal())*rng.uniform(.88,1.12,p)
        y=x@np.linalg.solve(g,z)+.07*rng.normal(size=n)
    elif kind=='paired':
        a=design(rng,n,(p+1)//2,.10)
        x=np.repeat(a,2,axis=1)[:,:p]+.005*rng.normal(size=(n,p))
        y=a@(rng.choice([-1.,1.],a.shape[1])*rng.uniform(.47,.59,a.shape[1]))+.04*rng.normal(size=n)
    else:
        x=design(rng,n,p,.997)
        w=np.zeros(p); w[:8]=rng.choice([-1.,1.],8)*rng.uniform(1.1,1.4,8)
        y=x@w+.055*rng.normal(size=n)
    t=time.perf_counter()
    try:
        ans=eq._Selector(x,y,lam2,lam0,m).solve()
        elapsed=time.perf_counter()-t
        print(seed,p,kind,'seconds',round(elapsed,4),'certificate',check(x,y,lam2,lam0,m,ans),flush=True)
    except RuntimeError as e:
        print(seed,p,kind,'seconds',round(time.perf_counter()-t,4),'FAIL',str(e),flush=True)
