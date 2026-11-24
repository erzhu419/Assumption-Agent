import pickle, time
from fractions import Fraction
from math import fsum
import numpy as np
from scipy.optimize import linprog
from facility_cert import solve_and_certify_facilities

p = pickle.load(open('/holdout_task/public_facility.pkl','rb'))
def make(seed, mode):
    rng = np.random.default_rng(seed)
    d = rng.uniform(0.5, 3, 24)
    f = rng.uniform(3, 40, 12)
    s = rng.uniform(3, 15, 12)
    c = rng.uniform(0, 15, (12, 24))
    if mode == 'symmetric':
        f[:] = 15; d[:] = 1; s[:] = 5; c[:] = 2
    if mode == 'clusters':
        c = rng.uniform(0, 0.1, (12, 24)) + 30 * (np.arange(12)[:, None] % 4 != np.arange(24)[None, :] % 4)
        f[:] = 15; d[:] = 1; s[:] = 5
    if mode == 'tight':
        s *= (sum(d) * 1.004 / sum(s))
    if mode == 'loose':
        s *= 5
    if mode == 'fixed_dominant':
        f *= 80; c *= 0.01
    if mode == 'transport_dominant':
        f *= 0.001; c *= 10
    return dict(fixed_cost=f, capacity=s, demand=d, cost_matrix=c)

def check(p, r):
    y = np.array(r['open_y']); x = r['flows_X']; U = r['upper_bound']; ls = r['proof_leaves']
    f,s,d,c = [p[k] for k in ('fixed_cost','capacity','demand','cost_matrix')]
    m,n=c.shape; masks = np.arange(1<<m)
    assert y.shape == (m,) and set(y) <= {0,1} and all(type(v) is int for v in r['open_y'])
    assert x.shape == (m,n) and np.isfinite(x).all() and x.min() >= 0
    residual = max(abs(x.sum(axis=0)-d).max(), (x.sum(axis=1)-s*y).max())
    assert residual < 1e-8 and (x[y==0]==0).all()
    assert abs(U-fsum([*(f*y),*(x*c).ravel()])) < 1e-8 and 1 <= len(ls) <= 80
    coverage = np.zeros(len(masks), int)
    total=sum(map(Fraction,d)); cap=list(map(Fraction,s))
    for a in ls:
        o=a['fixed_open']; z=a['fixed_closed']; b=a['lower_bound']
        assert len(set(o+z)) == len(o+z) and all(0 <= i < m for i in o+z)
        mask_o=sum(1<<i for i in o); mask_z=sum(1<<i for i in z)
        coverage[(masks&mask_o==mask_o)&(masks&mask_z==0)] += 1
        assert b >= U-1e-5
        if np.isinf(b):
            assert sum(v for i,v in enumerate(cap) if i not in z)<total
    assert (coverage==1).all()
    return len(ls),U-min(a['lower_bound'] for a in ls),residual

if __name__ == '__main__':
    worst={}
    for mode, seeds in [('public',range(1)), ('symmetric',range(1)), ('clusters',range(20)), ('tight',range(12)), ('loose',range(20)), ('fixed_dominant',range(15)), ('transport_dominant',range(15)), ('random',range(70))]:
        metrics=[]
        for seed in seeds:
            instance=p if mode=='public' else make(seed,mode)
            t=time.monotonic()
            try:
                r=solve_and_certify_facilities(instance)
                a=check(instance,r)
            except Exception:
                print('FAILED',mode,seed,flush=True)
                raise
            metrics.append((*a,time.monotonic()-t,seed))
        worst[mode]=max(metrics)
        print(mode,len(metrics),'largest tree (leaves,gap,residual,seconds,seed)',max(metrics),'max gap',max(a[1] for a in metrics),'max residual',max(a[2] for a in metrics),'max seconds',max(a[3] for a in metrics),flush=True)
