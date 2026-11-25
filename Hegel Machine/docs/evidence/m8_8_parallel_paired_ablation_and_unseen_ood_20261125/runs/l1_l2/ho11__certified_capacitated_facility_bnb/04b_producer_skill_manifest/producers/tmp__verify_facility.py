import math, pickle, sys, time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import eye, kron
sys.path.insert(0, '/holdout_task')
from facility_cert import solve_and_certify_facilities

BITS = (np.arange(4096)[:, None] >> np.arange(12)) & 1

def verify(name, case, reference=None):
    t = time.monotonic()
    f,u,d,c = (np.asarray(case[k]) for k in ('fixed_cost','capacity','demand','cost_matrix'))
    ans = solve_and_certify_facilities(case)
    y = np.asarray(ans['open_y']); x = ans['flows_X']; ub = ans['upper_bound']; leaves = ans['proof_leaves']
    assert y.shape == (12,) and np.isin(y,[0,1]).all()
    assert x.shape == (12,24) and np.isfinite(x).all() and x.min() >= -1e-9
    err = max(abs(x.sum(axis=0)-d).max(), (x.sum(axis=1)-u*y).max())
    assert err < 1e-8 and np.all(x[y==0]==0) and abs(f@y+np.sum(c*x)-ub) < 1e-9
    cover = np.zeros(len(BITS),int); leaf_min = np.empty(len(BITS)); assert 0<len(leaves)<=80
    for l in leaves:
        a,b,lb = (l[k] for k in ('fixed_open','fixed_closed','lower_bound'))
        assert len(set(a+b))==len(a+b) and set(a+b)<=set(range(12)) and lb>=ub-1e-5
        covered = (BITS[:,a]==1).all(axis=1) & (BITS[:,b]==0).all(axis=1)
        cover[covered]+=1; leaf_min[covered]=lb
        if lb == math.inf:
            from fractions import Fraction
            assert sum(Fraction(float(u[i])) for i in range(12) if i not in b) < sum(Fraction(float(v)) for v in d)
    assert np.all(cover==1)
    if reference is None:
        Aeq=kron(np.ones((1,12)),eye(24),format='csr'); Aub=kron(eye(12),np.ones((1,24)),format='csr')
        optimum=math.inf
        for i in np.flatnonzero(BITS@u >= d.sum()-1e-10):
            yi = BITS[i]; r=linprog(c.ravel(),A_eq=Aeq,b_eq=d,A_ub=Aub,b_ub=u*yi,method='highs')
            assert r.success,(name,i,r.message)
            z = f@yi+r.fun
            assert leaf_min[i]<=z+2e-9,(name,i,leaf_min[i],z)
            optimum=min(optimum,z)
    else:
        optimum=reference
    delta=ub-optimum
    print(f'{name}: obj={ub:.14g} error={delta:.3g} gap={ub-leaf_min.min():.3g} residual={err:.3g} leaves={len(leaves)} sec={time.monotonic()-t:.2f}',flush=True)
    assert abs(delta)<2e-9, (name,ub,optimum)

with open('/holdout_task/public_facility.pkl','rb') as p:
    public=pickle.load(p)
verify('public exhaustive',public)
rng=np.random.default_rng(804)
for k, scale in enumerate((1.05,1.35,2.0,2.7)):
    d=rng.uniform(.5,3,24); u=rng.uniform(3,13,12); u*=scale*d.sum()/u.sum()
    f=rng.uniform(6,38,12); c=rng.uniform(.1,12,(12,24))
    verify('dense %.2f'%scale,dict(fixed_cost=f,capacity=u,demand=d,cost_matrix=c))
base = dict(capacity=np.full(12,8.),demand=np.ones(24),cost_matrix=np.full((12,24),2.))
verify('flat ties, exact capacity',dict(base,fixed_cost=np.full(12,10.)),78.)
f = 10. + rng.permutation(12)*1e-8
verify('near ties, exact capacity',dict(base,fixed_cost=f),48.+sum(sorted(f)[:3]))
verify('all open necessary',dict(base,capacity=np.full(12,2.),fixed_cost=f),48.+sum(f))
