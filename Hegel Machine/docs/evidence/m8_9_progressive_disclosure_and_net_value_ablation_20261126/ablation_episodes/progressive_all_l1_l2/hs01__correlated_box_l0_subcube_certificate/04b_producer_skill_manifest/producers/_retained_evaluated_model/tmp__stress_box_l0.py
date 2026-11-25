import importlib.util
import itertools
import pickle
import time
import numpy as np
from scipy.optimize import lsq_linear

spec = importlib.util.spec_from_file_location('certified_box_l0', '/holdout_task/certified_box_l0.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

def objective(d, w):
    r = d['X'] @ w - d['y']
    return .5*(r @ r + d['ridge']*(w @ w)) + d['l0_penalty']*np.count_nonzero(abs(w)>1e-10)

def reference(d):
    X, y = d['X'], d['y']
    p = X.shape[1]
    A = np.vstack((X, np.sqrt(d['ridge'])*np.eye(p)))
    b = np.r_[y, np.zeros(p)]
    opt = np.inf
    for mask in itertools.product((False, True), repeat=p):
        lim = np.where(mask, d['box'], min(d['box'], 1e-10))
        w = lsq_linear(A, b, bounds=(-lim, lim), method='bvls', tol=1e-14).x
        r = A @ w - b
        opt = min(opt, .5*(r @ r) + d['l0_penalty']*sum(mask))
    return opt

def check(name, d, expected=None):
    t = time.perf_counter()
    ret = mod.solve_certified_box_l0(d)
    elapsed = time.perf_counter()-t
    w = np.asarray(ret['weights'])
    p = d['X'].shape[1]
    obj, lb, leaves = (ret[k] for k in ['objective', 'certified_lower_bound', 'certificate_leaves'])
    assert w.shape == (p,) and np.isfinite(w).all() and np.all(abs(w)<=d['box']), name
    assert np.isfinite([obj, lb]).all() and abs(obj-objective(d,w))<=1e-7, name
    assert 0<=obj-lb<=d['gap_tol'] and 1<=len(leaves)<=d['max_leaves'], name
    assert lb<=min(l['leaf_lower_bound'] for l in leaves)+1e-8, name
    covered = np.zeros(1<<p, dtype=int)
    ids = np.arange(1<<p)
    for l in leaves:
        z, a = l['fixed_zero'], l['fixed_nonzero']
        assert len(set(z+a))==len(z+a) and all(0<=i<p for i in z+a), name
        zm, am = (sum(1<<i for i in lst) for lst in (z,a))
        covered[(ids&zm==0)&(ids&am==am)] += 1
        if expected is not None and not z and not a:
            assert l['leaf_lower_bound']<=expected+1e-9, (name,l,expected)
    assert np.all(covered==1), name
    if expected is not None:
        assert abs(obj-expected)<=d['gap_tol']+1e-9, (name,obj,expected)
    print(f'{name}: gap={obj-lb:.3g} time={elapsed:.3f}s leaves={len(leaves)}',flush=True)
    return elapsed

def prob(X,y,ridge,penalty,box):
    return dict(X=X,y=y,ridge=ridge,l0_penalty=penalty,box=box,gap_tol=1e-6,max_leaves=1)

rng = np.random.default_rng(67)
for j,d in enumerate(pickle.load(open('/holdout_task/public_box_l0_cases.pkl','rb'))['cases']):
    check(f'public-{j}',d)
for j in range(12):
    n,p=11,7
    A=rng.normal(size=(n,3)); X=A@rng.normal(size=(3,p)) + (j%3)*.025*rng.normal(size=(n,p))
    y=rng.normal(size=n)+X@rng.normal(size=p)
    d=prob(X,y,[0.,0.,.04][j%3], [.01,.13,.5][j%3],[.08,.4,1.2][(j//3)%3])
    check(f'exhaustive-{j}',d,reference(d))
check('threshold-scale',prob(np.array([[1e8,1e8]]),np.array([.03]),0.,.1,2.),.5*.01**2)
check('box-at-threshold',prob(np.eye(2),np.ones(2),0.,.1,1e-10),(1-1e-10)**2)
check('zero-gram',prob(np.zeros((7,16)),np.ones(7),0.,.1,1.),3.5)
total=0.
for j,p in enumerate((14,15,15)):
    n=48; A=rng.normal(size=(n,4)); X=A@rng.normal(size=(4,p))
    X += (0.,.007,.14)[j]*rng.normal(size=X.shape)
    y=X@rng.normal(size=p)*1.5+.2*rng.normal(size=n)
    d=prob(X,y,[0.,1e-7,.03][j],.04,[.11,.24,.65][j])
    total+=check(f'full-size-correlated-{j}',d)
print(f'full-size trio: {total:.3f}s',flush=True)
assert total<20
