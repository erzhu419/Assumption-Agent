import importlib.util
import pickle
import time
import numpy as np
from scipy.optimize import lsq_linear

path = '/holdout_task/certified_box_l0.py'
spec = importlib.util.spec_from_file_location('certified_box_l0', path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

def check(name, d, ref=None):
    t = time.perf_counter()
    result = mod.solve_certified_box_l0(d)
    elapsed = time.perf_counter() - t
    X, y = d['X'], d['y']
    w = np.asarray(result['weights'])
    r = X @ w - y
    objective = .5 * (r @ r + d['ridge'] * (w @ w)) + d['l0_penalty'] * np.count_nonzero(abs(w) > 1e-10)
    lb = result['certified_lower_bound']
    leaves = result['certificate_leaves']
    assert w.shape == (X.shape[1],) and np.all(abs(w) <= d['box'])
    assert abs(objective-result['objective']) <= 1e-7
    assert 0 <= result['objective']-lb <= d['gap_tol']
    assert 1 <= len(leaves) <= d['max_leaves']
    assert lb <= min(leaf['leaf_lower_bound'] for leaf in leaves) + 1e-8
    support_ids = np.arange(1 << X.shape[1]); covered = np.zeros_like(support_ids)
    for leaf in leaves:
        z, a = leaf['fixed_zero'], leaf['fixed_nonzero']
        assert set(z).isdisjoint(a) and all(0 <= j < X.shape[1] for j in z+a)
        zm, am = (sum(1 << j for j in set(s)) for s in (z,a))
        covered += ((support_ids & zm) == 0) & ((support_ids & am) == am)
    assert np.all(covered == 1)
    if ref is not None:
        assert lb <= ref + 1e-9 and result['objective'] <= ref + d['gap_tol'], (name, lb, ref, objective)
    print(f'{name:24} time={elapsed:.3f}s gap={objective-lb:.3g} leaves={len(leaves)} support={np.count_nonzero(abs(w)>1e-10)}', flush=True)
    return elapsed

def data(X, y, ridge, lam, M):
    return dict(X=X, y=y, ridge=ridge, l0_penalty=lam, box=M, gap_tol=1e-6, max_leaves=512)

for i,d in enumerate(pickle.load(open('/holdout_task/public_box_l0_cases.pkl','rb'))['cases']):
    check(f'public-{i}',d)
    check(f'public-{i}-wide-box', dict(d, box=10*d['box']))

y = np.linspace(.3,1.8,8)
M, lam = .51,.13
X = np.repeat(np.eye(8),2,axis=1)
ref = sum(min(.5 * max(0., yi-k*M)**2+k*lam for k in range(3)) for yi in y)
check('duplicate-blocks-p16',data(X,y,0.,lam,M),ref)

def corr(seed,p,n,decoy):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n,p)) + 1.5*rng.normal(size=(n,1))
    if decoy:
        X[:, -3:] = X[:, :3] + decoy*rng.normal(size=(n,3))
    y = X @ rng.uniform(-1.4,1.4,p) + .1*rng.normal(size=n)
    return X,y

total = 0.
for seed,p,n,decoy,ridge,M in [(41,14,41,1e-2,.03,.21),(52,15,53,1e-6,0.,.45),(63,15,9,1e-7,0.,.16)]:
    X,y = corr(seed,p,n,decoy)
    total += check(f'correlated-{seed}-p{p}',data(X,y,ridge,.17,M))
print(f'private-size stress total={total:.3f}s',flush=True)
assert total < 20

X,y = corr(84,8,18,1e-4)
d = data(X,y,.03,.2,.37)
ref = .5 * (y @ y)
for mask in range(1,1<<8):
    a = [j for j in range(8) if mask & (1 << j)]
    A = np.vstack((X[:,a],np.sqrt(d['ridge'])*np.eye(len(a))))
    b = np.r_[y,np.zeros(len(a))]
    w = lsq_linear(A,b,bounds=(-d['box'],d['box']),method='bvls',tol=1e-14).x
    r = A@w-b
    ref = min(ref,.5*(r@r)+d['l0_penalty']*len(a))
check('independent-BVLS-p8',d,ref)

X = np.array([[1e4]])
y = np.array([.1])
check('free-threshold',data(X,y,0.,.02,1e-5),.5*(.1-1e-6)**2)
