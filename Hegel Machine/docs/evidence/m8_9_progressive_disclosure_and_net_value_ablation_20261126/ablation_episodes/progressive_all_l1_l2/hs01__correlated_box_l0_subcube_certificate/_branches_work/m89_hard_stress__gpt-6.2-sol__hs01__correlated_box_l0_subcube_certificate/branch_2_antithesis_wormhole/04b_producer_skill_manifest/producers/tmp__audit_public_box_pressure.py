import importlib.util
import pickle
import time
import numpy as np
from scipy.optimize import lsq_linear

spec = importlib.util.spec_from_file_location('box_l0', '/holdout_task/certified_box_l0.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('/holdout_task/public_box_l0_cases.pkl', 'rb') as f:
    cases = pickle.load(f)['cases']

def run(name, case):
    start = time.perf_counter()
    out = m.solve_certified_box_l0(case)
    elapsed = time.perf_counter() - start
    w = np.asarray(out['weights'])
    X, y, M = case['X'], case['y'], case['box']
    r = X @ w - y
    nz = np.abs(w) > 1e-10
    obj = .5*(r@r + case['ridge']*(w@w)) + case['l0_penalty']*nz.sum()
    lb = out['certified_lower_bound']
    leaves = out['certificate_leaves']
    assert w.shape == (X.shape[1],) and np.all(np.abs(w) <= M), name
    assert abs(out['objective']-obj) <= 1e-7 and 0 <= out['objective']-lb <= case['gap_tol'], name
    assert len(leaves) == 1 and leaves[0]['fixed_zero'] == leaves[0]['fixed_nonzero'] == [], name
    assert lb <= leaves[0]['leaf_lower_bound'] + 1e-8, name
    # An independent solve on the returned threshold pattern must not beat the certificate.
    p = X.shape[1]
    A = np.vstack((X, np.sqrt(case['ridge'])*np.eye(p)))
    b = np.r_[y, np.zeros(p)]
    limits = np.where(nz, M, min(M, 1e-10))
    u = lsq_linear(A, b, bounds=(-limits, limits), method='bvls', tol=1e-13).x
    dr = A@u-b
    assert lb <= .5*(dr@dr) + case['l0_penalty']*nz.sum() + 1e-9, name
    hits = int(np.count_nonzero(np.abs(w) >= M*(1-1e-9)))
    assert hits > 0, (name, 'stress did not bind the box')
    print(f'{name}: objective={obj:.10g} gap={out["objective"]-lb:.3g} active={nz.sum()} at_box={hits} time={elapsed:.3f}s', flush=True)
    return elapsed

a, b = cases
X = b['X']
trials = [
    ('public-0 tightened box', dict(a, box=.18)),
    ('public-1 tightened box lower activation cost', dict(b, box=.12, l0_penalty=.04)),
    ('public-1 tightened box zero ridge and contrast decoy', dict(b, X=np.c_[X, X[:,0]-X[:,1]], box=.24, ridge=0., l0_penalty=.04)),
]
total = sum(run(name, d) for name, d in trials)
print(f'Public-derived stress trio total: {total:.3f}s')
assert total < 20
