import importlib.util
import pickle
from time import perf_counter

import numpy as np
from scipy.optimize import lsq_linear

spec = importlib.util.spec_from_file_location('solver', '/holdout_task/certified_box_l0.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def audit(name, problem, reference=None):
    X, y = problem['X'], problem['y']
    M, r, cost = (problem[k] for k in ('box', 'ridge', 'l0_penalty'))
    t = perf_counter()
    out = mod.solve_certified_box_l0(problem)
    elapsed = perf_counter() - t
    w = np.asarray(out['weights'])
    assert w.shape == (X.shape[1],) and np.all(np.abs(w) <= M)
    residual = X @ w - y
    upper = .5 * (residual @ residual + r * (w @ w)) + cost * np.count_nonzero(abs(w) > 1e-10)
    lower = out['certified_lower_bound']
    assert abs(upper - out['objective']) <= 1e-7
    assert 0 <= out['objective'] - lower <= problem['gap_tol']
    leaves = out['certificate_leaves']
    assert 1 <= len(leaves) <= problem['max_leaves']
    assert lower <= min(leaf['leaf_lower_bound'] for leaf in leaves) + 1e-8
    ids = np.arange(1 << w.size)
    cover = np.zeros(ids.size, dtype=int)
    for leaf in leaves:
        z, a = leaf['fixed_zero'], leaf['fixed_nonzero']
        assert len(set(z + a)) == len(z + a) and all(0 <= i < w.size for i in z + a)
        zm, am = sum(1 << i for i in z), sum(1 << i for i in a)
        cover[((ids & zm) == 0) & ((ids & am) == am)] += 1
        if reference is not None and not (z or a):
            assert leaf['leaf_lower_bound'] <= reference + 1e-10
    assert np.all(cover == 1)
    if reference is not None:
        assert lower <= reference + 1e-10 and upper <= reference + problem['gap_tol']
    print(f'{name}: objective={upper:.12g}, gap={upper-lower:.3g}, time={elapsed:.3f}s', flush=True)
    return elapsed


def public_incumbent(problem):
    # Independently enumerate feasible zero-restricted supports with BVLS.
    X, y = problem['X'], problem['y']
    p = X.shape[1]
    r, M, cost = (problem[k] for k in ('ridge', 'box', 'l0_penalty'))
    A = np.vstack((X, np.sqrt(r)*np.eye(p)))
    b = np.r_[y, np.zeros(p)]
    best = .5 * (y @ y)
    bits = 1 << np.arange(p)
    for k in range(1, 1 << p):
        active = (k & bits) != 0
        if active.sum() * cost >= best:
            continue
        As = A[:, active]
        v = lsq_linear(As, b, bounds=(-M, M), method='bvls', tol=1e-13).x
        residual = As @ v - b
        best = min(best, .5*(residual @ residual) + cost*np.count_nonzero(abs(v) > 1e-10))
    return best


with open('/holdout_task/public_box_l0_cases.pkl', 'rb') as f:
    cases = pickle.load(f)['cases']
secs = 0.
for i, case in enumerate(cases):
    # Reorder columns and flip alternating signs; the optimum is invariant.
    X = case['X'][:, ::-1] * np.where(np.arange(case['X'].shape[1])%2, -1., 1.)
    secs += audit(f'public-{i}-signed-reversal', dict(case, X=X), public_incumbent(case))

# Signed exact duplicates, zero ridge, and enough signal to bind many boxes.
# For k paid coefficients the achievable interval has radius k*M+(p-k)*1e-10.
p = 15
base = np.asarray(cases[0]['X'][:, 0]).copy()
base /= np.linalg.norm(base)
X = base[:, None] * np.where(np.arange(p)%2, -1., 1.)
target, M, cost = 8.4, .71, .18
ks = np.arange(p+1)
reference = np.min(.5*np.maximum(target - ks*M - (p-ks)*1e-10, 0.)**2 + ks*cost)
secs += audit('public-column-signed-duplicates', dict(cases[0], X=X, y=target*base, ridge=0., box=M, l0_penalty=cost), reference)
print(f'three audit solves: {secs:.3f}s total (excluding reference solves)')
assert secs < 20
