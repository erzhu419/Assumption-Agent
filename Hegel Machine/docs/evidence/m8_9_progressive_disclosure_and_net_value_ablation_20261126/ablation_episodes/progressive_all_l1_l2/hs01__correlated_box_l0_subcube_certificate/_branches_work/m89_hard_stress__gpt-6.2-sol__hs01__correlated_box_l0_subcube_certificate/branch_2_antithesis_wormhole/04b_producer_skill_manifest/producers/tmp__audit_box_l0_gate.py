import importlib.util
import pickle
import time
import numpy as np

spec = importlib.util.spec_from_file_location('box_l0', '/holdout_task/certified_box_l0.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
cases = pickle.load(open('/holdout_task/public_box_l0_cases.pkl', 'rb'))['cases']

def verify(name, d):
    start = time.perf_counter()
    r = m.solve_certified_box_l0(d)
    elapsed = time.perf_counter() - start
    w = np.asarray(r['weights'])
    e = d['X'] @ w - d['y']
    obj = .5*(e @ e + d['ridge']*(w @ w)) + d['l0_penalty']*np.count_nonzero(abs(w) > 1e-10)
    lb = r['certified_lower_bound']
    leaves = r['certificate_leaves']
    assert w.shape == (d['X'].shape[1],) and np.all(np.abs(w) <= d['box']), name
    assert abs(obj-r['objective']) <= 1e-7 and 0 <= r['objective']-lb <= d['gap_tol'], name
    assert len(leaves) == 1 and leaves[0]['fixed_zero'] == leaves[0]['fixed_nonzero'] == [], name
    assert lb <= leaves[0]['leaf_lower_bound'] + 1e-8, name
    print(f'{name}: obj={obj:.12g}, gap={r["objective"]-lb:.3g}, runtime={elapsed:.3f}s', flush=True)
    return obj, lb, elapsed

def same_opt(name, first, second):
    assert max(first[1], second[1]) <= min(first[0], second[0]) + 1e-10, (name, first, second)

for i, d in enumerate(cases):
    base = verify(f'public-{i}', d)
    p = d['X'].shape[1]
    perm = np.roll(np.arange(p)[::-1], i+1)
    signs = 1-2*(np.arange(p)%2)
    flipped = verify(f'public-signed-permutation-{i}', dict(d, X=d['X'][:, perm]*signs))
    same_opt(f'public-symmetry-{i}', base, flipped)

d = dict(cases[0], ridge=0., max_leaves=1)
X = d['X']
dup = np.c_[X, X[:, 0], -X[:, 1]]
a = verify('public-zero-ridge-with-signed-duplicates', dict(d, X=dup))
b = verify('public-zero-ridge-duplicates-first', dict(d, X=dup[:, np.roll(np.arange(dup.shape[1]), 2)]))
same_opt('singular-duplicate-order', a, b)

# Seven independent sign-contrast pairs. The Gram blocks are strongly
# correlated, but the target is orthogonal to their shared direction.
from scipy.linalg import block_diag
for delta in (0.02, 0.002):
    p = 14
    box = .6
    block = np.array([[1., 1.], [delta, -delta]])
    X = block_diag(*([block] * (p//2)))
    y = np.tile([0., 3*delta], p//2)
    penalty = delta**2
    # The 2-active minimizer is (+box,-box); a 1-active fit has a
    # closed-form scalar minimizer. Compare both with the empty support.
    w1 = min(box, 3*delta**2/(1+delta**2))
    expected = (p//2)*min(.5*(3*delta)**2,
        .5*(w1**2+(delta*w1-3*delta)**2)+penalty,
        .5*((2*box-3)*delta)**2+2*penalty)
    c = verify(f'orthogonal-sign-contrast-{delta}', dict(d, X=X, y=y, l0_penalty=penalty, box=box))
    assert c[1] <= expected + 1e-10 and c[0] >= expected - 1e-10, (c, expected)
