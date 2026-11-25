import importlib.util
import pickle
from time import perf_counter
import numpy as np

spec = importlib.util.spec_from_file_location('solver', '/holdout_task/certified_box_l0.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

def verify(name, c, expected=None):
    t = perf_counter()
    out = mod.solve_certified_box_l0(c)
    dt = perf_counter()-t
    X, y = c['X'], c['y']
    w = np.asarray(out['weights'])
    assert w.shape == (X.shape[1],) and np.all(np.abs(w) <= c['box'])
    r = X @ w-y
    j = 0.5*(r@r + c['ridge']*(w@w)) + c['l0_penalty']*np.sum(np.abs(w)>1e-10)
    assert abs(j-out['objective']) <= 1e-7
    l = out['certified_lower_bound']
    assert 0 <= out['objective']-l <= c['gap_tol']
    leaves = out['certificate_leaves']
    assert 1 <= len(leaves) <= c['max_leaves']
    assert l <= min(z['leaf_lower_bound'] for z in leaves) + 1e-8
    cover = np.zeros(1 << w.size, dtype=int)
    ids = np.arange(cover.size)
    for leaf in leaves:
        zero, one = leaf['fixed_zero'], leaf['fixed_nonzero']
        assert len(set(zero + one)) == len(zero) + len(one)
        assert all(0 <= i < w.size for i in zero + one)
        cover[((ids & sum(1 << i for i in zero)) == 0) & ((ids & sum(1 << i for i in one)) == sum(1 << i for i in one))] += 1
        if expected is not None and not (zero or one):
            assert leaf['leaf_lower_bound'] <= expected + 1e-10
    assert np.all(cover == 1)
    if expected is not None:
        assert l <= expected + 1e-10 and j <= expected + c['gap_tol']
    print(f'{name}: J={j:.12g} gap={j-l:.3g} seconds={dt:.3f} active={sum(abs(w)>1e-10)} leaves={len(leaves)}', flush=True)
    return dt

with open('/holdout_task/public_box_l0_cases.pkl', 'rb') as f:
    for i, c in enumerate(pickle.load(f)['cases']):
        verify(f'public-{i}', c)

rng = np.random.default_rng(391)
base = dict(gap_tol=1e-6, max_leaves=512, box=.73, l0_penalty=.19)
seconds = 0.
for p, ridge, noise in [(14, .03, .01), (15, 0., 0.), (15, 0., .0001)]:
    L = rng.normal(size=(62, 5)); R = rng.normal(size=(5, p))
    X = L@R + noise*rng.normal(size=(62, p))
    X /= np.linalg.norm(X, axis=0)
    y = X@rng.uniform(-2, 2, p) + rng.normal(scale=.05, size=62)
    seconds += verify(f'correlated-{p}-{ridge}-{noise}', dict(base, X=X, y=y, ridge=ridge))
print(f'three correlated cases: {seconds:.3f}s total', flush=True)
assert seconds < 20

# Eight independent duplicate pairs: compare against coordinatewise 0/1/2 choices.
p = 16
X = np.repeat(np.eye(p//2), 2, axis=1)
y = np.linspace(.08, 1.6, p//2)
box, ridge, penalty, tiny = .58, .05, .12, 1e-10
# For a fixed paid count, both coefficients minimize the same convex loss;
# enumerate the free/upper state of each asymmetric box exactly.
opt = 0.
for target in y:
    values = []
    for k in range(3):
        b = np.array([box]*k+[tiny]*(2-k))
        for capped in range(4):
            fixed = (capped & np.array([1,2])) != 0
            v = np.where(fixed, b, (target-b[fixed].sum())/(ridge+(~fixed).sum()))
            if np.all(v >= -b) and np.all(v <= b):
                values.append(.5*((sum(v)-target)**2+ridge*(v@v))+k*penalty)
    opt += min(values)
verify('duplicate-pairs-analytic', dict(base, X=X, y=y, ridge=ridge, box=box, l0_penalty=penalty), opt)

# Every coefficient lies inside the unpenalized band when M is below the cutoff.
X = 1e8*np.eye(4); y = np.array([.003, -.005, .009, 0.]); box = 8e-11
v = np.clip(y/1e8, -box, box); r = X@v-y
verify('free-band-analytic', dict(base, X=X, y=y, ridge=0., box=box), .5*(r@r))
