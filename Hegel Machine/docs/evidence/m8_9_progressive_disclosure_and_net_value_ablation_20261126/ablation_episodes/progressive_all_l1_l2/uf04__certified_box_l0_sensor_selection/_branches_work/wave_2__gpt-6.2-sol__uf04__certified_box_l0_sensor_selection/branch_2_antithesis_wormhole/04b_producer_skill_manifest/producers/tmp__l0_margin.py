import sys
sys.path.insert(0, '/holdout_task')
import numpy as np
from certified_l0_solver import solve_certified_box_l0

worst_nodes = 0
worst_gap = 0.
failures = []
for seed in (31, 47, 58, 92):
    rng = np.random.default_rng(seed)
    Q = np.linalg.qr(rng.normal(size=(42, 15)))[0]
    for ridge, fee in ((.12, .55), (.35, .85)):
        for coupling in (1e-4, 5e-4):
            X = Q[:, np.arange(15)//2] + coupling*rng.normal(size=Q.shape)
            y = Q[:, :8] @ np.full(8, np.sqrt(2*fee*(1+ridge))+.12)
            key = (seed, ridge, coupling)
            try:
                out = solve_certified_box_l0(dict(X=X, y=y, ridge=ridge, box=2., l0_penalty=fee))
                w = np.asarray(out['weights'])
                value = .5*np.sum((X@w-y)**2) + .5*ridge*(w@w) + fee*np.count_nonzero(abs(w)>1e-9)
                gap = value - out['certified_lower_bound']
                nodes = out['nodes_explored']
                assert np.max(abs(w)) <= 2. and abs(value-out['objective']) < 1e-11
                assert -1e-13 <= gap <= 1e-5 and 1 <= nodes <= 600, (gap, nodes)
                worst_nodes = max(worst_nodes, nodes)
                worst_gap = max(worst_gap, gap)
            except (RuntimeError, AssertionError) as exc:
                failures.append(key)
                print('FAIL', key, exc, flush=True)
print('margin cases:', 16, 'failed:', len(failures), 'worst passing nodes:', worst_nodes, 'worst passing gap:', worst_gap, flush=True)
assert not failures
