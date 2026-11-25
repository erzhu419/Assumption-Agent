import pickle
import sys
import numpy as np
from scipy.optimize import lsq_linear
sys.path.insert(0, '/holdout_task')
from certified_l0_solver import solve_certified_box_l0

with open('/holdout_task/public_box_l0_instances.pkl', 'rb') as f:
    cases = pickle.load(f)

worst_gap = 0.
worst_nodes = 0
for i, raw in enumerate(cases):
    X0, y = np.asarray(raw['X']), np.asarray(raw['y'])
    ridge = raw['ridge']
    cols = np.argsort(np.abs(X0.T @ y))[-8:]
    for p, X in ((8, X0[:, cols]), (X0.shape[1], X0)):
        for box, fee in ((.125, .20), (.30, 1.7), (1e-9, .85)):
            ans = solve_certified_box_l0(dict(X=X, y=y, ridge=ridge, box=box, l0_penalty=fee))
            w = np.asarray(ans['weights'])
            def value(a):
                e = X @ a - y
                return .5 * (e @ e + ridge * (a @ a)) + fee * np.count_nonzero(np.abs(a) > 1e-9)
            U, L, N = ans['objective'], ans['certified_lower_bound'], ans['nodes_explored']
            label = (i, p, box, fee)
            assert w.shape == (p,) and np.max(np.abs(w)) <= box, label
            assert abs(U - value(w)) <= 1e-11 and 0 <= U-L <= 1e-5 and 1 <= N <= 600, (label, U, L, N)
            if p == 8 or box == 1e-9:
                # Independent box least-squares oracle; enumerate only the reduced test inputs.
                A = np.vstack((X, np.sqrt(ridge)*np.eye(p)))
                Y = np.r_[y, np.zeros(p)]
                oracle_upper = np.inf
                oracle_lower = np.inf
                for bits in range(1 << p if box > 1e-9 else 1):
                    active = (bits >> np.arange(p)) & 1
                    lim = np.where(active, box, 1e-9)
                    a = lsq_linear(A, Y, bounds=(-lim, lim), tol=1e-13, max_iter=250).x
                    r = A @ a - Y
                    g = A.T @ r
                    smooth_mode = .5 * (r @ r) + fee * active.sum()
                    oracle_lower = min(oracle_lower, smooth_mode - g @ a - np.abs(g) @ lim)
                    oracle_upper = min(oracle_upper, value(a))
                assert oracle_upper-oracle_lower <= 1e-8, (label, oracle_upper-oracle_lower)
                assert L <= oracle_upper + 1e-11 and U >= oracle_lower - 1e-11 and U-oracle_lower <= 1.001e-5, (label, L, U, oracle_lower, oracle_upper)
            print(label, 'gap', U-L, 'nodes', N, 'at box', np.count_nonzero(np.abs(w) == box), flush=True)
            worst_gap = max(worst_gap, U-L)
            worst_nodes = max(worst_nodes, N)
print('public-derived worst gap', worst_gap, 'max nodes', worst_nodes)
