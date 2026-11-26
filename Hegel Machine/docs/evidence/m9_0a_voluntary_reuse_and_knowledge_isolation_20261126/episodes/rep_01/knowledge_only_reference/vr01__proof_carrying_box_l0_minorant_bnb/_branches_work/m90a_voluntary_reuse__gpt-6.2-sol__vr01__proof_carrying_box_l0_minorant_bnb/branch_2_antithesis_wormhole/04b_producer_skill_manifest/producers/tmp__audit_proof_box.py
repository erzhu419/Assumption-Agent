import pickle
import runpy
import numpy as np
from scipy.optimize import lsq_linear

ns = runpy.run_path('/tmp/check_proof_box.py')
solve = ns['m'].solve_proof_carrying_box_l0
check = ns['check']
for row in pickle.load(open('/holdout_task/public_proof_box_l0_cases.pkl', 'rb')):
    q = row['problem']
    s = solve(q)
    gap, count, mineig, over = check(q, s)
    err = abs(s['objective'] - row['gt']['exact_objective'])
    assert err <= 1e-5
    w = s['weights']
    X, y, l2, M = (q[k] for k in ('X', 'y', 'lam2', 'M'))
    a = w != 0
    if a.any():
        fit = lsq_linear(np.vstack([X[:, a], np.sqrt(l2)*np.eye(a.sum())]), np.r_[y, np.zeros(a.sum())], bounds=(-M, M), tol=1e-13)
        coeferr = max(abs(fit.x - w[a]))
        assert coeferr < 1e-6
    else:
        coeferr = 0.
    print('public', X.shape, 'gap', gap, 'leaves', count, 'objective error', err, 'independent support fit error', coeferr, 'witness overshoot', over)

# The API charges only coefficients strictly above 1e-8. Orthogonal data
# permit an independent exact comparison of the uncharged and charged optima.
for scale in (1., 1.e8):
    p = 22
    X, y = scale*np.eye(p), np.full(p, .5)
    q = dict(X=X, y=y, lam0=.22, lam2=.14, M=1.1)
    try:
        s = solve(q)
    except RuntimeError as exc:
        print('threshold stress', scale, 'solver error', exc)
        continue
    v = scale*.5/(scale*scale + q['lam2'])
    z = min(v, 1e-8)
    uncharged = p*(.5*(scale*z-.5)**2+.5*q['lam2']*z*z)
    charged = p*(.5*(scale*v-.5)**2+.5*q['lam2']*v*v+q['lam0']) if v>1e-8 else float('inf')
    exact = min(uncharged, charged)
    print('threshold stress', scale, 'analytic optimum', exact, 'returned objective', s['objective'], 'reported lower minus optimum', s['certified_lower_bound']-exact)
