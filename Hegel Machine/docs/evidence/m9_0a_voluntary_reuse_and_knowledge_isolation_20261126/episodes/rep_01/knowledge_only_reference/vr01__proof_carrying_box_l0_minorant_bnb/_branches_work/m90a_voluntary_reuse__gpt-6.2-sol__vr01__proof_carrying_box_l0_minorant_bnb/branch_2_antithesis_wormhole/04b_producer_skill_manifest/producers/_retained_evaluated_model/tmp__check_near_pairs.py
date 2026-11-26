import runpy
import numpy as np
ns = runpy.run_path('/tmp/check_proof_box.py')
solve, check = ns['m'].solve_proof_carrying_box_l0, ns['check']
p = 22
for pairs in (9, 10):
    for rho in (.99, .999999):
        for signed in (False, True):
            G = np.eye(p)
            h = np.r_[np.ones(2*pairs), np.zeros(p-2*pairs)]
            for j in range(0, 2*pairs, 2):
                sign = -1 if signed and j % 4 == 0 else 1
                G[j, j+1] = G[j+1, j] = sign * rho
                h[j+1] *= sign
            X = np.linalg.cholesky(G).T
            y = np.linalg.solve(X.T, h)
            q = dict(X=X, y=y, lam0=.15, lam2=.12, M=1.)
            expected = .5*(y@y) + pairs*min(0., .15-.5/1.12, .30-1/(1.12+rho))
            s = solve(q)
            assert abs(s['objective']-expected) <= 1e-5
            print('near pairs', pairs, rho, 'signed', signed, check(q, s), flush=True)
