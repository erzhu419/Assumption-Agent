import runpy
import sys
import numpy as np

check = runpy.run_path('/tmp/proof_verify_trace.py')['check']
k = int(sys.argv[1])
rho = float(sys.argv[2])
p = 22
l0, l2, M = .2, .1, 1.
G = np.eye(p)
i = np.arange(0, 2*k, 2)
G[i, i+1] = G[i+1, i] = rho
if rho == 1.:
    X = np.eye(p)
    X[:, i+1] = X[:, i]
    y = np.zeros(p)
    y[i] = 1.1
else:
    X = np.linalg.cholesky(G).T
    h = np.zeros(p)
    h[:2*k] = 1.1
    y = np.linalg.solve(X.T, h)
# Block-wise exact objective: compare zero, either singleton, and both active.
one = l0 - 1.1 + .5*(1+l2)
two = 2*l0 - 1.1**2/(1+l2+rho)
exact = .5*(y@y) + k*min(0., one, two)
perm = np.random.default_rng(27).permutation(p)
check(f'paired-k{k}-rho{rho}', dict(X=X[:,perm],y=y,lam0=l0,lam2=l2,M=M), exact)
