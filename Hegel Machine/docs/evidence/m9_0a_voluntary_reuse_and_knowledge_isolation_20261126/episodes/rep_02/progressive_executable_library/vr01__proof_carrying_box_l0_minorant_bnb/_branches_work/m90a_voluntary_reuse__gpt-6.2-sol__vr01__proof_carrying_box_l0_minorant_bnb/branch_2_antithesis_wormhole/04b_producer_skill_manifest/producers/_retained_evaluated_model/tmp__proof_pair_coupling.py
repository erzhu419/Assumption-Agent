import runpy
import numpy as np
check = runpy.run_path('/tmp/proof_verify_final_slice.py')['check']
p = 22
k, rho, l0, l2, M = 10, .85, .2, .1, 1.
G = np.eye(p)
i = np.arange(0, 2*k, 2)
G[i, i+1] = G[i+1, i] = rho
h = np.r_[np.full(2*k, 1.1), np.zeros(p-2*k)]
rng = np.random.default_rng(27)
perm = rng.permutation(p)
# A PSD rank-one cross-pair perturbation gives an independent objective
# bracket: at most .5*delta*(M*sum(abs(v)))**2 over the entire box.
v = np.resize(np.array([1., -1.]), p)
for delta in (1e-10, 1e-8):
    X = np.linalg.cholesky(G + delta*np.outer(v,v)).T
    y = np.linalg.solve(X.T, h)
    # A singleton per pair is optimal in the unperturbed objective.
    # Choose five of each sign so the rank-one penalty is exactly zero;
    # this attains the unperturbed lower bound in the perturbed problem.
    exact = .5*(y@y) + k*(l0 - 1.1 + .5*(1+l2))
    check(f'coupled-pairs-delta{delta:g}',dict(X=X[:,perm],y=y,lam0=l0,lam2=l2,M=M),exact)
