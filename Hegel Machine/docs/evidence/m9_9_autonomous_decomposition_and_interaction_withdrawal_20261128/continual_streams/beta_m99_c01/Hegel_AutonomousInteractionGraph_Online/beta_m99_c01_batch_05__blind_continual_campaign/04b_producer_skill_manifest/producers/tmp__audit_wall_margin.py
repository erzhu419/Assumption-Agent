import sys
sys.path.insert(0, '/holdout_task')
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import csc_matrix
from probe_walls import load, axis, forward
from downstream_solver import _forward

s, W, Y = load()
S, _ = _forward(s)
modes = s['source_modes']
rhs = modes.reshape(len(modes), -1, order='F').T @ W
blocks = []
for d, law in (('x', 'periodic'), ('y', 'sealed')):
    h = s['L'+d] / (s['n'+d]-1)
    k = s['kappa_'+d]
    A = axis(k, s['vel_'+d], h, s['robin_'+d], law).toarray()
    blocks.append((A, h, k))

def predict(p):
    ops = []
    for (A, h, k), q in zip(blocks, p.reshape(2, 2, 3)):
        B = A.copy()
        for (i, j, opposite), (inward, wrap, leak) in zip(((0, 1, len(k)-1), (len(k)-1, len(k)-2, 0)), q):
            f = -A[i,j] * inward
            w = (0.5*(k[i]+k[opposite])/h**2) * wrap
            B[i,j] -= f
            B[i,opposite] -= w
            B[i,i] += f+w+leak/h
        ops.append(csc_matrix(B))
    return forward(s, *ops, rhs)

base = predict(np.zeros(12))
np.testing.assert_allclose(base, S @ W, atol=1e-13, rtol=0)
Z = s['Z_nuisance']
nshift = len(s['mu_shifts'])
shift = np.repeat(np.arange(nshift), len(s['sensor_ij']))
rows = []
for j in range(nshift):
    test = shift == j
    train = ~test
    z = Z[train]
    P = np.linalg.solve(z.T @ z + s['nuisance_ridge']*np.eye(z.shape[1]), z.T)
    raw = lambda R: R[train]
    projected = lambda R: R[train]-z@(P@R[train])
    for name, transform in (('raw', raw), ('profiled', projected)):
        fit = least_squares(lambda p: 1e5*transform(predict(p)-Y).ravel(), np.zeros(12), bounds=(-.05,.05), max_nfev=100)
        rms = lambda R: np.sqrt(np.mean(R[test]**2, axis=0))
        old, new = rms(base-Y), rms(predict(fit.x)-Y)
        rows.append((j, name, float(old.max()), float(new.max()), float(np.max(np.abs(fit.x))), bool(fit.success)))
for row in rows:
    print('shift, training loss, baseline worst pulse RMS, expanded worst pulse RMS, max wall correction, converged:', row)
print('Worst held-out baseline/expanded RMS:', max(r[2] for r in rows), max(r[3] for r in rows))
