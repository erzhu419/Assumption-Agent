import itertools
import pickle
import sys

import numpy as np
from scipy.sparse import diags, eye, kron, lil_matrix
from scipy.sparse.linalg import splu

sys.path.insert(0, '/holdout_task')
from _verified_upstream import calibrate_upstream
from batch_solver import solve_batch
from downstream_solver import _responses

with open('/holdout_task/public_batch.pkl', 'rb') as f:
    item = pickle.load(f)[0]
study = item['study']
s = calibrate_upstream(dict(study, reference_tracer_pulses=item['campaign_diagnostics']))
W = np.column_stack([p['w_pulse'] for p in s['reference_tracer_pulses']])
Y = np.column_stack([p['y_pulse_raw'] for p in s['reference_tracer_pulses']])

# Assemble endpoint rows independently, allowing each face its own law.
def axis(d, ends):
    k, v = s['kappa_'+d], s['vel_'+d]
    n = len(k)
    h = s['L'+d]/(n-1)
    A = lil_matrix((n,n))
    for i in range(n):
        law = ends[0] if i == 0 else ends[1] if i == n-1 else 'periodic'
        if law == 'periodic':
            for step in (-1, 1):
                j = (i+step) % n
                a = (k[i]+k[j])/(2*h*h) + max(0., -step*v[i])/h
                A[i,j] -= a
                A[i,i] += a
        else:
            j = 1 if i == 0 else n-2
            a = k[i]/(h*h)
            A[i,j] = -a
            A[i,i] = a + (s['robin_'+d]/h if law == 'robin' else 0.)
    return A.tocsc()

pairs = list(itertools.product(('sealed','periodic','robin'), repeat=2))
ax, ay = ([axis(d, p) for p in pairs] for d in 'xy')
nx, ny = s['nx'], s['ny']
I = eye(nx*ny, format='csc')
R = diags(s['V_reaction_2d'].ravel(order='F'))
rhs = np.column_stack([m.ravel(order='F') for m in s['source_modes']]) @ W
i, j = s['sensor_ij'].T
ns = len(i)
laws, predictions = [], []
for a, b in itertools.product(range(len(pairs)), repeat=2):
    L = (kron(eye(ny),ax[a]) + kron(ay[b],eye(nx)) + R).tocsc()
    predictions.append(np.vstack([splu(L+mu*I).solve(rhs)[i+nx*j] for mu in s['mu_shifts']]))
    laws.append(pairs[a]+pairs[b])
P = np.array(predictions)
E = P - Y
want = laws.index(('periodic','periodic','sealed','sealed'))
S, U = _responses(s)
np.testing.assert_allclose(S @ W, P[want], atol=1e-12, rtol=0)

# Hold out complete rosettes across shifts, entire shifts, and individual pulses.
row = np.arange(len(Y))
folds = [(row % ns // 4 == a, slice(None)) for a in range(ns//4)]
folds += [(row//ns == a, slice(None)) for a in range(len(s['mu_shifts']))]
folds += [(np.zeros(len(Y), dtype=bool), a) for a in range(Y.shape[1])]
worst, margin = 0., np.inf
Z = s['Z_nuisance']
for test, col in folds:
    if isinstance(col, int):
        train = E[:,:,np.arange(Y.shape[1])!=col]
        design = Z
        held = E[:,:,col]
    else:
        train = E[:,~test,:]
        design = Z[~test]
        held = E[:,test,:]
    # Check selection both before and after fitting nuisance on training rows only.
    q = np.linalg.qr(design, mode='reduced')[0]
    for residual in (train, train - q @ (q.T @ train)):
        rmse = np.sqrt(np.mean(residual**2, axis=(1,2)))
        order = np.argsort(rmse)
        assert order[0] == want, (laws[order[0]], rmse[order[:2]])
        margin = min(margin, rmse[order[1]]-rmse[order[0]])
    worst = max(worst, float(np.sqrt(np.mean(held[want]**2))))
assert worst < 2e-6
out = solve_batch(study)
c = out['kkt_certificate']
gap = abs(c['primal_objective']-c['dual_lower_bound'])
assert gap < 1e-11
print(f'{len(laws)} face-wise laws, {len(folds)} held-out folds: worst RMSE={worst:.3e}, smallest runner-up training RMSE margin={margin:.3e}, public KKT gap={gap:.3e}')
