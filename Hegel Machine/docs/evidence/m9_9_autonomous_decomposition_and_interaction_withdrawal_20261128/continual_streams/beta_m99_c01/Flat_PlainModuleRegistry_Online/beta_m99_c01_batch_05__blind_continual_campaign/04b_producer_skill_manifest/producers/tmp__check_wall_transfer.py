import pickle
import sys
from pathlib import Path
import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu

root = Path('/holdout_task')
sys.path.insert(0, str(root))
from _verified_upstream import calibrate_upstream
from downstream_solver import _forward
from batch_solver import solve_batch

with (root / 'public_batch.pkl').open('rb') as stream:
    item = pickle.load(stream)[0]
raw = item['study']
s = calibrate_upstream(dict(raw, reference_tracer_pulses=item['campaign_diagnostics']))
pulses = s['reference_tracer_pulses']
W = np.stack([p['w_pulse'] for p in pulses])
Y = np.stack([p['y_pulse_raw'] for p in pulses])
Z = s['Z_nuisance']
P = np.eye(Y.shape[1]) - Z @ np.linalg.solve(Z.T @ Z + s['nuisance_ridge'] * np.eye(Z.shape[1]), Z.T)

def axis(d, kind):
    k, v = s['kappa_'+d], s['vel_'+d]
    n = len(k)
    h = s['L'+d] / (n-1)
    a = np.zeros((n, n))
    for i in range(n):
        for direction in (-1, 1):
            j = i + direction
            if i in (0, n-1) and kind in ('reflect', 'robin'):
                if 0 <= j < n:
                    a[i, j] = -k[i] / h**2
            else:
                wrap = not 0 <= j < n
                j %= n
                adv = 0. if wrap and kind == 'diffusive wrap' else max(-direction*v[i], 0.) / h
                a[i, j] -= (k[i]+k[j]) / (2*h**2) + adv
        a[i, i] = -a[i].sum()
        if i in (0, n-1) and kind == 'robin':
            a[i, i] += s['robin_'+d] / h
    return a

nx, ny = s['nx'], s['ny']
ix, iy = s['sensor_ij'].T
rhs = np.column_stack([m.ravel(order='F') for m in s['source_modes']])
I = eye(nx*ny, format='csc')

def design(x, y):
    op = (kron(eye(ny), axis('x', x)) + kron(axis('y', y), eye(nx))
          + diags(s['V_reaction_2d'].ravel(order='F'))).tocsc()
    return np.vstack([splu(op + mu*I).solve(rhs)[ix+nx*iy] for mu in s['mu_shifts']])

selected = ('periodic', 'reflect')
S = _forward(s)[0]
assert np.max(np.abs(S - design(*selected))) < 1e-11
residuals = {(x, y): W @ (S if (x, y) == selected else design(x, y)).T - Y
             for x in ('periodic', 'diffusive wrap', 'reflect', 'robin')
             for y in ('periodic', 'reflect', 'robin')}
banks = np.asarray(s['row_bank_id']).reshape(len(s['mu_shifts']), -1)
for label, proj in [('calibrated', np.eye(Y.shape[1])), ('profiled', P)]:
    Es = {model: E @ proj.T for model, E in residuals.items()}
    worst = 0.
    nearest_ratio = np.inf
    for hold in range(len(W)):
        train = np.arange(len(W)) != hold
        model = min(Es, key=lambda m: np.mean(Es[m][train]**2))
        assert model == selected, (label, hold, model)
        def worst_slice(m):
            e = Es[m][hold].reshape(banks.shape)
            return max(np.sqrt(np.mean(row[b == bank]**2)) for row, b in zip(e, banks) for bank in np.unique(b))
        curr = worst_slice(model)
        nearest_ratio = min(nearest_ratio, min(worst_slice(m) for m in Es if m != model)/curr)
        worst = max(worst, curr)
    assert worst < 2e-6 and nearest_ratio > 10
    print(label, 'leave-one-pulse-out worst shift/bank RMSE', worst, 'minimum nearest-alternative ratio', nearest_ratio)

c = solve_batch(raw)['kkt_certificate']
gap = abs(c['primal_objective'] - c['dual_lower_bound'])
assert gap < 1e-10
print('Isolated public solve primal-dual gap', gap)
