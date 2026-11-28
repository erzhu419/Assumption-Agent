import pickle
import sys

import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu

sys.path.insert(0, '/holdout_task')
from batch_solver import solve_batch
from downstream_solver import _axis, _responses
from _verified_upstream import calibrate_upstream

with open('/holdout_task/public_batch.pkl', 'rb') as f:
    items = pickle.load(f)
for item in items:
    raw = item['study']
    s = calibrate_upstream(dict(raw, reference_tracer_pulses=item.get('campaign_diagnostics')))
    pulses = s['reference_tracer_pulses']
    W = np.column_stack([p['w_pulse'] for p in pulses])
    Y = np.column_stack([p['y_pulse_raw'] for p in pulses])
    S, U = _responses(s)
    nx, ny = s['nx'], s['ny']
    hx, hy = s['Lx']/(nx-1), s['Ly']/(ny-1)
    ax = _axis(s['kappa_x'], s['vel_x'], hx, True)
    ay = _axis(s['kappa_y'], s['vel_y'], hy, False)
    ablated = ax.tolil()
    for i, j in ((0, nx-1), (nx-1, 0)):
        ablated[i, i] += float(ax[i, j])
        ablated[i, j] = 0.
    I = eye(nx*ny, format='csc')
    rest = kron(ay, eye(nx, format='csc'), format='csc') + diags(s['V_reaction_2d'].ravel(order='F'), format='csc')
    L0 = kron(eye(ny, format='csc'), ablated.tocsc(), format='csc') + rest
    ix, iy = s['sensor_ij'].T
    rows = ix + nx*iy
    rhs = np.column_stack([a.ravel(order='F') for a in s['source_modes']]) @ W
    no_wrap = np.vstack([splu(L0 + mu*I).solve(rhs)[rows] for mu in s['mu_shifts']])
    err = (S @ W - Y).reshape(len(s['mu_shifts']), len(rows), -1)
    ablated_err = (no_wrap - Y).reshape(err.shape)
    rmse = np.sqrt(np.mean(err**2, axis=1))
    no_wrap_rmse = np.sqrt(np.mean(ablated_err**2, axis=1))
    print('Periodic-x RMSE by shift x pulse:', rmse.tolist())
    print('Wrap-only ablation RMSE by shift x pulse:', no_wrap_rmse.tolist())
    print('Largest periodic-x point residual:', float(np.max(np.abs(err))))
    assert rmse.max() < 3e-6 and np.max(np.abs(err)) < 5e-6
    assert np.all(no_wrap_rmse > 10*rmse)
    out = solve_batch(raw)
    w = out['weights']
    cert = out['kkt_certificate']
    Z = s['Z_nuisance']
    P = np.eye(len(S)) - Z @ np.linalg.solve(Z.T @ Z + s['nuisance_ridge']*np.eye(Z.shape[1]), Z.T)
    Q = P @ S
    tangent = np.eye(len(w))[:, :-1] - np.eye(len(w))[:, [-1]]
    hv = np.linalg.eigvalsh(tangent.T @ (Q.T @ Q + s['source_ridge']*np.eye(len(w))) @ tangent)
    gap = cert['primal_objective'] - cert['dual_lower_bound']
    estimate = np.sqrt(2*max(gap, 0.)/s['source_ridge'])
    print('Nuisance-projected budget tangent curvature min/max:', float(hv[0]), float(hv[-1]))
    print('KKT gap and gap-implied weight error bound (up to floating-point precision):', gap, estimate)
    assert hv[0] > 0 and abs(gap) < 1e-11
