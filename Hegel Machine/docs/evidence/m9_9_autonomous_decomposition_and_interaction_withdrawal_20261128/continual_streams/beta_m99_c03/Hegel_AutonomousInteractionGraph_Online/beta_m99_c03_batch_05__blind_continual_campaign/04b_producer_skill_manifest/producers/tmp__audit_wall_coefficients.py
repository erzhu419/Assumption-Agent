import pickle, runpy, sys

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import csc_matrix, diags, eye, kron
from scipy.sparse.linalg import splu

root = '/holdout_task'
sys.path.insert(0, root)
from _verified_upstream import calibrate_upstream
from downstream_solver import _axis_operator

worst_fixed = worst_fit = worst_delta = 0.
min_rank = 8
folds = 0
with open(root + '/public_batch.pkl', 'rb') as f:
    items = pickle.load(f)
for item in items:
    s = calibrate_upstream(dict(item['study'], reference_tracer_pulses=item['campaign_diagnostics']))
    nx, ny = s['nx'], s['ny']
    pulses = s['reference_tracer_pulses']
    w = np.column_stack([p['w_pulse'] for p in pulses])
    y = np.column_stack([p['y_pulse_raw'] for p in pulses])
    rows = s['sensor_ij'] @ np.array([1, nx])
    rhs = np.column_stack([m.ravel(order='F') for m in s['source_modes']]) @ w
    axes = [_axis_operator(s['kappa_'+a], s['vel_'+a], s['L'+a], a == 'x').toarray() for a in 'xy']
    ident = eye(nx*ny, format='csc')
    reaction = diags(s['V_reaction_2d'].ravel(order='F'), format='csc')

    def residual(theta):
        perturbed = []
        for base, params, a in zip(axes, theta.reshape(2, 2, 2), 'xy'):
            op = base.copy()
            h = s['L'+a] / (len(base)-1)
            for (i, j), (wrap, leak) in zip([(0, -1), (-1, 0)], params):
                op[i, j] -= wrap / h**2
                op[i, i] += wrap / h**2 + leak / h
            perturbed.append(csc_matrix(op))
        ax, ay = perturbed
        L = kron(eye(ny), ax, format='csc') + kron(ay, eye(nx), format='csc') + reaction
        return np.vstack([splu(L + shift * ident).solve(rhs)[rows] for shift in s['mu_shifts']]) - y

    zero = np.zeros(8)
    base_error = residual(zero)
    for p in range(w.shape[1]):
        train = np.arange(w.shape[1]) != p
        fit = least_squares(lambda t: residual(t)[:, train].ravel(), zero, bounds=(-0.05, 0.05), gtol=1e-12)
        assert fit.success, fit.message
        min_rank = min(min_rank, np.linalg.matrix_rank(fit.jac))
        worst_delta = max(worst_delta, np.max(abs(fit.x)))
        for e, fitted in [(base_error, False), (residual(fit.x), True)]:
            rmse = np.sqrt(np.mean(e[:, p].reshape(len(s['mu_shifts']), -1)**2, axis=1)).max()
            if fitted:
                worst_fit = max(worst_fit, rmse)
            else:
                worst_fixed = max(worst_fixed, rmse)
        folds += 1
print(f'boundary coefficient audit: {folds} held-pulse folds; minimum Jacobian rank={min_rank}/8; max absolute fitted perturbation={worst_delta:.3g}; worst held-out shift RMSE fixed={worst_fixed:.3g}, fitted={worst_fit:.3g}')
sys.argv = [root + '/validate_public_schema.py', root + '/batch_solver.py']
runpy.run_path(sys.argv[0], run_name='__main__')
