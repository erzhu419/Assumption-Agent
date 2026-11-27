import itertools
import pickle
import runpy
import numpy as np

m = runpy.run_path('/holdout_task/batch_solver.py')
item = pickle.load(open('/holdout_task/public_batch.pkl', 'rb'))[0]
s = item['study']
x, raw = (s['sensor_calibration_sweeps'][k] for k in ('ref_rosette', 'raw_rosette'))
S, _ = m['_forward'](s)
truth = [S @ p['w_pulse'] for p in item['campaign_diagnostics']]
pulses = [p['y_pulse_raw'].reshape(-1, 4) for p in item['campaign_diagnostics']]
ns = len(s['mu_shifts'])

# Fit extra terms only on sweep training rows, never on a tracer or test sweep.
def terms(z, j, model):
    cols = [np.ones(len(z)), z[:, j]]
    if model == 'full affine':
        cols = [np.ones(len(z)), *z.T]
    if model == 'quadratic':
        cols = [np.ones(len(z)), *z.T]
        cols += [z[:, a] * z[:, b] for a, b in itertools.combinations_with_replacement(range(4), 2)]
    if model == 'rational pair':
        cols += [z[:, j] / (1 + a * (z[:, j] + sign * z[:, b])**2) for b in range(4) if b != j for a in (.25, 1.) for sign in (-1,1)]
    return np.column_stack(cols)

for model in ('diagonal affine', 'full affine', 'quadratic', 'rational pair'):
    sweep_errors, pulse_errors = [], []
    for k in range(5):
        test = np.arange(len(x)) % 5 == k
        for j in range(4):
            fit = np.linalg.lstsq(terms(x[~test], j, model), raw[~test, j], rcond=None)[0]
            sweep_errors.append(np.sqrt(np.mean((terms(x[test], j, model) @ fit - raw[test, j])**2)))
            for z, y in zip(truth, pulses):
                r = (terms(z.reshape(-1,4), j, model) @ fit - y[:, j]).reshape(ns,-1)
                pulse_errors.extend(np.sqrt(np.mean(r**2, axis=1)))
    print(model, 'worst_fold_bank_rmse', max(sweep_errors), 'worst_fold_pulse_shift_bank_rmse', max(pulse_errors))

# Nuisance stress: omit one physical sensor at every shift. A large change in
# weights or group-level calibration residual would warrant localization.
y = m['_calibrate'](s)
Z = s['Z_nuisance']; ridge = s['source_ridge']; cap = s['u_box']; budget = s['total_budget']

def fit_rows(keep, project):
    T, r, z = S[keep], y[keep], Z[keep]
    if project:
        prof = np.linalg.solve(z.T @ z + s['nuisance_ridge'] * np.eye(z.shape[1]), z.T)
        T, r = T - z @ (prof @ T), r - z @ (prof @ r)
    H = T.T @ T + ridge * np.eye(T.shape[1]); c = -T.T @ r
    w, nu = m['_box_simplex'](H, c, cap, budget)
    g = H @ w + c
    gap = g @ w - nu * budget - cap * np.minimum(0, g-nu).sum()
    gamma = np.linalg.solve(z.T @ z + s['nuisance_ridge'] * np.eye(z.shape[1]), z.T @ (y[keep] - S[keep] @ w)) if project else np.zeros(z.shape[1])
    return w, gamma, gap

all_rows = np.ones(len(y), dtype=bool)
w0, _, _ = fit_rows(all_rows, True)
for project in (True, False):
    residual = np.zeros_like(y); drift, gaps = [], []
    for j in range(len(s['sensor_ij'])):
        held = np.arange(len(y)) % len(s['sensor_ij']) == j
        w, gamma, gap = fit_rows(~held, project)
        residual[held] = S[held] @ w + Z[held] @ gamma - y[held]
        drift.append(np.linalg.norm(w-w0)); gaps.append(abs(gap))
    print('nuisance_projection', project, 'RMSE by shift', np.sqrt(np.mean(residual.reshape(ns,-1)**2, axis=1)), 'RMSE by bank', np.sqrt(np.mean(residual.reshape(-1,4)**2, axis=0)), 'max_weight_drift_l2', max(drift), 'max_abs_dual_gap', max(gaps))
