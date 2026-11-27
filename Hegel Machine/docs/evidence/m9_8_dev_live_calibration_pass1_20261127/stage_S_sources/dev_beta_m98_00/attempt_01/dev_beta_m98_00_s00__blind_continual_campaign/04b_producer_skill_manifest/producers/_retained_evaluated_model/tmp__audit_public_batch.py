import importlib.util
import pickle
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

root = Path('/holdout_task')
spec = importlib.util.spec_from_file_location('batch_solver', root / 'batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with (root / 'public_batch.pkl').open('rb') as f:
    batch = pickle.load(f)
for index, item in enumerate(batch):
    s = item['study']
    c = s['sensor_calibration_sweeps']
    x, raw = map(np.asarray, (c['ref_rosette'], c['raw_rosette']))
    folds = np.arange(len(x)) % 4
    corrected = np.empty_like(x)
    rational = np.empty_like(x)
    for fold in range(4):
        t = folds != fold
        fit = m._calibration({'ref_rosette': x[t], 'raw_rosette': raw[t], 'single_bank_sweeps': c.get('single_bank_sweeps', {})})
        corrected[~t] = fit(raw[~t]) - x[~t]
        for j in range(4):
            q, a = m._rational_fit(x[t], (raw-x)[t, j])
            rational[~t, j] = m._rational_predict(x[~t], q, a) + x[~t, j] - raw[~t, j]
    score = lambda e: float(np.max([np.sqrt(np.mean(e[folds == k]**2, axis=0)) for k in range(4)]))
    print('study', index, 'worst fold/channel sweep RMSE, identity / nested selected inverse / forced rational forward:', score(raw-x), score(corrected), score(rational))
    source = np.asarray(s['source_modes'])
    unmix = m._calibration(c)
    response, fields = m._response(s, np.ones(8), source)
    for j, p in enumerate(s.get('reference_tracer_pulses') or []):
        e = (response @ p['w_pulse'] - unmix(p['y_pulse_raw'])).reshape(len(s['mu_shifts']), -1, 4)
        print('pulse', j, 'worst shift/channel RMSE', np.sqrt(np.mean(e**2, axis=1)).max(), 'worst row', np.max(abs(e)))
    Z = s['Z_nuisance']; y = unmix(s['y_obs_raw']); nr = s['nuisance_ridge']
    rr = s['source_ridge']; cap = s['u_box']; budget = s['total_budget']
    P = np.eye(len(Z)) - Z @ np.linalg.solve(Z.T @ Z + nr * np.eye(Z.shape[1]), Z.T)
    X, Y = P @ response, P @ y
    def objective(w):
        r = X @ w - Y
        return float((r @ r + rr * (w @ w)) / 2)
    def grad(w):
        return X.T @ (X @ w - Y) + rr*w
    out = m.solve_batch(s); w = out['weights']; cert = out['kkt_certificate']
    oracle = minimize(objective, np.full(len(w), budget / len(w)), jac=grad, method='SLSQP', bounds=[(0, cap)] * len(w), constraints=[{'type': 'eq', 'fun': lambda p: p.sum()-budget, 'jac': lambda p: np.ones_like(p)}], options={'ftol': 1e-14, 'maxiter': 300})
    lo, hi = np.asarray(cert['lam_lower']), np.asarray(cert['lam_upper'])
    stationarity = np.max(abs(grad(w) - cert['nu_sum'] - lo + hi))
    gap = cert['primal_objective'] - cert['dual_lower_bound']
    field_error = np.max(abs(out['terminal_field'] - np.tensordot(oracle.x, fields, axes=1)))
    print('public oracle:', {'success': oracle.success, 'budget_error': abs(w.sum()-budget), 'weight_max_error': np.max(abs(w-oracle.x)), 'terminal_field_max_error': field_error, 'certificate_stationarity': stationarity, 'certificate_gap': gap})
    assert oracle.success and abs(w.sum()-budget) < 1e-9 and np.min(w) >= -1e-10 and np.max(w) <= cap + 1e-10
    assert abs(objective(w)-cert['primal_objective']) < 1e-9 and abs(gap) < 1e-9 and stationarity < 1e-9
    assert np.max(abs(w-oracle.x)) < 1e-5 and field_error < 1e-5
    print('nuisance rank:', np.linalg.matrix_rank(Z), 'ridge projection algebra residual:', np.max(abs(P @ Z - nr * Z @ np.linalg.inv(Z.T @ Z + nr*np.eye(Z.shape[1])))))
