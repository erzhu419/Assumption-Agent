import importlib.util
import pickle

import numpy as np
from scipy.optimize import least_squares

spec = importlib.util.spec_from_file_location('candidate', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('/holdout_task/public_batch.pkl', 'rb') as f:
    item = pickle.load(f)[0]
d, g = item['dataset'], item['campaign_diagnostics']
tr, te = d['train'], d['test']
dt, t = d['past_dt'], d['eval_times']
b = m._bridge(d['calibration_load_cells'])
xtr, _ = m._features(tr, te)
tau, theta = m._material(tr, b, xtr, dt, t)
x, _ = m._features(g, {k: v[:0] for k, v in g.items()})
past, hold = m._loads(g, b)
c, r = m._basis(tau, past, hold, x, dt, t)
base, rate = c @ theta, r @ theta
y = g['strain_curves_obs'] - g['strain_0'][:, None]
z = hold[:, None] - r[:, :2] * tau
drift = theta[2] + x @ theta[3:]
u = t[:, None] / tau
F = -np.expm1(-u) - u
lab = g['lab_id']
complete = np.isfinite(g['coupon_features_raw']).all(1)
for name in ('linear drift', 'rational drift', 'rational gap difference', 'paired history differences'):
    err = np.empty_like(y)
    extras = []
    for ell in np.unique(lab):
        held = lab == ell
        fit = ~held & complete
        signal = drift if name != 'rational gap difference' else z[:, 0] - z[:, 1]
        scale = max(np.sqrt(np.mean(signal[fit]**2)), 1e-8)
        q2 = (signal / scale)**2
        D = drift[:, None, None] * F[None, :, :]
        if name == 'paired history differences':
            D = np.concatenate((D, ((hold[:, None] - z)[:, None, :] * F[None, :, :])), axis=2)
        target = (y - base)[fit].ravel()
        A = D[fit].reshape(len(target), -1)
        init = np.linalg.lstsq(A, target, rcond=None)[0]
        if name.startswith('rational'):
            def predict(q):
                return (D @ q[:-1]) / (1 + q[-1] * q2[:, None])
            init = np.r_[init, 0.001]
            bounds = (np.r_[[-np.inf] * (len(init) - 1), 0.], np.inf)
        else:
            def predict(q):
                return D @ q
            bounds = (-np.inf, np.inf)
        q = least_squares(lambda q: predict(q)[fit].ravel() - target, init,
                          bounds=bounds, loss='soft_l1', f_scale=.003).x
        if name.startswith('rational'):
            extras.append(round(float(q[-1]), 6))
        err[held] = abs((base + predict(q) - y)[held])
    worst_lab = max(err[lab == ell].mean() for ell in np.unique(lab))
    worst_complete_lab = max(err[(lab == ell) & complete].mean() for ell in np.unique(lab) if np.any((lab == ell) & complete))
    print(f'{name}: worst lab={worst_lab:.6f}, worst complete lab={worst_complete_lab:.6f}, complete={err[complete].mean():.6f}, missing={err[~complete].mean():.6f}, worst horizon={err.mean(0).max():.6f}, saturation={extras}')
re = abs(rate - g['instant_rate_obs'])
print(f'All derivative-conserving candidates: rate MAE complete={re[complete].mean():.6f}, missing={re[~complete].mean():.6f}, worst lab={max(re[lab == ell].mean() for ell in np.unique(lab)):.6f}')
