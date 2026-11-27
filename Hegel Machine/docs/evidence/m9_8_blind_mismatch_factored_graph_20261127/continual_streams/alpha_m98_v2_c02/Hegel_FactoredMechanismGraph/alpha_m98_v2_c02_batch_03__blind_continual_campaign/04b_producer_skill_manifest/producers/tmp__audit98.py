import importlib.util
import pickle
import numpy as np
from scipy.optimize import least_squares

spec = importlib.util.spec_from_file_location('candidate', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('/holdout_task/public_batch.pkl', 'rb') as f:
    item = pickle.load(f)[0]
d, c = item['dataset'], item['campaign_diagnostics']
o = m.solve_batch(d)
assert all(np.isfinite(v).all() for v in o.values())
tau, theta = o['estimated_taus'], o['hereditary_weights']
t, dt = d['eval_times'], d['past_dt']
stress = m._stress_calibration(d['calibration_load_cells'])
_, x = m._standardize(c, c)
y, r = m._predict_chamber(c, x, stress, dt, t, tau, theta)
yerr = c['strain_curves_obs'] - y
rerr = c['instant_rate_obs'] - r
rho = theta[2] + x @ theta[3:]
g = -np.expm1(-t)
complete = np.isfinite(c['coupon_features_raw']).all(axis=1)
# Fit each proposed correction on other diagnostic labs only. Incomplete
# coupons are scored separately so imputation does not select the time law.
models = {'unit decay': (None, None),
          'decay perturbation': (rho[:, None] * (t * np.exp(-t) - g), np.zeros(len(rho))),
          'rational drift gain': ((rho**3 / (1 + rho**2))[:, None] * g, rho**3 / (1 + rho**2))}
for a, b in [(a,b) for a in range(4) for b in range(a+1,4)]:
    delta = x[:, a] - x[:, b]
    models[f'feature difference {a}-{b}'] = (delta[:, None] * g, delta)
for name, (curve, rate) in models.items():
    scores = []
    for lab in np.unique(c['lab_id']):
        q = c['lab_id'] == lab
        fit = ~q & complete
        held = q & complete
        w = 0.0 if curve is None else least_squares(lambda p: np.r_[((p[0]*curve[fit]-yerr[fit])/.004).ravel(), (p[0]*rate[fit]-rerr[fit])/.006], [0.], loss='soft_l1').x[0]
        ey = yerr[held] if curve is None else yerr[held] - w*curve[held]
        er = rerr[held] if rate is None else rerr[held] - w*rate[held]
        scores.append([np.mean(abs(ey)), np.mean(abs(ey[:, -3:])), np.mean(abs(er))])
    print(name, 'worst lab [curve, tail, onset] MAE:', np.round(np.max(scores, axis=0), 7))
print('imputed [curve, tail, onset] MAE:', np.round([np.mean(abs(yerr[~complete])), np.mean(abs(yerr[~complete,-3:])), np.mean(abs(rerr[~complete]))],7))
# Test whether extra bridge terms correct a measurable cross-talk error on
# held-out sweeps, rather than fitting in-sample instrument noise.
cal = d['calibration_load_cells']
raw = cal['bridge_raw']; target = cal['true_channel_loads'].sum(axis=1)
affine = np.column_stack((np.ones(len(raw)), raw))
quadratic = np.column_stack((affine, *[raw[:,a]*raw[:,b] for a in range(4) for b in range(a,4)]))
for label, a in [('affine bridge', affine), ('quadratic bridge', quadratic)]:
    errors = []
    for j in range(5):
        q = np.arange(len(raw)) % 5 == j
        coef = np.linalg.lstsq(a[~q], target[~q], rcond=None)[0]
        errors.append(np.mean(abs(a[q]@coef - target[q])))
    print(label, 'worst held-out sweep-fold total-load MAE:', max(errors))
