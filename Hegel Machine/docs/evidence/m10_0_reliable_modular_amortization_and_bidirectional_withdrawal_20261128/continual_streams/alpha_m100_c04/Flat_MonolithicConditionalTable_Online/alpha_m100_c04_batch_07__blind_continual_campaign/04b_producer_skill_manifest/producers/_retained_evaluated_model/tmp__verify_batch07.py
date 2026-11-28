import importlib.util
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('batch_solver', '/holdout_task/batch_solver.py')
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
item = pickle.load(open('/holdout_task/public_batch.pkl', 'rb'))[0]
d, diag = item['dataset'], item['campaign_diagnostics']
tr, te = d['train'], d['test']
dt, t = d['past_dt'], d['eval_times']
out = s.solve_batch(d)
assert set(out) == {'estimated_taus', 'hereditary_weights', 'predicted_strain', 'initial_rate_pred'}
tau, theta = out['estimated_taus'], out['hereditary_weights']
assert tau.shape == (2,) and .05 < tau[0] < tau[1] < 25
assert theta.shape == (7,)
assert out['predicted_strain'].shape == (len(te['lab_id']), len(t))
assert out['initial_rate_pred'].shape == (len(te['lab_id']),)
assert all(np.all(np.isfinite(v)) for v in out.values())
b = s._calibrate(d['calibration_load_cells'])
x = s._features(tr, te)
inp = s._inputs(te, x[len(tr['lab_id']):], b)
past, hold, cov = inp
mem = np.zeros((len(hold), 2))
for k, interval in enumerate(dt):
    e = np.exp(-interval / tau)
    mem = e * mem + (1 - e) * past[:, k, None]
amp = (hold[:, None] - mem) * theta[:2] + (cov @ theta[2:])[:, None] * s._CHAMBER_DRIFT[:2]
creep = (cov @ theta[2:]) * s._CHAMBER_DRIFT[2]
np.testing.assert_allclose(out['predicted_strain'], te['strain_0'][:, None] + amp @ (-np.expm1(-t[:, None] / tau)).T + creep[:, None] * t, atol=1e-12)
np.testing.assert_allclose(out['initial_rate_pred'], amp @ (1 / tau) + creep, atol=1e-12)
h = 1e-5
small, _ = s._chamber_response(tau, theta, inp, dt, np.array([h, 2*h]))
err = np.max(abs((4*small[:, 0] - small[:, 1])/(2*h) - out['initial_rate_pred']))
assert err < 1e-7, err
print('Analytic/finite-difference rate max difference:', err)
itr = s._inputs(tr, x[:len(tr['lab_id'])], b)
obs = tr['strain_curves_obs'] - tr['strain_0'][:, None]
rates = tr['instant_rate_obs']
fits, errors = [], []
for lab in np.unique(tr['lab_id']):
    held = tr['lab_id'] == lab
    q, w = s._fit(tuple(v[~held] for v in itr), dt, t, obs[~held], rates[~held])
    a, r = s._design(q, tuple(v[held] for v in itr), dt, t)
    fits.append(q)
    errors.append([np.max(np.median(abs(a @ w - obs[held]), axis=0)), np.median(abs(r @ w - rates[held]))])
print('Benchtop held-lab tau ranges:', np.min(fits, axis=0), np.max(fits, axis=0))
print('Benchtop worst held-lab median absolute strain (over horizons) / rate:', np.max(errors, axis=0))
xd = s._features(diag, diag)[:len(diag['lab_id'])]
idg = s._inputs(diag, xd, b)
a, r = s._design(tau, idg, dt, t)
drift = idg[2] @ theta[2:]
y = np.column_stack((diag['strain_curves_obs'] - diag['strain_0'][:, None], diag['instant_rate_obs']))
base = np.column_stack((a @ theta - drift[:, None] * t, r @ theta - drift))
f = np.vstack((np.column_stack((-np.expm1(-t[:, None]/tau), t)), np.r_[1/tau, 1.]))
mat = drift[:, None, None] * f
complete = np.isfinite(diag['coupon_features_raw']).all(axis=1)
res = np.empty_like(y)
for lab in np.unique(diag['lab_id']):
    held = diag['lab_id'] == lab
    fit = ~held & complete
    w = np.linalg.lstsq(mat[fit].reshape(-1, 3), (y-base)[fit].ravel(), rcond=None)[0]
    res[held] = base[held] + mat[held] @ w - y[held]
for name, mask in [('complete', complete), ('missing', ~complete), ('all', np.ones(len(y), bool))]:
    rms = np.array([np.sqrt(np.mean(res[mask & (diag['lab_id'] == lab)]**2, axis=0)) for lab in np.unique(diag['lab_id'])])
    print(name, 'worst diagnostic held-lab/horizon strain RMSE / worst held-lab rate RMSE:', rms[:, :-1].max(), rms[:, -1].max())
