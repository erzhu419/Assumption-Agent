import importlib.util
import pickle
import numpy as np

root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('solver', root + 'alpha_w02_s00_solver.py')
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
with open(root + 'public_alpha_w02_s00.pkl', 'rb') as f:
    d = pickle.load(f)[0]['dataset']
tr = d['train']
t = d['eval_times']
dt = d['past_dt']
x, _, complete = s._align(d)
cal = s._calibration(d)
h, p = s._state_inputs(tr, cal)
y = tr['strain_curves_obs'] - tr['strain_0'][:, None]
r = tr['instant_rate_obs']

# Independent recurrence check, including a nonzero calibration offset.
taus = np.array([0.20, 8.0])
weights = np.array([.6, .5, .09, .16, -.13, .11, -.08])
hh, pp = h[:12] + .16, p[:12] + .16
z = np.zeros((len(hh), 2))
for k, step in enumerate(dt):
    a = np.exp(-step/taus)
    z = a*z + (1-a)*hh[:, k, None]
amp = (pp[:, None]-z)*weights[:2]
drift = weights[2] + x[:12] @ weights[3:]
c, v = s._basis(taus, hh, pp, dt, np.r_[0., t], x[:12])
np.testing.assert_allclose(c @ weights, amp @ (1-np.exp(-np.r_[0., t][None, :]/taus[:, None])) + drift[:, None]*np.r_[0., t], atol=1e-12)
np.testing.assert_allclose(v @ weights, (amp/taus).sum(axis=1) + drift, atol=1e-12)
print('independent endpoint and rate checks passed')

worst = np.zeros(3)
for l in np.unique(tr['lab_id']):
    holdout = tr['lab_id'] == l
    fit = ~holdout & complete
    tau, w = s._fit(h[fit], p[fit], dt, t, x[fit], y[fit], r[fit])
    b, v = s._basis(tau, h[holdout], p[holdout], dt, t, x[holdout])
    ce = np.abs(b @ w - y[holdout]); re = np.abs(v @ w - r[holdout]); mc = complete[holdout]
    vals = [np.nanmedian(ce[mc], axis=0).max(), np.nanmedian(re[mc]), np.nanmedian(ce[~mc], axis=0).max()]
    worst = np.maximum(worst, vals)
    print('lab', l, 'n complete/missing', int(mc.sum()), int((~mc).sum()), 'tau', tau.round(5), 'worst horizon complete / rate complete / horizon missing median abs errors', np.round(vals, 6))
print('worst LOLO errors:', worst.round(6))

rng = np.random.default_rng(102)
for n, glitch in [(10, 2), (4, 1), (2, 0), (1, 0)]:
    errors = []
    for i in range(100):
        a, b = rng.uniform(.88, 1.12), rng.uniform(-.12, .12)
        xc = np.linspace(-2, 2, n) if n > 1 else np.array([1.2])
        yc = a*xc + b + rng.normal(0, 1e-4, n)
        yc[rng.choice(n, glitch, replace=False)] += rng.choice([-1., 1.], glitch)*.15
        aa, bb = s._affine(xc, yc, (1., 0.))
        errors.append(max(abs((aa-a)*q + bb-b) for q in (-2.5, 2.5)))
    print('cal stress checks/glitches', n, glitch, 'maximum extrapolation error p95/max', np.round([np.quantile(errors,.95),max(errors)], 6))
