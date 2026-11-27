import importlib.util
import pickle
import numpy as np
spec = importlib.util.spec_from_file_location('solver', '/holdout_task/batch_solver.py')
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
with open('/holdout_task/public_batch.pkl', 'rb') as f:
    item = pickle.load(f)[0]
d, di = item['dataset'], item['campaign_diagnostics']
tr, te = d['train'], d['test']
t, dt = d['eval_times'], d['past_dt']
out = s.solve_batch(d)
tau, beta = out['estimated_taus'], out['hereditary_weights']
assert 0.05 < tau[0] < tau[1] < 25
bridge = s._bridge(d['calibration_load_cells'])
p, h = s._loads(tr, bridge)
x, xt = s._features(tr, te)
strain = tr['strain_curves_obs'] - tr['strain_0'][:, None]
rate = tr['instant_rate_obs']
for lab in np.unique(tr['lab_id']):
    k = tr['lab_id'] == lab
    tj, bj = s._fit(x[~k], p[~k], h[~k], dt, t, strain[~k], rate[~k])
    a, b = s._design(tj, x[k], p[k], h[k], dt, t)
    print('intrinsic held-out lab', lab, 'taus', tj.round(4), 'strain/rate p90', np.quantile(abs(strain[k] - a@bj), .9), np.quantile(abs(rate[k] - b@bj), .9))

xd = di['coupon_features_raw'].copy()
complete = np.isfinite(xd).all(1)
for lab in np.unique(di['lab_id']):
    k = di['lab_id'] == lab
    r = di['ref_coupons'][k]
    xd[k] = (xd[k] - np.median(r.mean(0))) / np.median(r.std(0))
xd = np.where(np.isfinite(xd), xd, np.nanmean(xd, 0))
_, hd = s._loads(di, bridge)
y = di['strain_curves_obs'] - di['strain_0'][:, None]
v = di['instant_rate_obs']
z, w = s._chamber(tau, beta, xd, hd, t)
for label,k in [('complete',complete),('missing',~complete)]:
    print('artifact diagnostic replay', label, 'strain/rate max', np.max(abs(y[k] - z[k])), np.max(abs(v[k] - w[k])))
assert max(np.max(abs(y-z)), np.max(abs(v-w))) < 0.01

# Verify strain and rate stay coupled, including at deployment drift extremes.
_, ht = s._loads(te, bridge)
eps = 1e-6
delta, rate = s._chamber(tau, beta, xt, ht, np.array([0., eps]))
np.testing.assert_array_equal(delta[:,0], 0.)
np.testing.assert_allclose(delta[:,1]/eps, rate, atol=3e-6)
np.testing.assert_allclose(rate, out['initial_rate_pred'])
