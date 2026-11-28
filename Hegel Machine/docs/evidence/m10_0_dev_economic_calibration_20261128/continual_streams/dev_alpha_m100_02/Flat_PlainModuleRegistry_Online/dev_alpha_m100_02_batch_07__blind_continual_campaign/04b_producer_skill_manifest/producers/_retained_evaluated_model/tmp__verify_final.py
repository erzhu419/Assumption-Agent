import json, pickle, sys
import numpy as np
sys.path.insert(0, '/holdout_task')
import batch_solver as api
import downstream_solver as s
from _verified_upstream import calibrate_upstream

with open('/holdout_task/public_batch.pkl', 'rb') as f:
    item = pickle.load(f)[0]
with open('/holdout_task/_upstream_param_payload.json') as f:
    pay = json.load(f)
raw = item['dataset']
d = calibrate_upstream(raw, None)
q = calibrate_upstream(dict(raw, test=item['campaign_diagnostics']), pay)['test']
out = api.solve_batch(raw)
for name, val in s.solve_downstream(d).items():
    np.testing.assert_array_equal(val, out[name])
tau, theta = out['estimated_taus'], out['hereditary_weights']
assert .05 < tau[0] < tau[1] < 25
for val in out.values():
    assert np.isfinite(val).all()
t, dt = d['eval_times'], d['past_dt']
x, = s._features(q)
b, br = s._basis(tau, t)
gap = s._history_gap(tau, q['past_channel_raw'].sum(-1), q['hold_channel_raw'].sum(-1), dt)
mechanical_amp = gap * theta[:2]
mechanical = mechanical_amp @ b[:, :2].T
force = x @ theta[2:]
rate = q['instant_rate_obs']
observed = q['strain_curves_obs'] - q['strain_0'][:, None]
redistribution = b[:, :2] * tau - t[:, None]
latent = rate - mechanical_amp @ br[:2]
response = observed - mechanical - latent[:, None] * t
predictors = latent[:, None, None] * redistribution
miss = np.isnan(q['coupon_features_raw']).any(axis=1)
err = np.empty((len(rate), len(t)+1))
fits = []
for lab in np.unique(q['lab_id']):
    mask = q['lab_id'] == lab
    coef = np.linalg.lstsq(predictors[~mask].reshape(-1,2), response[~mask].ravel(), rcond=None)[0]
    fits.append(coef)
    strain, rp = s._chamber(q, x, tau, theta, dt, t, coef)
    err[mask] = np.c_[strain-q['strain_curves_obs'],rp-rate][mask]
def rmse(e):
    return np.sqrt([np.mean(e[:,:-1]**2), np.mean(e[:,-1]**2)])
worst = np.max([rmse(err[q['lab_id'] == lab]) for lab in np.unique(q['lab_id'])], axis=0)
strain, rp = s._chamber(q, x, tau, theta, dt, t)
fixed = np.c_[strain - q['strain_curves_obs'], rp - rate]
print('LOLO fold fractions:', np.round(fits, 5))
print('LOLO worst-lab strain/rate RMSE:', worst)
print('LOLO incomplete / complete strain/rate RMSE:', rmse(err[miss]), rmse(err[~miss]))
print('Fixed-coefficient worst-lab strain/rate RMSE:', np.max([rmse(fixed[q['lab_id'] == lab]) for lab in np.unique(q['lab_id'])], axis=0))
assert np.all(worst < [0.022, 0.024]); assert np.all(rmse(fixed[~miss]) < [0.008,0.004])
# Check the returned rate against the chamber curve derivative independently.
xt = s._features(d['train'], d['test'])[1]
eps = 1e-6
small, r = s._chamber(d['test'], xt, tau, theta, dt, np.array([0., eps]))
np.testing.assert_allclose((small[:,1]-small[:,0])/eps, r, atol=1e-5, rtol=0)
print('Module composition and chamber rate derivative agree.')
# Same coupons, same material fit: isolate the chamber-law difference.
base_s, base_r = s._chamber(q, x, tau, theta, dt, t, np.zeros(2))
base_err = np.c_[base_s - q['strain_curves_obs'], base_r - rate]
print('Benchtop versus chamber strain/rate RMSE:', rmse(base_err), rmse(fixed))
assert rmse(fixed)[0] < 0.05 * rmse(base_err)[0]
np.testing.assert_array_equal(base_r, rp)
# A different affine bridge encoding must be neutral after refitting its sweeps.
from copy import deepcopy
encoded = deepcopy(raw)
factor = np.array([0.93, 0.96, 0.92, 0.95])
offset = np.array([0.07, -0.03, 0.05, -0.01])
bridge = encoded['calibration_load_cells']['bridge_raw']
mask = np.all(abs(bridge) <= 8.5, axis=1)
bridge[mask] = bridge[mask] * factor + offset
assert np.array_equal(mask, np.all(abs(bridge) <= 8.5, axis=1))
for part in ('train', 'test'):
    for key in ('past_channel_raw', 'hold_channel_raw'):
        encoded[part][key] = encoded[part][key] * factor + offset
for name, value in api.solve_batch(encoded).items():
    np.testing.assert_allclose(value, out[name], rtol=0, atol=2e-7)
print('Refitting the supplied bridge sweeps preserves predictions under affine re-encoding.')
