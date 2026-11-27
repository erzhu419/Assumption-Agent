import importlib.util
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('candidate', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
p = pickle.load(open('/holdout_task/public_batch.pkl', 'rb'))[0]
d = p['dataset']; tr = d['train']; q = p['campaign_diagnostics']
dt = d['past_dt']; t = d['eval_times']

def obs(z):
    return np.c_[z['strain_curves_obs'] - z['strain_0'][:, None], z['instant_rate_obs']]

def metrics(e):
    return tuple(np.round([np.sqrt(np.mean(e[:, :-1]**2)), np.sqrt(np.mean(e[:, -1]**2)),
                           np.max(np.sqrt(np.mean(e[:, :-1]**2, axis=1))), max(abs(e[:, -1]))], 6))

out = m.solve_batch(d)
a, b = m._prepare(d)
taus = out['estimated_taus']; theta = out['hereditary_weights']
assert 0.05 < taus[0] < taus[1] < 25
x = q['coupon_features_raw'].copy()
missing = np.isnan(x).any(axis=1)
for lab in np.unique(q['lab_id']):
    i = q['lab_id'] == lab
    r = q['ref_coupons'][i]
    x[i] = (x[i] - np.median(r.mean(axis=0))) / np.median(r.std(axis=0))
x = np.where(np.isnan(x), np.nanmean(x, axis=0), x)
off, w = m._calibration(d['calibration_load_cells'])
c = ((q['past_channel_raw'] - off) @ w, (q['hold_channel_raw'] - off) @ w,
     np.c_[np.ones(len(x)), x])
yq = obs(q)
f, r = m._chamber(c, t, taus, theta)
e = np.c_[f, r] - yq
print('metrics: strain RMSE, rate RMSE, max coupon strain RMSE, max absolute rate error')
print('chamber baseline', metrics(m._design(c, dt, t, taus) @ theta - yq))
print('chamber fixed split', metrics(e))
for lab in np.unique(q['lab_id']):
    print('chamber lab', lab, metrics(e[q['lab_id'] == lab]))
print('chamber missing features', metrics(e[missing]))
print('chamber by time RMSE', np.round(np.sqrt(np.mean(e[:, :-1]**2, axis=0)), 6))
assert max(metrics(e)[:2]) < .008 and max(metrics(e)[2:]) < .013
for g in np.unique(tr['lab_id']):
    keep = tr['lab_id'] != g
    tt, th = m._fit(tuple(v[keep] for v in a), dt, t, obs(tr)[keep])
    err = m._design(tuple(v[~keep] for v in a), dt, t, tt) @ th - obs(tr)[~keep]
    ff, rr = m._chamber(c, t, tt, th)
    print('benchtop held-out lab', g, 'tau delta', np.round(tt-taus, 6),
          'max weight delta', round(np.max(abs(th-theta)), 6), 'rate RMSE', round(np.sqrt(np.mean(err[:, -1]**2)), 6),
          'median absolute strain error', round(np.median(abs(err[:, :-1])), 6),
          'chamber transfer', metrics(np.c_[ff, rr] - yq))
for step in (1e-5, 1e-6):
    inc, slope = m._chamber(b, np.array([0., step]), taus, theta)
    assert np.all(inc[:, 0] == 0)
    np.testing.assert_allclose(inc[:, 1] / step, slope, atol=5e-5, rtol=1e-5)
print('initial-rate derivative and zero-time boundary passed')
