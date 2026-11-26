import importlib.util
import pickle
import numpy as np

root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('solver', root + 'alpha_w04_batch_04_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
case = pickle.load(open(root + 'public_alpha_w04_batch_04.pkl', 'rb'))[0]
d, g = case['dataset'], case['gt']
tr, te = d['train'], d['test']
dt, t, y = (d[k] for k in ('past_dt', 'eval_times', 'yield_stress'))
panel = d['calibration_load_cells']
aff = m._calibrate(panel)
x, _ = m._align(tr, te)
h, s = m._stress(tr, aff)
c = tr['strain_curves_obs'] - tr['strain_0'][:, None]
r = tr['instant_rate_obs']
complete = np.isfinite(tr['coupon_features_raw']).all(axis=1)
plastic = (np.abs(h).max(axis=1)>y) | (np.abs(s)>y)
for lab in range(4):
    fit = tr['lab_id'] != lab
    tau, w = m._fit(dt, t, h[fit], s[fit], x[fit], c[fit], r[fit], y)
    a, b = m._design(tau, dt, t, h, s, x, y)
    ce, re = c-a@w, r-b@w
    for regime in (False, True):
        q = ~fit & complete & (plastic==regime)
        if q.any():
            print('held lab / plastic / n', lab, regime, q.sum(), 'curve/rate RMSE', np.sqrt(np.mean(ce[q]**2)), np.sqrt(np.mean(re[q]**2)), 'signed means', np.mean(ce[q]), np.mean(re[q]))
    assert np.max(np.abs(tau-g['taus'])) < .15
    assert np.max(np.abs(w[7:]-g['vp_theta'])) < .01
# Cross measured sub-yield and over-yield histories and holds so the two
# plastic basis terms are tested independently, not just where both co-occur.
h0 = h[np.flatnonzero((np.abs(h)<=y).all(axis=1))[0]]
h1 = h[np.flatnonzero((np.abs(h)>y).any(axis=1))[0]]
s0 = s[np.flatnonzero((np.abs(s)<=y) & (s!=0))[0]]
s1 = s[np.flatnonzero(np.abs(s)>y)[0]]
H = np.array([h0,h0,h1,h1,h0,h1]); S = np.array([s0,s1,s0,s1,-s1,-s1])
a,b = m._design(g['taus'], dt, t, H, S, np.zeros((6,4)), y)
over = np.sign(S)*np.maximum(np.abs(S)-y,0)**2
work = (np.maximum(np.abs(H)-y,0)**2 @ dt)*S
np.testing.assert_allclose(a[:,:,7], np.outer(over,t))
np.testing.assert_allclose(a[:,:,8], np.outer(work,-np.expm1(-t/g['taus'][0])))
np.testing.assert_allclose(b[:,7], over)
np.testing.assert_allclose(b[:,8], work/g['taus'][0])
# Both wings reduced to spot checks, with one EMI glitch on each wing.
idx = np.concatenate([np.flatnonzero(panel['lab_id']==lab)[:2] for lab in range(4)])
sparse = {k:v[idx].copy() for k,v in panel.items()}
sparse['true_stress'][[0,4]] += [.2,-.2]
fit = m._calibrate(sparse)
print('both wings sparse+EMI max gain/bias error', np.max(np.abs(fit-aff),axis=0))
assert np.max(np.abs(fit-aff)) < .001
out = m.solve_alpha_w04_batch_04(dict(d, calibration_load_cells=sparse))
missing = ~np.isfinite(te['coupon_features_raw']).all(axis=1)
th,ts = g['te_hist_true'],g['te_hold_true']
tp = (np.abs(th).max(axis=1)>y)|(np.abs(ts)>y)
errors = []
for lab in range(4):
    for regime in (False,True):
        for miss in (False,True):
            q = (te['lab_id']==lab)&(tp==regime)&(missing==miss)
            if q.any():
                errors.append([np.sqrt(np.mean((out[k][q]-g[v][q])**2)) for k,v in [('predicted_strain','te_clean'),('initial_rate_pred','te_true_rate0')]])
print('both wings sparse+EMI, worst lab × yield × missingness slice RMSE', np.max(errors,axis=0))
assert np.all(np.max(errors,axis=0) < [.09,.021])
