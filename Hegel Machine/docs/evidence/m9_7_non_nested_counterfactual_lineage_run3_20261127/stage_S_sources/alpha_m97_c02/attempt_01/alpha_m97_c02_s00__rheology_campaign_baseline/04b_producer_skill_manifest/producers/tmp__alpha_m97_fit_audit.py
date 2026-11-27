import importlib.util
import pickle
import numpy as np

root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('solver', root+'alpha_m97_c02_s00_solver.py')
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
with open(root+'public_alpha_m97_c02_s00.pkl', 'rb') as f:
    d = pickle.load(f)[0]['dataset']
tr, te = d['train'], d['test']
lab = tr['lab_id']
x = s._align(tr, te)[:len(lab)]
p, h = s._physical(tr, s._calibrate(d['calibration_load_cells']))
dt, t = d['past_dt'], d['eval_times']
y = tr['strain_curves_obs'] - tr['strain_0'][:, None]
r = tr['instant_rate_obs']
missing = np.isnan(tr['coupon_features_raw']).any(axis=1)
for l in np.unique(lab):
    val = lab == l
    for name, fit in [('all', ~val), ('complete_only', ~val & ~missing)]:
        tau, w = s._fit(p[fit], h[fit], x[fit], dt, t, y[fit], r[fit])
        a, b = s._basis(p[val], h[val], dt, t, tau)
        drift = w[2] + x[val] @ w[3:]
        strain_error = a @ w[:2] + drift[:, None]*t - y[val]
        rate_error = b @ w[:2] + drift - r[val]
        print('lab', l, name, 'taus', np.round(tau, 6))
        for label, group in [('complete', ~missing[val]), ('missing', missing[val])]:
            if not group.any():
                continue
            e, er = strain_error[group], rate_error[group]
            print(label, group.sum(), 'early/late/rate median absolute errors', np.round([np.median(np.abs(e[:, :2])), np.median(np.abs(e[:, -2:])), np.median(np.abs(er))], 6))
            if label == 'complete':
                print('signed late/rate medians', np.round([np.median(e[:, -2:]), np.median(er)], 6))
