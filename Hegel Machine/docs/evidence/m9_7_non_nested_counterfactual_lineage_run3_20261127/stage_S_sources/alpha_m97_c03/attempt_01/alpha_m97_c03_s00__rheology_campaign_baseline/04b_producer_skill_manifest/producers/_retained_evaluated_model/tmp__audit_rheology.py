import importlib.util
import pickle
import numpy as np

p = '/holdout_task/'
spec = importlib.util.spec_from_file_location('solver', p+'alpha_m97_c03_s00_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open(p+'public_alpha_m97_c03_s00.pkl', 'rb') as stream:
    d = pickle.load(stream)[0]['dataset']
tr = d['train']
lab, stratum = tr['lab_id'], tr['stratum']
n = len(lab)
t, dt = d['eval_times'], d['past_dt']
past, hold, x, base = (v[:n] for v in m._prepare(d))
y, rates = tr['strain_curves_obs']-base[:, None], tr['instant_rate_obs']
complete = np.isfinite(tr['coupon_features_raw']).all(axis=1)
print('Regime: max calibrated |history|, critical stress, banding index:', np.abs(past).max(), d['crit_stress'], d['shear_banding_index'])
# Audit alignment by comparing against raw features with the same physical model.
raw = tr['coupon_features_raw']
models = {'aligned': x, 'raw': np.column_stack((np.ones(n), raw))}
for label, features in models.items():
    err = np.full_like(y, np.nan)
    rate_err = np.full(n, np.nan)
    for c in np.unique(lab):
        fit = complete & (lab != c)
        val = complete & (lab == c)
        taus, w = m._fit(past[fit], hold[fit], features[fit], dt, t, y[fit], rates[fit])
        a, b = m._design(taus, past[val], hold[val], features[val], dt, t)
        err[val] = y[val] - a @ w
        rate_err[val] = rates[val] - b @ w
    cells = [complete & (lab == c) & (stratum == s) for c, s in np.unique(np.column_stack((lab, stratum)), axis=0)]
    print(label, 'worst cell/horizon MedAE', max(np.median(np.abs(err[v]),axis=0).max() for v in cells if v.any()), 'worst cell rate MedAE', max(np.median(np.abs(rate_err[v])) for v in cells if v.any()))
    if label == 'aligned':
        for c in np.unique(lab):
            v = complete & (lab == c)
            drift_error = np.median(err[v]/t, axis=1)
            signed = [np.median(drift_error[q]) for q in [hold[v]<0, hold[v]>=0]]
            print('lab', c, 'median residual drift by hold sign', np.round(signed,5), 'median rate residual', round(float(np.median(rate_err[v])),5))
