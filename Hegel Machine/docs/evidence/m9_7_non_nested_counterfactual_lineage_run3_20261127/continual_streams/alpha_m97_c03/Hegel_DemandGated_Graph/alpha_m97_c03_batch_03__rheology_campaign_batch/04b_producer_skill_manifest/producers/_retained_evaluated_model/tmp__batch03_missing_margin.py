import importlib.util
import pickle
import numpy as np

root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('solver', root + 'alpha_m97_c03_batch_03_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open(root + 'public_alpha_m97_c03_batch_03.pkl', 'rb') as f:
    d = pickle.load(f)[0]['dataset']
tr = d['train']
n = len(tr['lab_id'])
p, h, x, b = (a[:n] for a in m._prepare(d))
y = tr['strain_curves_obs'] - b[:, None]
r = tr['instant_rate_obs']
lab = np.asarray(tr['lab_id'])
missing = np.isnan(tr['coupon_features_raw']).any(axis=1)
for grouping in ('lab_id', 'stratum'):
    groups = np.asarray(tr[grouping])
    for complete_only in (False, True):
        ec, er = np.empty_like(y), np.empty_like(r)
        fitted = []
        for g in np.unique(groups):
            val = groups == g
            fit = ~val & (~missing if complete_only else True)
            tau, weights = m._fit(p[fit], h[fit], x[fit], d['past_dt'], d['eval_times'], y[fit], r[fit])
            a, q = m._design(tau, p[val], h[val], x[val], d['past_dt'], d['eval_times'])
            ec[val], er[val] = a @ weights - y[val], q @ weights - r[val]
            fitted.append(np.r_[tau, weights])
        med = lambda a: np.nanmedian(np.abs(a))
        rms = lambda a: np.sqrt(np.nanmean(a*a))
        print(grouping, 'complete_only', complete_only,
              'missing curve medAE / rate medAE / rate RMSE', np.round([med(ec[missing]), med(er[missing]), rms(er[missing])], 6),
              'complete worst-group curve / rate medAE', np.round([max(med(ec[(groups == g) & ~missing]) for g in np.unique(groups)), max(med(er[(groups == g) & ~missing]) for g in np.unique(groups))], 6),
              'tau range', np.round(np.ptp(fitted, axis=0)[:2], 6))
