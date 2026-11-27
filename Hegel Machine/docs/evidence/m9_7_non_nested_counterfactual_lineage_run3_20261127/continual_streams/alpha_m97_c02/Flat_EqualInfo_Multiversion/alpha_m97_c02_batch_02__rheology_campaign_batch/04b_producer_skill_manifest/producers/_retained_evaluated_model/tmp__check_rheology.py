import importlib.util
import pickle
import sys
import numpy as np

def load(path):
    spec = importlib.util.spec_from_file_location('solver', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

with open('/holdout_task/public_alpha_m97_c02_batch_02.pkl', 'rb') as f:
    d = pickle.load(f)[0]['dataset']
tr = d['train']; n = len(tr['lab_id']); lab = tr['lab_id']; stratum = tr['stratum']
dt, t = d['past_dt'], d['eval_times']
y = tr['strain_curves_obs'] - tr['strain_0'][:, None]; r = tr['instant_rate_obs']
good = np.isfinite(tr['coupon_features_raw']).all(axis=1)
for path in sys.argv[1:]:
    s = load(path); data = tuple(a[:n] for a in s._prepare(d)); C = np.empty_like(y); R = np.empty_like(r)
    print(path)
    for k in [-1, 0, 1, 2, 3]:
        fit = good & (lab != k); check = (lab == k) if k >= 0 else np.ones(n, bool)
        tau, w = s._fit(*(a[fit] for a in data), dt, t, y[fit], r[fit]); c, z = s._design(*(a[check] for a in data), dt, t, tau)
        e, f = c @ w - y[check], z @ w - r[check]
        m = good[check]
        print(k, tau.round(5), w.round(5), 'complete curve abs q50/90/95/99/max', np.round(np.quantile(np.abs(e[m]), [.5,.90,.95,.99,1]), 5), 'rate RMSE complete/all', np.sqrt(np.mean(f[m]**2)), np.sqrt(np.mean(f**2)), 'curve abs >1 complete', np.sum(np.abs(e[m]) > 1))
        if k >= 0: C[check], R[check] = e, f
    for k,j in np.unique(np.c_[lab,stratum],axis=0):
        use=(lab==k)&(stratum==j)
        print('cell', k,j, use.sum(), 'curve MAE/median', np.mean(np.abs(C[use])), np.median(np.abs(C[use])), 'rate RMSE', np.sqrt(np.mean(R[use]**2)))
    print('curve spikes by horizon', (np.abs(C[good])>1).sum(axis=0), 'rows with multiple', np.sum((np.abs(C[good])>1).sum(axis=1)>1))
