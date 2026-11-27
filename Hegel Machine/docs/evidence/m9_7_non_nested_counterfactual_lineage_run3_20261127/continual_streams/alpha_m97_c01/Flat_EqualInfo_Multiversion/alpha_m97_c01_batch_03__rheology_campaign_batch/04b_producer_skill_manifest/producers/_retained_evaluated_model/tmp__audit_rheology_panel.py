import importlib.util
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('rheology', '/holdout_task/alpha_m97_c01_batch_03_solver.py')
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
with open('/holdout_task/public_alpha_m97_c01_batch_03.pkl', 'rb') as f:
    p = pickle.load(f)[0]
d, truth = p['dataset'], p['gt']
tr, te = d['train'], d['test']
n = len(tr['lab_id'])
lab = np.r_[tr['lab_id'], te['lab_id']]
ref = np.vstack((tr['ref_coupons'], te['ref_coupons']))
dt, t, past, hold, aligned = s._prepare(d)
# Undo just the lab-specific location and scale after cell imputation.
# A global affine rescaling is immaterial because the drift includes an intercept.
pooled = aligned.copy()
for c in np.unique(lab):
    m = lab == c
    center = np.nanmedian(np.nanmean(ref[m], axis=0))
    scale = np.nanmedian(np.nanstd(ref[m], axis=0, ddof=0))
    assert np.isfinite(center) and np.isfinite(scale) and scale > 0
    pooled[m] = aligned[m] * scale + center
curve = tr['strain_curves_obs'] - tr['strain_0'][:, None]
rate = tr['instant_rate_obs']
complete = np.isfinite(tr['coupon_features_raw']).all(axis=1)
for name, x in [('lab-aligned', aligned), ('pooled', pooled)]:
    results = []
    for c in np.unique(lab[:n]):
        fit = lab[:n] != c
        val = ~fit & complete
        tau, w = s._fit(dt, t, past[:n][fit], hold[:n][fit], x[:n][fit], curve[fit], rate[fit])
        a, b = s._basis(tau, dt, t, past[:n][val], hold[:n][val], x[:n][val])
        results.append([np.quantile(abs(a @ w - curve[val]), .9), np.quantile(abs(b @ w - rate[val]), .9)])
    print(name, 'per-lab complete-coupon p90 curve/rate:', np.round(results, 5).tolist(), 'worst:', np.round(np.max(results, axis=0), 5))
o = s.solve_alpha_m97_c01_batch_03(d)
e = o['predicted_strain'] - truth['te_clean']
r = o['initial_rate_pred'] - truth['te_true_rate0']
missing = ~np.isfinite(te['coupon_features_raw']).all(axis=1)
for label, m in [('complete', ~missing), ('imputed', missing)]:
    print(label, int(m.sum()), 'clean strain/rate RMSE', np.sqrt(np.mean(e[m]**2)), np.sqrt(np.mean(r[m]**2)), 'last-horizon strain RMSE', np.sqrt(np.mean(e[m, -1]**2)))
