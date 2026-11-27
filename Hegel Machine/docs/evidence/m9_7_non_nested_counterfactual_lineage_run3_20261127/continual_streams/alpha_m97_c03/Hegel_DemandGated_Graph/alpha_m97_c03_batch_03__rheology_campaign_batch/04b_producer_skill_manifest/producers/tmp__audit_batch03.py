import importlib.util
import pickle
import numpy as np
from scipy.optimize import least_squares

root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('solver', root + 'alpha_m97_c03_batch_03_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
d = pickle.load(open(root + 'public_alpha_m97_c03_batch_03.pkl', 'rb'))[0]['dataset']
tr = d['train']
n = len(tr['lab_id'])
dt, t = d['past_dt'], d['eval_times']
p, h, x, b = [a[:n] for a in m._prepare(d)]
y, r = tr['strain_curves_obs'] - b[:, None], tr['instant_rate_obs']
missing = ~np.isfinite(tr['coupon_features_raw']).all(axis=1)
lab = np.asarray(tr['lab_id'])
for field in ('train', 'test'):
    print(field, 'fields', sorted(d[field]))
print('dataset fields', sorted(d), 'shear banding index', d['shear_banding_index'])
for g in np.unique(lab):
    print('lab', g, 'rows', (lab == g).sum(), 'missing', ((lab == g) & missing).sum(), 'positive stress fraction', np.mean(h[lab == g] > 0))

modes = ('specified', 'linear flow', 'no branch contrast', 'no signed slip')
err = {name: [np.empty_like(y), np.empty_like(r)] for name in modes}
for g in np.unique(lab):
    train, val = lab != g, lab == g
    tau, w = m._fit(p[train], h[train], x[train], dt, t, y[train], r[train])
    a, q = m._design(tau, p, h, x, dt, t)
    # Audit the two nontrivial constitutive bases separately, not merely their sum.
    np.testing.assert_allclose(a[:, :, 2], t[None, :] / (1 + .85*t[None, :]) * np.ones((n, 1)))
    np.testing.assert_allclose(q[:, 2], 1.)
    decay = -np.expm1(-t/tau[0])
    gap_difference = (q[:, 0] * tau[0] - q[:, 1] * tau[1])
    np.testing.assert_allclose(a[:, :, 8], gap_difference[:, None] * decay[None, :], atol=1e-12)
    np.testing.assert_allclose(q[:, 8], gap_difference / tau[0], atol=1e-12)
    yc = train[:, None] & np.isfinite(y)
    yr = train & np.isfinite(r)
    targets = np.r_[y[yc], r[yr]]
    residual = np.r_[(a @ w - y)[yc], (q @ w - r)[yr]]
    def mad(v):
        return max(1e-6, 1.4826*np.median(np.abs(v-np.median(v))))
    scale = np.r_[np.full(yc.sum(), 2*mad(residual[:yc.sum()])), np.full(yr.sum(), 2*mad(residual[yc.sum():]))]
    for name in modes:
        A, Q = a.copy(), q.copy()
        cols = np.arange(9)
        if name == 'linear flow':
            A[:, :, 2:8] *= (1 + .85*t[None, :, None])
        if name == 'no branch contrast':
            cols = cols[cols != 8]
        if name == 'no signed slip':
            cols = cols[cols != 7]
        A, Q = A[:, :, cols], Q[:, cols]
        design = np.vstack((A[yc], Q[yr]))
        fit = w if name == 'specified' else least_squares(lambda v: (design @ v - targets)/scale, w[cols], loss='cauchy', x_scale='jac').x
        err[name][0][val] = (A @ fit - y)[val]
        err[name][1][val] = (Q @ fit - r)[val]

def med(a):
    return float(np.nanmedian(np.abs(a)))
for name, (e, f) in err.items():
    print(name, 'pooled curve/rate medAE', round(med(e), 5), round(med(f), 5), 'worst lab curve/rate medAE', round(max(med(e[lab == g]) for g in np.unique(lab)), 5), round(max(med(f[lab == g]) for g in np.unique(lab)), 5))
e, f = err['specified']
for g in np.unique(lab):
    for mask, label in ((~missing, 'complete'), (missing, 'missing')):
        rows = (lab == g) & mask
        if rows.any():
            print('held-out lab', g, label, 'n', rows.sum(), 'signed curve/rate medians', np.round([np.nanmedian(e[rows]), np.nanmedian(f[rows])], 5), 'rate medAE', round(med(f[rows]), 5))
