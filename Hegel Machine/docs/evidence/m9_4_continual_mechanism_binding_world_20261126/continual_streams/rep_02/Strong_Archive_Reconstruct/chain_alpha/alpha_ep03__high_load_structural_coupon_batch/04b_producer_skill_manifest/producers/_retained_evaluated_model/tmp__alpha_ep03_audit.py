import importlib.util
import pickle
from itertools import combinations
import numpy as np
from scipy.optimize import least_squares

spec = importlib.util.spec_from_file_location('alpha_ep03', '/holdout_task/alpha_ep03_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('/holdout_task/public_alpha_ep03.pkl', 'rb') as f:
    d = pickle.load(f)[0]['dataset']
tr, _, y, complete = m._prepare(d)
x = m._aligned_panels(d)[0][:len(y)]
c = complete[:len(y)]
lab = d['train']['lab_id']
stratum = d['train']['stratum']
sigma = d['yield_stress']
high = (np.abs(tr.past).max(axis=1) > sigma) | (np.abs(tr.hold) > sigma)
positive = tr.hold > 0
overstress = np.sign(tr.hold) * np.maximum(np.abs(tr.hold) - sigma, 0)**2
saturation = 1 + (tr.hold / sigma)**2
features = {
    'coupon_products': np.column_stack([x[:, i]*x[:, j] for i, j in combinations(range(4), 2)] + [x[:, i]*x[:, j]*x[:, k] for i, j, k in combinations(range(4), 3)]),
    'stress_saturation': np.column_stack([tr.hold/saturation, overstress/saturation, tr.plastic_memory/saturation]),
}
# Linear sums/differences are already in the span of the four aligned features.
# These fixed challenger families add genuinely different drift columns.
errors = {name: np.empty_like(y[:, :-1]) for name in ('baseline', *features)}
rates = np.empty(len(y))
for k in np.unique(lab):
    held = lab == k
    fit = c & ~held
    tau, beta = m._fit(tr, y, fit)
    a = tr.matrix(np.log(tau))
    for name, out in errors.items():
        if name == 'baseline':
            p = a[held] @ beta
            rates[held] = p[:, -1] - y[held, -1]
        else:
            ae = np.concatenate([a, features[name][:, None, :] * tr.linear_time[None, :, None]], axis=2)
            scale = m._noise(a[fit] @ beta - y[fit])
            mat = (ae[fit] / scale[None, :, None]).reshape(-1, ae.shape[2])
            obs = (y[fit] / scale).ravel()
            b = least_squares(lambda b: mat @ b - obs, np.r_[beta, np.zeros(features[name].shape[1])], jac=lambda b: mat.copy(), loss='cauchy', f_scale=2, x_scale='jac').x
            p = ae[held] @ b
        innovation = y[held, -1] - p[:, -1]
        out[held] = p[:, :-1] + np.where(c[held], .25, 1.)[:, None]*innovation[:, None]*tr.times - y[held, :-1]
groups = [(lab == k) & (high == h) & (positive == s) for k in np.unique(lab) for h in (False, True) for s in (False, True)]
for name, e in errors.items():
    print(name, 'worst signed load x lab RMSE / median abs on complete rows:', round(max(np.sqrt(np.mean(e[g]**2)) for g in groups if g.any()), 6), round(max(np.median(np.abs(e[g & c])) for g in groups if (g & c).any()), 6), 'pooled median abs:', round(np.median(np.abs(e)), 6), 'missing median abs:', round(np.median(np.abs(e[~c])), 6))
e = errors['baseline']
cells = [(lab == k) & (stratum == j) & (high == h) & c for k in np.unique(lab) for j in np.unique(stratum) for h in (False, True)]
print('signed median strain residual by lab (low/high):', [[round(np.median(e[(lab == k) & (high == h) & c]), 6) for h in (False, True)] for k in np.unique(lab)])
print('maximum absolute signed median strain residual by lab x stratum x load (>=4 complete coupons):', round(max(abs(np.median(e[g])) for g in cells if g.sum() >= 4), 6))
print('median absolute onset residual by lab:', [round(np.median(abs(rates[(lab == k) & c])), 6) for k in np.unique(lab)])
print('baseline median absolute residual by horizon:', np.round(np.median(abs(e), axis=0), 6))
print('baseline absolute residual above 0.2 by horizon:', (abs(e) > .2).sum(axis=0).tolist())
