import importlib.util
import pickle
from itertools import combinations

import numpy as np
from scipy.optimize import least_squares

spec = importlib.util.spec_from_file_location('solver', '/holdout_task/alpha_m97_c01_batch_05_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('/holdout_task/public_alpha_m97_c01_batch_05.pkl', 'rb') as f:
    data = pickle.load(f)[0]['dataset']
a = data['train']
past, hold, x, strain = m._inputs(data)
model = m._Model(data, past, hold, x)
n = len(a['lab_id'])
lab = a['lab_id']
active = model.act[:n] > 0
complete = np.isfinite(a['coupon_features_raw']).all(axis=1)
y = np.column_stack((a['strain_curves_obs'] - strain[:n, None], a['instant_rate_obs']))
families = {
    'base': np.zeros((n, 0)),
    'lab_drift': np.eye(4)[lab, 1:],
    'telemetry_control': a['aux_chamber_telemetry'],
    'pair_and_saturation': np.column_stack([x[:n,i]*x[:n,j] for i,j in combinations(range(4), 2)] + [hold[:n]/(1+hold[:n]**2)]),
}
# Pairwise linear sums/differences are already in the span of the four
# aligned coupon columns. Probe nonlinear terms and nuisance effects in drift.
fold = np.full(n, -1)
rng = np.random.default_rng(117)
for c in range(4):
    for g in (False, True):
        rows = rng.permutation(np.flatnonzero(complete & (lab == c) & (active == g)))
        fold[rows] = np.arange(len(rows)) % 4
errors = {name: np.full_like(y, np.nan) for name in families}
for f in range(4):
    tr = np.flatnonzero((fold >= 0) & (fold != f))
    va = np.flatnonzero(fold == f)
    t, w = m._fit(model, y, tr)
    tr_resid = y[tr] - model.design(t, tr) @ w
    va_resid = y[va] - model.design(t, va) @ w
    for name, features in families.items():
        errors[name][va] = va_resid
        if features.shape[1]:
            fit = least_squares(lambda b: (tr_resid - (features[tr] @ b)[:,None] * model.flow[tr]).ravel(), np.zeros(features.shape[1]), loss='cauchy', f_scale=.04)
            errors[name][va] -= (features[va] @ fit.x)[:,None] * model.flow[va]
for name, resid in errors.items():
    scores = []
    for c in range(4):
        for g in (False, True):
            q = complete & (lab == c) & (active == g)
            if not q.any():
                continue
            r = resid[q]
            scores.append([np.sqrt(np.mean(np.minimum(abs(r[:,:-1]),.1)**2)), np.sqrt(np.mean(r[:,-1]**2)), abs(np.median(r[:,-1]))])
    print(name, 'mean/worst slices: clipped strain RMSE, rate RMSE, absolute median rate bias', np.round(np.mean(scores,axis=0),6), np.round(np.max(scores,axis=0),6))
