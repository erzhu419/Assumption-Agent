import importlib.util
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('candidate', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('/holdout_task/public_batch.pkl', 'rb') as f:
    p = pickle.load(f)[0]
d = p['dataset']; q = p['campaign_diagnostics']; t = d['eval_times']; dt = d['past_dt']
out = m.solve_batch(d)
taus = out['estimated_taus']; theta = out['hereditary_weights']
cal = d['calibration_load_cells']; loads = cal['true_channel_loads']; raw = cal['bridge_raw']
i, j = np.triu_indices(4)

def design(x, quadratic):
    a = np.c_[np.ones(len(x)), x]
    return np.c_[a, x[:, i]*x[:, j]] if quadratic else a

rng = np.random.default_rng(17)
folds = np.array_split(rng.permutation(len(loads)), 5)
for quadratic in (False, True):
    err = np.empty(len(loads))
    for fold in folds:
        keep = np.ones(len(loads), dtype=bool); keep[fold] = False
        fit = np.linalg.lstsq(design(raw[keep], quadratic), loads[keep].sum(axis=1), rcond=None)[0]
        err[fold] = design(raw[fold], quadratic) @ fit - loads[fold].sum(axis=1)
    print('bridge inverse diagnostic quadratic=', quadratic, 'RMSE', np.sqrt(np.mean(err**2)), 'worst', max(abs(err)))

x = q['coupon_features_raw'].copy(); lab = q['lab_id']
for g in np.unique(lab):
    rows = lab == g; r = q['ref_coupons'][rows]
    x[rows] = (x[rows] - np.median(r.mean(0))) / np.median(r.std(0))
x = np.where(np.isnan(x), np.nanmean(x, axis=0), x)
drift = theta[2] + x @ theta[3:]
off, w = m._calibration(cal)
h = (q['hold_channel_raw'] - off) @ w
past = (q['past_channel_raw'] - off) @ w
obs = np.c_[q['strain_curves_obs']-q['strain_0'][:, None], q['instant_rate_obs']]
b = np.r_[-np.expm1(-t[:, None] / taus), [1/taus]]
f = b * taus
base = (h[:, None]*theta[:2]) @ b.T + drift[:, None]*f[:, 1]
res = obs - base
partition = drift[:, None]*(f[:, 0]-f[:, 1])
z = np.zeros((len(h), 2))
for s, delta in enumerate(dt):
    decay = np.exp(-delta/taus)
    z = decay*z + (1-decay)*past[:, s, None]
contrast = (past[:, -1]-past[:, 0])[:, None]
models = {
    'conserved split': [partition],
    'retained-memory challenger': [partition, (z*theta[:2]) @ b.T],
    'pairwise history contrast challenger': [partition, contrast*b[:, 0], contrast*b[:, 1]],
    'rational split challenger': [partition, partition*drift[:, None]**2/(1+drift[:, None]**2)],
    'rational gain challenger': [partition, drift[:, None]*f[:, 1]*drift[:, None]**2/(1+drift[:, None]**2)],
}
for name, terms in models.items():
    a = np.stack(terms, axis=-1); error = np.empty_like(res)
    for g in np.unique(lab):
        keep = lab != g
        fit = np.linalg.lstsq(a[keep].reshape(-1, len(terms)), res[keep].ravel(), rcond=None)[0]
        error[~keep] = a[~keep] @ fit - res[~keep]
    print(name, 'worst lab strain/rate RMSE', np.round(np.max([[np.sqrt(np.mean(error[lab==g, :-1]**2)), np.sqrt(np.mean(error[lab==g, -1]**2))] for g in np.unique(lab)], axis=0),6), 'worst coupon strain RMSE/rate abs', np.round([max(np.sqrt(np.mean(error[:, :-1]**2, axis=1))), max(abs(error[:, -1]))],6))
