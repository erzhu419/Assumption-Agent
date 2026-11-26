import importlib.util
import pickle
import numpy as np

path = '/holdout_task/alpha_w03_batch_04_solver.py'
spec = importlib.util.spec_from_file_location('solver', path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
case = pickle.load(open('/holdout_task/public_alpha_w03_batch_04.pkl', 'rb'))[0]
d, gt = case['dataset'], case['gt']
train, test = d['train'], d['test']
x, h, p = m._prep(d)
n = len(train['lab_id'])
dt, t, yld = d['past_dt'], d['eval_times'], d['yield_stress']
y = train['strain_curves_obs'] - train['strain_0'][:, None]
r = train['instant_rate_obs']
missing = np.r_[~np.isfinite(train['coupon_features_raw']).all(axis=1), ~np.isfinite(test['coupon_features_raw']).all(axis=1)]

def fit(mask):
    ids = np.flatnonzero(mask)
    return m._fit(x[ids], h[ids], p[ids], dt, t, yld, y[ids], r[ids])

def predict(tau, weights, ids, observed, anchor):
    c, a = m._design(tau, x[ids], h[ids], p[ids], dt, t, yld)
    curve, rate = c @ weights, a @ weights
    if anchor:
        use = missing[ids] & np.isfinite(observed)
        curve[use] += (observed[use] - rate[use])[:, None] * t
        rate[use] = observed[use]
    return curve, rate

def rmse(z):
    return np.sqrt(np.mean(z[np.isfinite(z)]**2))

for complete_only, anchor in [(False, False), (False, True), (True, True)]:
    print('\ncomplete-only training:', complete_only, 'missing-row rate anchor:', anchor)
    usable = ~missing[:n] if complete_only else np.ones(n, dtype=bool)
    for lab in range(4):
        val = train['lab_id'] == lab
        ids = np.flatnonzero(val)
        tau, weights = fit(usable & ~val)
        c, rate = predict(tau, weights, ids, r[val], anchor)
        err = c-y[val]
        print('LOLO', lab, 'curve RMSE', rmse(err), 'complete rate RMSE', rmse((rate-r[val])[~missing[ids]]), 'missing curve RMSE', rmse(err[missing[ids]]))
    tau, weights = fit(usable)
    ids = np.arange(n, len(x))
    c, rate = predict(tau, weights, ids, test['instant_rate_obs'], anchor)
    err_c = c+test['strain_0'][:, None]-gt['te_clean']
    err_r = rate-gt['te_true_rate0']
    over = (np.abs(h[n:])>yld) | (np.abs(p[n:]).max(axis=1)>yld)
    print('tau, weights', tau, weights)
    for name, mask in [('all', np.ones(len(ids), bool)), ('missing', missing[ids]), ('complete', ~missing[ids]), ('above', over), ('below', ~over), *[(f'lab{lab}', test['lab_id']==lab) for lab in range(4)]]:
        if mask.any():
            print('public', name, 'curve/rate RMSE', rmse(err_c[mask]), rmse(err_r[mask]), 'final-time RMSE', rmse(err_c[mask,-1]))
