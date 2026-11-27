import importlib.util
import pickle
from pathlib import Path
import numpy as np

root = Path('/holdout_task')
spec = importlib.util.spec_from_file_location('candidate', root/'alpha_m97_c01_batch_06_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with (root/'public_alpha_m97_c01_batch_06.pkl').open('rb') as f:
    d = pickle.load(f)[0]['dataset']
train, test = d['train'], d['test']
inputs = tuple(test)
lab, st = train['lab_id'], train['stratum']
missing = ~np.isfinite(train['coupon_features_raw']).all(axis=1)

def rmse(x):
    return np.sqrt(np.mean(x*x))

for title, group in [('lab', lab), ('stratum', st)]:
    curve_err = np.empty_like(train['strain_curves_obs'])
    rate_err = np.empty_like(train['instant_rate_obs'])
    for value in np.unique(group):
        valid = group == value
        fold = dict(d, train={k: v[~valid] for k, v in train.items()},
                    test={k: np.concatenate((test[k], train[k][valid])) for k in inputs})
        pred = m.solve_alpha_m97_c01_batch_06(fold)
        n = len(test['lab_id'])
        curve_err[valid] = pred['predicted_strain'][n:] - train['strain_curves_obs'][valid]
        rate_err[valid] = pred['initial_rate_pred'][n:] - train['instant_rate_obs'][valid]
        print(title, value, 'tau', np.round(pred['estimated_taus'], 4), 'curve/rate RMSE', *(round(v, 5) for v in (rmse(curve_err[valid]), rmse(rate_err[valid]))))
    if title == 'lab':
        for h in range(curve_err.shape[1]):
            e = curve_err[:,h]
            med = np.median(e)
            print('horizon', h, 'median', round(med,5), 'MAD', round(np.median(abs(e-med)),5), 'abs q90/q95/q99/max', np.round(np.quantile(abs(e), [.90,.95,.99,1]),5))
        idx = np.argwhere(abs(curve_err) > .25)
        print('large curve residuals:', len(idx), 'of', curve_err.size)
        for i,j in sorted(idx,key=lambda ij:-abs(curve_err[tuple(ij)]))[:18]:
            print('large', int(i), int(j), 'lab/stratum', int(lab[i]),int(st[i]), 'obs',round(train['strain_curves_obs'][i,j],4), 'residual',round(curve_err[i,j],4), 'other-time median abs',round(np.median(abs(np.delete(curve_err[i],j))),4), 'rate residual',round(rate_err[i],4))
        print('rate absolute q50/q90/q95/q99/max: ', np.round(np.quantile(abs(rate_err), [.50,.90,.95,.99,1]),5))
    for label, take in [('complete', ~missing), ('imputed', missing)]:
        print(title, label, 'curve/rate RMSE', *(round(v, 5) for v in (rmse(curve_err[take]), rmse(rate_err[take]))))
    worst = max((rmse(curve_err[(lab==i)&(st==j)]), i, j) for i,j in set(zip(lab,st)))
    worst_rate = max((rmse(rate_err[(lab==i)&(st==j)]), i, j) for i,j in set(zip(lab,st)))
    horizons = np.sqrt(np.mean(curve_err**2, axis=0))
    print(title, 'worst cell curve/rate (RMSE, lab, stratum)', worst, worst_rate, 'worst horizon', float(horizons.max()), int(horizons.argmax()))

# Independent vectorized memory and zero-time derivative, with reversed and near-zero holds.
rng = np.random.default_rng(906)
dt = rng.uniform(.1,.9,18)
t = np.array([0.,1e-7,.3,4.7])
tau = np.array([.57,3.9])
past = rng.normal(size=(5,18))
hold = np.array([-2.,-1e-8,0.,1e-8,2.])
x = rng.normal(size=(5,4))
w = np.array([.66,.55,.14,.16,-.12,.12,-.09,.3,-.2])
future_dt = np.r_[np.cumsum(dt[:0:-1])[::-1],0.]
mem = past @ ((1-np.exp(-dt[:,None]/tau))*np.exp(-future_dt[:,None]/tau))
g0,g1 = (hold[:,None]-mem).T
phi = t/(1+.85*t)
flow = w[2]+x@w[3:7]+w[7]*np.copysign(np.sqrt(abs(hold)),hold)
ref = np.outer(w[0]*g0+w[8]*(g0-g1), 1-np.exp(-t/tau[0])) + np.outer(w[1]*g1,1-np.exp(-t/tau[1]))+np.outer(flow,phi)
c,r = m._design(tau,dt,t,past,hold,x)
np.testing.assert_allclose(c@w,ref,atol=2e-14)
deriv_error = np.max(np.abs((c[:,1]@w)/t[1]-r@w))
assert deriv_error < 1e-6
print('independent kernel maximum error', np.max(np.abs(c@w-ref)), 'zero-time finite-difference maximum error', deriv_error)
