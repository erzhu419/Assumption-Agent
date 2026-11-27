import importlib.util, pickle
import numpy as np
from scipy.optimize import minimize_scalar
s = importlib.util.spec_from_file_location('solver', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
i = pickle.load(open('/holdout_task/public_batch.pkl', 'rb'))[0]
d = i['dataset']; tr, te, dg = d['train'], d['test'], i['campaign_diagnostics']
t, dt = d['eval_times'], d['past_dt']; lab = dg['lab_id']; missing = np.isnan(dg['coupon_features_raw']).any(axis=1)
y = np.column_stack([dg['strain_curves_obs']-dg['strain_0'][:, None], dg['instant_rate_obs']])
xt, _ = m._features(tr, te)
empty = {k: dg[k][:0] for k in ('lab_id', 'stratum', 'coupon_features_raw', 'ref_coupons')}
xd, _ = m._features(dg, empty)
bridge = m._axial_load_map(d['calibration_load_cells'])
src, dst = m._specimen(tr, xt, bridge), m._specimen(dg, xd, bridge)
labels = ('production', 'free decay', 'rational drift', 'branch difference', 'missingness offset')
err = {k: np.empty_like(y) for k in labels}
for l in np.unique(lab):
    keep = tr['lab_id'] != l; h = lab == l; fit = ~h & ~missing
    tau, theta = m._fit(tuple(a[keep] for a in src), dt, t, (tr['strain_curves_obs']-tr['strain_0'][:, None])[keep], tr['instant_rate_obs'][keep])
    b, r = m._branches(dst, dt, t, tau)
    base = np.column_stack([b@theta[:2], r@theta[:2]])
    drift = dst[2]@theta[2:]; kernel = lambda a: np.r_[-np.expm1(-a*t)/a, 1.]
    pred = base + drift[:, None]*kernel(1.)
    err['production'][h] = (pred-y)[h]
    for name, candidate in [('free decay', lambda a: base+drift[:, None]*kernel(a)), ('rational drift', lambda a: base+(drift/(1+a*drift**2))[:, None]*kernel(1.))]:
        bounds = (.5, 1.5) if name == 'free decay' else (0., 4.)
        p = minimize_scalar(lambda a: np.mean((candidate(a)[fit]-y[fit])**2), bounds=bounds, method='bounded').x
        err[name][h] = (candidate(p)-y)[h]
    q = np.column_stack([b[:, :, 0]*theta[0]-b[:, :, 1]*theta[1], r[:, 0]*theta[0]-r[:, 1]*theta[1]])
    a = np.sum(q[fit]*(y-pred)[fit])/np.sum(q[fit]**2)
    err['branch difference'][h] = (pred+a*q-y)[h]
    q = missing[:, None]*kernel(1.); fit = ~h & missing
    a = np.sum(q[fit]*(y-pred)[fit])/np.sum(q[fit]**2)
    err['missingness offset'][h] = (pred+a*q-y)[h]
for k, e in err.items():
    for name, mask in [('complete', ~missing), ('missing', missing)]:
        slices = [mask & (lab == l) for l in np.unique(lab) if np.any(mask & (lab == l))]
        rmse = lambda x: np.sqrt(np.mean(x*x))
        print(k, name, 'pooled curve/rate', *(round(rmse(e[mask, z]), 6) for z in [slice(None,-1), -1]), 'worst lab curve/rate', *(round(max(rmse(e[h,z]) for h in slices), 6) for z in [slice(None,-1), -1]))
