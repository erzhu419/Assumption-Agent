import importlib.util
import pickle
import numpy as np
from scipy.optimize import least_squares

p = '/holdout_task/batch_solver.py'
s = importlib.util.spec_from_file_location('solver', p)
m = importlib.util.module_from_spec(s)
s.loader.exec_module(m)
with open('/holdout_task/public_batch.pkl', 'rb') as f:
    item = pickle.load(f)[0]
d = item['dataset']; g = item['campaign_diagnostics']; dt, t = d['past_dt'], d['eval_times']
out = m.solve_batch(d); tau, b = out['estimated_taus'], out['hereditary_weights']
_, coupon = m._prepare(dict(d, test=g))
past, hold = coupon[:2]
x = m._aligned_features(g, g)[-len(hold):]
y, r = m._deployment(tau, b, dt, t, (past, hold, x))
e = g['strain_curves_obs'] - g['strain_0'][:, None] - y
f = g['instant_rate_obs'] - r
drift = x @ b[2:]
z = np.zeros((len(hold), 2))
for ds, stress in zip(dt, past.T):
    z += -np.expm1(-ds/tau) * (stress[:, None] - z)
u = t[:, None]/tau
h = (-np.expm1(-u) - u) @ m._CHAMBER_FORCING
rm = lambda a: np.sqrt(np.mean(a*a))

# Challenge the linear forcing with saturation and a signed contrast between the carried states.
# Fit one correction at a time using only the other labs. Neither changes the rate at zero.
lab = g['lab_id']; groups = np.unique(lab)
for name in ('baseline', 'saturation', 'state_contrast'):
    scores = []; parameters = []
    for k in groups:
        test = lab == k; train = ~test
        if name == 'baseline':
            pred = np.zeros_like(e)
        else:
            v = drift if name == 'saturation' else z[:, 0] - z[:, 1]
            v = v / np.sqrt(np.mean(v[train]**2))
            base = drift[:, None] * h
            def delta(a):
                factor = 1/(1+a*v*v)-1 if name == 'saturation' else a*v
                return base * factor[:, None]
            bounds = (0., 10.) if name == 'saturation' else (-2., 2.)
            a = least_squares(lambda p: ((delta(p[0])-e)[train]/.007).ravel(), [0.01], bounds=bounds, loss='soft_l1').x[0]
            parameters.append(a); pred = delta(a)
        err = e[test] - pred[test]
        scores.append([rm(err), rm(err[:, :4]), rm(err[:, -4:]), rm(f[test])])
    print(name, 'worst lab strain/early/late/rate', np.max(scores, axis=0).round(6), 'fold coefficients', np.round(parameters, 5))

# For this continuous task, compare pooled and within-lab residual slopes, not class accuracy.
def slopes(a, v):
    def fit(aa, vv):
        aa = aa-aa.mean(); vv = vv-vv.mean()
        return float(vv@aa/(vv@vv))
    aa = np.concatenate([a[lab==k] - a[lab==k].mean() for k in groups])
    vv = np.concatenate([v[lab==k] - v[lab==k].mean() for k in groups])
    return np.round([fit(a,v), fit(aa,vv), *[fit(a[lab==k],v[lab==k]) for k in groups]], 5)
for name, v in [('intrinsic drift', drift), ('carried-state contrast',z[:,0]-z[:,1])]:
    print(name, 'rate residual slopes (pooled, within, each lab)', slopes(f, v))
missing = np.isnan(g['coupon_features_raw']).any(axis=1)
shifted = np.max(np.abs(g['hold_channel_raw']), axis=1) > m._stress_calibration(d['calibration_load_cells'])[1]
for name, mask in [('missing', missing), ('hold bridge state', shifted)]:
    for s in (False, True):
        select = mask == s
        if np.any(select):
            print(name, s, 'n', select.sum(), 'strain/rate', round(rm(e[select]),6), round(rm(f[select]),6))
