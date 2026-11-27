import importlib.util
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('solver', '/holdout_task/alpha_m97_c01_batch_01_solver.py')
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
with open('/holdout_task/public_alpha_m97_c01_batch_01.pkl', 'rb') as f:
    d = pickle.load(f)[0]['dataset']
q = d['train']
lab = q['lab_id']
n = len(lab)
a = tuple(v[:n] for v in s._prepare(d))
dt, t = d['past_dt'], d['eval_times']
y = q['strain_curves_obs'] - q['strain_0'][:, None]
r = q['instant_rate_obs']
e = np.empty_like(y)
er = np.empty_like(r)
for j in np.unique(lab):
    m = lab == j
    train = tuple(v[~m] for v in a)
    test = tuple(v[m] for v in a)
    p, w = s._fit(dt, t, train, y[~m], r[~m])
    c, b = s._basis(p, dt, t, test)
    e[m] = c @ w - y[m]
    er[m] = b @ w - r[m]
act = a[-1]
shear_all = np.asarray(d['shear_banding_index'])
shear = np.broadcast_to(shear_all, (n,)) if shear_all.ndim == 0 else shear_all[:n]
if shear_all.ndim == 0:
    print('Batch-wide shear index:', shear_all.item(), '; no within-batch shear contrast')
missing = np.isnan(q['coupon_features_raw']).any(axis=1)
slices = {'no missing features': ~missing, 'imputed features': missing,
          'partial activation': (act > 0) & (act < .95), 'saturated activation': act >= .95,
          'near gate, lower side': (shear >= .4) & (shear <= .5),
          'near gate, upper side': (shear > .5) & (shear <= .6)}
slices.update({f'stratum {v}': q['stratum'] == v for v in np.unique(q['stratum'])})
for name, m in slices.items():
    if m.any():
        print(name, 'n=', m.sum(), 'curve/rate median abs, worst horizon median abs and bias=',
              np.round([np.median(abs(e[m])), np.median(abs(er[m])),
                        np.max(np.median(abs(e[m]), axis=0)),
                        np.max(abs(np.median(e[m], axis=0)) )], 5))
    else:
        print(name, 'n=0; unsupported by public labels')
