import importlib.util
import pickle
import numpy as np

path = '/holdout_task/alpha_m96_w02_batch_04_solver.py'
spec = importlib.util.spec_from_file_location('coupon', path)
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
with open('/holdout_task/public_alpha_m96_w02_batch_04.pkl', 'rb') as f:
    d = pickle.load(f)[0]['dataset']
tr, te = d['train'], d['test']
cal = s._calibrate(d['calibration_load_cells'])
x = s._align(tr, te)[:len(tr['lab_id'])]
lab = tr['lab_id']
full = s._Design(d, tr, x, cal)
y = np.column_stack((tr['strain_curves_obs'] - tr['strain_0'][:, None], tr['instant_rate_obs']))
res = np.empty_like(y)
for c in range(4):
    m = lab != c
    sub = {k: v[m] for k, v in tr.items()}
    tau, weights = s._fit(s._Design(d, sub, x[m], cal), sub)
    res[~m] = full.make(tau)[~m] @ weights - y[~m]
    print('held lab', c, 'taus', np.round(tau, 5), 'plastic', np.round(weights[7:], 6))

def metrics(mask):
    err = res[mask]
    if len(err):
        return tuple(np.round([np.sqrt(np.nanmean(err[:, :-1]**2)), np.sqrt(np.nanmean(err[:, -1]**2)), np.nanmedian(np.abs(err[:, :-1])), np.nanmedian(np.abs(err[:, -1]))], 6))

h, o = full.hist != 0, full.base[:, -1, 7] != 0
print('metric order: strain RMSE, onset RMSE, strain median absolute error, onset median absolute error')
for c in range(4):
    for name, use in [('subyield', ~h & ~o), ('history-only', h & ~o), ('hold-active', o)]:
        m = (lab == c) & use
        print(c, name, int(m.sum()), metrics(m))
missing = np.isnan(tr['coupon_features_raw']).any(axis=1)
print('missing features', int(missing.sum()), metrics(missing), 'complete', metrics(~missing))
print('horizon RMSE', np.round(np.sqrt(np.nanmean(res[:, :-1]**2, axis=0)), 6))

checks = d['calibration_load_cells']
for c in range(4):
    own = np.flatnonzero(checks['lab_id'] == c)
    for case in ('single spot', 'three EMI spikes'):
        keep = (checks['lab_id'] != c) | (np.arange(len(checks['lab_id'])) == own[len(own)//2]) if case == 'single spot' else np.ones(len(checks['lab_id']), bool)
        corrupt = {k: v[keep].copy() for k, v in checks.items()}
        if case == 'three EMI spikes':
            corrupt['true_stress'][own[:3]] += np.array([.27, -.35, .42])
        g,b = s._calibrate(corrupt)[c] - cal[c]
        worst = max(abs(b-3*g), abs(b+3*g))
        print('calibration', c, case, 'max stress delta', round(worst, 6))
        assert worst < .002
