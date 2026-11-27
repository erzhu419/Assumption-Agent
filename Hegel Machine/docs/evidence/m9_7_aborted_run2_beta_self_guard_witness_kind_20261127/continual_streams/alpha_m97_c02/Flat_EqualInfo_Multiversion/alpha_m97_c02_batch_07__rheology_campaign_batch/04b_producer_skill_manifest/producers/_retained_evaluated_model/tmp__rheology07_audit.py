import importlib.util
import pickle
import numpy as np

root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('solver', root + 'alpha_m97_c02_batch_07_solver.py')
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
with open(root + 'public_alpha_m97_c02_batch_07.pkl', 'rb') as f:
    item = pickle.load(f)[0]
d = item['dataset']
a, b = d['train'], d['test']
dt, t = d['past_dt'], d['eval_times']
x, _ = s._aligned(a, b)
inputs = s._inputs(a, x, s._calibration(d), dt, d['crit_stress'], d['shear_banding_index'])
active = inputs[-1] > 0
complete = np.isfinite(a['coupon_features_raw']).all(axis=1)
lab = a['lab_id']
y = a['strain_curves_obs'] - a['strain_0'][:, None]
r = a['instant_rate_obs']
original = s._design

def switched(tau, dt, times, inp):
    curve, rate = original(tau, dt, times, inp)
    inactive = inp[-1] == 0
    curve[inactive, :, 2:7] *= (times / np.log1p(times))[None, :, None]
    return curve, rate

def summarize(e, er, rows):
    if not rows.any():
        return []
    return [int(rows.sum()), *np.round([np.quantile(np.abs(e[rows]), .90), np.quantile(np.abs(er[rows]), .90), np.median(e[rows, -1]), np.median(er[rows])], 5).tolist()]

for name, design in [('log throughout', original), ('linear at zero damage', switched)]:
    s._design = design
    e, er = np.empty_like(y), np.empty_like(r)
    for c in np.unique(lab):
        fit, val = (lab != c) & complete, lab == c
        tau, w = s._fit(dt, t, tuple(v[fit] for v in inputs), y[fit], r[fit])
        C, R = design(tau, dt, t, tuple(v[val] for v in inputs))
        e[val], er[val] = C @ w - y[val], R @ w - r[val]
    print(name, 'n, p90 strain, p90 rate, signed median final strain, signed median rate')
    for state in (False, True):
        for c in np.unique(lab):
            print('  lab', c, 'active', state, summarize(e, er, (lab == c) & (active == state) & complete))
    print('  missing coupons:', summarize(e, er, ~complete))
    print('  p90 at each horizon (complete):', np.round(np.quantile(np.abs(e[complete]), .90, axis=0), 5).tolist())
s._design = original
out = s.solve_alpha_m97_c02_batch_07(d)
gt = item.get('gt')
print('Public reference fields:', {k: np.shape(v) for k, v in gt.items()} if isinstance(gt, dict) else type(gt).__name__)
if isinstance(gt, dict):
    for k in out.keys() & gt.keys():
        ref, pred = np.asarray(gt[k]), np.asarray(out[k])
        if ref.shape == pred.shape and ref.dtype.kind in 'fiu':
            print('public reference', k, 'RMSE', float(np.sqrt(np.mean((ref-pred)**2))), 'max abs error', float(np.max(np.abs(ref-pred))))
