import importlib.util
import pickle
import numpy as np

root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('solver', root + 'alpha_m97_c02_batch_06_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open(root + 'public_alpha_m97_c02_batch_06.pkl', 'rb') as f:
    d = pickle.load(f)[0]['dataset']
tr, te = d['train'], d['test']
cal = m._calibration(d)
xt, _ = m._aligned(tr, te)
raw = tr['coupon_features_raw']
complete = np.isfinite(raw).all(axis=1)
lab = tr['lab_id']
dt, times = d['past_dt'], d['eval_times']
y = tr['strain_curves_obs'] - tr['strain_0'][:, None]
r = tr['instant_rate_obs']
for name, x in [('reference-aligned', xt), ('unaligned control', raw)]:
    inputs = m._inputs(tr, x, cal)
    all_metrics = []
    for held in np.unique(lab):
        train = complete & (lab != held)
        test = complete & (lab == held)
        tau, w = m._fit(dt, times, tuple(v[train] for v in inputs), y[train], r[train])
        a, b = m._design(tau, dt, times, tuple(v[test] for v in inputs))
        error = a @ w - y[test]
        rate_error = b @ w - r[test]
        metric = np.array([np.median(abs(error)), np.median(abs(rate_error)), abs(np.median(error)), abs(np.median(rate_error))])
        print(name, 'held lab', held, 'median absolute strain/rate and absolute median bias', np.round(metric, 6))
        all_metrics.append(metric)
    print(name, 'worst held-lab metrics', np.round(np.max(all_metrics, axis=0), 6))
