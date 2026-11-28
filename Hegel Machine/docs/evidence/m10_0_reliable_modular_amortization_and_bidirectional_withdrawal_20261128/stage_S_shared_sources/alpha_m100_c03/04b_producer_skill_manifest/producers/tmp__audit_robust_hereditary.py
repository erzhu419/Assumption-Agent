import importlib.util
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('solver', '/holdout_task/batch_solver.py')
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
with open('/holdout_task/public_batch.pkl', 'rb') as f:
    d = pickle.load(f)[0]['dataset']
tr, te = d['train'], d['test']
cal = d['calibration_load_cells']
x = s._features(tr, te)[:len(tr['lab_id'])]
p, h = s._loads(tr, s._bridge(cal))
y = tr['strain_curves_obs'] - tr['strain_0'][:, None]
r = tr['instant_rate_obs']
dt, t = d['past_dt'], d['eval_times']
labs = tr['lab_id']
real_ls = s.least_squares
metrics = {}
for cutoff in (1.5, 2.5, 4.0, None):
    def ls(fun, x0, **kw):
        if kw['loss'] == 'cauchy':
            if cutoff is None:
                return real_ls(fun, x0, loss='linear', bounds=kw['bounds'], x_scale=kw['x_scale'], max_nfev=kw['max_nfev'])
            kw['f_scale'] = cutoff / 2.5
        return real_ls(fun, x0, **kw)
    s.least_squares = ls
    result = []
    for lab in np.unique(labs):
        m = labs != lab
        tau, theta = s._fit(x[m], p[m], h[m], dt, t, y[m], r[m])
        a, b = s._design(tau, x[~m], p[~m], h[~m], dt, t)
        ec, er = abs(a @ theta - y[~m]), abs(b @ theta - r[~m])
        result.append([max(np.quantile(ec[:, cols], .9) for cols in np.array_split(np.arange(len(t)), 3)), np.quantile(er, .9), np.median(ec), np.median(er), *tau])
    v = np.asarray(result)
    metrics[cutoff] = v
    print('Cauchy cutoff in robust SD', cutoff, 'worst p90 strain/rate', v[:, :2].max(axis=0), 'mean lab median strain/rate', v[:, 2:4].mean(axis=0), 'tau ranges', np.ptp(v[:, 4:], axis=0))
s.least_squares = real_ls
raw, true = cal['bridge_raw'], cal['true_channel_loads']
a = np.c_[raw, np.ones(len(raw))]
y_total = true.sum(axis=1)
# Contiguous stress folds, so the extreme held-out sweep groups test extrapolation.
order = np.argsort(y_total)
for i, test in enumerate(np.array_split(order, 5)):
    keep = np.ones(len(raw), dtype=bool)
    keep[test] = False
    affine = np.linalg.lstsq(a[keep], y_total[keep], rcond=None)[0]
    err = abs(a[test] @ affine - y_total[test])
    print('contiguous load fold', i, 'p90/max total stress error', np.quantile(err, .9), err.max())
