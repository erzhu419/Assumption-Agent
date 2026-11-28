import importlib.util, json, pickle
import numpy as np
from scipy.optimize import least_squares

s = importlib.util.spec_from_file_location('solver', '/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(s)
s.loader.exec_module(m)
item = pickle.load(open('/holdout_task/public_batch.pkl', 'rb'))[0]
d = item['dataset']; tr = d['train']; te = d['test']
lab = tr['lab_id']; labs = np.unique(lab)
dt, t = d['past_dt'], d['eval_times']
p, h = m._loads(tr, m._bridge(d['calibration_load_cells']))
x = m._features(tr, te)[:len(h)]
y = tr['strain_curves_obs'] - tr['strain_0'][:, None]; r = tr['instant_rate_obs']

def fit(mask):
    tau, w = m._fit(p[mask], h[mask], x[mask], dt, t, y[mask], r[mask])
    c, z = m._design(tau, p, h, x, dt, t)
    return tau, c, z, c @ w - y, z @ w - r

def bases(tau, c, z, mask, name):
    if name == 'saturation':
        gap = z[:, :2] * tau
        unit = np.maximum(np.sqrt(np.mean(gap[mask]**2, axis=0)), 1e-12)
        f = 1 / (1 + (gap / unit)**2) - 1
        return c[:, :, :2] * f[:, None, :], z[:, :2] * f
    if name == 'interactions':
        i, j = np.triu_indices(x.shape[1], 1)
        f = np.column_stack((x[:, i] * x[:, j], (h-p[:, -1]) * x[:, 0] * x[:, 1]))
        return t[None, :, None] * f[:, None, :], f
    if name == 'lab_centered':
        g = np.column_stack((np.ones(len(h)), x))
        for l in labs:
            q = mask & (lab == l)
            if q.any():
                g[q] -= g[q].mean(axis=0)
        return t[None, :, None] * g[:, None, :], g
    raise ValueError(name)

def predict(fit_result, train, test, name):
    tau, c, z, ec, er = fit_result
    if name == 'baseline':
        return ec[test], er[test]
    u, v = bases(tau, c, z, train, name)
    def scale(a):
        return 1.4826 * np.median(abs(a-np.median(a)))
    sc, sr = scale(ec[train]), scale(er[train]); k = v.shape[1]
    if name == 'lab_centered':
        # Fit slopes within each training lab, then set a shared intercept.
        # Never estimate a nuisance offset from a held-out lab's responses.
        ec, er = ec.copy(), er.copy()
        for l in labs:
            q = train & (lab == l)
            if q.any():
                ec[q] -= np.median(ec[q], axis=0)
                er[q] -= np.median(er[q])
    a = np.vstack((u[train].reshape(-1, k)/sc, v[train]/sr))
    b = np.r_[ec[train].ravel()/sc, er[train]/sr]
    w = least_squares(lambda w: np.r_[a @ w+b, 2*w], np.zeros(k), loss='soft_l1', f_scale=1.5).x
    return ec[test]+u[test]@w, er[test]+v[test]@w

def metrics(errors):
    ec, er = errors
    return np.array([np.quantile(abs(ec), .9), np.quantile(abs(er), .9)])

names = ('baseline', 'saturation', 'interactions', 'lab_centered')
results = {n: [] for n in (*names, 'nested')}
cache = {}
for l in labs:
    trn = lab != l; tst = ~trn
    inner = {n: [] for n in names}
    for hld in labs[labs != l]:
        a = trn & (lab != hld); b = lab == hld
        key = tuple(sorted((l,hld)))
        if key not in cache: cache[key] = fit(a)
        ref = metrics(predict(cache[key], a, b, 'baseline'))
        for name in names:
            inner[name].append(metrics(predict(cache[key], a, b, name))/ref)
    selected = min(names, key=lambda n: np.max(inner[n]))
    basefit = fit(trn)
    for name in names:
        results[name].append(metrics(predict(basefit, trn, tst, name)))
    results['nested'].append(metrics(predict(basefit, trn, tst, selected)))
    print('outer lab', l, 'selected', selected, 'baseline signed median strain/rate', np.median(basefit[3][tst]), np.median(basefit[4][tst]))
for name, rows in results.items():
    print(name, 'per lab p90', np.round(rows, 6).tolist(), 'worst', np.max(rows, axis=0))
print('archive prior fit count:', len(json.load(open('/tmp/hegel_continual_archive/archive_manifest.json'))), 'chamber observations:', item.get('campaign_diagnostics') is not None)
