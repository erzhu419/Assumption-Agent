import pickle, runpy
import numpy as np
from scipy.optimize import minimize_scalar

root = '/holdout_task/'
m = runpy.run_path(root + 'batch_solver.py')
d = pickle.load(open(root + 'public_batch.pkl', 'rb'))[0]['dataset']
c = d['calibration_load_cells']
x, y = c['true_channel_loads'], c['bridge_raw']
basis, fit, recover = (m[k] for k in ('_bridge_basis', '_calibration', '_recover_loads'))
a = basis(x)
folds = np.array_split(np.random.default_rng(230).permutation(len(x)), 5)
errors = {k: [] for k in ('sparse_cubic', 'pairwise_quotient', 'dense_cubic')}
strengths = []
for fold in folds:
    train = np.ones(len(x), dtype=bool)
    train[fold] = False
    b = fit({'bridge_raw': y[train], 'true_channel_loads': x[train]})
    residual = np.linalg.norm(a @ b - y, axis=1)
    clean = train & (residual < 0.01)
    test = fold[residual[fold] < 0.01]
    errors['sparse_cubic'].extend(np.linalg.norm(a[test] @ b - y[test], axis=1))
    dense = np.linalg.lstsq(a[clean], y[clean], rcond=None)[0]
    errors['dense_cubic'].extend(np.linalg.norm(a[test] @ dense - y[test], axis=1))
    pred = np.empty_like(y[test])
    for j in range(4):
        use = b[:, j] != 0
        p = a[:, use]
        nxt = x[:, (j + 1) % 4]
        qs = (x[:, j], nxt, x[:, j] - nxt, x[:, j] + nxt)
        def project(q, strength):
            z = p / (1 + strength * q[:, None]**2)
            w = np.linalg.lstsq(z[clean], y[clean, j], rcond=None)[0]
            return np.mean((z[clean] @ w - y[clean, j])**2), z[test] @ w
        candidates = [(project(qs[0], 0), 0.)]
        for q in qs:
            opt = minimize_scalar(lambda s: project(q, s)[0], bounds=(0, 1), method='bounded', options={'xatol': 1e-9})
            candidates.append((project(q, opt.x), opt.x))
        (_, pred[:, j]), s = min(candidates, key=lambda v: v[0][0])
        strengths.append(s)
    errors['pairwise_quotient'].extend(np.linalg.norm(pred - y[test], axis=1))
for k, e in errors.items():
    print(k, 'clean OOF forward bridge error p95/max', np.percentile(e, [95, 100]))
print('quotient denominator strength median/max', np.percentile(strengths, [50, 100]))
b = fit(c)
print('calibration inverse axial worst by applied load sign:', [(str(sign), float(np.max(abs(recover(y[mask], b).sum(1) - x[mask].sum(1))))) for sign, mask in [(sign, (np.sign(x.sum(1)) == sign) & (np.linalg.norm(a @ b - y, axis=1) < 0.01)) for sign in np.unique(np.sign(x.sum(1)))] if mask.any()])
tr, te = d['train'], d['test']
n = len(tr['lab_id'])
u = m['_coupon_inputs'](tr, m['_features'](tr, te)[:n], b)
t, dt = d['eval_times'], d['past_dt']
curve = tr['strain_curves_obs'] - tr['strain_0'][:, None]
rate = tr['instant_rate_obs']
errors = []
for lab in np.unique(tr['lab_id']):
    test = tr['lab_id'] == lab
    pars = m['_fit'](tuple(z[~test] for z in u), dt, t, curve[~test], rate[~test])
    a, b = m['_design'](pars[0], tuple(z[test] for z in u), dt, t)
    ce, re = curve[test] - a @ pars[1], rate[test] - b @ pars[1]
    print('held-out lab', lab, 'signed medians (early curve, late curve, rate)', np.round([np.median(ce[:, :len(t)//2]), np.median(ce[:, len(t)//2:]), np.median(re)], 6))
    for st in np.unique(tr['stratum'][test]):
        cell = tr['stratum'][test] == st
        errors.append((int(lab), int(st), int(cell.sum()), float(np.mean(abs(ce[cell]))), float(np.mean(abs(re[cell])))))
print('worst lab/stratum cells (lab, stratum, n, curve MAE, rate MAE):', max(errors, key=lambda e:e[3]), max(errors, key=lambda e:e[4]))
