import importlib.util
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('solver', '/holdout_task/alpha_ep03_solver.py')
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
with open('/holdout_task/public_alpha_ep03.pkl', 'rb') as f:
    d = pickle.load(f)[0]['dataset']
tr, _, y, complete = s._prepare(d)
complete = complete[:len(y)]
lab = d['train']['lab_id']
stratum = d['train']['stratum']
t = tr.times
pred = np.empty_like(y)
for c in np.unique(lab):
    fit = (lab != c) & complete
    logs, theta = s._fit(tr, y, fit)
    pred[lab == c] = (tr.matrix(logs) @ theta)[lab == c]
    print('held-out lab', c, 'taus', np.exp(logs), 'plastic weights', theta[-2:])
logs, theta = s._fit(tr, y, complete)
print('full fit', 'taus', np.exp(logs), 'weights', theta)
_, dm, dh = tr.matrix(logs, True)
for j in range(2):
    delta = np.eye(2)[j] * 1e-5
    fd = (tr.matrix(logs+delta) @ theta - tr.matrix(logs-delta) @ theta) / 2e-5
    analytic = dm[...,j] * theta[j] + (dh * theta[-1] if j == 0 else 0)
    np.testing.assert_allclose(fd, analytic, rtol=1e-6, atol=1e-8)
rate_error = y[:, -1] - pred[:, -1]
resid = y[:,:-1] - pred[:,:-1]
slices = {'all': np.ones(len(y), dtype=bool), 'missing': ~complete, 'complete': complete,
          'below_hold_yield': tr.over == 0, 'positive_hold_yield': tr.over > 0,
          'negative_hold_yield': tr.over < 0, 'top_history_work': np.abs(tr.work) >= np.quantile(np.abs(tr.work), .9),
          **{f'lab{c}': lab == c for c in np.unique(lab)},
          **{f'stratum{c}': stratum == c for c in np.unique(stratum)}}
correction = np.where(complete, .25, 1.)
raw = resid
adj = resid - (correction * rate_error)[:, None] * t
for key, rows in slices.items():
    if rows.any():
        print(key, rows.sum(), 'rate_MAE/RMSE', np.round([np.mean(abs(rate_error[rows])), np.sqrt(np.mean(rate_error[rows]**2))], 6), 'curve_MAE raw/adjusted', np.round([np.mean(abs(r[rows])) for r in (raw, adj)], 6), 'curve_RMSE raw/adjusted', np.round([np.sqrt(np.mean(r[rows]**2)) for r in (raw, adj)], 6), 'horizon_MAE adjusted', np.round(np.mean(abs(adj[rows]), axis=0), 6))

# Isolate constitutive regimes: signed hold flow and history work need not activate together.
base = np.empty_like(y)
for c in np.unique(lab):
    use = (lab != c) & complete
    held = lab == c
    reduced = s._Design(tr.past, tr.hold, tr.dt, t, np.zeros((len(y), 4)), np.inf)
    reduced.linear = tr.linear
    logs, theta = s._fit(reduced, y, use)
    base[held] = (reduced.matrix(logs) @ theta)[held]

history_active = np.any(np.abs(tr.past) > d['yield_stress'], axis=1)
for hold_active in (False, True):
    for hist_active in (False, True):
        rows = ((tr.over != 0) == hold_active) & (history_active == hist_active)
        if rows.any():
            print('regime', hold_active, hist_active, rows.sum(), 'rate MAE base/full',
                  np.mean(abs(y[rows,-1]-base[rows,-1])), np.mean(abs(rate_error[rows])),
                  'curve median AE base/full', np.median(abs(y[rows,:-1]-base[rows,:-1])),
                  np.median(abs(raw[rows])))

# Exact threshold and sign tests for both strain and its t=0+ derivative.
sy = d['yield_stress']
probe = s._Design(np.array([[sy], [-sy], [sy+.5], [-sy-.5]]),
                  np.array([sy, -sy, sy+.25, -sy-.25]), np.array([.4]),
                  np.array([0., 1.e-6, .9]), np.zeros((4,4)), sy)
logs = np.log([.56, 3.82])
a = probe.matrix(logs)
np.testing.assert_array_equal(a[:2, :, -2:], 0.)
np.testing.assert_allclose(a[2:3, :, -2:], -a[3:4, :, -2:])
np.testing.assert_allclose((a[:, 1]-a[:, 0])/1.e-6, a[:, -1], atol=1.e-5, rtol=2.e-6)
np.testing.assert_allclose(a[2, -1, -2:], [.25**2, .5**2*.4*(sy+.25)/.56])
print('threshold, signed response, and onset derivative verified')
