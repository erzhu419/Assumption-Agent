import importlib.util, json, pickle
import numpy as np

root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('solver', root + 'alpha_m97_c02_batch_07_solver.py')
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
item = pickle.load(open(root + 'public_alpha_m97_c02_batch_07.pkl','rb'))[0]
d = item['dataset']
a, b = d['train'], d['test']
dt, times = d['past_dt'], d['eval_times']
x, _ = s._aligned(a, b)
base = s._calibration(d)
history = json.load(open(s._ARCHIVE))
previous = base.copy()
previous[2] = s._line(*s._points(history[-2], 2))
print('latest-clean', base.tolist(), 'older-channel-2', previous[2].tolist())
lab, strata = a['lab_id'], a['stratum']
measured = np.isfinite(a['coupon_features_raw']).all(axis=1)
curve = a['strain_curves_obs'] - a['strain_0'][:, None]
rate = a['instant_rate_obs']
rng = np.random.default_rng(7)
fold = np.empty(len(x), int)
for c in np.unique(lab):
    for g in np.unique(strata):
        rows = rng.permutation(np.flatnonzero((lab == c) & (strata == g)))
        fold[rows] = np.arange(len(rows)) % 4
for name, cal in [('latest-clean', base), ('older-channel-2', previous)]:
    inputs = s._inputs(a, x, cal, dt, d['crit_stress'], d['shear_banding_index'])
    ec, er = np.empty_like(curve), np.empty_like(rate)
    edc, edr = np.empty_like(curve), np.empty_like(rate)
    taus = []
    for k in range(4):
        fit = (fold != k) & measured
        val = fold == k
        tau, w = s._fit(dt, times, tuple(v[fit] for v in inputs), curve[fit], rate[fit])
        C, R = s._design(tau, dt, times, tuple(v[val] for v in inputs))
        ec[val], er[val] = C @ w - curve[val], R @ w - rate[val]
        taus.append(tau)
    for c in np.unique(lab):
        fit = (lab != c) & measured
        val = lab == c
        tau, w = s._fit(dt, times, tuple(v[fit] for v in inputs), curve[fit], rate[fit])
        C, R = s._design(tau, dt, times, tuple(v[val] for v in inputs))
        edc[val], edr[val] = C @ w - curve[val], R @ w - rate[val]
    metrics = lambda p, q, m: np.round([*np.quantile(np.abs(p[m]), [.5, .9]), *np.quantile(np.abs(q[m]), [.5, .9])], 6).tolist()
    print(name, 'fold taus', np.round(taus, 5).tolist())
    for c in np.unique(lab):
        m = lab == c
        print('lab', c, 'active', int(np.count_nonzero(inputs[-1][m])), 'four-fold', metrics(ec, er, m & measured), 'leave-lab-out', metrics(edc, edr, m & measured))
    print('all four-fold', metrics(ec, er, measured), 'missing coupons', metrics(ec, er, ~measured))
    print('all leave-lab-out', metrics(edc, edr, measured))
