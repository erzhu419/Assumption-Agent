import importlib.util
import itertools
import pickle
import numpy as np

root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('solver', root + 'alpha_w02_batch_04_solver.py')
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
d = pickle.load(open(root + 'public_alpha_w02_batch_04.pkl', 'rb'))[0]['dataset']
tr, te, c = d['train'], d['test'], d['calibration_load_cells']
dt, t, sy = d['past_dt'], d['eval_times'], d['yield_stress']
cal = s._calibrate(c, sy)
lab = tr['lab_id']
complete = np.isfinite(tr['coupon_features_raw']).all(axis=1)
x = s._inputs(tr, s._features(tr, te)[:len(lab)], cal, dt, sy)
y = np.c_[tr['strain_curves_obs'] - tr['strain_0'][:, None], tr['instant_rate_obs']]
rmse = lambda a: np.sqrt(np.nanmean(a*a))
for name, cols in [('full', ()), ('no_hold_overstress', (7,)), ('no_history_work', (8,))]:
    scores = []
    for i in range(4):
        train = complete & (lab != i)
        test = complete & (lab == i)
        u = tuple(a[train].copy() for a in x)
        if 7 in cols: u[2][:, -1] = 0
        if 8 in cols: u[3][:] = 0
        tau, w = s._fit(u, dt, t, y[train])
        e = s._design(tuple(a[test] for a in x), dt, t, tau) @ w - y[test]
        active = (x[2][test, -1] != 0) | (x[3][test] != 0)
        # Median per-lab residuals expose a systematic instrument offset without fitting one to a held-out lab.
        scores.append([rmse(e[active, :-1]), rmse(e[active, -1]), np.nanmedian(e[:, -1]), np.nanmedian(e[:, -2])])
    print(name, 'LOLO [above-yield strain RMSE, above-yield rate RMSE, median rate residual, median final-strain residual]', np.round(scores, 6).tolist())
for i in range(4):
    ix = np.flatnonzero(c['lab_id'] == i)
    rows = lab == i
    cmd = np.r_[tr['past_stress_cmd'][rows].ravel(), tr['stress_hold_cmd'][rows]]
    lo, hi = cmd.min(), cmd.max()
    worst = (0., None)
    for size in (3, 4):
        for use in itertools.combinations(ix, size):
            keep = (c['lab_id'] != i) | np.isin(np.arange(len(c['lab_id'])), use)
            for bad in use:
                for emi in (-.22, .22):
                    cc = {k: v.copy() for k, v in c.items()}
                    cc['true_stress'][bad] += emi
                    gain, bias = s._calibrate({k: v[keep] for k, v in cc.items()}, sy)[i] - cal[i]
                    err = max(abs(gain*lo+bias), abs(gain*hi+bias))
                    if err > worst[0]: worst = (err, (use, bad, emi))
    print('lab', i, 'worst sparse-sweep physical stress delta and witness', worst)
    assert worst[0] < .01, (i, worst)
    if worst[0] > .01:
        use, bad, emi = worst[1]
        keep = (c['lab_id'] != i) | np.isin(np.arange(len(c['lab_id'])), use)
        cc = {k: v.copy() for k, v in c.items()}
        cc['true_stress'][bad] += emi
        orig, changed = s.solve_alpha_w02_batch_04(d), s.solve_alpha_w02_batch_04(dict(d, calibration_load_cells={k:v[keep] for k,v in cc.items()}))
        m = te['lab_id'] == i
        print('affected lab strain/rate prediction RMSE shift', [rmse((changed[k]-orig[k])[m]) for k in ('predicted_strain','initial_rate_pred')])

# A new, well-measured instrument regime must not be pulled back to the archive.
for i in range(4):
    cc = {k: v.copy() for k, v in c.items()}
    m = cc['lab_id'] == i
    cc['true_stress'][m] += .15 * cc['cmd_stress'][m] + .09
    cc['true_stress'][np.flatnonzero(m)[0]] += .22
    error = np.max(abs(s._calibrate(cc, sy)[i] - cal[i] - [.15, .09]))
    assert error < .001, (i, error)
    print('shifted current sweep with EMI', i, 'max coefficient error', round(error, 6))
