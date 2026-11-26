import importlib.util
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('solver', '/holdout_task/alpha_w02_batch_04_solver.py')
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
rmse = lambda e: np.sqrt(np.nanmean(e**2))
for case in pickle.load(open('/holdout_task/public_alpha_w02_batch_04.pkl', 'rb')):
    d, g = case['dataset'], case['gt']
    tr, te = d['train'], d['test']
    dt, t, y = d['past_dt'], d['eval_times'], d['yield_stress']
    cal = s._calibrate(d['calibration_load_cells'])
    f = s._features(tr, te)
    n = len(tr['lab_id'])
    tr_in = s._inputs(tr, cal, dt, y, f[:n])
    te_in = s._inputs(te, cal, dt, y, f[n:])
    tau = s.solve_alpha_w02_batch_04(d)['estimated_taus']
    reflected = dict(te)
    gain, bias = cal[te['lab_id']].T
    reflected['stress_hold_cmd'] = -te['stress_hold_cmd'] - 2*bias/gain
    reflected['past_stress_cmd'] = -te['past_stress_cmd'] - (2*bias/gain)[:, None]
    a = s._design(te_in, dt, t, np.log(tau))
    b = s._design(s._inputs(reflected, cal, dt, y, f[n:]), dt, t, np.log(tau))
    np.testing.assert_allclose(a[:, :, [0,1,7,8]], -b[:, :, [0,1,7,8]], rtol=1e-11, atol=1e-11)
    np.testing.assert_array_equal(a[:, :, 2:7], b[:, :, 2:7])
    print('Stress-reversal parity passed for Prony and both plastic terms, including onset rate')
    lab = tr['lab_id']
    complete = np.isfinite(tr['coupon_features_raw']).all(1)
    yielded = (tr_in[3] != 0) | (tr_in[4] != 0)
    obs = np.c_[tr['strain_curves_obs']-tr['strain_0'][:, None], tr['instant_rate_obs']]
    fold = np.full(n, -1)
    rng = np.random.default_rng(318)
    for l in range(4):
        for j in (False, True):
            ids = rng.permutation(np.flatnonzero(complete & (lab == l) & (yielded == j)))
            fold[ids] = np.arange(len(ids)) % 4
    errors = np.full((2, *obs.shape), np.nan)
    public = []
    for k in range(4):
        fit, val = complete & (fold != k), fold == k
        tau, c = s._fit(tuple(v[fit] for v in tr_in), dt, t, obs[fit])
        raw = s._design(tr_in, dt, t, np.log(tau)) @ c
        res = obs - raw
        slope = np.nanmedian(res[:, :-1]/t, axis=1)
        shift = np.array([np.nanmedian(slope[fit & (lab == l)]) for l in range(4)])
        errors[0,val] = res[val]
        errors[1,val] = res[val] - shift[lab[val],None]*np.r_[t,1.]
        pred = s._design(te_in, dt, t, np.log(tau)) @ c
        pred_plus = pred + shift[te['lab_id'],None]*np.r_[t,1.]
        clean = np.c_[g['te_clean']-te['strain_0'][:,None], g['te_true_rate0']]
        public.append(np.stack((pred-clean, pred_plus-clean)))
    def report(name, e, labs, regime, use, metric):
        values = []
        for l in range(4):
            for r in (False, True):
                m = use & (labs == l) & (regime == r)
                if m.any():
                    values.append([metric(x[m,:-1]) for x in e] + [metric(x[m,-1]) for x in e])
        print(name, 'worst slice: curve base/centered then rate base/centered', np.round(np.max(values,axis=0),7))
    report('Complete-row cross-fitted observations', errors, lab, yielded, complete, lambda e: np.nanmedian(abs(e)))
    report('Complete-row independent clean test (fold ensemble)', np.mean(public,axis=0), te['lab_id'], (te_in[3]!=0)|(te_in[4]!=0), np.isfinite(te['coupon_features_raw']).all(1), rmse)
