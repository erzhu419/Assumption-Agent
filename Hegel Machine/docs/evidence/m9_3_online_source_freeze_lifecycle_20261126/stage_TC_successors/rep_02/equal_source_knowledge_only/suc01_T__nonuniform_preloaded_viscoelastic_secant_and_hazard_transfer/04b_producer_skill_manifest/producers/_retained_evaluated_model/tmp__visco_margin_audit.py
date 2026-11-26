import importlib.util
import pickle
import numpy as np

root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('solver', root + 'preloaded_hereditary_secant_hazard_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open(root + 'public_suc01_T_datasets.pkl', 'rb') as f:
    items = pickle.load(f)

for item in items:
    d, g = item['dataset'], item['gt']
    tr, te = d['train'], d['test']
    dt, t = d['past_dt'], d['eval_times']
    out = m.solve_preloaded_hereditary_transfer(d)
    for tau in [out['estimated_taus'], np.array([.2, 9.])]:
        for data in (tr, te):
            z = data['initial_memory_states'].copy()
            for step, stress in zip(dt, data['past_stress'].T):
                z += -np.expm1(-step / tau) * (stress[:, None] - z)
            gap = data['stress_hold_0'][:, None] - z
            curve, rate = m._design(data, dt, t, tau)
            np.testing.assert_allclose(rate, gap / tau, rtol=1e-11, atol=1e-12)
            np.testing.assert_allclose(curve, gap[:, None, :] * -np.expm1(-t[:, None] / tau), rtol=1e-11, atol=1e-12)
    late = t >= np.median(t)
    print('Leave-one-lab-out absolute drift errors against robust late-curve measurements; columns: aligned fusion, rate-only, unaligned coupon')
    for lab in np.unique(tr['lab_id']):
        is_test = tr['lab_id'] == lab
        a = {k: v[~is_test] for k, v in tr.items()}
        b = {k: v[is_test] for k, v in tr.items()}
        x, xx = m._panel(a, b)
        tau, c = m._fit(a, dt, t, x)
        aa, _ = m._design(a, dt, t, tau)
        ab, rb = m._design(b, dt, t, tau)
        def late_drift(data, design):
            return np.median((data['strain_curves_obs'][:, late] - data['strain_0'][:, None] - design[:, late] @ c[:2]) / t[late], axis=1)
        complete = np.isfinite(b['coupon_features_raw']).all(axis=1)
        measured = late_drift(b, ab)[complete]
        rate = (b['instant_rate_obs'] - rb @ c[:2])[complete]
        blend = m._rate_weight(a, dt, t, x, tau, c)
        fusion = rate + (1. - blend) * ((xx @ c[2:])[complete] - rate)
        good = np.isfinite(a['coupon_features_raw']).all(axis=1)
        raw = np.column_stack((np.ones(good.sum()), a['coupon_features_raw'][good]))
        w = np.sqrt(m._balanced(a['lab_id'][good], a['stratum'][good]))
        raw_c = np.linalg.lstsq(raw * w[:, None], late_drift(a, aa)[good] * w, rcond=None)[0]
        raw_pred = raw_c[0] + b['coupon_features_raw'][complete] @ raw_c[1:]
        print('lab', lab, 'n', len(measured), 'blend', round(blend, 3), 'median', np.round(np.median(abs(np.column_stack([fusion, rate, raw_pred]) - measured[:, None]), axis=0), 5))
    true = g['te_cross']
    initial = te['strain_0'] >= te['critical_strain']
    for name, mask in [('at onset', initial), ('interior', (true > 0) & (true < t[-1])), ('terminal/censored', (true == t[-1]) & ~initial)]:
        if np.any(mask):
            print('public crossing', name, 'n', int(mask.sum()), 'MAE', round(np.mean(abs(out['hazard_crossing_times'][mask] - true[mask])), 6))
