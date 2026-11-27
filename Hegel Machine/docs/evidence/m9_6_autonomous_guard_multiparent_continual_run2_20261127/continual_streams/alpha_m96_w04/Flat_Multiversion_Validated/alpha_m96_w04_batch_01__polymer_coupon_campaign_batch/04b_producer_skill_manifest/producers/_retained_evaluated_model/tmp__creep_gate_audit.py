import importlib.util
import pickle
import numpy as np
from scipy.optimize import least_squares

root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('solver', root + 'alpha_m96_w04_batch_01_solver.py')
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
rows = pickle.load(open(root + 'public_alpha_m96_w04_batch_01.pkl', 'rb'))
print('public campaigns:', len(rows))
for row in rows:
    d = row['dataset']; a, b = d['train'], d['test']; labs = a['lab_id']; n = len(labs)
    coef = s._calibrate(d['calibration_load_cells']); x = s._align(a, b)[:n]
    dt, t = d['past_dt'], d['eval_times']; m = s._Creep(a, x, coef, dt, t)
    y = np.column_stack((a['strain_curves_obs'] - a['strain_0'][:, None], a['instant_rate_obs']))
    assert np.all(dt > 0) and np.all(t > 0)
    print('dataset keys:', sorted(d))
    miss = ~np.isfinite(a['coupon_features_raw']).all(axis=1)
    residuals = {name: np.zeros_like(y[:, :-1]) for name in ('physical', 'nonlinear')}
    transient_error = 0.
    # Fixed low-order challenger: pairwise feature/stress terms and a bounded rational stress term.
    q = np.column_stack((x * m.hold[:, None], x[:, 0] * x[:, 1], x[:, 2] * x[:, 3], m.hold / (1 + m.hold**2)))
    for lab in np.unique(labs):
        held = labs == lab; p = s._fit(m, y, ~held); design = m.basis(p[:2])[0]
        for mode, tau in enumerate(p[:2]):
            z = a['initial_memory_states'][:, mode].copy()
            for j, h in enumerate(dt):
                z += (m.past[:, j] - z) * -np.expm1(-h / tau)
            f = np.r_[-np.expm1(-t / tau), 1 / tau]
            transient_error = max(transient_error, float(np.max(abs(design[..., mode] - (m.hold-z)[:, None]*f))))
        pv = design @ p[2:]
        # Fit only on the outer training labs; missing features are not evidence for nonlinear drift.
        train = ~held & ~miss
        base = pv-y
        new = q[:, None, :] * np.r_[t, 1.][None, :, None]
        scale = np.maximum(2*s._mad(base[train], axis=0), 1e-6)
        def error(w):
            return ((base[train] + new[train] @ w)/scale).ravel()
        w = least_squares(error, np.zeros(q.shape[1]), loss='soft_l1').x
        for name, response in [('physical', pv.copy()), ('nonlinear', pv + new @ w)]:
            response[miss, :-1] += (y[miss, -1] - response[miss, -1])[:, None]*t
            residuals[name][held] = (response-y)[held, :-1]
    assert transient_error < 1e-12, transient_error
    print('direct recurrence max error across both branches and held-out fits:', transient_error)
    for name, r in residuals.items():
        for lab in np.unique(labs):
            sel = labs == lab
            complete = sel & ~miss; missing = sel & miss
            r0 = r[complete]
            print(name, 'held lab', lab, 'complete median coupon curve AE/endpoint AE/signed endpoint bias:', np.round([np.median(np.median(abs(r0), axis=1)), np.median(abs(r0[:, -1])), np.median(r0[:, -1])], 6), 'incomplete endpoint RMSE:', np.sqrt(np.mean(r[missing, -1]**2)) if np.any(missing) else 'no slice')
    # Three spot checks put the candidate just above its explicit sparse cutoff.
    checks = {k: np.asarray(v) for k, v in d['calibration_load_cells'].items()}; worst = 0.
    for wing in (0, 1):
        group = [np.where(checks['lab_id'] == c)[0] for c in (wing*2, wing*2+1)]
        ordered = [ix[np.argsort(checks['cmd_stress'][ix])] for ix in group]
        for crop in ((0, 1, 2), (0, 4, -1), (-3, -2, -1)):
            for spike in range(3):
                altered = {k: v.copy() for k,v in checks.items()}
                keep = checks['lab_id']//2 != wing
                for ix in ordered:
                    kept = ix[list(crop)]; keep[kept] = True; altered['true_stress'][kept[spike]] += .15
                c = s._calibrate({k:v[keep] for k,v in altered.items()})
                worst = max(worst, float(np.max(abs(c-coef))))
    print('worst three-check calibration coefficient shift:', worst)
