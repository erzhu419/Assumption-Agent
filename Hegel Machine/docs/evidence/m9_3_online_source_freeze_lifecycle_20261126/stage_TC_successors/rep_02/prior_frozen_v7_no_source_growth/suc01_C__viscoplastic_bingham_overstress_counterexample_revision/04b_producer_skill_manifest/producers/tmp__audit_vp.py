import importlib.util
import pickle
import numpy as np

path = '/holdout_task/viscoplastic_overstress_hereditary_solver.py'
spec = importlib.util.spec_from_file_location('solver', path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

for p in pickle.load(open('/holdout_task/public_suc01_C_datasets.pkl', 'rb')):
    d, g = p['dataset'], p['gt']
    tr, te = d['train'], d['test']
    t, dt, tau, sy = d['eval_times'], d['past_dt'], g['taus'], d['yield_stress']
    h, s = te['past_stress'], te['stress_hold_0']
    hold, hist = abs(s) > sy, (abs(h) > sy).any(axis=1)
    F = -np.expm1(-t[:, None] / tau)
    lag = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.]
    z = h @ (-np.expm1(-dt[:, None] / tau) * np.exp(-lag[:, None] / tau))
    ov = s * 0 + np.sign(s) * np.maximum(abs(s)-sy, 0.)**2
    work = s * (np.maximum(abs(h)-sy, 0.)**2 @ dt)
    creep = g['vp_theta'][0] * ov[:, None] * t
    surge = g['vp_theta'][1] * work[:, None] * F[:, 0]
    base = te['strain_0'][:, None] + ((s[:, None]-z)*g['theta'][:2]) @ F.T
    remainder = g['te_clean'] - base - creep - surge
    drift = (remainder @ t) / (t @ t)
    residual = np.max(abs(remainder - drift[:, None]*t))
    print('Clean-law maximum residual:', residual)
    assert residual < 1e-9
    print('MAE if plastic terms are omitted, with all other physical terms unchanged:')
    for lab in np.unique(te['lab_id']):
        for mode, mask in [('sub', ~(hold|hist)), ('above', hold|hist)]:
            mask = mask & (te['lab_id'] == lab)
            if mask.any():
                print(int(lab), mode, mask.sum(), round(abs((creep+surge)[mask]).mean(), 6))
    def subset(split, mask):
        return {key: val[mask] if isinstance(val, np.ndarray) and val.ndim and len(val) == len(mask) else val for key, val in split.items()}
    residuals = np.zeros_like(g['te_clean'])
    for lab in np.unique(te['lab_id']):
        mask = te['lab_id'] == lab
        q = m.solve_viscoplastic_hereditary_revision(dict(d, train=subset(tr, tr['lab_id'] != lab), test=subset(te, mask)))
        e = abs(q['predicted_strain'] - g['te_clean'][mask])
        residuals[mask] = e
        err_t = max(abs(q['estimated_taus']/tau-1))
        err_v = np.linalg.norm(q['viscoplastic_weights']-g['vp_theta'])/np.linalg.norm(g['vp_theta'])
        print('held-out lab', int(lab), 'tau error', round(err_t,6), 'weight error', round(err_v,6), 'MAE', round(e.mean(),6), 'sub/above MAE', [round(e[cohort[mask]].mean(),6) for cohort in (~(hold|hist), hold|hist) if cohort[mask].any()])
        assert err_t <= .10 and err_v <= .14 and e.mean() <= .070
        for cohort, tolerance in [(~(hold|hist), .065), (hold|hist, .080)]:
            if cohort[mask].any():
                assert e[cohort[mask]].mean() <= tolerance
    complete = np.isfinite(te['coupon_features_raw']).all(axis=1)
    print('held-out lab MAE, complete / incomplete coupons:', [round(residuals[mask].mean(), 6) for mask in (complete, ~complete) if mask.any()])
    print('held-out lab MAE by horizon:', residuals.mean(axis=0).round(6))
