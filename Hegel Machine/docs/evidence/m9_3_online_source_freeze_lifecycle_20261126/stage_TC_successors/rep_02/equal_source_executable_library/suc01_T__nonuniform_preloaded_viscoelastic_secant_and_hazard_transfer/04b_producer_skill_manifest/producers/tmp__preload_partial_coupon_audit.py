import importlib.util
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('solver', '/holdout_task/preloaded_hereditary_secant_hazard_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

def compare(a, b, t, dt):
    x, y, ca, cb = m._aligned_features(a, b)
    tau, gains, coef = m._fit(a, t, dt, x, ca)
    amp = m._deficit(tau, dt, b['past_stress'], b['initial_memory_states'], b['stress_hold_0']) * gains
    rate = b['instant_rate_obs'] - amp @ (1 / tau)
    base = b['strain_0'][:, None] + amp @ (-np.expm1(-t[:, None] / tau)).T
    missing = ~np.isfinite(b['coupon_features_raw'])
    weight = 0.7 * cb
    # Estimate the variance of the omitted contribution for each actual mask.
    # Keep the existing alignment, imputation and complete-coupon blend fixed.
    cells = np.column_stack((a['lab_id'][ca], a['stratum'][ca]))
    _, inv, count = np.unique(cells, axis=0, return_inverse=True, return_counts=True)
    cell_weight = 1.0 / count[inv.reshape(-1)]
    omitted = (x[ca, 1:] * coef[1:]) @ missing.T
    mu = np.average(omitted, axis=0, weights=cell_weight)
    var = np.average((omitted - mu)**2, axis=0, weights=cell_weight)
    informed_weight = .7 / (1 + .7 * var / .012**2)
    predictions = []
    for w in (weight, informed_weight):
        flow = rate + w * (y @ coef - rate)
        predictions.append(base + flow[:, None] * t)
    return predictions, informed_weight

for u in pickle.load(open('public_suc01_T_datasets.pkl', 'rb')):
    d, g = u['dataset'], u['gt']
    a, b, t, dt = d['train'], d['test'], d['eval_times'], d['past_dt']
    heldout = []
    for lab in np.unique(a['lab_id']):
        test = a['lab_id'] == lab
        tr = {k:v[~test] for k,v in a.items()}
        te = {k:v[test] for k,v in a.items()}
        mask = ~np.isfinite(te['coupon_features_raw']).all(axis=1)
        pred, w = compare(tr, te, t, dt)
        e = np.column_stack([np.minimum(abs(p[mask] - te['strain_curves_obs'][mask]), .2).mean(axis=1) for p in pred])
        heldout.extend(e)
        print('held-out lab', lab, 'missing n', len(e), 'baseline/challenger clipped strain MAE', e.mean(axis=0))
    print('pooled held-out missing clipped strain MAE', np.mean(heldout, axis=0))
    mask = ~np.isfinite(b['coupon_features_raw']).all(axis=1)
    pred, w = compare(a, b, t, dt)
    for method, p in zip(('baseline','partial coupons'), pred):
        h = m._crossing_times(t, b['strain_0'], p, b['critical_strain'])
        s = (p[:,-1] - b['strain_0']) / np.maximum(abs(b['stress_hold_0']), .5)
        e = np.c_[abs(p-g['te_clean']).mean(1), abs(s-g['te_secant']), abs(h-g['te_cross'])]
        print(method, 'public overall', e.mean(0), 'missing', e[mask].mean(0), 'largest missing crossing miss', e[mask,2].max())
    print('partial coupon weights on missing rows', w[mask].round(5))
    # Direct check of the documented floor, which is inactive on the public hold loads.
    probe = dict(d, test=dict(b, stress_hold_0=b['stress_hold_0'].copy()))
    probe['test']['stress_hold_0'][:3] = [-.25, 0., .25]
    out = m.solve_preloaded_hereditary_transfer(probe)
    np.testing.assert_allclose(out['secant_compliance'][:3], (out['predicted_strain'][:3,-1] - b['strain_0'][:3]) / .5)
    print('secant floor branch verified')
