import importlib.util
import pickle
import numpy as np

spec = importlib.util.spec_from_file_location('solver', '/holdout_task/viscoplastic_overstress_hereditary_solver.py')
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)
for item in pickle.load(open('/holdout_task/public_suc01_C_datasets.pkl', 'rb')):
    d = item['dataset']
    tr = d['train']
    y, t = d['yield_stress'], d['eval_times']
    missing = ~np.isfinite(tr['coupon_features_raw']).all(axis=1)
    above = (np.abs(tr['past_stress']).max(axis=1) > y) | (np.abs(tr['stress_hold_0']) > y)
    base = np.empty_like(tr['strain_curves_obs'])
    shift = np.zeros_like(base)
    for lab in np.unique(tr['lab_id']):
        val = tr['lab_id'] == lab
        a = {k: v[~val] for k, v in tr.items()}
        b = {k: v[val] for k, v in tr.items() if k != 'strain_curves_obs'}
        out = solver.solve_viscoplastic_hereditary_revision(dict(d, train=a, test=b))
        base[val] = out['predicted_strain']
        # The constitutive onset derivative is used directly, not extrapolated from sampled curves.
        dt = d['past_dt']
        x = solver._align(a, b)
        n = len(a['lab_id'])
        tau, coef = solver._fit(solver._inputs(a, x[:n], dt, y), dt, t, a['strain_curves_obs'] - a['strain_0'][:, None], a['instant_rate_obs'], solver._cell_weights(a['lab_id'], a['stratum']))
        _, rate = solver._design(solver._inputs(b, x[n:], dt, y), dt, t, tau)
        shift[val] = ((~missing[val]) * (b['instant_rate_obs'] - rate @ coef))[:, None] * t
    cells = [(lab, s, (tr['lab_id'] == lab) & (tr['stratum'] == s)) for lab in np.unique(tr['lab_id']) for s in np.unique(tr['stratum'])]
    for gain in (0., 0.5, 1.):
        e = np.minimum(np.abs(base + gain * shift - tr['strain_curves_obs']), 0.3)
        worst = max((float(e[mask].mean()), int(lab), int(s)) for lab, s, mask in cells if mask.any())
        late = [(int(lab), round(float(e[(tr['lab_id'] == lab) & ~missing, -1].mean()), 6)) for lab in np.unique(tr['lab_id'])]
        print('gain', gain, 'OOF clipped MAE all/above/sub', [round(float(z.mean()), 6) for z in (e, e[above], e[~above])], 'worst cell', worst, 'late complete by lab', late)
