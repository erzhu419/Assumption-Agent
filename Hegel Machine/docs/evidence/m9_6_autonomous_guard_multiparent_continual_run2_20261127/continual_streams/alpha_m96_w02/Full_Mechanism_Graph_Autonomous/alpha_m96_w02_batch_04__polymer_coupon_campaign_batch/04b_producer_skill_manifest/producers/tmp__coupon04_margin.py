import importlib.util, pickle
import numpy as np
p = '/holdout_task/alpha_m96_w02_batch_04_solver.py'
spec = importlib.util.spec_from_file_location('coupon', p)
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
with open('/holdout_task/public_alpha_m96_w02_batch_04.pkl', 'rb') as f:
    d = pickle.load(f)[0]['dataset']
tr = d['train']
cal = s._calibrate(d['calibration_load_cells'])
worst = 0.
for wing in (0, 1):
    held = tr['lab_id'] // 2 == wing
    train = {k: v[~held] for k, v in tr.items()}
    test = {k: v[held] for k, v in tr.items()}
    without = {k: v for k, v in test.items() if k != 'instant_rate_obs'}
    obs = test['strain_curves_obs']
    out = s.solve_alpha_m96_w02_batch_04(dict(d, train=train, test=test))
    prior = s.solve_alpha_m96_w02_batch_04(dict(d, train=train, test=without))
    a = s._Design(d, test, s._align(train, test)[len(train['lab_id']):], cal)
    h, o = a.hist != 0, a.base[:, -1, 7] != 0
    missing = ~np.isfinite(test['coupon_features_raw']).all(axis=1)
    print('held wing', wing, 'taus', np.round(out['estimated_taus'], 5), 'plastic', np.round(out['viscoplastic_weights'], 5))
    for name, m in [('subyield',~h&~o),('history-only',h&~o),('hold-active',o),('missing',missing),('complete',~missing)]:
        if not m.any(): continue
        err = (out['predicted_strain'] - obs)[m]
        base = (prior['predicted_strain'] - obs)[m]
        rmse = np.sqrt(np.nanmean(err**2))
        worst = max(worst, rmse)
        print(name, m.sum(), 'prior/new RMSE', np.round([np.sqrt(np.nanmean(base**2)),rmse],6), 'median abs', round(np.nanmedian(abs(err)),6), 'late signed median', round(np.nanmedian(err[:,-1]),6))
print('worst held-wing strain RMSE', round(worst,6))
worst_stress = 0.
checks = d['calibration_load_cells']
for lab in range(4):
    own = np.flatnonzero(checks['lab_id'] == lab)
    order = own[np.argsort(checks['cmd_stress'][own])]
    pair = order[[0,-1]]
    keep = (checks['lab_id'] != lab) | np.isin(np.arange(len(checks['lab_id'])), pair)
    for idx in pair:
        for jump in (-.27,.27):
            c = {k:v[keep].copy() for k,v in checks.items()}
            c['true_stress'][np.count_nonzero(keep[:idx])] += jump
            g,b = s._calibrate(c)[lab] - cal[lab]
            delta = max(abs(b-3*g),abs(b+3*g))
            worst_stress = max(worst_stress, delta)
print('worst of 16 sparse + EMI cases', round(worst_stress, 6))
assert worst_stress < .002
