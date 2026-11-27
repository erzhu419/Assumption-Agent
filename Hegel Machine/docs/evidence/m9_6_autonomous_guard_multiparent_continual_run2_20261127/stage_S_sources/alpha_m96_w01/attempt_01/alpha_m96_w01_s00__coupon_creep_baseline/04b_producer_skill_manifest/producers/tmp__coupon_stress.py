import copy, importlib.util, pickle
import numpy as np

p = '/holdout_task/alpha_m96_w01_s00_solver.py'
spec = importlib.util.spec_from_file_location('solver', p)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
case = pickle.load(open('/holdout_task/public_alpha_m96_w01_s00.pkl', 'rb'))[0]
d, gt = case['dataset'], case['gt']
solve = m.solve_alpha_m96_w01_s00
base = solve(d)
rmse = lambda x: float(np.sqrt(np.mean(np.square(x))))
for j in range(4):
    fold = copy.deepcopy(d)
    tr = fold['train']
    held = tr['lab_id'] == j
    target = tr['strain_curves_obs'][held]
    rates = tr['instant_rate_obs'][held]
    # Preserve the original unlabeled reference panel on both sides of the fold.
    fold['test'] = {k: np.concatenate((fold['test'][k], tr[k][held])) for k in fold['test'] if k in tr}
    fold['train'] = {k: v[~held] for k, v in tr.items()}
    o = solve(fold)
    n = int(held.sum())
    complete = np.isfinite(tr['coupon_features_raw'][held]).all(axis=1)
    err = o['predicted_strain'][-n:] - target
    clean_err = o['predicted_strain'][:-n] - gt['te_clean']
    print('held lab', j, 'tau', o['estimated_taus'], 'observed curve median abs', np.median(abs(err)), 'complete rate RMSE', rmse((o['initial_rate_pred'][-n:] - rates)[complete]), 'public clean curve RMSE', rmse(clean_err))
    assert np.max(abs(o['estimated_taus'] - base['estimated_taus'])) < .25
    assert rmse(clean_err) < .025

for mode in ('affine_emi', 'sparse_two', 'sparse_one'):
    mod = copy.deepcopy(d)
    c = mod['calibration_load_cells']
    g, b = np.array([.89, 1.13, 1.20, .79]), np.array([.12, -.14, .24, -.17])
    ll = c['lab_id']
    c['cmd_stress'] = (c['cmd_stress'] - b[ll])/g[ll]
    for split in (mod['train'], mod['test']):
        ll = split['lab_id']
        split['past_stress_cmd'] = (split['past_stress_cmd'] - b[ll, None])/g[ll, None]
        split['stress_hold_cmd'] = (split['stress_hold_cmd'] - b[ll])/g[ll]
    if mode == 'affine_emi':
        c['true_stress'][[20, 23, 31, 36]] += [1.2, -1.5, -.8, 1.6]
    else:
        q = np.flatnonzero(c['lab_id'] == 3)
        order = q[np.argsort(c['cmd_stress'][q])]
        keep = (c['lab_id'] != 3) | np.isin(np.arange(len(c['lab_id'])), order[[0, -1]] if mode == 'sparse_two' else order[len(order)//2:len(order)//2+1])
        mod['calibration_load_cells'] = c = {k: v[keep] for k, v in c.items()}
    a, z = m._load_calibration(c)
    o = solve(mod)
    e = o['predicted_strain']-gt['te_clean']
    print(mode, 'gain max error', max(abs(a-g)), 'bias max error', max(abs(z-b)), 'curve RMSE by lab', [rmse(e[mod['test']['lab_id']==j]) for j in range(4)])
    if mode != 'sparse_one':
        assert max(abs(a-g)) < .001 and max(abs(z-b)) < .001
        assert rmse(o['predicted_strain'] - base['predicted_strain']) < .002

miss = copy.deepcopy(d)
rows = ~np.isfinite(miss['test']['coupon_features_raw']).all(axis=1)
miss['test']['instant_rate_obs'][rows] = np.nan
o = solve(miss)
assert np.all(np.isfinite(o['predicted_strain'])) and np.all(np.isfinite(o['initial_rate_pred']))
assert np.array_equal(o['predicted_strain'][~rows], base['predicted_strain'][~rows])
assert np.array_equal(o['viscoplastic_weights'], np.zeros(2))
assert .20 <= o['estimated_taus'][0] < o['estimated_taus'][1] <= 8
print('missing-rate fallback curve RMSE', rmse((o['predicted_strain']-gt['te_clean'])[rows]))
