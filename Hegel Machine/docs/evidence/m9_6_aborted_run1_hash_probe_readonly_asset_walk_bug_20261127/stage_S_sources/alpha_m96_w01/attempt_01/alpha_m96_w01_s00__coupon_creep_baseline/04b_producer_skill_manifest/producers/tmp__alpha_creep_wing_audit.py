import importlib.util
import pickle
import numpy as np

root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('solver', root + 'alpha_m96_w01_s00_solver.py')
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)
with open(root + 'public_alpha_m96_w01_s00.pkl', 'rb') as f:
    p = pickle.load(f)[0]
d, gt = p['dataset'], p['gt']
tr, te = d['train'], d['test']
solve = solver.solve_alpha_m96_w01_s00
base = solve(d)
missing = ~np.isfinite(te['coupon_features_raw']).all(axis=1)
rmse = lambda x: np.sqrt(np.mean(x**2))

def report(tag, out, m):
    e = (out['predicted_strain'] - gt['te_clean'])[m]
    r = (out['initial_rate_pred'] - gt['te_true_rate0'])[m]
    print(tag, 'n', m.sum(), 'strain/late-strain/rate RMSE', rmse(e), rmse(e[:, -3:]), rmse(r))

for wing in (0, 1):
    held = te['lab_id']//2 == wing
    fit = tr['lab_id']//2 != wing
    out = solve(dict(d, train={k:v[fit] for k,v in tr.items()}))
    print('held wing', wing, 'tau shift vs full fit', out['estimated_taus'] - base['estimated_taus'])
    for name, mask in [('complete', held & ~missing), ('incomplete', held & missing)]:
        if mask.any():
            report(f'W{wing} {name} full', base, mask)
            report(f'W{wing} {name} cross-wing', out, mask)

# Two high-leverage spot-check glitches on the same wing, with the other wing untouched.
checks = d['calibration_load_cells']
for wing in (0, 1):
    x, ids = checks['cmd_stress'], checks['lab_id']
    y = checks['true_stress'].copy()
    for lab in (2*wing, 2*wing+1):
        j = np.flatnonzero(ids == lab)
        ends = j[np.argsort(x[j])[[0,-1]]]
        y[ends] += np.array([-.6,.6])
    out = solve(dict(d, calibration_load_cells=dict(checks, true_stress=y)))
    report(f'W{wing} EMI complete', out, (te['lab_id']//2 == wing) & ~missing)
    print('EMI calibration gains', solver._load_calibration(dict(checks, true_stress=y), range(4)))
