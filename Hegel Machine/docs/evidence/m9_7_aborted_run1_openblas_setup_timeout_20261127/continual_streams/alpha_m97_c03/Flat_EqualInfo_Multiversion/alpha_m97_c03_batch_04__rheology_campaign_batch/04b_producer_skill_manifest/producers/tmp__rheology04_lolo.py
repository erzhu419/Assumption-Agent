import importlib.util
import pickle
import numpy as np
from pathlib import Path
root = Path('/holdout_task')
spec = importlib.util.spec_from_file_location('candidate', root / 'alpha_m97_c03_batch_04_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with (root / 'public_alpha_m97_c03_batch_04.pkl').open('rb') as f:
    d = pickle.load(f)[0]['dataset']
a, b = d['train'], d['test']
solve = m.solve_alpha_m97_c03_batch_04
xc, _ = m._align(a,b)
stress, _, _ = m._inputs(a,xc,m._calibrate(d['calibration_load_cells']))
active = (np.maximum(abs(stress)-d['crit_stress'],0)**2 @ d['past_dt']) > 0
missing = np.isnan(a['coupon_features_raw']).any(axis=1)
curve = np.empty_like(a['strain_curves_obs']); rate = np.empty_like(a['instant_rate_obs'])
for lab in np.unique(a['lab_id']):
    val = a['lab_id'] == lab
    fold = dict(d, train={k: v[~val] for k,v in a.items()}, test={k: np.concatenate((a[k][val],v)) for k,v in b.items()})
    o = solve(fold)
    curve[val] = o['predicted_strain'][:val.sum()]; rate[val] = o['initial_rate_pred'][:val.sum()]
    print('fold taus', lab, o['estimated_taus'])
ce = curve-a['strain_curves_obs']; re = rate-a['instant_rate_obs']
def row(name, use):
    if use.any():
        print(name, int(use.sum()), 'curve_rmse', round(np.sqrt(np.mean(ce[use]**2)),6), 'rate_rmse', round(np.sqrt(np.mean(re[use]**2)),6))
for lab in np.unique(a['lab_id']):
    val=a['lab_id']==lab
    row(('lab',lab),val)
    row(('lab inactive',lab),val & ~active)
    row(('lab active',lab),val & active)
row('missing',missing); row('complete',~missing)
print('time RMSE',np.sqrt(np.mean(ce**2,axis=0)))
o=solve(d)
for k in ('estimated_taus','hereditary_weights','regime_mode_weights'): print(k,o[k])
