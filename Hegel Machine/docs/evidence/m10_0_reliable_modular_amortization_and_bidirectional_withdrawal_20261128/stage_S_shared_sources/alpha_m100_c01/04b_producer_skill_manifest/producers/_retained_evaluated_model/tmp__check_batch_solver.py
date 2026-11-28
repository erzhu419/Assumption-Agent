import importlib.util
from pathlib import Path
import numpy as np
import pickle

path = Path('/holdout_task/batch_solver.py')
spec = importlib.util.spec_from_file_location('batch_solver', path)
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)
d = pickle.load(open('/holdout_task/public_batch.pkl', 'rb'))[0]['dataset']
tr, te = d['train'], d['test']
lab = tr['lab_id']
missing = np.isnan(tr['coupon_features_raw']).any(axis=1)

def hold_out(rows):
    fields = tuple(te)
    sub = dict(d)
    sub['train'] = {key: value[~rows] for key, value in tr.items()}
    sub['test'] = {key: np.concatenate((te[key], tr[key][rows]), axis=0) for key in fields}
    return sub

def report(label, cf, rf):
    print(label, 'curve med/p90', np.round(np.quantile(abs(cf), [.5,.9]), 6),
          'rate med/p90', np.round(np.quantile(abs(rf), [.5,.9]), 6),
          'curve/rate signed median', np.round([np.median(cf), np.median(rf)], 6))

for scale in (.004, .008, .016, .032):
    if scale == .008:
        solve = s.solve_batch
    else:
        ns = {}
        exec(compile(path.read_text().replace('f_scale=.008', f'f_scale={scale}'), str(path), 'exec'), ns)
        solve = ns['solve_batch']
    c, r = np.zeros_like(tr['strain_curves_obs']), np.zeros_like(tr['instant_rate_obs'])
    print('scale', scale)
    for l in np.unique(lab):
        rows = lab == l
        o = solve(hold_out(rows))
        c[rows] = o['predicted_strain'][len(te['lab_id']):] - tr['strain_curves_obs'][rows]
        r[rows] = o['initial_rate_pred'][len(te['lab_id']):] - tr['instant_rate_obs'][rows]
        report(f'lab {l}', c[rows], r[rows])
        print('  taus', np.round(o['estimated_taus'], 5))
    report('all', c, r)
    report('missing-feature rows', c[missing], r[missing])
    report('earliest', c[:,:3], r)
    report('latest', c[:,-3:], r)
