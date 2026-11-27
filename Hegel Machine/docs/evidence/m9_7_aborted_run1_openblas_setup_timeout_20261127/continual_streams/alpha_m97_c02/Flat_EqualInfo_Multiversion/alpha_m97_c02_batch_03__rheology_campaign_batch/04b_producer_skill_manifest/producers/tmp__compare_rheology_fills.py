import importlib.util
from pathlib import Path
import pickle
import numpy as np

path = Path('/holdout_task/alpha_m97_c02_batch_03_solver.py')
spec = importlib.util.spec_from_file_location('baseline', path)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
source = path.read_text()
old = '    fill = np.nanmean(means, axis=0)\n    x = np.where(np.isfinite(x), x, fill)'
new = '''    pooled = np.nanmean(means, axis=0)
    for group in np.unique(stratum):
        mask = stratum == group
        donors = means[cells[:, 1] == group]
        count = np.isfinite(donors).sum(axis=0)
        fill = np.divide(np.nansum(donors, axis=0), count, out=pooled.copy(), where=count > 0)
        x[mask] = np.where(np.isfinite(x[mask]), x[mask], fill)'''
assert source.count(old) == 1
candidate = source.replace(old, new)
Path('/tmp/rheology_stratum_candidate.py').write_text(candidate)
alt = {}
exec(compile(candidate, '/tmp/rheology_stratum_candidate.py', 'exec'), alt)
with open('/holdout_task/public_alpha_m97_c02_batch_03.pkl', 'rb') as file:
    item = pickle.load(file)[0]
d, gt = item['dataset'], item['gt']
tr, te = d['train'], d['test']
n = len(tr['lab_id'])
missing = ~np.isfinite(te['coupon_features_raw']).all(axis=1)
dt, times = d['past_dt'], d['eval_times']
age = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.]
# Hold out entire training labs and mask a known coupon feature per complete held-out row.
# Both methods see the same masks, and no held-out response enters fitting.
raw = tr['coupon_features_raw']
observed = np.isfinite(raw).all(axis=1)
for name, ns in [('pooled', vars(base)), ('stratum-balanced', alt)]:
    print(name)
    out = ns['solve_alpha_m97_c02_batch_03'](d)
    ec = out['predicted_strain'] - gt['te_clean']
    er = out['initial_rate_pred'] - gt['te_true_rate0']
    rmse = lambda a: float(np.sqrt(np.mean(a*a)))
    for label, sel in [('all', np.ones(len(er), bool)), ('natural missing', missing), ('complete', ~missing)]:
        print(label, 'clean strain/rate RMSE', rmse(ec[sel]), rmse(er[sel]))
    print('tau/weight max abs errors', *(float(np.max(np.abs(out[k] - gt[g]))) for k, g in [('estimated_taus', 'taus'), ('hereditary_weights', 'theta'), ('regime_mode_weights', 'vp_theta')]))
    for lab in np.unique(tr['lab_id']):
        holdout = tr['lab_id'] == lab
        masked = holdout & observed
        idx = np.flatnonzero(masked)
        changed = raw.copy()
        changed[idx, np.arange(len(idx)) % raw.shape[1]] = np.nan
        dd = dict(d, train=dict(tr, coupon_features_raw=changed))
        p,h,x,e = ns['_prepare'](dd)
        fit = np.flatnonzero(~holdout)
        y = tr['strain_curves_obs'] - e[:n, None]
        r = tr['instant_rate_obs']
        lt, w = ns['_fit'](p[fit], h[fit], x[fit], age, dt, times, y[fit], r[fit])
        c,v = ns['_basis'](lt, p[idx], h[idx], x[idx], age, dt, times)
        print('masked holdout lab', lab, 'strain/rate median, P90 AE', np.round(np.nanpercentile(np.abs(c@w-y[idx]), [50,90]), 6), np.round(np.nanpercentile(np.abs(v@w-r[idx]), [50,90]), 6))
