import importlib.util
import pickle

import numpy as np

root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('cand', root+'viscoplastic_overstress_hereditary_solver.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
p, = pickle.load(open(root+'public_suc01_C_datasets.pkl', 'rb'))
d, g = p['dataset'], p['gt']
time = d['eval_times']
fits = []
least_squares = m.least_squares

def capture(*args, **kwargs):
    fit = least_squares(*args, **kwargs)
    fits.append(fit)
    return fit

m.least_squares = capture

def compare(dataset):
    fits.clear()
    train, test = dataset['train'], dataset['test']
    out = m.solve_viscoplastic_hereditary_revision(dataset)
    coef = min(fits, key=lambda fit: fit.cost).x[2:]
    panel = [np.concatenate((train[k], test[k])) for k in
             ('coupon_features_raw', 'ref_coupons', 'lab_id', 'stratum')]
    x, _ = m.align_reference_panel(*panel)
    hist = m._history(test, d['past_dt'], d['yield_stress'])
    _, rate = m._design(hist, d['past_dt'], time, x[len(train['lab_id']):], out['estimated_taus'])
    missing = ~np.isfinite(test['coupon_features_raw']).all(axis=1)
    drift = (test['instant_rate_obs'] - rate @ coef) * missing
    base = out['predicted_strain']
    return base, base + drift[:, None] * time, missing

def report(label, split, target, a, b, missing):
    over = (np.abs(split['past_stress']).max(axis=1) > d['yield_stress']) | (np.abs(split['stress_hold_0']) > d['yield_stress'])
    e0, e1 = np.abs(a-target), np.abs(b-target)
    print(label)
    for name, mask in [('all', np.ones(len(a), dtype=bool)), ('over', over), ('sub', ~over), ('missing', missing)]:
        if mask.any():
            print(name, mask.sum(), 'MAE baseline/conditioned', np.round([e0[mask].mean(), e1[mask].mean()], 5), 'last horizon', np.round([e0[mask, -1].mean(), e1[mask, -1].mean()], 5))
    return e0, e1

s = d['train']
test_keys = d['test'].keys()
for source in ('lab_id', 'stratum'):
    a = np.empty_like(s['strain_curves_obs'])
    b = np.empty_like(a)
    missing = np.zeros(len(a), dtype=bool)
    for label in np.unique(s[source]):
        mask = s[source] == label
        fold = dict(d, train={k:v[~mask] for k,v in s.items()}, test={k:s[k][mask] for k in test_keys})
        a[mask], b[mask], missing[mask] = compare(fold)
    e0, e1 = report('Out of fold by '+source+' (against observed strain)', s, s['strain_curves_obs'], a, b, missing)
    print('missing-only fold mean absolute errors', [(int(k), round(e0[(s[source]==k)&missing].mean(), 5), round(e1[(s[source]==k)&missing].mean(), 5)) for k in np.unique(s[source][missing])])
a, b, missing = compare(d)
report('Public clean test', d['test'], g['te_clean'], a, b, missing)
