import importlib.util
import pickle
import numpy as np
from sklearn.metrics import roc_auc_score

spec = importlib.util.spec_from_file_location('cal', '/holdout_task/viscoelastic_panel_calibration.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('/holdout_task/public_viscoelastic_cohort.pkl', 'rb') as f:
    entries = pickle.load(f)
summary = {key: [] for key in ('aligned', 'raw pooled', 'expanded aligned')}
for i, entry in enumerate(entries):
    cohort = entry['cohort']
    b = cohort['train']
    c, s, y = (np.asarray(b[key]) for key in ('center_id', 'stratum', 'y'))
    raw = m._extract(b, np.asarray(cohort['taus']))
    z = m._align(raw, b['controls'], c)
    # Shared positive affine changes within each center must cancel in alignment.
    gain = (1 + .35*c)[:, None]
    offset = (3 - 2*c)[:, None]
    assert np.allclose(m._align(raw*gain + offset, b['controls']*gain + offset, c), z, equal_nan=True, atol=1e-12, rtol=0)
    def expanded(x):
        a, b, d = x[:, :3].T
        return np.column_stack((x, a*a, b*b, d*d, a*b, a*d, b*d, (a-b)/(1+d*d), (a+d)/(1+b*b), (b-d)/(1+a*a)))
    for center in np.unique(c):
        hold = c == center
        tr = ~hold
        w = m._cell_weights(c[tr], s[tr])
        for kind in summary:
            v = raw if kind == 'raw pooled' else z
            wt = np.ones(tr.sum()) if kind == 'raw pooled' else w
            train, test = m._resolve_missing(v[tr], v[hold], wt)
            x, xv = m._features(train, s[tr]), m._features(test, s[hold])
            if kind == 'expanded aligned':
                x, xv = expanded(x), expanded(xv)
            coef = m._logistic(x, y[tr].astype(float), wt)
            p = m.expit(xv @ coef) if kind == 'raw pooled' else m._risk(xv, coef, ~np.isfinite(v[hold]), v[tr])
            truth = y[hold]
            r0, r1 = np.mean(p[truth == 0] < .5), np.mean(p[truth == 1] >= .5)
            row = (roc_auc_score(truth, p), np.mean((p-truth)**2), (r0+r1)/2, r0, r1)
            summary[kind].append(row)
            print(i, center, kind, 'auc brier BA recall0 recall1:', np.round(row, 4))
for kind, rows in summary.items():
    v = np.asarray(rows)
    print(kind, 'worst auc brier BA recall0 recall1:', np.round([v[:,0].min(), v[:,1].max(), *v[:,2:].min(axis=0)], 4))
