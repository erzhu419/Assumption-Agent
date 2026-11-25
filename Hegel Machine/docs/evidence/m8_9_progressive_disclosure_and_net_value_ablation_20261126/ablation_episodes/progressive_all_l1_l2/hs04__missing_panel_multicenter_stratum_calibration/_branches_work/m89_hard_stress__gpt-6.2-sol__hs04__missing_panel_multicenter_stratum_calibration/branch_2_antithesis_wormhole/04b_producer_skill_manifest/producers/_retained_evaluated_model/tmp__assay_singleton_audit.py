import sys
sys.path.insert(0, '/tmp')
import pickle
import numpy as np
from sklearn.metrics import roc_auc_score, brier_score_loss
from assay_stress_helpers import m, simulate

base = m._reference_transforms

def singleton_update(ref):
    panels, pooled = base(ref)
    x, groups = np.asarray(ref['X']), np.asarray(ref['center_id'])
    for c, (location, scale) in panels.items():
        panel = x[groups == c]
        columns = [v[np.isfinite(v)] for v in panel.T]
        good = [v for v in columns if len(v) >= 2 and v.std(ddof=1) > 0]
        if len(good) < 2:
            continue
        means = np.array([v.mean() for v in good])
        for j, v in enumerate(columns):
            if len(v) != 1:
                continue
            variance = scale[j] ** 2
            w = np.array([len(a) for a in good]) / variance
            center = np.average(means, weights=w)
            q = w @ (means - center) ** 2
            between = max(0., (q - len(w) + 1) / (w.sum() - (w @ w) / w.sum()))
            # A singleton can inform location, but not the column's gain.
            location[j] += (between / (between + variance)) * (v[0] - location[j])
    return panels, pooled

def compare(name, cases):
    scores = [[], []]
    for d, g in cases:
        r, q = d['reference_controls'], d['query']
        for k, op in enumerate((base, singleton_update)):
            m._reference_transforms = op
            out = m.calibrate_multicenter_assay(d)
            error = np.abs(out['aligned_query_features'] - g['Z_query'])
            for c in np.unique(q['center_id']):
                take = q['center_id'] == c
                single = np.isfinite(r['X'][r['center_id'] == c]).sum(0) == 1
                measured = np.isfinite(q['X'][take]) & single
                y, p = g['y_query'][take], out['query_probabilities'][take]
                scores[k].append([error[take].mean(), error[take][measured].mean() if measured.any() else np.nan, roc_auc_score(y,p), brier_score_loss(y,p)])
    for label, rows in zip(('fallback', 'singleton posterior'), scores):
        a = np.asarray(rows)
        print(name, label, 'mean', np.nanmean(a, axis=0).round(4), 'worst', np.array([a[:,0].max(), np.nanmax(a[:,1]), a[:,2].min(), a[:,3].max()]).round(4))

print('Metrics: center MAE, observed singleton-column MAE, AUROC, Brier')
with open('/holdout_task/public_multicenter_assay.pkl','rb') as f:
    compare('public', [(e['dataset'],e['gt']) for e in pickle.load(f)['cohorts']])
for h in (.05, .23):
    compare(('synthetic: residual center effect 2.8; heterogeneity',h), [simulate(seed,h,2.8) for seed in range(60,92)])
