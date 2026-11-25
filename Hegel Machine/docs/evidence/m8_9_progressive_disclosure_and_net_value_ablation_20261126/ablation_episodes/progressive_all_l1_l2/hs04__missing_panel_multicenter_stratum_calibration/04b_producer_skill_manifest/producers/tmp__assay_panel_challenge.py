import sys
sys.path.insert(0, '/tmp')
import pickle
import numpy as np
from scipy.special import digamma, polygamma
from sklearn.metrics import roc_auc_score, brier_score_loss
from assay_stress_helpers import simulate, m

baseline = m._reference_transforms

def estimate(values, variance, huber):
    w = 1 / variance
    anchor = np.average(values, weights=w)
    # Estimate excess dispersion relative to the known sampling variances.
    q = np.dot(w, (values - anchor)**2)
    df = len(w) - 1
    denom = w.sum() - np.dot(w,w) / w.sum()
    between = max(.25 * len(w) / w.sum(), (q - df) / denom if df else 0.)
    w = 1 / (variance + between)
    anchor = np.average(values, weights=w)
    if huber:
        for _ in range(4):
            u = np.abs(values - anchor) / np.sqrt(variance + between)
            anchor = np.average(values, weights=w / np.maximum(1., u / 1.5))
    return anchor + between / (between + variance) * (values - anchor), anchor

def transforms(reference, huber):
    x, c = reference['X'], reference['center_id']; panels = {}
    for center in np.unique(c):
        cols, means, sds, counts = [], [], [], []
        for j, v in enumerate(x[c == center].T):
            v = v[np.isfinite(v)]
            if len(v) < 2: continue
            sd = v.std(ddof=1)
            if sd <= 0: continue
            cols.append(j); means.append(v.mean()); sds.append(sd); counts.append(len(v))
        if not cols: continue
        n, sd, mu = map(np.asarray, (counts, sds, means)); nu = n - 1.
        logs = np.log(sd) + .5 * (np.log(nu / 2) - digamma(nu / 2))
        fitted_logs, gain = estimate(logs, .25 * polygamma(1, nu / 2), huber)
        fitted_means, location = estimate(mu, np.exp(2 * gain) / n, huber)
        loc = np.full(x.shape[1], location); scale = np.full(x.shape[1], np.exp(gain))
        loc[cols] = fitted_means; scale[cols] = np.exp(fitted_logs)
        panels[center] = loc, scale
    if not panels: return baseline(reference)
    return panels, tuple(np.median(v, axis=0) for v in zip(*panels.values()))

def evaluate(entries, operator):
    m._reference_transforms = operator
    rows = []
    for d,g in entries:
        q = d['query']; out = m.calibrate_multicenter_assay(d)
        observed = np.isfinite(q['X']); error = np.abs(out['aligned_query_features'] - g['Z_query']); p = out['query_probabilities']
        for c in np.unique(q['center_id']):
            i = q['center_id'] == c
            rr = d['reference_controls']; controls = rr['X'][rr['center_id']==c]
            sparse = np.array([len(v) < 2 or np.ptp(v) == 0 for col in controls.T for v in [col[np.isfinite(col)]]])
            mask = observed[i] & sparse
            rows.append([error[i].mean(), error[i][mask].mean() if mask.any() else np.nan, roc_auc_score(g['y_query'][i], p[i]), brier_score_loss(g['y_query'][i], p[i])])
    return np.asarray(rows)

with open('/holdout_task/public_multicenter_assay.pkl', 'rb') as f:
    public = [(e['dataset'], e['gt']) for e in pickle.load(f)['cohorts']]
for name, data in [('public',public)] + [(('sparse', h, c), [simulate(seed,h,c) for seed in range(28,60)]) for h in (.05,.23) for c in (0,2.8)]:
    print(name)
    for method, op in [('median',baseline), ('precision',lambda r: transforms(r,False)), ('robust precision',lambda r: transforms(r,True))]:
        a = evaluate(data,op)
        print(method, 'mean MAE/sparse MAE/AUC/Brier', np.nanmean(a,axis=0).round(4), 'worst', np.array([np.max(a[:,0]),np.nanmax(a[:,1]),np.min(a[:,2]),np.max(a[:,3])]).round(4))
