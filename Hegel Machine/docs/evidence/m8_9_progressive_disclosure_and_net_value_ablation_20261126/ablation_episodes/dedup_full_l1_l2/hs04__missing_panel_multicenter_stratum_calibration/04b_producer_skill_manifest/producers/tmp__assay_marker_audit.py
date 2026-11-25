import pickle, runpy
import numpy as np
from scipy.special import digamma, polygamma
from sklearn.metrics import roc_auc_score

m = runpy.run_path('/holdout_task/multicenter_calibration.py')
with open('/holdout_task/public_multicenter_assay.pkl', 'rb') as f:
    cohorts = pickle.load(f)['cohorts']

for i, entry in enumerate(cohorts):
    d, gt = entry['dataset'], entry['gt']
    ref, q = d['reference_controls'], d['query']
    centers = np.unique(ref['center_id'])
    moments = np.empty((2, len(centers), ref['X'].shape[1]))
    noise = np.empty_like(moments)
    for k, c in enumerate(centers):
        rows = ref['X'][ref['center_id'] == c]
        n = np.isfinite(rows).sum(axis=0)
        valid = [(j, v[np.isfinite(v)]) for j,v in enumerate(rows.T)]
        stats = [(j, v.mean(), v.std(ddof=1)) for j,v in valid if len(v)>=2 and np.ptp(v)>0]
        moments[:, k] = np.nan
        noise[:, k] = np.inf
        if not stats:
            continue
        j, mu, sd = map(np.asarray, zip(*stats))
        df = n[j] - 1
        typical = np.median(sd)
        moments[:, k, j] = np.vstack([mu / typical, np.log(sd) - (digamma(df / 2) - np.log(df / 2)) / 2])
        noise[:, k, j] = np.vstack([(sd / typical)**2/n[j], polygamma(1, df / 2)/4])
    for name, estimate, variance in zip(('location','log-scale'), moments, noise):
        print('cohort', i, name)
        errors = []
        for k, c in enumerate(centers):
            others = np.arange(len(centers)) != k
            residual = estimate[others] - np.nanmedian(estimate[others], axis=1, keepdims=True)
            precision = 1 / variance[others]
            predicted = np.divide(np.nansum(residual * precision, axis=0), precision.sum(axis=0), out=np.zeros(estimate.shape[1]), where=precision.sum(axis=0)>0)
            held = np.isfinite(estimate[k]); target = estimate[k, held]; predicted = predicted[held]
            if len(target)<2:
                continue
            err=[]
            for gamma in (0., .5, 1.):
                adj = target - gamma*predicted
                residual = np.array([adj[j] - np.median(np.delete(adj,j)) for j in range(len(adj))])
                err.append(np.mean(residual**2))
            errors.append(err)
        print('masked-marker MSE center-only / half / full transfer',np.round(errors,5).tolist(), 'mean',np.round(np.mean(errors,axis=0),5).tolist())
    p = m['calibrate_multicenter_assay'](d)['query_probabilities']; y = gt['y_query']; true = gt['prob_query']
    for c in np.unique(q['center_id']):
        h = q['center_id'] == c
        print('center',int(c),'fit/oracle AUROC',np.round([roc_auc_score(y[h], pp[h]) for pp in (p,true)],4),'fit/oracle Brier',np.round([np.mean((y[h]-pp[h])**2) for pp in (p,true)],4))
