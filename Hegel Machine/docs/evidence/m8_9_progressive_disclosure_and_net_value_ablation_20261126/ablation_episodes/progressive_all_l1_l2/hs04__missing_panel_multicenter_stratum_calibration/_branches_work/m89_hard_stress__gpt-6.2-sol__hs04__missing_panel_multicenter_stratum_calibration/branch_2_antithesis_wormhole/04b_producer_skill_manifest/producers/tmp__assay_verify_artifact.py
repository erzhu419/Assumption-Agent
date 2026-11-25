import sys
import pickle
import numpy as np
from sklearn.metrics import roc_auc_score, brier_score_loss, balanced_accuracy_score
sys.path.insert(0, '/tmp')
from assay_stress_helpers import simulate, m

def evaluate(d, g):
    q = d['query']; out = m.calibrate_multicenter_assay(d)
    assert set(out) == {'aligned_query_features', 'query_probabilities', 'feature_weights'}
    z, p, b = (out[k] for k in ('aligned_query_features', 'query_probabilities', 'feature_weights'))
    assert z.shape == q['X'].shape and p.shape == (len(z),) and b.shape == (z.shape[1],)
    assert all(np.issubdtype(a.dtype, np.floating) and np.isfinite(a).all() for a in (z, p, b))
    assert ((p > 0) & (p < 1)).all() and (z[np.isnan(q['X'])] == 0.).all()
    rows = []
    for c in np.unique(q['center_id']):
        ix = q['center_id'] == c
        y, prob = g['y_query'][ix], p[ix]
        ma = np.mean(np.abs(z[ix] - g['Z_query'][ix])) if 'Z_query' in g else 0.
        oracle = g.get('prob_query')
        rows.append([ma, roc_auc_score(y, prob), brier_score_loss(y, prob), balanced_accuracy_score(y, prob >= .5),
                     roc_auc_score(y, oracle[ix]) if oracle is not None else 0.,
                     brier_score_loss(y, oracle[ix]) if oracle is not None else 0.])
    return np.asarray(rows)

def summary(tag, rows):
    a = np.asarray(rows).reshape(-1, 6)
    print(tag, 'mean', a.mean(0).round(4), 'worst', a[[a[:, 0].argmax(), a[:, 1].argmin(), a[:, 2].argmax(), a[:, 3].argmin(), a[:, 4].argmin(), a[:, 5].argmax()], np.arange(6)].round(4))

print('metrics: MAE/AUROC/Brier/balanced accuracy/oracle AUROC/oracle Brier')
with open('/holdout_task/public_multicenter_assay.pkl', 'rb') as f:
    for j, e in enumerate(pickle.load(f)['cohorts']):
        d, g = e['dataset'], e['gt']; t = d['train']
        real = evaluate(d, g)
        assert (real[:, 0] <= .24).all()
        print('public', j, real.round(4).tolist())
        folds = []
        for center in np.unique(t['center_id']):
            keep = t['center_id'] != center
            fold = {'reference_controls': d['reference_controls'],
                    'train': {k: v[keep] for k, v in t.items()},
                    'query': {k: v[~keep] for k, v in t.items() if k != 'y'}}
            folds.append(evaluate(fold, {'y_query': t['y'][~keep]}))
        summary(('public LOCO', j), folds)
for confound in (0., 2.8):
    for heter in (.05, .23):
        summary(('sparse synthetic', confound, heter), [evaluate(*simulate(seed, heter, confound)) for seed in range(12, 28)])
