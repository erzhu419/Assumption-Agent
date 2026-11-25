import importlib.util
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, roc_auc_score, recall_score

spec = importlib.util.spec_from_file_location('assay', '/holdout_task/multicenter_calibration.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

def metrics(y, p):
    pred = p >= .5
    recall = recall_score(y, pred, labels=[0, 1], average=None, zero_division=0)
    return np.r_[roc_auc_score(y, p), brier_score_loss(y, p), balanced_accuracy_score(y, pred), recall]

def nonlinear(z, s):
    i, j = np.triu_indices(z.shape[1], 1)
    return np.column_stack((z, s == 1, s == 2, .5 * z**2, .5 * z[:, i] * z[:, j], .5 * z / (1 + z**2)))

with open('/holdout_task/public_multicenter_assay.pkl', 'rb') as f:
    entries = pickle.load(f)['cohorts']
print('columns: AUROC, Brier, balanced accuracy, negative recall, positive recall')
for number, entry in enumerate(entries):
    d, gt = entry['dataset'], entry['gt']
    t, q, r = (d[k] for k in ('train', 'query', 'reference_controls'))
    y, s, c = (t[k] for k in ('y', 'stratum', 'center_id'))
    panels, pooled = m._reference_transforms(r)
    z = m._align(t, panels, pooled)
    raw = m._design(np.nan_to_num(t['X']), s)[:, 1:]
    extended = nonlinear(z, s)
    # Orthogonal sum/difference coordinates preserve the entire linear basis.
    h = np.eye(z.shape[1]); h[:2, :2] = np.array([[1., 1.], [1., -1.]]) / np.sqrt(2)
    oof = {k: np.empty(y.shape) for k in ('raw', 'aligned', 'rotated', 'extended')}
    print('cohort', number)
    for center in np.unique(c):
        te = c == center; tr = ~te
        for name, train_z, test_z in [('aligned', z[tr], z[te]), ('rotated', z[tr] @ h, z[te] @ h)]:
            b = m._fit(train_z, s[tr], c[tr], y[tr]); oof[name][te] = m.expit(m._design(test_z, s[te]) @ b)
        for name, x in [('raw', raw), ('extended', extended)]:
            fit = LogisticRegression(C=1/3, max_iter=1000).fit(x[tr], y[tr]); oof[name][te] = fit.predict_proba(x[te])[:, 1]
        print(' center', center, 'prevalence', round(y[te].mean(), 3), {k: metrics(y[te], v[te]).round(4).tolist() for k, v in oof.items() if k != 'rotated'})
    print(' pooled', {k: metrics(y, v).round(4).tolist() for k, v in oof.items() if k != 'rotated'})
    difference = np.max(np.abs(oof['aligned'] - oof['rotated']))
    print(' max sum/difference reparameterization change', round(difference, 8))
    assert difference < 1e-4
    out = m.calibrate_multicenter_assay(d)
    for center in np.unique(q['center_id']):
        ix = q['center_id'] == center
        print(' query', center, 'current/oracle', metrics(gt['y_query'][ix], out['query_probabilities'][ix]).round(4), metrics(gt['y_query'][ix], gt['prob_query'][ix]).round(4))
