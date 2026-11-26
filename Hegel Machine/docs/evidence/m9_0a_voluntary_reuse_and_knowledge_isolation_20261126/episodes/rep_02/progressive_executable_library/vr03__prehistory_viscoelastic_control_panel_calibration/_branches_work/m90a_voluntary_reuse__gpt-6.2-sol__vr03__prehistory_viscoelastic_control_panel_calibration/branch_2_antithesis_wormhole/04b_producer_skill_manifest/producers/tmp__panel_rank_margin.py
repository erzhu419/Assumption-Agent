import pickle, sys
import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
sys.path.insert(0, '/holdout_task')
import viscoelastic_panel_calibration as m

nodes, quad = np.polynomial.hermite.hermgauss(24)
quad /= np.sqrt(np.pi)

def compare(x, s, center, y, test, test_s):
    w = m.balanced_cell_weights(center, s)
    fill = m.weighted_column_fill(x, w)
    a = m._features(np.where(np.isnan(x), fill, x), s)
    b = m._features(np.where(np.isnan(test), fill, test), test_s)
    fit = LogisticRegression(C=5., max_iter=500, tol=1e-9).fit(a, y, sample_weight=w)
    logit = fit.decision_function(b)
    complete = np.isfinite(x).all(axis=1)
    cov = np.cov(x[complete], rowvar=False, aweights=w[complete], ddof=0)
    d = np.isnan(test) * fit.coef_[0, :3]
    sd = np.sqrt(np.maximum(0, np.einsum('ij,jk,ik->i', d, cov, d)))
    return np.column_stack([expit(logit), expit(logit[:,None] + np.sqrt(2.)*sd[:,None]*nodes) @ quad])

def pair_metrics(y,p,miss):
    i, j = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    d = p[i,None] - p[j][None,:]
    loss = (d < 0) + .5*(d == 0)
    affected = miss[i,None] | miss[j][None,:]
    return int(affected.sum()), int(affected.size), float(loss[affected].sum()), float(loss[~affected].sum())

with open('/holdout_task/public_viscoelastic_cohort.pkl','rb') as f:
    cases = pickle.load(f)
rows = []
for k, case in enumerate(cases):
    cohort, gt = case['cohort'], case['gt']
    tr, qu = cohort['train'], cohort['query']
    x = m._aligned(tr, cohort['taus'])
    c, s, y = (tr[a] for a in ('center_id','stratum','y'))
    for g in np.unique(c):
        keep = c != g
        p = compare(x[keep],s[keep],c[keep],y[keep],x[~keep],s[~keep])
        scores = [(roc_auc_score(y[~keep], a), np.mean((y[~keep]-a)**2)) for a in p.T]
        print('LOCO',k,g,np.round(scores,5).tolist())
        rows.append(scores)
    out = m.calibrate_viscoelastic_cohort(cohort)
    for g in np.unique(qu['center_id']):
        take = qu['center_id'] == g
        yy, p = gt['y_query'][take], out['query_probabilities'][take]
        miss = qu['missing_channel_mask'][take].any(axis=1)
        print('public baseline',k,g,'AUC',roc_auc_score(yy,p),'pair count with missing / total, rank loss with missing / complete', pair_metrics(yy,p,miss))
a = np.asarray(rows)
print('LOCO minimum AUC baseline/integrated',a[:,:,0].min(0),'maximum Brier',a[:,:,1].max(0),'mean Brier',a[:,:,1].mean(0))
