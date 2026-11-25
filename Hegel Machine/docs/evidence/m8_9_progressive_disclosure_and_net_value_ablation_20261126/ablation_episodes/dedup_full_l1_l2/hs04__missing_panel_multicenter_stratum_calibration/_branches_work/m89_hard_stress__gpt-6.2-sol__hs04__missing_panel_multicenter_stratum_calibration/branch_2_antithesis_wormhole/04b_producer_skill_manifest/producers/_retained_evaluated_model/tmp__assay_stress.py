import importlib.util
import pickle
import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_auc_score

spec = importlib.util.spec_from_file_location('assay', '/holdout_task/multicenter_calibration.py')
assay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(assay)

def score(q, y, z, o):
    x, p, w = (o[k] for k in ('aligned_query_features', 'query_probabilities', 'feature_weights'))
    assert x.shape == z.shape and p.shape == y.shape and w.shape == (z.shape[1],)
    assert np.isfinite(x).all() and np.isfinite(w).all() and ((0 < p) & (p < 1)).all()
    assert (x[~np.isfinite(q['X'])] == 0.).all()
    return np.array([[np.mean(abs(x[m]-z[m])), roc_auc_score(y[m], p[m]), np.mean((y[m]-p[m])**2)] for c in np.unique(q['center_id']) for m in [q['center_id']==c]])

def sub(a, mask):
    return {k: v[mask] for k,v in a.items()}

with open('/holdout_task/public_multicenter_assay.pkl', 'rb') as f:
    cohorts = pickle.load(f)['cohorts']
for i, e in enumerate(cohorts):
    d, g = e['dataset'],e['gt']
    tr = d['train']
    folds = []
    for c in np.unique(tr['center_id']):
        mask = tr['center_id']==c
        test = sub(tr, mask)
        y = test.pop('y')
        p = assay.calibrate_multicenter_assay(dict(d, train=sub(tr, ~mask), query=test))['query_probabilities']
        folds.append((roc_auc_score(y, p), np.mean((y-p)**2)))
    print('public', i, 'LOCO AUROC/Brier', np.round(folds, 4).tolist())
    ref = d['reference_controls']
    degraded = {k:v.copy() for k,v in ref.items()}
    for c in np.unique(d['query']['center_id']):
        ind = np.flatnonzero(ref['center_id']==c)
        degraded['X'][ind, 0] = np.nan
        degraded['X'][ind[1:], 1] = np.nan
        degraded['X'][ind[:2], 2] = ref['X'][ind[2], 2] if np.isfinite(ref['X'][ind[2],2]) else 1.
        degraded['X'][ind[2:], 2] = np.nan
    stress = dict(d, reference_controls=degraded)
    print('public', i, 'degraded MAE/AUROC/Brier', np.round(score(d['query'],g['y_query'],g['Z_query'],assay.calibrate_multicenter_assay(stress)), 4).tolist())

for hetero in (0.08, 0.30):
    scores=[]; oracle=[]
    for seed in range(12):
        rng = np.random.default_rng(seed)
        centers = np.arange(6)
        loc = np.array([3.5,3.2,-3.5,-3.2,-3.,3.])[:,None]+hetero*rng.normal(size=(6,8))
        scale = np.array([1.3, .8, 1.2, .85, 1.35, .75])[:,None]*np.exp(hetero*rng.normal(size=(6,8)))
        def observed(z,c,rate):
            x = loc[c]+scale[c]*z
            x[rng.random(x.shape)<rate]=np.nan
            return x
        rc = np.repeat(centers, [80,75,80,75,24,24]); rx = observed(rng.normal(size=(len(rc),8)),rc,.15)
        for c in (4,5):
            ids = np.flatnonzero(rc==c)
            rx[ids,0]=np.nan; rx[ids[1:],1]=np.nan
        def patients(c):
            z = rng.normal(size=(len(c),8)) + np.where(np.isin(c, [0,1,4]), .45,-.45)[:,None]
            s = np.array([rng.choice(3,p=[.25,.30,.45] if i in (0,1,4) else [.55,.25,.20]) for i in c])
            w = np.array([1.25,-1.05,.9,-.8,.65,-.6,.55,.55])*1.2
            p = expit(z@w+np.array([-.9,-.05,.75])[s])
            y = (rng.random(len(c))<p).astype(int)
            return dict(center_id=c, stratum=s, X=observed(z,c,.03)), z, y, p
        t, _, yt, _ = patients(np.repeat(centers[:4], 95))
        q, z, y, p = patients(np.repeat(centers[4:], 180))
        d=dict(reference_controls=dict(center_id=rc,X=rx),train=dict(t,y=yt),query=q)
        scores.extend(score(q, y, z, assay.calibrate_multicenter_assay(d)))
        oracle.extend([(roc_auc_score(y[m],p[m]),np.mean((y[m]-p[m])**2)) for c in (4,5) for m in [q['center_id']==c]])
    a=np.array(scores)
    print('synthetic heterogeneity', hetero, 'MAE/AUROC/Brier mean',np.round(a.mean(0),4),'worst',np.round([a[:,0].max(),a[:,1].min(),a[:,2].max()],4),'oracle AUROC/Brier mean',np.round(np.mean(oracle,0),4))
