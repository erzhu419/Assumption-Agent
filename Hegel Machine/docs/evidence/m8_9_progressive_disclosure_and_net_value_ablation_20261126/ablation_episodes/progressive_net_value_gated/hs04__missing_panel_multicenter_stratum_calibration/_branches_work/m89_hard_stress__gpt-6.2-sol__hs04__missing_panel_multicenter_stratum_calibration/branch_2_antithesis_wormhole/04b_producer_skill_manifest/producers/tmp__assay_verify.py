import importlib.util
import pickle
import numpy as np
from sklearn.metrics import roc_auc_score, brier_score_loss

spec = importlib.util.spec_from_file_location('assay', '/holdout_task/multicenter_calibration.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('/holdout_task/public_multicenter_assay.pkl','rb') as handle:
    cohorts = pickle.load(handle)['cohorts']

def subset(part, keep):
    return {k: v[keep] for k,v in part.items()}

def check(d):
    o = m.calibrate_multicenter_assay(d)
    n,p = d['query']['X'].shape
    assert set(o) == {'aligned_query_features','query_probabilities','feature_weights'}
    x, prob, w = (o[k] for k in ('aligned_query_features','query_probabilities','feature_weights'))
    assert x.shape == (n,p) and prob.shape == (n,) and w.shape == (p,)
    assert all(a.dtype.kind=='f' and np.isfinite(a).all() for a in (x,prob,w))
    assert np.all((0 < prob) & (prob < 1)) and np.all(x[np.isnan(d['query']['X'])]==0)
    return x,prob,w

def scores(y,p):
    return f'{roc_auc_score(y,p):.3f}/{brier_score_loss(y,p):.3f}'

for i, co in enumerate(cohorts):
    d,g = co['dataset'],co['gt']
    x,p,w = check(d)
    print('cohort',i,'pooled',scores(g['y_query'],p),'weights',np.round(w,2))
    for c in np.unique(d['query']['center_id']):
        z = d['query']['center_id']==c
        print(' query',c,'MAE',round(np.abs(x[z]-g['Z_query'][z]).mean(),3),'AUROC/Brier',scores(g['y_query'][z],p[z]),'oracle',scores(g['y_query'][z],g['prob_query'][z]))
    t=d['train']
    for c in np.unique(t['center_id']):
        keep=t['center_id']==c
        _,p,_=check(dict(d,train=subset(t,~keep),query=subset(t,keep)))
        print(' LOCO',c,scores(t['y'][keep],p))
    # Exercise zero finite controls, a single finite value, and zero spread.
    r = {k: v.copy() for k,v in d['reference_controls'].items()}
    ix = np.flatnonzero(r['center_id']==5)
    r['X'][ix,0] = np.nan
    r['X'][ix[1:],1] = np.nan
    r['X'][ix,2] = 1
    x,_,_=check(dict(d,reference_controls=r))
    keep=d['query']['center_id']==5
    mae=np.abs(x[keep]-g['Z_query'][keep]).mean()
    print(' stressed sparse-panel center 5 MAE',round(mae,3))
    assert mae <= 0.24
