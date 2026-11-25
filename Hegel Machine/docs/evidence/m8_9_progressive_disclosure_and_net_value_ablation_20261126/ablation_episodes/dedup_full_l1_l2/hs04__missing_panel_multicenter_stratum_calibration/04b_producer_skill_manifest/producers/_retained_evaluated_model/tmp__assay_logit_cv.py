import pickle, runpy
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

mod = runpy.run_path('/holdout_task/multicenter_calibration.py')
with open('/holdout_task/public_multicenter_assay.pkl', 'rb') as f:
    cohorts = pickle.load(f)['cohorts']

base = (.25, 3., 0., False)
families = {
    'baseline': [base],
    'ridge_and_strata': [(C, s, 0., False) for C in (.12, .25, .7, 2.) for s in (1., 3., 10.)],
    'center_intercepts': [(C, 3., a, False) for C in (.12, .25, .7, 2.) for a in (1., 3., 10.)],
    'equal_center_mass': [(C, 3., 0., True) for C in (.12, .25, .7, 2.)],
}

def fit(x, t, v, q, cfg):
    C, stratum_scale, nuisance, equal_mass = cfg
    levels = np.unique(t['stratum'])
    a = np.c_[x, stratum_scale*(t['stratum'][:, None] == levels)]
    b = np.c_[v, stratum_scale*(q['stratum'][:, None] == levels)]
    centers, inverse, counts = np.unique(t['center_id'], return_inverse=True, return_counts=True)
    if nuisance:
        a = np.c_[a, nuisance*((inverse[:, None] == np.arange(len(centers))) - 1/len(centers))]
        b = np.c_[b, np.zeros((len(b), len(centers)))]
    w = len(a)/(len(centers)*counts[inverse]) if equal_mass else None
    reg = LogisticRegression(C=C, max_iter=500, tol=1e-8).fit(a, t['y'], sample_weight=w)
    return reg.predict_proba(b)[:, 1]

def take(data, m):
    return {k: v[m] for k,v in data.items()}

def score(y, p):
    return np.array([roc_auc_score(y, p), np.mean((y-p)**2)])

def loco(x, data, cfg):
    results = []
    for c in np.unique(data['center_id']):
        v = data['center_id'] == c
        results.append(score(data['y'][v], fit(x[~v], take(data, ~v), x[v], take(data,v), cfg)))
    return np.array(results)

for i,e in enumerate(cohorts):
    d,g = e['dataset'], e['gt']; t,q = d['train'], d['query']
    a,b = mod['_reference_affines'](d['reference_controls'])
    x,z = mod['_align'](t,a,b), mod['_align'](q,a,b)
    for name, candidates in families.items():
        def select(x,t):
            return min(candidates, key=lambda cfg: loco(x,t,cfg)[:,1].mean()) if len(candidates)>1 else base
        outer = []
        for c in np.unique(t['center_id']):
            held = t['center_id'] == c
            cfg = select(x[~held], take(t,~held))
            outer.append(score(t['y'][held], fit(x[~held],take(t,~held),x[held],take(t,held),cfg)))
        cfg = select(x,t)
        p = fit(x,t,z,q,cfg)
        observed = np.array([score(g['y_query'][k],p[k]) for c in np.unique(q['center_id']) for k in [q['center_id']==c]])
        outer = np.array(outer)
        print(i, name, cfg, 'nested AUROC min/Brier max/mean', np.round([outer[:,0].min(),outer[:,1].max(),outer[:,1].mean()],4), 'query', np.round(observed,4).tolist(), 'prob MSE',round(np.mean((p-g['prob_query'])**2),5))
