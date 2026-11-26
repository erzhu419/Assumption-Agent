import pickle, sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
sys.path.insert(0, '/holdout_task')
import viscoelastic_panel_calibration as m

summaries = {}
with open('/holdout_task/public_viscoelastic_cohort.pkl', 'rb') as f:
    cases = pickle.load(f)
for i, case in enumerate(cases):
    data, gt = case['cohort'], case['gt']
    tr, qu = data['train'], data['query']
    taus = data['taus']
    c, s, y = (tr[k] for k in ('center_id', 'stratum', 'y'))
    z = m._aligned(tr, taus)
    controls = tr['controls']
    raw = z.copy()
    for g in np.unique(c):
        a = controls[c == g]
        raw[c == g] = z[c == g] * np.median(np.nanstd(a, axis=0)) + np.median(np.nanmean(a, axis=0))
    ranks, conds, memory_diffs = [], [], []
    for split in (tr, qu):
        for t in split['traces']:
            a = m._stress_design(t, taus)
            ranks.append(np.linalg.matrix_rank(a))
            conds.append(np.linalg.cond(a))
            # Independent formula uses the exponential weights on all earlier inputs.
            tt = np.r_[t['past_timestamps'], t['timestamps']]
            uu = np.r_[t['past_strain_history'], t['strain_u']]
            direct = []
            for k in range(len(tt)-len(a), len(tt)):
                weights = -np.expm1(-np.diff(tt[:k+1])[:,None]/taus)
                state = uu[0]*np.exp(-(tt[k]-tt[0])/np.asarray(taus))
                state += (uu[1:k+1,None]*weights*np.exp(-(tt[k]-tt[1:k+1,None])/taus)).sum(axis=0)
                direct.append(state)
            memory_diffs.append(np.max(np.abs(a[:,2:] - direct)))
    print('case', i, 'min design rank', min(ranks), 'max condition', max(conds), 'max independent memory error', max(memory_diffs))
    assert min(ranks) == 4 and max(memory_diffs) < 1e-11
    for name, source in [('raw', raw), ('aligned', z), ('augmented', z)]:
        pooled = np.empty(len(y))
        for g in np.unique(c):
            k = c != g
            w = m.balanced_cell_weights(c[k], s[k])
            fill = m.weighted_column_fill(source[k], w)
            x = np.where(np.isnan(source), fill, source)
            design = m._features(x, s)
            if name == 'augmented':
                pairs = [(0,1),(0,2),(1,2)]
                design = np.column_stack([design, *[x[:,a]*x[:,b] for a,b in pairs], x.prod(axis=1), *[x[:,a]/(1+x[:,b]**2) for a,b in pairs]])
            fit = LogisticRegression(C=5, max_iter=500, tol=1e-9).fit(design[k], y[k], sample_weight=w)
            p = pooled[~k] = fit.predict_proba(design[~k])[:,1]
            yy = y[~k]
            recalls = [(p[yy == 0] < .5).mean(), (p[yy == 1] >= .5).mean()]
            vals = (roc_auc_score(yy,p), np.mean((yy-p)**2), np.mean(recalls), min(recalls))
            summaries.setdefault(name, []).append(vals)
            print('loco', i, name, int(g), 'AUC Brier BA min-recall', np.round(vals, 4))
        print('pooled LOCO AUC', i, name, roc_auc_score(y,pooled))
    out = m.calibrate_viscoelastic_cohort(data)
    for g in np.unique(qu['center_id']):
        k = qu['center_id'] == g
        p = out['query_probabilities'][k]
        yy = gt['y_query'][k]
        mae = np.mean(np.abs(out['aligned_query_moduli'][k]-gt['Z_query'][k]))
        auc = roc_auc_score(yy,p)
        brier = np.mean((p-yy)**2)
        print('query', i, int(g), 'margins MAE AUC Brier', .22-mae, auc-.88, .16-brier)
        assert mae <= .22 and auc >= .88 and brier <= .16
for name, rows in summaries.items():
    a = np.asarray(rows)
    print('worst LOCO', name, a[:,0].min(), a[:,1].max(), a[:,2].min(), a[:,3].min())
