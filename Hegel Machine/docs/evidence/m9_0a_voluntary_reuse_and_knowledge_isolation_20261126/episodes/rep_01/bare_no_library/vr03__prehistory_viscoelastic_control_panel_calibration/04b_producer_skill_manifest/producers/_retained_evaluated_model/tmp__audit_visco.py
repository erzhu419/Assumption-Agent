import importlib.util
import pickle
import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_auc_score

spec = importlib.util.spec_from_file_location('calibration', '/holdout_task/viscoelastic_panel_calibration.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with open('/holdout_task/public_viscoelastic_cohort.pkl', 'rb') as f:
    cases = pickle.load(f)
rng = np.random.default_rng(518)
subsets = np.array([rng.choice(36, 4, replace=False) for _ in range(96)])
for k, case in enumerate(cases):
    cohort = case['cohort']
    taus = np.asarray(cohort['taus'])
    t = cohort['train']
    c, s, y = (t[key] for key in ('center_id', 'stratum', 'y'))
    raw = m._extract(t, taus, subsets)
    aligned = m._align(raw, t)
    # The exact affine basis predicts 32 clean observations. A large residual
    # before the glitch tail would motivate investigating the assumed basis or fit.
    order = []
    for block in (t, cohort['query']):
        restored = m._extract(dict(block, missing_channel_mask=np.zeros_like(block['missing_channel_mask'])), taus, subsets)
        for trace, b in zip(block['traces'], restored):
            r = trace['stress_obs'] - m._design(trace, taus)[:, 1:] @ b
            r = np.sort(np.abs(r - np.median(r)))
            order.append(r[[30, 31, 32]])
    print('cohort', k, 'residual orders 31..33 median/max', np.round(np.median(order, axis=0), 4), np.round(np.max(order, axis=0), 4))
    results = {}
    for h in np.unique(c):
        test = c == h
        train = ~test
        idx, counts = m._cell_index(c[train], s[train])
        w = train.sum() / (len(counts) * counts[idx])
        for name, z in [('raw', raw), ('panel', aligned)]:
            mean = m._imputation(z[train], idx, len(counts))
            x = m._risk_design(np.where(np.isnan(z), mean, z), s)
            p = expit(x[test] @ m._fit_risk(x[train], y[train], w))
            yh, sh = y[test], s[test]
            recalls = [np.mean((p[yh == cl] >= .5) == cl) for cl in (0, 1)]
            cells = []
            for level in np.unique(sh):
                r = sh == level
                if np.unique(yh[r]).size == 2:
                    cells.append((int(level), int(r.sum()), *[np.mean((p[r & (yh == cl)] >= .5) == cl) for cl in (0, 1)]))
            results.setdefault(name, []).append((int(h), roc_auc_score(yh, p), np.mean((yh-p)**2), *recalls))
            print(' ', name, 'held center', h, 'stratum (s,n,recall0,recall1)', [(a,b,round(d,3),round(e,3)) for a,b,d,e in cells])
    for name, values in results.items():
        print(' ', name, '(center,AUC,Brier,recall0,recall1)', np.round(values, 4).tolist())
