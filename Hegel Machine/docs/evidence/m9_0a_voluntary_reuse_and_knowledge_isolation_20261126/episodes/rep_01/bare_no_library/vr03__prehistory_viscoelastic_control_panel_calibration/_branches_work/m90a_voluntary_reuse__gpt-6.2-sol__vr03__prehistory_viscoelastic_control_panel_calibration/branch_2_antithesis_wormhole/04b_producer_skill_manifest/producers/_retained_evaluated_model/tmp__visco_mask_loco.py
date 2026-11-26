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
for j, case in enumerate(cases):
    cohort = case['cohort']
    t = cohort['train']
    c, s, y = (t[key] for key in ('center_id', 'stratum', 'y'))
    z = m._align(m._extract(t, np.asarray(cohort['taus']), subsets), t)
    complete = np.isfinite(z).all(axis=1)
    metrics = {'imputed': [], 'complete fit': []}
    masked_losses = {key: [] for key in metrics}
    for held in np.unique(c):
        test = c == held
        train = ~test
        idx, counts = m._cell_index(c[train], s[train])
        mean = m._imputation(z[train], idx, len(counts))
        x = m._risk_design(np.where(np.isnan(z), mean, z), s)
        for name, eligible in [('imputed', train), ('complete fit', train & complete)]:
            idx, counts = m._cell_index(c[eligible], s[eligible])
            if len(counts) != 9:
                print(j, held, name, 'missing complete-case training cells', len(counts))
            w = eligible.sum() / (len(counts) * counts[idx])
            b = m._fit_risk(x[eligible], y[eligible], w)
            p = expit(x[test] @ b)
            error = (p - y[test]) ** 2
            masked_losses[name].extend(error[~complete[test]])
            metrics[name].append((roc_auc_score(y[test], p), error.mean()))
    for name, rows in metrics.items():
        a = np.asarray(rows)
        print(f'cohort {j} {name}: held-center AUC/Brier={np.round(a, 5).tolist()}, worst={a[:, 0].min():.5f}/{a[:, 1].max():.5f}, masked Brier={np.mean(masked_losses[name]):.5f} (n={len(masked_losses[name])})')
