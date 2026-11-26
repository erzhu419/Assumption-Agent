import pickle
import runpy
import contextlib
import io
import numpy as np
from scipy.special import expit, logit
with contextlib.redirect_stdout(io.StringIO()):
    env = runpy.run_path('/tmp/visco_eval.py')
m, cases, auc = (env[k] for k in ('m', 'data', 'auc'))
for i, case in enumerate(cases):
    co, gt = case['cohort'], case['gt']
    q = co['query']
    r = m.calibrate_viscoelastic_cohort(co)
    x, pred = r['aligned_query_moduli'], r['query_probabilities']
    z, y, p = (gt[k] for k in ('Z_query', 'y_query', 'prob_query'))
    mask = q['missing_channel_mask']
    # Diagnostic counterfactuals only: oracle substitutions never enter the estimator.
    eta = logit(p)
    beta = gt['beta']
    aligned_only = expit(eta + np.where(mask, 0, x-z) @ beta)
    imputed_only = expit(eta + np.where(mask, x-z, 0) @ beta)
    oracle_on_estimated = expit(eta + (x-z) @ beta)
    for c in np.unique(q['center_id']):
        w = q['center_id'] == c
        scores = [(round(auc(y[w], v[w]), 5), round(np.mean((v[w]-y[w])**2), 5)) for v in (pred, oracle_on_estimated, aligned_only, imputed_only, p)]
        print(f'cohort {i}, center {c}, missing rows {mask[w].any(axis=1).sum()}: fitted / oracle on estimated inputs / observed alignment error only / imputation error only / full oracle (AUROC, Brier): {scores}')
