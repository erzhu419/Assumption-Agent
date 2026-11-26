import pickle, sys
import numpy as np
sys.path.insert(0, '/holdout_task')
from multicenter_certified_l0 import _align, _potency, solve_multicenter_certified_l0

for i, item in enumerate(pickle.load(open('/holdout_task/public_multicenter_l0_studies.pkl', 'rb'))):
    study, truth = item['study'], item['gt']
    c = study['cohort']; center, strata, y = (c[k] for k in ('center_id', 'stratum', 'y_potency'))
    out = solve_multicenter_certified_l0(study); a, w = _align(c)
    np.testing.assert_array_equal(out['aligned_biomarkers'], a)
    ref_beta = truth['beta_potency']; full = out['robust_potency_weights']
    missing = ~np.isfinite(c['X_raw']); delta = abs(a-truth['Z_true'])
    cells = [(center == g) & (strata == s) for g in np.unique(center) for s in np.unique(strata)]
    np.testing.assert_allclose([w[m].sum() for m in cells], len(a)/len(cells))
    print(i, 'MAE all/observed/missing:', delta.mean(), delta[~missing].mean(), delta[missing].mean())
    for g in np.unique(center):
        mask = center != g
        # Re-estimate imputation on the training centers only. Held-center scanner calibration is not needed to assess training-slope stability.
        train = {key: value[mask] for key, value in c.items()}
        train_a, _ = _align(train)
        b = _potency(train_a, strata[mask], y[mask])
        held = ~mask
        resid = y[held] - a[held] @ b
        per_cell = [float(np.median(resid[strata[held] == s])) for s in np.unique(strata)]
        print(' omitted center', g, 'relative slope error/change', np.linalg.norm(b-ref_beta)/np.linalg.norm(ref_beta), np.linalg.norm(b-full)/np.linalg.norm(full), 'held residual medians by stratum', np.round(per_cell, 3))
    sparse = study['sparse_selection']; X = sparse['X']; G = X.T@X
    z = np.linspace(-sparse['M'], sparse['M'], G.shape[0]); yy = sparse['y']
    direct = .5*np.sum((X@z-yy)**2)
    algebra = .5*(yy@yy) - (X.T@yy)@z + .5*np.diag(G)@(z*z) + np.triu(G, 1)[np.triu_indices(len(z), 1)] @ np.outer(z,z)[np.triu_indices(len(z),1)]
    np.testing.assert_allclose(direct, algebra, atol=1e-12, rtol=0)
    cert = out['certified_selection']; gap = cert['objective']-cert['certified_lower_bound']
    assert abs(cert['objective']-truth['exact_l0_objective'])<=1e-5 and 0<=gap<=1e-5 and 1<=len(cert['subcube_leaves'])<=2048
    print(' public certificate gap/leaves:', gap, len(cert['subcube_leaves']))
