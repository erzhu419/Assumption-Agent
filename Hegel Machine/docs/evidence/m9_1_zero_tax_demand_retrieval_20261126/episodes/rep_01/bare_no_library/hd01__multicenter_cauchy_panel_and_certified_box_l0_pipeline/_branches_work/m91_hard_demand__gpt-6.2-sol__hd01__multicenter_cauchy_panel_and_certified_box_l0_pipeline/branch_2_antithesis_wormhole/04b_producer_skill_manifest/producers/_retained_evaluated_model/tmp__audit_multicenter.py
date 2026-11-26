import importlib.util
import pickle
from pathlib import Path
import numpy as np

root = Path('/holdout_task')
spec = importlib.util.spec_from_file_location('pipeline', root / 'multicenter_certified_l0.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

def fit(design, y, base):
    def wls(w):
        return np.linalg.lstsq(w[:, None] * design, w * y, rcond=None)[0]
    beta = wls(np.sqrt(base))
    e = y - design @ beta
    scale = np.median(abs(e - np.median(e)))
    for _ in range(200):
        new = wls(np.sqrt(base) / np.hypot(1., (y - design @ beta) / scale))
        if np.linalg.norm(new - beta) <= 1e-10 * (1 + np.linalg.norm(new)):
            return new
        beta = new
    return beta

for k, row in enumerate(pickle.loads((root / 'public_multicenter_l0_studies.pkl').read_bytes())):
    study, gt = row['study'], row['gt']
    c = study['cohort']; y = c['y_potency']; s = c['stratum']
    names, center = np.unique(c['center_id'], return_inverse=True)
    out = mod.solve_multicenter_certified_l0(study)
    z = out['aligned_biomarkers']; obs = np.isfinite(c['X_raw'])
    a = np.c_[np.ones(len(y)), s == 1, s == 2, z]
    indicators = np.eye(len(names))[center, 1:]
    cell = center * 3 + s
    errors = {key: [] for key in ('baseline', 'balanced', 'center_adjusted')}
    for held in range(len(names)):
        train = center != held
        test = ~train
        normal = np.ones(train.sum())
        balance = 1 / np.bincount(cell[train], minlength=3 * len(names))[cell[train]]
        for key, d, w in [('baseline', a, normal), ('balanced', a, balance), ('center_adjusted', np.c_[a, indicators], normal)]:
            beta = fit(d[train], y[train], w)
            e = y[test] - a[test] @ beta[:a.shape[1]]
            # A held center has no fitted offset; score within-stratum contrasts.
            errors[key].extend(float(np.median(abs(v - np.median(v)))) for j in range(3) if (v := e[s[test] == j]).size)
    ref = gt['beta_potency']
    for key, d, w in [('baseline', a, np.ones(len(y))), ('balanced', a, 1 / np.bincount(cell)[cell]), ('center_adjusted', np.c_[a, indicators], np.ones(len(y)))]:
        beta = fit(d, y, w)[3:a.shape[1]]
        if key == 'baseline':
            np.testing.assert_allclose(beta, out['robust_potency_weights'], atol=1e-9)
        print(k, key, 'held-center cell median / worst', np.median(errors[key]), max(errors[key]), 'slope relL2', np.linalg.norm(beta-ref)/np.linalg.norm(ref))
    err = abs(z-gt['Z_true']); cert = out['certified_selection']
    print(k, 'alignment MAE', err.mean(), 'observed contribution', np.where(obs,err,0).mean(), 'imputed contribution', np.where(obs,0,err).mean(), 'missing fraction', (~obs).mean(), 'certificate gap', cert['objective']-cert['certified_lower_bound'])
