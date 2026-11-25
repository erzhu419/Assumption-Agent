import importlib.util
import pickle
from pathlib import Path
from unittest.mock import patch

import numpy as np

path = Path('/holdout_task/prony_rheology.py')
spec = importlib.util.spec_from_file_location('prony_rheology', path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
# Ablate spacing regularization without changing screening, mode selection or integration.
unregularized = {'__name__': 'unregularized'}
source = path.read_text()
assert source.count('for strength in (18, 10, 6, 3):') == 1
exec(compile(source.replace('for strength in (18, 10, 6, 3):', 'for strength in ():'), str(path), 'exec'), unregularized)
with open('/holdout_task/public_rheology.pkl', 'rb') as f:
    public = pickle.load(f)
e = public['experiment']
t, x, y, tt, xx = (e[k] for k in ('train_t', 'train_strain', 'train_stress_raw', 'test_t', 'test_strain'))
D = np.c_[np.ones(len(t)), (t - t.mean()) / np.ptp(t)]
ref = public['gt_test_stress']

def audit(label, namespace):
    fit, screen = (namespace[k] for k in ('_fit', '_robust_screen'))
    rows = []
    for block in np.array_split(np.arange(len(t)), 10):
        train = np.ones(len(t), dtype=bool)
        train[max(0, block[0] - 2):block[-1] + 3] = False
        def fold_screen(A, drift, obs):
            local, s = screen(A[train], drift[train], obs[train])
            keep = np.zeros(len(t), dtype=bool)
            keep[np.flatnonzero(train)[local]] = True
            fold_screen.keep, fold_screen.s = keep, s
            return keep, s
        with patch.dict(namespace, {'_robust_screen': fold_screen}):
            tau, c = fit(t, x, y)
        physical = m._design(t, x, tau) @ c
        keep = fold_screen.keep
        b = np.linalg.lstsq(D[keep], (y - physical)[keep], rcond=None)[0]
        err = (y - physical - D @ b)[block]
        d = m._design(tt, xx, tau) @ c - ref
        rows.append((np.sqrt(np.mean(np.minimum(abs(err), 6*fold_screen.s)**2)), np.linalg.norm(d)/np.linalg.norm(ref), abs(c[0]/public['gt_G_inf'] - 1), c[0]))
    a = np.asarray(rows)
    print(label, 'folds worst clipped RMSE / public stress RRMSE / public modulus relative error:', a[:, :3].max(axis=0), 'modulus range:', a[:, 3].min(), a[:, 3].max(), flush=True)
    result = namespace['fit_prony_and_predict_stress'](e)
    d = result['test_stress'] - ref
    stress = np.linalg.norm(d) / np.linalg.norm(ref)
    worst = max(np.linalg.norm(d[q]) / np.linalg.norm(ref[q]) for q in np.array_split(np.arange(len(ref)), 4))
    modulus = abs(result['G_inf'] / public['gt_G_inf'] - 1)
    print(label, 'full stress RRMSE / worst quartile / modulus relative error:', stress, worst, modulus, flush=True)
    return stress, modulus

s, g = audit('saved', vars(m))
audit('spacing ablation', unregularized)
assert s <= .022 and g <= .06
