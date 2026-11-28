import importlib.util
import pickle
from itertools import combinations

import numpy as np

spec = importlib.util.spec_from_file_location('candidate', '/holdout_task/node_00.py')
node = importlib.util.module_from_spec(spec)
spec.loader.exec_module(node)
with open('/holdout_task/public_batch.pkl', 'rb') as f:
    batch = pickle.load(f)
for i, entry in enumerate(batch):
    s = entry['study']
    p = s['panel_a']
    u, v = p['u'], p['v']
    print(f'study {i}: entry={sorted(entry)}, study={sorted(s)}, train={sorted(s["train"])}, panel_a={sorted(p)}')
    coef = node._fit(p)
    b, g, c, k = coef
    n = np.roll(u, -1, axis=1)
    residual = b + g*u + c*(u-1)*n + k*u**3 - v
    order = np.argsort(np.linalg.norm(residual, axis=1))
    clean = order[:-3]
    print('outlier indices and norm gap', order[-3:].tolist(), np.linalg.norm(residual[order[-4]]), np.linalg.norm(residual[order[-3]]))
    x = u[clean]
    y = v[clean]
    nb = np.roll(x, -1, axis=1)
    ones = np.ones_like(x)
    cross = (x-1)*nb
    features = [ones, x, cross, x**3]
    # Pairwise linear contrasts are already spanned by the affine model.
    # Test nonlinear contrasts and rational corrections by held-row prediction.
    variants = {
        'affine': [ones]+[np.broadcast_to(x[:, j:j+1], x.shape) for j in range(x.shape[1])],
        'cyclic': features,
        'separate_neighbor_terms': [ones, x, nb, x*nb, x**3],
        'contrast_correction': features + [(x-nb)**3, (x+nb)**3],
        'rational_correction': features + [cross/(1+(x-nb)**2), x**3/(1+(x+nb)**2)],
    }
    for name, cols in variants.items():
        a = np.stack(cols, axis=-1)
        err = np.empty_like(y)
        for row in range(len(x)):
            tr = np.arange(len(x)) != row
            for j in range(x.shape[1]):
                beta = np.linalg.lstsq(a[tr,j], y[tr,j], rcond=None)[0]
                err[row,j] = a[row,j] @ beta - y[row,j]
        rowerr = np.linalg.norm(err, axis=1)
        outer = np.max(np.abs(x), axis=1) >= np.quantile(np.max(np.abs(x), axis=1), .75)
        print(name, 'LOO row RMS / worst / outer-quartile RMS:', *(f'{z:.8g}' for z in (np.sqrt(np.mean(rowerr**2)), np.max(rowerr), np.sqrt(np.mean(rowerr[outer]**2)))))
