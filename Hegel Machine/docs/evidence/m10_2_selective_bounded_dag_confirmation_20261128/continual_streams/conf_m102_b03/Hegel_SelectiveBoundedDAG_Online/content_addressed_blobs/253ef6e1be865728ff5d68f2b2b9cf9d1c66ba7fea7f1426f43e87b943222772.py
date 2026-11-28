"""Reconstruct the calibrated branch features consumed by node_01."""
import numpy as np
from scipy.optimize import minimize_scalar


def _ones(x):
    return np.column_stack((np.ones(len(x)), x))


def _ls(x, y):
    return np.linalg.lstsq(x, y, rcond=None)[0]


def _calibrate(panel, x):
    u, v = panel['u'], panel['v']
    i, j = np.triu_indices(len(u), 1)
    coefficients = []
    for k in range(u.shape[1]):
        d = u[i, k] - u[j, k]
        valid = np.abs(d) > 1e-8
        slope = np.median((v[i[valid], k] - v[j[valid], k]) / d[valid])
        residual = v[:, k] - slope * u[:, k]
        residual -= np.median(residual)
        good = np.abs(residual) < max(.001, 6 * np.median(np.abs(residual)))
        coefficients.append(_ls(_ones(u[good, k:k+1]), v[good, k]))
    offset, scale = np.array(coefficients).T
    return (x - offset) / scale


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    x = np.vstack((study['train']['x'], study['test']['x']))
    z = _calibrate(study['panel_a'], x)
    b, c = study['panel_b'], study['panel_c']
    linear = _ones(z) @ _ls(_ones(b['u']), b['v'])
    u, v1, v2 = c['u'], c['v1'], c['v2']

    def first(w, k):
        return _ones(np.c_[w, w / (1 + k * np.abs(w))])

    def residual(k):
        f = first(u, k)
        return np.sum((f @ _ls(f, v1) - v1)**2)

    k = minimize_scalar(residual, bounds=(.15, 3.), method='bounded').x
    q = first(z, k) @ _ls(first(u, k), v1)
    i, j = np.triu_indices(z.shape[1], 1)

    def second(w):
        return _ones(np.c_[w, np.cos(2.4 * w), np.sin(2.4 * w), w[:, i] * w[:, j]])

    r = second(z) @ _ls(second(u), v2)
    return {'z': z, 'L': linear, 'Q': q, 'R': r, 'product': q, 'cosine': r}
