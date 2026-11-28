import numpy as np
from scipy.linalg import solve


def _center(k, q):
    mean = k.mean(axis=0)
    grand = mean.mean()
    return k - mean - mean[:, None] + grand, q - mean - q.mean(axis=1, keepdims=True) + grand


def _gram(x, q, degree):
    mean = x.mean(axis=0)
    scale = np.maximum(x.std(axis=0), 1e-10)
    x, q = (x - mean) / scale, (q - mean) / scale
    return _center((1 + x @ x.T / x.shape[1]) ** degree,
                   (1 + q @ x.T / x.shape[1]) ** degree)


def run_node(inputs: dict) -> dict:
    p = inputs['node_00']
    state = inputs['node_02']['state_pred']
    k, q = _center(p['K'], p['Kq'])
    values, vectors = np.linalg.eigh(k)
    rank = min(p['sy'].shape[1], len(values))
    root = np.sqrt(np.maximum(values[-rank:], 1e-10))
    vectors = vectors[:, -rank:]
    terms = (
        (.05, _gram(vectors * root, (q @ vectors) / root, 4)),
        (.03, _gram(p['st'][:, :p['sy'].shape[1]], p['qt'][:, :p['sy'].shape[1]], 2)),
        (.03, _gram(p['scores'], state @ p['modes'].T, 1)),
    )
    for weight, (kt, qt) in terms:
        k += weight * kt
        q += weight * qt
    weight = solve(k + .003 * np.eye(len(k)), p['y_centered'], assume_a='pos')
    return {'y_hat': np.asarray(p['y_mean'] + q @ weight)}
