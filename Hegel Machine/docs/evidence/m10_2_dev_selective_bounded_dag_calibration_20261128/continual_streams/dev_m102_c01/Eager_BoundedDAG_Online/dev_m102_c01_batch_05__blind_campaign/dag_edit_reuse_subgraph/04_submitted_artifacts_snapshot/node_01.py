import numpy as np
from scipy.linalg import solve
from scipy.spatial.distance import cdist


def _center(k, q):
    m = k.mean(axis=0)
    g = m.mean()
    return k - m - m[:, None] + g, q - m - q.mean(axis=1, keepdims=True) + g


def _standardize(x, q):
    m = x.mean(axis=0)
    s = np.maximum(x.std(axis=0), 1e-10)
    return (x - m) / s, (q - m) / s


def _poly(x, q, degree):
    d = x.shape[1]
    return _center((1 + (x @ x.T) / d) ** degree,
                   (1 + (q @ x.T) / d) ** degree)


def _radial(dx, dq, width):
    return _center(np.exp(-dx / width), np.exp(-dq / width))


def _distances(x, q):
    return cdist(x, x, 'sqeuclidean'), cdist(q, x, 'sqeuclidean')


def run_node(inputs: dict) -> dict:
    p = inputs['node_00']
    state = inputs['node_02']['state_pred']
    k, q = _center(p['K'], p['Kq'])
    values, vectors = np.linalg.eigh(k)
    d = p['sy'].shape[1]
    rank = min(d, len(values))
    root = np.sqrt(np.maximum(values[-rank:], 1e-10))
    vectors = vectors[:, -rank:]
    x, xq = _standardize(vectors * root, (q @ vectors) / root)
    t, tq = _standardize(p['st'][:, :d], p['qt'][:, :d])
    s, sq = _standardize(p['sy'], p['qy'])
    a, aq = _standardize(p['scores'], state @ p['modes'].T)
    dt, dtq = _distances(t, tq)
    for w, (kt, qt) in (
        (.0003852, _poly(x, xq, 3)),
        (.0561139, _poly(x, xq, 4)),
        (.6507786, _radial(*_distances(x, xq), 8.)),
        (.0047207, _radial(dt, dtq, 4.)),
        (.9814442, _radial(dt, dtq, 8.)),
        (.084672, _poly(t, tq, 2)),
        (.0009055, _poly(s, sq, 3)),
        (.0085005, _poly(a, aq, 1)),
    ):
        k += w * kt
        q += w * qt
    alpha = solve(k + 4.7e-6 * np.eye(len(k)), p['y_centered'], assume_a='pos')
    return {'y_hat': np.asarray(p['y_mean'] + q @ alpha)}
