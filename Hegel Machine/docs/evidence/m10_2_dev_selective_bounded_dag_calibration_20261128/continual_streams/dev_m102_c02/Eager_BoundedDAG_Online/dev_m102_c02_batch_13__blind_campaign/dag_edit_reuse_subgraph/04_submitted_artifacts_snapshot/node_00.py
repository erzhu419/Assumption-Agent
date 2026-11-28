"""Infer and invert the sensor map, then apply the reference cosine map."""
from itertools import combinations_with_replacement
import numpy as np
from scipy.optimize import least_squares, minimize_scalar


def _powers(x, terms):
    return np.column_stack([np.prod(x[:, term], axis=1) for term in terms])


def _sensor_map(u, v):
    d = u.shape[1]
    terms = [t for n in range(4) for t in combinations_with_replacement(range(d), n)]
    phi = _powers(u, terms)
    deviation = np.max(np.abs(v - np.median(v, axis=0)), axis=1)
    keep = deviation < max(3.0, 2.5 * np.median(deviation))
    if keep.sum() < len(terms):
        keep = np.ones(len(v), dtype=bool)
    b = np.linalg.lstsq(phi[keep], v[keep], rcond=None)[0]
    support = np.abs(b) > 0.005
    support[0] = True
    for j in range(d):
        cols = support[:, j]
        b[~cols, j] = 0
        b[cols, j] = np.linalg.lstsq(phi[keep][:, cols], v[keep, j], rcond=None)[0]
    use = support.any(axis=1)
    return b, [t for t, on in zip(terms, use) if on], b[use]


def _invert(x, b, terms, coef):
    z = np.linalg.lstsq(b[1:1+x.shape[1]].T, (x-b[0]).T, rcond=None)[0].T
    deriv = [[(k, t[:i] + t[i+1:]) for i, v in enumerate(t) if v == axis]
             for axis in range(x.shape[1]) for k, t in enumerate(terms)]
    def jac(row):
        j = np.zeros((x.shape[1], x.shape[1]))
        for axis in range(x.shape[1]):
            for k in range(len(terms)):
                for _, rest in deriv[axis*len(terms)+k]:
                    j[:, axis] += coef[k]*np.prod(row[list(rest)])
        return j
    return np.asarray([least_squares(lambda w: _powers(w[None], terms)[0] @ coef - target,
                                    guess, jac=jac, gtol=1e-11, max_nfev=80).x
                       for target, guess in zip(x, z)]).reshape(x.shape)


def _cosines(x, w):
    return np.column_stack((np.ones(len(x)), np.cos(w*x)))


def run_node(inputs: dict) -> dict:
    s = inputs['__input__']
    u, v = (np.asarray(s['panel_a'][k], dtype=float) for k in ('u', 'v'))
    b, terms, coef = _sensor_map(u, v)
    u, v = (np.asarray(s['panel_b'][k], dtype=float) for k in ('u', 'v'))
    def loss(w):
        a = _cosines(u, w)
        return np.linalg.norm(a @ np.linalg.lstsq(a, v, rcond=None)[0] - v)**2
    mesh = np.linspace(.15, 8, 100)
    i = int(np.argmin([loss(w) for w in mesh]))
    w = minimize_scalar(loss, bounds=(mesh[max(i-1, 0)], mesh[min(i+1, len(mesh)-1)]),
                        method='bounded', options={'xatol': 1e-11}).x
    a = _cosines(u, w)
    mixing = np.linalg.lstsq(a, v, rcond=None)[0]
    def coords(split):
        x = np.asarray(s[split]['x'], dtype=float)
        return _cosines(_invert(x, b, terms, coef), w) @ mixing
    return {'train_coords': coords('train'), 'query_coords': coords('test')}
