import numpy as np
from scipy.optimize import least_squares


def lin(f, y):
    return np.linalg.lstsq(f, y, rcond=1e-9)[0]


def one(z):
    return np.c_[np.ones(len(z)), z]


def _align(panel):
    u = np.asarray(panel['u'], dtype=float)
    v = np.asarray(panel['v'], dtype=float)
    offset, scale = [], []
    for a, b in zip(u.T, v.T):
        F = one(a)
        ij = np.triu_indices(len(a), 1)
        da = (a[:, None] - a)[ij]
        db = (b[:, None] - b)[ij]
        mask = abs(da) > 1e-8
        slope = np.median(db[mask] / da[mask])
        inter = np.median(b - slope*a)
        for _ in range(5):
            r = abs(b - inter - slope*a)
            keep = r <= max(5*np.median(r), 0.00035)
            inter, slope = lin(F[keep], b[keep])
        offset.append(inter)
        scale.append(slope)
    return np.asarray(offset), np.asarray(scale)


def _panel_maps(panel, z):
    u = np.asarray(panel['u'], dtype=float)

    # Both responses share their nonlinear shape parameters across coordinates.
    def softsign(w, q):
        return one(w / (1 + q[0]*abs(w)) + q[1]*w)

    def curvature(w, q):
        w2 = w*w
        return np.c_[one(w), w2, w2/(1+q[0]*w2)]

    def fit(features, target, initial, bounds, **options):
        def residual(q):
            f = features(u, q)
            return (f @ lin(f, target) - target).ravel()
        q = least_squares(residual, initial, bounds=bounds, **options).x
        return features(z, q) @ lin(features(u, q), target)

    c = fit(softsign, panel['v1'], [.85, -.01], ([.02, -1], [10, 1]))
    d = fit(curvature, panel['v2'], [.25], (.08, .65), gtol=1e-13)
    return np.c_[c, d]


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    x, test = study['train']['x'], study['test']['x']
    n = len(x)
    offset, scale = _align(study['panel_a'])
    w = _panel_maps(study['panel_c'], (np.vstack((x, test))-offset)/scale)
    w = (w-w[:n].mean(0))/np.maximum(w[:n].std(0), .1)
    return {'features': w, 'n_train': n}
