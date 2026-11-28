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
        # Cyclic nearest-neighbor products and shared odd saturation.
        cross = w * np.roll(w, -1, axis=1)
        saturation = w / np.sqrt(1 + q[0]*w*w)
        return np.c_[np.ones(len(w)), cross, saturation]

    def fit(features, target, initial, bounds, **options):
        def residual(q):
            f = features(u, q)
            return (f @ lin(f, target) - target).ravel()
        q = least_squares(residual, initial, bounds=bounds, **options).x
        return features(z, q) @ lin(features(u, q), target)

    c = fit(softsign, panel['v1'], [.85, -.01], ([.02, -1], [10, 1]))
    d = fit(curvature, panel['v2'], [.85], (.05, 6.), gtol=1e-13)
    return np.c_[c, d]


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    train = study['train']['x']
    n = len(train)
    offset, scale = _align(study['panel_a'])
    z = (np.vstack((train, study['test']['x'])) - offset)/scale
    w = _panel_maps(study['panel_c'], z)
    w = (w-w[:n].mean(0))/np.maximum(w[:n].std(0), .1)
    return {'features': w, 'n_train': n}
