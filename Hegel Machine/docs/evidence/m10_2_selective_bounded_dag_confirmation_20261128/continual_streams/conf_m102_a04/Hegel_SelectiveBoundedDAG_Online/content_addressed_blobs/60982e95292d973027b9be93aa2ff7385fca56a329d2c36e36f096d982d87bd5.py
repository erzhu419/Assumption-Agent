import numpy as np
from scipy.optimize import least_squares


def _linear(design, target):
    return np.linalg.lstsq(design, target, rcond=1e-9)[0]


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _alignment(panel):
    u, v = np.asarray(panel['u'], float), np.asarray(panel['v'], float)
    ij = np.triu_indices(len(u), 1)
    params = []
    for a, b in zip(u.T, v.T):
        da = (a[:, None] - a[None, :])[ij]
        db = (b[:, None] - b[None, :])[ij]
        valid = np.abs(da) > 1e-8
        slope = np.median(db[valid] / da[valid])
        intercept = np.median(b - slope * a)
        f = _affine(a)
        for _ in range(5):
            error = np.abs(b - intercept - slope * a)
            keep = error <= max(5 * np.median(error), 0.00035)
            intercept, slope = _linear(f[keep], b[keep])
        params.append((intercept, slope))
    return np.asarray(params).T


def _couplings(x):
    # The four supported cross terms form a cycle; opposite pairs are absent.
    a, b, c, d = x.T
    return np.column_stack((np.ones(len(x)), a*b, a*d, b*c, c*d))


def _curvature(x, q):
    x2 = x*x
    return np.column_stack((_affine(x), x2, x2/(1 + q*x2)))


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    train = np.asarray(study['train']['x'], float)
    test = np.asarray(study['test']['x'], float)
    offset, scale = _alignment(study['panel_a'])
    z = (np.vstack((train, test)) - offset) / scale
    panel = study['panel_c']
    u = np.asarray(panel['u'], float)
    v1 = np.asarray(panel['v1'], float)
    v2 = np.asarray(panel['v2'], float)
    c = _couplings(z) @ _linear(_couplings(u), v1)

    def residual(q):
        f = _curvature(u, q[0])
        return (f @ _linear(f, v2) - v2).ravel()

    q = least_squares(residual, [.25], bounds=(.08, .65), gtol=1e-13).x[0]
    d = _curvature(z, q) @ _linear(_curvature(u, q), v2)
    w = np.column_stack((c, d))
    reference = w[:len(train)]
    return {'features': (w - reference.mean(axis=0)) / np.maximum(reference.std(axis=0), .1)}
