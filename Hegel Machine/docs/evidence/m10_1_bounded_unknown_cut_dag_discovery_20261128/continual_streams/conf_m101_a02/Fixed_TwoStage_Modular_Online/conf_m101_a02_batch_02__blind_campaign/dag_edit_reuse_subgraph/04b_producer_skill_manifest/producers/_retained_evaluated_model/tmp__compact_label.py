import numpy as np
from scipy.optimize import least_squares


def _affine(z):
    return np.column_stack((np.ones(len(z)), z))


def _features(z, shared):
    cross = z[:, :2] * z[:, 2:]
    if shared:
        cross = cross.sum(axis=1, keepdims=True)
    return np.column_stack((z, cross))


def _label(z, z_test, y, shared=True, tied=True):
    x, xt = _features(z, shared), _features(z_test, shared)
    lin = _affine(x)
    start = np.linalg.lstsq(lin, y, rcond=None)[0][1:]

    def design(x, p):
        index = x @ p[:-1] + p[-1]
        base = _affine(x) if not tied else np.column_stack((np.ones(len(x)), index))
        return np.column_stack((base, np.tanh(index)))

    def residual(p):
        d = design(x, p)
        return d @ np.linalg.lstsq(d, y, rcond=None)[0] - y

    fits = [least_squares(residual, np.r_[scale * start, 0.0], max_nfev=100)
            for scale in (0.6, 1.0, 1.6)]
    p = min(fits, key=lambda r: r.cost).x
    b = np.linalg.lstsq(design(x, p), y, rcond=None)[0]
    return design(xt, p) @ b
