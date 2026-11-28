"""Predict scalar and trajectory responses from fitted calibration panels."""
import numpy as np
from scipy.optimize import least_squares


def _ones(x):
    return np.column_stack((np.ones(len(x)), x))


def _ls(a, b):
    return np.linalg.lstsq(a, b, rcond=None)[0]


def _align(panel):
    u, v = panel['u'], panel['v']
    # Median of pairwise slopes, followed by inlier least squares.
    i, j = np.triu_indices(len(u), 1)
    dx, dy = u[i] - u[j], v[i] - v[j]
    result = []
    for k in range(u.shape[1]):
        valid = abs(dx[:, k]) > 1e-8
        slope = np.median(dy[valid, k] / dx[valid, k])
        r = v[:, k] - u[:, k] * slope
        r -= np.median(r)
        good = abs(r) <= max(.001, 5 * np.median(abs(r)))
        result.append(_ls(_ones(u[good, k:k+1]), v[good, k]))
    return np.asarray(result).T


def _quadratic(z):
    # The reference panel uses the four edges between coordinates {0, 2} and {1, 3}.
    return _ones(z[:, [0, 0, 1, 2]] * z[:, [1, 3, 2, 3]])


def _cosine(u, v, z):
    def design(x, p):
        return _ones(np.c_[x, np.cos(p[0] * x + p[1:])])

    def residual(p):
        f = design(u, p)
        return (f @ _ls(f, v) - v).ravel()

    p = least_squares(residual, [1.27, 0., 0., 0., 0.], max_nfev=150).x
    return design(z, p) @ _ls(design(u, p), v)


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    train = study['train']
    n = len(train['x'])
    offset, scale = _align(study['panel_a'])
    z = (np.vstack((train['x'], study['test']['x'])) - offset) / scale
    panel = study['panel_c']
    u = panel['u']
    c = _quadratic(z) @ _ls(_quadratic(u), panel['v1'])
    d = _cosine(u, panel['v2'], z)
    coordinate_branch = (z[:, :, None] * c[:, None, :]).reshape(len(z), 16)
    base = _ones(np.c_[c, d, coordinate_branch])
    scalar = np.c_[base, c * d]
    response = (d[:, :, None] * c[:, None, :]).reshape(len(z), 16)
    trajectory = np.c_[base, response]
    return {
        'y_hat': scalar[n:] @ _ls(scalar[:n], train['y']),
        'unprojected': trajectory[n:] @ _ls(trajectory[:n], train['traj']),
    }
