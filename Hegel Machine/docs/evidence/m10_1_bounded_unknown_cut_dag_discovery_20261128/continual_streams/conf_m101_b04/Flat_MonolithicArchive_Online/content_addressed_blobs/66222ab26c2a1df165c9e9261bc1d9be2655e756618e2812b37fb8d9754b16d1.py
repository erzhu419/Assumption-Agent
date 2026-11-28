"""Invert the reference calibration and fit readouts of the panel B operator."""
from itertools import combinations_with_replacement
import numpy as np
from scipy.optimize import least_squares, root

def _fit(design, target):
    return np.linalg.lstsq(design, target, rcond=None)[0]

def _affine(z):
    return np.column_stack((np.ones(len(z)), z))

def _calibrate(panel, observed):
    u, v = panel['u'], panel['v']
    next_u = np.roll(u, -1, axis=1)
    designs = [np.column_stack((np.ones(len(u)), a, b, a * b, a**3))
               for a, b in zip(u.T, next_u.T)]
    # The sparse corruptions are much larger than the calibration nonlinearity.
    affine = _affine(u)
    keep = np.arange(len(u))
    for _ in range(3):
        residual = np.linalg.norm(affine @ _fit(affine[keep], v[keep]) - v, axis=1)
        keep = np.flatnonzero(residual < 5 * np.median(residual))
    coef = np.array([_fit(p[keep], v[keep, j]) for j, p in enumerate(designs)]).T

    def forward(z):
        w = np.roll(z, -1)
        return coef[0] + coef[1]*z + coef[2]*w + coef[3]*z*w + coef[4]*z**3

    return np.array([root(lambda z: forward(z) - row, row, tol=1e-10).x
                     for row in observed])

def _panel_b(panel, x):
    u, v = panel['u'], panel['v']

    def features(q, p):
        return _affine(q + p[1] * np.sin(p[0] * q))

    def residual(p):
        a = features(u, p)
        return (a @ _ls(a, v) - v).ravel()

    p = least_squares(residual, [2.85, -.4], bounds=([1.5, -1.], [4.5, -.05])).x
    return features(x, p) @ _ls(features(u, p), v)


def _ls(a, b):
    return np.linalg.lstsq(a, b, rcond=None)[0]


def _scalar_features(z):
    return np.column_stack((_affine(z), z[:, 0] * z[:, 2], z[:, 1] * z[:, 3]))


def solve_batch(study: dict) -> dict:
    train = study['train']
    n = len(train['x'])
    x = np.concatenate((train['x'], study['test']['x']), axis=0)
    z = _panel_b(study['panel_b'], _calibrate(study['panel_a'], x))
    a = _scalar_features(z)
    y = a[n:] @ _ls(a[:n], train['y'])
    a = _affine(z)
    state = a[n:] @ _ls(a[:n], train['traj'])
    end = state[:, -1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), end.mean(), end.std()])}
