"""Fit the calibrated spatial cascade and its scalar and trajectory readouts."""
from itertools import combinations_with_replacement

import numpy as np
from scipy.optimize import least_squares


def _ls(a, b):
    return np.linalg.lstsq(a, b, rcond=None)[0]


def _aff(x):
    return np.column_stack((np.ones(len(x)), x))


def _calibrate(panel, x):
    params = []
    for u, v in zip(panel['u'].T, panel['v'].T):
        d = _aff(u)
        fit = _ls(d, v)
        for _ in range(10):
            ii = np.argsort(np.abs(d @ fit - v))[:max(2, int(0.84 * len(u)))]
            fit = _ls(d[ii], v[ii])
        residual = np.abs(d @ fit - v)
        ii = residual <= max(.002, 10 * np.median(residual))
        params.append(_ls(d[ii], v[ii]))
    offset, scale = np.asarray(params).T
    return (x - offset) / scale


def _dead(x, p):
    return _aff(np.sign(x) * np.maximum(np.abs(x) - p[0], 0.) ** p[1])


def _interaction(x):
    return _aff(np.column_stack([x[:, i] * x[:, j] for i, j in ((0, 1), (0, 3), (1, 2), (2, 3))]))


def _latent(study, x):
    z = _calibrate(study['panel_a'], x)
    b, c = study['panel_b'], study['panel_c']
    def residual(p):
        d = _dead(b['u'], p)
        return (d @ _ls(d, b['v']) - b['v']).ravel()
    p = least_squares(residual, [.35, 1.5], bounds=([0, .25], [.8, 4])).x
    w = _dead(z, p) @ _ls(_dead(b['u'], p), b['v'])
    return _interaction(w) @ _ls(_interaction(c['u']), c['v1'])


def _quadratic(z):
    return np.column_stack((_aff(z), *[z[:, i] * z[:, j] for i, j in combinations_with_replacement(range(z.shape[1]), 2)]))


def solve_batch(study: dict) -> dict:
    train, test = study['train'], study['test']
    n = len(train['x'])
    z = _latent(study, np.concatenate((train['x'], test['x']), axis=0))
    d = _quadratic(z)
    # Keep the two dominant interactions and shrink the remaining curvature.
    pair = d[:, [0, 1, 2, 3, 4, 7, 11]]  # affine terms, z0*z2, z1*z3
    sparse = pair[n:] @ _ls(pair[:n], train['y'])
    full = d[n:] @ _ls(d[:n], train['y'])
    y = sparse + .25 * (full - sparse)
    a = _aff(z)
    coeff = _ls(a[:n], train['traj'])
    modes = np.linalg.svd(a[:n] @ coeff, full_matrices=False)[2][:2]
    state = ((a[n:] @ coeff) @ modes.T) @ modes
    tail = state[:, -1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), tail.mean(), tail.std()])}
