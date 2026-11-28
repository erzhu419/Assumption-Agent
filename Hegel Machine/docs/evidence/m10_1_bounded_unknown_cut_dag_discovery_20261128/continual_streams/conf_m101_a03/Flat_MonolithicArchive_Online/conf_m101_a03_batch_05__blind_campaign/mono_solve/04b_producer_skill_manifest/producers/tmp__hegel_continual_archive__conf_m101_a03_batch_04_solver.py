"""Calibrated regression for the scalar readout and sampled trajectory."""
from itertools import combinations_with_replacement
import numpy as np
from scipy.optimize import least_squares, root

_I, _J = np.array(list(combinations_with_replacement(range(4), 2))).T


def _fit(x, y):
    return np.linalg.lstsq(x, y, rcond=None)[0]


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _products(x):
    return x[:, _I] * x[:, _J]


def _calibration(panel):
    u, v = panel['u'], panel['v']
    d = np.column_stack((_affine(u), _products(u), u**3))
    w = np.ones(len(u))
    for _ in range(30):
        c = _fit(d * w[:, None], v * w[:, None])
        e = np.linalg.norm(d @ c - v, axis=1)
        w = np.minimum(1., max(1.4 * np.median(e), .003) / np.maximum(e, 1e-12))
    keep = e < .003
    c = _fit(d[keep], v[keep])
    for j in range(4):
        support = abs(c[:, j]) > .00025
        c[support, j] = _fit(d[keep][:, support], v[keep, j])
        c[~support, j] = 0.

    return c, keep


def _coordinates(panel, observed):
    c, _ = _calibration(panel)

    def forward(z):
        return np.r_[1., z, z[_I] * z[_J], z**3] @ c

    initial = np.linalg.solve(c[1:5].T, (observed - c[0]).T).T
    return np.array([root(lambda z: forward(z) - row, guess, tol=1e-10).x
                     for row, guess in zip(observed, initial)])


def _operator_b(panel, coordinates):
    u, v = panel['u'], panel['v']

    def design(x, params):
        h = x * np.exp(params[:4]) + params[4:]
        return np.column_stack((_affine(x), h / (1 + h*h)))

    def residual(params):
        d = design(u, params)
        return (d @ _fit(d, v) - v).ravel()

    initial = np.r_[np.full(4, np.log(.87)), np.zeros(4)]
    params = least_squares(residual, initial, bounds=(-3, 3), max_nfev=100).x
    return design(coordinates, params) @ _fit(design(u, params), v)


def solve_batch(study: dict) -> dict:
    train = study['train']
    n = len(train['x'])
    z = _coordinates(study['panel_a'], np.concatenate((train['x'], study['test']['x'])))
    b = _operator_b(study['panel_b'], z)
    state_design = np.column_stack((_affine(z), b))
    scalar_design = np.column_stack((state_design, _products(b)))
    y = scalar_design[n:] @ _fit(scalar_design[:n], train['y'])

    traj = train['traj']
    center = traj.mean(axis=0)
    centered = traj - center
    modes = np.linalg.svd(centered, full_matrices=False)[2][:2].T
    state = center + (state_design[n:] @ _fit(state_design[:n], centered @ modes)) @ modes.T
    last = state[:, -1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), last.mean(), last.std()])}
