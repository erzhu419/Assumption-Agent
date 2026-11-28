"""Cyclic panel calibration and profiled latent-coordinate readouts."""
from itertools import combinations_with_replacement

import numpy as np
from scipy.optimize import minimize_scalar, root


def _aff(x):
    return np.column_stack((np.ones(len(x)), x))


def _ls(x, y):
    return np.linalg.lstsq(x, y, rcond=None)[0]


def _cal_basis(x):
    next_x = np.roll(x, -1, axis=-1)
    return np.stack((np.ones_like(x), x, next_x, x * next_x, x**3), axis=-1)


def _coordinates(study, x):
    panel = study['panel_a']
    u, v = panel['u'], panel['v']
    magnitude = np.linalg.norm(v, axis=1)
    med = np.median(magnitude)
    mad = np.median(np.abs(magnitude - med))
    keep = magnitude <= med + 10 * max(mad, 1e-6)
    basis = _cal_basis(u)
    for _ in range(3):
        coef = np.array([_ls(basis[keep, j], v[keep, j]) for j in range(u.shape[1])])
        residual = np.linalg.norm(np.sum(basis * coef, axis=-1) - v, axis=1)
        keep = residual <= max(10 * np.median(residual), 1e-3)
    def forward(z):
        return np.sum(_cal_basis(z) * coef, axis=-1)
    def inverse(row):
        fit = root(lambda z: forward(z) - row, row, tol=1e-10)
        if np.max(np.abs(fit.fun)) > 1e-7:
            from scipy.optimize import least_squares
            fit = least_squares(lambda z: forward(z) - row, row)
        return fit.x
    z = np.array([inverse(row) for row in x])
    p = study['panel_b']
    return _aff(z) @ _ls(_aff(p['u']), p['v'])


def _design(w, gain, kind):
    if kind == 'scalar':
        h = np.arcsinh(gain * w)
        quadratic = [h[:, i] * h[:, j] for i, j in combinations_with_replacement(range(w.shape[1]), 2)]
        return np.column_stack((_aff(h), *quadratic, w))
    h = (2 / np.sqrt(3)) * np.sinh(np.arcsinh((3 * np.sqrt(3) / 2) * gain * w) / 3)
    return np.column_stack((_aff(h), w))


def _readout(w, target, kind):
    n = len(target)
    def loss(gain):
        f = _design(w[:n], gain, kind)
        return np.mean((f @ _ls(f, target) - target)**2)
    bounds = (0.3, 9.) if kind == 'scalar' else (0.25, 4.)
    gain = minimize_scalar(loss, bounds=bounds, method='bounded').x
    f = _design(w, gain, kind)
    return f[n:] @ _ls(f[:n], target)


def solve_batch(study: dict) -> dict:
    train, test = study['train'], study['test']
    w = _coordinates(study, np.concatenate((train['x'], test['x']), axis=0))
    y = _readout(w, train['y'], 'scalar')
    state = _readout(w, train['traj'], 'trajectory')
    tail = state[:, -1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), tail.mean(), tail.std()])}
