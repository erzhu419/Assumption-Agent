"""Calibrate the input coordinates and fit separate scalar and time readouts."""
from itertools import combinations_with_replacement

import numpy as np
from scipy.optimize import least_squares, minimize_scalar


def _fit(a, y):
    return np.linalg.lstsq(a, y, rcond=1e-10)[0]


def _aff(*blocks):
    return np.column_stack((np.ones(len(blocks[0])), *blocks))


def _calibration_features(x):
    adj = np.roll(x, -1, axis=1)
    return np.stack((np.ones_like(x), x, adj, x * adj, x**3), axis=-1)


def _coordinates(panel, x):
    u, v = panel['u'], panel['v']
    features = _calibration_features(u)
    clean = np.max(np.abs(v), axis=1) < 3
    for _ in range(2):
        c = np.stack([_fit(features[clean, j], v[clean, j]) for j in range(4)])
        clean = np.max(np.abs((features * c).sum(axis=-1) - v), axis=1) < .01
    z = (x - c[:, 0]) / c[:, 1]
    j = np.arange(4)
    for _ in range(16):
        r = (_calibration_features(z) * c).sum(axis=-1) - x
        if np.max(np.abs(r), initial=0.) < 1e-12:
            break
        jac = np.zeros((len(z), 4, 4))
        jac[:, j, j] = c[:, 1] + c[:, 3] * np.roll(z, -1, axis=1) + 3 * c[:, 4] * z*z
        jac[:, j, (j+1) % 4] = c[:, 2] + c[:, 3] * z
        z -= np.linalg.solve(jac, r[..., None])[..., 0]
    return z


def _operators(panel, z):
    u = panel['u']

    def edges(w):
        return _aff(w[:, [0, 0, 1, 2]] * w[:, [1, 3, 2, 3]])

    f = edges(z) @ _fit(edges(u), panel['v1'])
    v = panel['v2']

    def design(w, rate):
        return _aff(w, np.log1p(rate*w*w))

    def loss(rate):
        a = design(u, rate)
        return np.sum((a @ _fit(a, v) - v)**2)

    rate = minimize_scalar(loss, bounds=(.001, 2.), method='bounded').x
    g = design(z, rate) @ _fit(design(u, rate), v)
    return f, g


def _spatial(study, x):
    z = _coordinates(study['panel_a'], x)
    f, g = _operators(study['panel_c'], z)
    q = np.column_stack([z[:, i]*z[:, j] for i, j in combinations_with_replacement(range(4), 2)])
    return z, f, g, q


def _ridge(a, y, n, penalty):
    a = a[:, 1:] - a[:n, 1:].mean(axis=0)
    a /= np.maximum(a[:n].std(axis=0), 1e-12)
    u, s, vt = np.linalg.svd(a[:n], full_matrices=False)
    return y.mean(axis=0) + (a @ vt.T * (s / (s*s + penalty))) @ (u.T @ y)


def _time_fit(t, trajectories):
    def basis(rates):
        return -np.expm1(-t[:, None]*rates)

    def residual(rates):
        a = basis(rates)
        return (a @ _fit(a, trajectories.T) - trajectories.T).ravel()

    rates = least_squares(residual, [.4, 1.8], bounds=([.05, .8], [.8, 5.])).x
    return basis(rates)


def solve_batch(study: dict) -> dict:
    tr = study['train']
    n = len(tr['x'])
    z, f, g, q = _spatial(study, np.vstack((tr['x'], study['test']['x'])))
    a = _aff(z, q, f*g)
    # Keep individual nearest-neighbor products around the four-channel ring.
    b = _aff(z, q, f*g, f*np.roll(g, -1, axis=1), f*np.roll(g, 1, axis=1))
    y = _ridge(a, tr['y'], n, .0001)[n:]
    h = _time_fit(study['eval_times'], tr['traj'])
    modes = _fit(h, tr['traj'].T).T
    state = _ridge(b, modes, n, .01)[n:] @ h.T
    last = state[:, -1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), last.mean(), last.std()])}
