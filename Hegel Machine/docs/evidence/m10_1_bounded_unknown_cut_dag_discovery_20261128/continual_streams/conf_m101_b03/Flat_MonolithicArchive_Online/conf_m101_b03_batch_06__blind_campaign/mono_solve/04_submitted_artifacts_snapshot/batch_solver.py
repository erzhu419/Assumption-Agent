"""Calibrated operator readouts with two shared relaxation rates."""

import numpy as np
from scipy.optimize import least_squares, minimize_scalar


def _affine(*blocks):
    return np.column_stack((np.ones(len(blocks[0])), *blocks))


def _fit(design, target):
    return np.linalg.lstsq(design, target, rcond=1e-10)[0]


def _designs(study, x):
    panel = study['panel_a']
    u, v = panel['u'], panel['v']
    clean = np.max(np.abs(v), axis=1) < 3
    offset, scale = np.array([
        _fit(_affine(u[clean, j]), v[clean, j]) for j in range(u.shape[1])
    ]).T
    z = (x - offset) / scale

    panel = study['panel_c']
    u = panel['u']

    def edges(w):
        return _affine(w * np.roll(w, -1, axis=1))

    def cosine(w, rate):
        return _affine(np.cos(rate * w))

    def loss(rate):
        d = cosine(u, rate)
        residual = d @ _fit(d, panel['v2']) - panel['v2']
        return np.sum(residual**2)

    rate = minimize_scalar(loss, bounds=(1.5, 3.5), method='bounded').x
    f = edges(z) @ _fit(edges(u), panel['v1'])
    g = cosine(z, rate) @ _fit(cosine(u, rate), panel['v2'])
    dot = np.sum(f * g, axis=1)
    cross = f[:, 0] * g[:, 1]
    return _affine(f, g, dot), _affine(g, cross), _affine(f, g, dot, cross)


def _trajectory_design(times, rates, slow, fast):
    h = -np.expm1(-times[:, None] * rates)
    d = np.concatenate((slow[:, None, :] * h[None, :, 0, None],
                        fast[:, None, :] * h[None, :, 1, None]), axis=2)
    return d.reshape(len(slow) * len(times), d.shape[2])


def solve_batch(study: dict) -> dict:
    train = study['train']
    test = study['test']['x']
    n = len(train['x'])
    y_design, slow, fast = _designs(study, np.vstack((train['x'], test)))
    y = y_design[n:] @ _fit(y_design[:n], train['y'])

    times = study['eval_times']
    target = train['traj'].ravel()

    def training_design(rates):
        return _trajectory_design(times, rates, slow[:n], fast[:n])

    def residual(rates):
        d = training_design(rates)
        return d @ _fit(d, target) - target

    rates = least_squares(residual, [.4, 1.8], bounds=([.05, .65], [.9, 6])).x
    weights = _fit(training_design(rates), target)
    state = (_trajectory_design(times, rates, slow[n:], fast[n:]) @ weights).reshape(len(test), len(times))
    last = state[:, -1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), last.mean(), last.std()])}
