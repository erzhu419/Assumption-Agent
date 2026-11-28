"""Fit panel operators and predict the scalar response and trajectory."""

import numpy as np
from scipy.optimize import minimize_scalar


def _design(*blocks):
    return np.column_stack((np.ones(len(blocks[0])), *blocks))


def _least_squares(x, y):
    return np.linalg.lstsq(x, y, rcond=None)[0]


def _odd(x, rate):
    return _design(-np.sign(x) * np.expm1(-rate * np.abs(x)))


def _even(x, rate):
    squared = x * x
    return _design(x, squared / np.sqrt(1 + rate * squared))


def _fit_operator(u, v, design, bounds):
    def loss(rate):
        g = design(u, rate)
        return np.sum((g @ _least_squares(g, v) - v) ** 2)

    rate = minimize_scalar(loss, bounds=bounds, method='bounded').x
    coefficients = _least_squares(design(u, rate), v)
    return lambda x: design(x, rate) @ coefficients


def _cross(a, b):
    return (a[:, :, None] * b[:, None, :]).reshape(len(a), 16)


def _features(study, x):
    panel = study['panel_b']
    f = _design(x) @ _least_squares(_design(panel['u']), panel['v'])
    panel = study['panel_c']
    c = _fit_operator(panel['u'], panel['v1'], _odd, (.1, 5))(f)
    d = _fit_operator(panel['u'], panel['v2'], _even, (.0001, 2))(f)
    return _design(f, c, d, _cross(f, c)), .3 * _cross(f, d), .3 * _cross(c, d)


def _ridge(g, target, penalty):
    n = len(target)
    training = g[:n]
    normal = training.T @ training + penalty * np.eye(g.shape[1])
    normal[0, 0] -= penalty
    return g[n:] @ np.linalg.solve(normal, training.T @ target)


def solve_batch(study: dict) -> dict:
    train = study['train']
    g, fd, cd = _features(study, np.vstack((train['x'], study['test']['x'])))
    traj = train['traj']
    basis = np.linalg.svd(traj, full_matrices=False)[2][:2]
    target = np.column_stack((train['y'], traj @ basis.T))
    base = _ridge(g, target, .0003)
    # Average interaction readouts, keeping the base model at half the weight.
    prediction = .5 * base + (
        _ridge(np.column_stack((g, fd)), target, .003)
        + _ridge(np.column_stack((g, cd)), target, .0003)
        + _ridge(np.column_stack((g, fd, cd)), target, .001)
    ) / 6
    y_hat = prediction[:, 0]
    state_pred = prediction[:, 1:] @ basis
    terminal = state_pred[:, -1]
    return {'y_hat': y_hat, 'state_pred': state_pred,
            'diagnostics_vec': np.array([y_hat.mean(), y_hat.std(),
                                         terminal.mean(), terminal.std()])}
