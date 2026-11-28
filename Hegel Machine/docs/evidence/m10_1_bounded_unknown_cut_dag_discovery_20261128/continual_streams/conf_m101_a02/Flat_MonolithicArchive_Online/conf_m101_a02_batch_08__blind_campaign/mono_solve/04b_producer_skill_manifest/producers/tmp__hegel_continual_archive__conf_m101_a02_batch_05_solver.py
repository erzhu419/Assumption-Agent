"""Recover calibrated coordinates and predict scalar and trajectory readouts."""
import numpy as np
from scipy.optimize import least_squares


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _fit(d, y):
    return np.linalg.lstsq(d, y, rcond=None)[0]


def _calibrate(panel, x):
    u, v = panel['u'], panel['v']
    distance = np.linalg.norm(u - v, axis=1)
    keep = distance < 4 * np.median(distance)
    offset, gain = np.column_stack([
        _fit(_affine(a[keep]), b[keep]) for a, b in zip(u.T, v.T)
    ])
    return (x - offset) / gain


def _operator_design(x, params):
    rate, gain = params
    return _affine(x * (1 - gain * np.exp(-rate * x**2)))


def _operator(panel):
    u, v = panel['u'], panel['v']

    def residual(params):
        d = _operator_design(u, params)
        return (d @ _fit(d, v) - v).ravel()

    q = least_squares(residual, [1., 1.1], bounds=([.15, .2], [4., 3.])).x
    return q, _fit(_operator_design(u, q), v)


def _label_design(z):
    a, b, c, d = z.T
    pair = a * c + b * d
    contrast = a + b - c - d
    return np.column_stack((_affine(z), pair, np.tanh(contrast),
                            pair / (1 + contrast**2)))


def solve_batch(study: dict) -> dict:
    train = study['train']
    n = len(train['x'])
    x = _calibrate(study['panel_a'], np.concatenate((train['x'], study['test']['x'])))
    params, mixer = _operator(study['panel_b'])
    z = _operator_design(x, params) @ mixer
    label = _label_design(z)
    state = _affine(z)
    y_hat = label[n:] @ _fit(label[:n], train['y'])
    state_pred = state[n:] @ _fit(state[:n], train['traj'])
    terminal = state_pred[:, -1]
    return {'y_hat': y_hat, 'state_pred': state_pred,
            'diagnostics_vec': np.array([y_hat.mean(), y_hat.std(), terminal.mean(), terminal.std()])}
