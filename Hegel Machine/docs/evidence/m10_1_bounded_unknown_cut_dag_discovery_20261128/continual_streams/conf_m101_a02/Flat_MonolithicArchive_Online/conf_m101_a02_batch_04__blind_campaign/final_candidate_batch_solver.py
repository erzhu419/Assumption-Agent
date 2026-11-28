"""Calibrate the input coordinates and fit compact scalar and trajectory readouts."""

import numpy as np
from scipy.optimize import least_squares


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _fit(design, values):
    return np.linalg.lstsq(design, values, rcond=None)[0]


def _calibrate(panel, measured):
    u, v = panel['u'], panel['v']
    design = _affine(u)
    error = np.linalg.norm(v - design @ _fit(design, v), axis=1)
    keep = error < 4 * np.median(error)
    i = np.arange(u.shape[1])
    j = np.roll(i, -1)
    a = u[keep]
    c = np.column_stack([
        _fit(np.column_stack((np.ones(len(a)), a[:, k], a[:, l],
                              a[:, k] * a[:, l], a[:, k] ** 3)), v[keep, k])
        for k, l in zip(i, j)
    ])
    z = (measured - c[0]) / c[1]
    jac = np.zeros((len(z), len(i), len(i)))
    for _ in range(20):
        b = z[:, j]
        residual = c[0] + c[1] * z + c[2] * b + c[3] * z * b + c[4] * z**3 - measured
        if np.max(np.abs(residual), initial=0) < 1e-12:
            break
        jac[:, i, i] = c[1] + c[3] * b + 3 * c[4] * z**2
        jac[:, i, j] = c[2] + c[3] * z
        z -= np.linalg.solve(jac, residual[..., None])[..., 0]
    return z


def _operator(panel, x):
    u, v = panel['u'], panel['v']

    def design(a, params):
        rate, gain = params
        return _affine(a * (1 - gain * np.exp(-rate * a**2)))

    def residual(params):
        d = design(u, params)
        return (d @ _fit(d, v) - v).ravel()

    params = least_squares(residual, [1., 1.1], bounds=([.15, .2], [4., 3.])).x
    return design(x, params) @ _fit(design(u, params), v)


def _label_design(z):
    a, b, c, d = z.T
    return np.column_stack((_affine(z), a * c + b * d))


def solve_batch(study: dict) -> dict:
    train = study['train']
    n = len(train['x'])
    measured = np.concatenate((train['x'], study['test']['x']))
    z = _operator(study['panel_b'], _calibrate(study['panel_a'], measured))
    label = _label_design(z)
    state = _affine(z)
    y_hat = label[n:] @ _fit(label[:n], train['y'])
    state_pred = state[n:] @ _fit(state[:n], train['traj'])
    last = state_pred[:, -1]
    return {'y_hat': y_hat, 'state_pred': state_pred,
            'diagnostics_vec': np.array([y_hat.mean(), y_hat.std(), last.mean(), last.std()])}
