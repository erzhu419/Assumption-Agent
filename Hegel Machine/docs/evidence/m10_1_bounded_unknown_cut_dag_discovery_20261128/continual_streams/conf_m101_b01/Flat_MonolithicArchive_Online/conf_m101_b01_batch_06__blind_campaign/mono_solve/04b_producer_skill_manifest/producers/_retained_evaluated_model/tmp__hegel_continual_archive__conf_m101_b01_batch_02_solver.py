"""Fit the panel-calibrated mixed-product model and its two readouts."""
import numpy as np
from scipy.optimize import least_squares


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _fit(design, target):
    return np.linalg.lstsq(design, target, rcond=None)[0]


def _calibration(panel):
    u, v = panel['u'], panel['v']
    i, j = np.triu_indices(len(u), 1)
    du, dv = u[j] - u[i], v[j] - v[i]
    gain = np.array([np.median(dv[m, k] / du[m, k])
                     for k in range(u.shape[1])
                     for m in [np.abs(du[:, k]) > 0.05]])
    offset = np.median(v - u * gain, axis=0)
    residual = np.linalg.norm(v - u * gain - offset, axis=1)
    keep = residual <= 6 * np.median(residual)
    u, v = u[keep], v[keep]
    uc = u - u.mean(axis=0)
    gain = np.sum(uc * v, axis=0) / np.sum(uc**2, axis=0)
    return gain, v.mean(axis=0) - u.mean(axis=0) * gain


def _products(x):
    return x[:, [0, 0, 1, 2]] * x[:, [1, 3, 2, 3]]


def _coordinates(study, x):
    gain, offset = _calibration(study['panel_a'])
    b, c = study['panel_b'], study['panel_c']
    z = _affine((x - offset) / gain) @ _fit(_affine(b['u']), b['v'])
    q = _affine(_products(z)) @ _fit(_affine(_products(c['u'])), c['v1'])
    return z, q


def _trajectory(design, train_state, test_design, times):
    def basis(log_rates):
        return -np.expm1(-np.exp(log_rates[:, None]) * times)

    coefficients = _fit(design, train_state)
    def project(log_rates):
        modes = basis(log_rates)
        amplitudes = _fit(modes.T, coefficients.T).T
        return amplitudes, modes

    def residual(log_rates):
        amplitudes, modes = project(log_rates)
        return (design @ amplitudes @ modes - train_state).ravel()

    fit = least_squares(residual, np.log([0.4, 1.8]), bounds=(-6, 4))
    amplitudes, modes = project(fit.x)
    return test_design @ amplitudes @ modes


def solve_batch(study: dict) -> dict:
    train, test = study['train'], study['test']
    n = len(train['x'])
    _, q = _coordinates(study, np.concatenate((train['x'], test['x'])))
    state_design = _affine(q)
    scalar_design = np.column_stack((state_design, q[:, :2] * q[:, 2:]))
    y = scalar_design[n:] @ _fit(scalar_design[:n], train['y'])
    state = _trajectory(state_design[:n], train['traj'], state_design[n:],
                        np.asarray(study['eval_times']))
    terminal = state[:, -1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), terminal.mean(), terminal.std()])}
