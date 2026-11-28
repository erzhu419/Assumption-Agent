"""Calibrated dual-panel response and two-mode trajectory prediction."""

import numpy as np
from scipy.optimize import least_squares


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _fit(x, y):
    return np.linalg.lstsq(x, y, rcond=None)[0]


def _calibrate(panel, x):
    u, v = panel['u'], panel['v']
    du = u[:, None, :] - u[None, :, :]
    dv = v[:, None, :] - v[None, :, :]
    slope = np.array([np.median(dv[:, :, j][abs(du[:, :, j]) > .2] /
                               du[:, :, j][abs(du[:, :, j]) > .2])
                      for j in range(u.shape[1])])
    offset = np.median(v - u * slope, axis=0)
    residual = np.max(abs(v - u * slope - offset), axis=1)
    clean = residual <= max(.001, 8 * np.median(residual))
    offset, slope = np.array([_fit(_affine(u[clean, j:j+1]), v[clean, j])
                              for j in range(u.shape[1])]).T
    return (x - offset) / slope


def _scaled_rotation(u, v, z):
    u_mean, v_mean = u.mean(axis=0), v.mean(axis=0)
    centered = u - u_mean
    left, singular, right = np.linalg.svd(centered.T @ (v - v_mean))
    matrix = (left @ right) * (singular.sum() / np.sum(centered**2))
    return (z - u_mean) @ matrix + v_mean


def _panel_features(study, x):
    z = _calibrate(study['panel_a'], x)
    c = study['panel_c']
    def saturation(u):
        return -np.sign(u) * np.expm1(-1.8 * np.abs(u))
    def softplus(u):
        return np.logaddexp(0., 2.2 * u)
    h = _scaled_rotation(saturation(c['u']), c['v1'], saturation(z))
    j = _scaled_rotation(softplus(c['u']), c['v2'], softplus(z))
    return h, j


def _time_basis(t, rates):
    return -np.expm1(-t[:, None] * rates)


def _modes(t, traj):
    def residual(rates):
        d = _time_basis(t, rates)
        return (d @ _fit(d, traj.T) - traj.T).ravel()
    rates = least_squares(residual, [.4, 1.8], bounds=([.05, .8], [.8, 4.])).x
    basis = _time_basis(t, rates)
    return basis, _fit(basis, traj.T).T


def _predict(study):
    train = study['train']
    x = np.concatenate((train['x'], study['test']['x']))
    h, j = _panel_features(study, x)
    n = len(train['x'])
    scalar = _affine(np.column_stack((h, j, np.sum(h * j, axis=1))))
    crosses = (h[:, :, None] * j[:, None, :]).reshape(len(x), -1)
    spatial = _affine(np.column_stack((h, j, crosses)))
    time_basis, amplitudes = _modes(study['eval_times'], train['traj'])
    y_hat = scalar[n:] @ _fit(scalar[:n], train['y'])
    state_pred = spatial[n:] @ _fit(spatial[:n], amplitudes) @ time_basis.T
    return y_hat, state_pred


def solve_batch(study: dict) -> dict:
    y_hat, state_pred = _predict(study)
    terminal = state_pred[:, -1]
    return {'y_hat': y_hat, 'state_pred': state_pred,
            'diagnostics_vec': np.array([y_hat.mean(), y_hat.std(),
                                         terminal.mean(), terminal.std()])}
